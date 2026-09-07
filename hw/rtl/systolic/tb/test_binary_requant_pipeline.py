import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


MODE_ADD = 0
MODE_SUB = 1
MODE_MUL = 2
LANES = 32


def scale_product(value, multiplier, shift, double_round_shift):
    product = value * multiplier
    if shift == 0:
        return product
    product += 1 << (shift - 1)
    if double_round_shift and shift > 31 - double_round_shift:
        offset = 1 << (30 - double_round_shift)
        product += offset if value >= 0 else -offset
    return product >> shift


def reference(lhs, rhs, mode, params):
    result = []
    for lhs_value, rhs_value in zip(lhs, rhs):
        lhs_centered = lhs_value - params["lhs_zero_point"]
        rhs_centered = rhs_value - params["rhs_zero_point"]
        if mode == MODE_MUL:
            combined = lhs_centered * rhs_centered
        else:
            lhs_scaled = scale_product(
                lhs_centered,
                params["lhs_multiplier"],
                params["lhs_shift"],
                params["double_round_shift"],
            )
            rhs_scaled = scale_product(
                rhs_centered,
                params["rhs_multiplier"],
                params["rhs_shift"],
                params["double_round_shift"],
            )
            combined = lhs_scaled + rhs_scaled if mode == MODE_ADD else lhs_scaled - rhs_scaled
        output = scale_product(
            combined,
            params["output_multiplier"],
            params["output_shift"],
            params["double_round_shift"],
        ) + params["output_zero_point"]
        result.append(max(params["clamp_min"], min(params["clamp_max"], output)) & 0xFF)
    return result


def pack(values):
    word = 0
    for lane, value in enumerate(values):
        word |= (value & 0xFF) << (lane * 8)
    return word


async def reset(dut):
    dut.rst_ni.value = 0
    dut.flush_i.value = 0
    dut.in_valid_i.value = 0
    dut.out_ready_i.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def push(dut, lhs, rhs, mode, params):
    dut.lhs_i.value = pack(lhs)
    dut.rhs_i.value = pack(rhs)
    dut.mode_i.value = mode
    for name, value in params.items():
        getattr(dut, f"{name}_i").value = value
    dut.in_valid_i.value = 1
    while True:
        await RisingEdge(dut.clk_i)
        if dut.in_ready_o.value:
            break
    dut.in_valid_i.value = 0


@cocotb.test()
async def binary_requant_pipeline_all_modes_and_backpressure(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, units="ns").start())
    await reset(dut)

    rng = random.Random(0xB1A2)
    cases = []
    for mode in (MODE_ADD, MODE_SUB, MODE_MUL):
        for index in range(6):
            lhs = [rng.randint(-128, 127) for _ in range(LANES)]
            rhs = [rng.randint(-128, 127) for _ in range(LANES)]
            params = {
                "lhs_multiplier": rng.randint(1, 1 << 20),
                "lhs_shift": rng.randint(12, 24),
                "rhs_multiplier": rng.randint(1, 1 << 20),
                "rhs_shift": rng.randint(12, 24),
                "output_multiplier": rng.randint(1, 1 << 20),
                "output_shift": rng.randint(12, 24),
                "lhs_zero_point": rng.randint(-128, 127),
                "rhs_zero_point": rng.randint(-128, 127),
                "output_zero_point": rng.randint(-128, 127),
                "clamp_min": -110 + index,
                "clamp_max": 105 - index,
                "double_round_shift": 20 if index & 1 else 0,
            }
            cases.append((lhs, rhs, mode, params, reference(lhs, rhs, mode, params)))

    expected = []
    received = []
    producer_done = False

    async def producer():
        nonlocal producer_done
        for lhs, rhs, mode, params, result in cases:
            await push(dut, lhs, rhs, mode, params)
            expected.append(result)
        producer_done = True

    async def consumer():
        cycle = 0
        while not producer_done or len(received) < len(expected):
            dut.out_ready_i.value = 0 if cycle % 7 in (2, 3) else 1
            await RisingEdge(dut.clk_i)
            if dut.out_valid_o.value and dut.out_ready_i.value:
                assert not dut.invalid_o.value
                received.append([
                    (int(dut.packed_o.value) >> (lane * 8)) & 0xFF
                    for lane in range(LANES)
                ])
            cycle += 1
            assert cycle < 500

    await cocotb.start(producer())
    await consumer()
    assert received == expected


@cocotb.test()
async def binary_requant_pipeline_flush_and_invalid_config(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, units="ns").start())
    await reset(dut)

    params = {
        "lhs_multiplier": 1,
        "lhs_shift": 0,
        "rhs_multiplier": 1,
        "rhs_shift": 0,
        "output_multiplier": 1,
        "output_shift": 0,
        "lhs_zero_point": 0,
        "rhs_zero_point": 0,
        "output_zero_point": 0,
        "clamp_min": -128,
        "clamp_max": 127,
        "double_round_shift": 0,
    }
    await push(dut, [1] * LANES, [2] * LANES, MODE_ADD, params)
    dut.flush_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.flush_i.value = 0
    dut.out_ready_i.value = 1
    for _ in range(8):
        await RisingEdge(dut.clk_i)
        assert not dut.out_valid_o.value

    params["clamp_min"] = 4
    params["clamp_max"] = -4
    await push(dut, [1] * LANES, [2] * LANES, MODE_ADD, params)
    for _ in range(12):
        await RisingEdge(dut.clk_i)
        if dut.out_valid_o.value:
            assert dut.invalid_o.value
            break
    else:
        raise AssertionError("invalid transaction did not leave the pipeline")
