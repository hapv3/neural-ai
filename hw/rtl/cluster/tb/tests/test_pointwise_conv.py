import os

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


L2_INPUT = 0x80000000
L2_WEIGHT = 0x80020000
L2_OUTPUT = 0x80030000

H = 48
W = 48
C = 32


def to_i8(value):
    value &= 0xFF
    return value - 256 if value >= 128 else value


def clamp_i8(value):
    return max(-128, min(127, value))


def c32_groups(channels):
    return (channels + 31) // 32


def c32_offset(rows, channel, row):
    return (((channel >> 5) * rows + row) * 32) + (channel & 31)


def input_value(row, channel):
    return ((row * 11 + channel * 7 + 3) % 9) - 4


def weight_value(k, n):
    return ((k * 5 + n * 3 + 1) % 7) - 3


def make_input_bytes(height, width, input_c):
    rows = height * width
    data = [0] * (rows * c32_groups(input_c) * 32)
    for row in range(rows):
        for channel in range(input_c):
            data[c32_offset(rows, channel, row)] = input_value(row, channel) & 0xFF
    return data


def make_weight_bytes(input_c, output_c):
    data = []
    for ocg in range(c32_groups(output_c)):
        for icg in range(c32_groups(input_c)):
            for k_lane in range(32):
                k = icg * 32 + k_lane
                for n_lane in range(32):
                    n = ocg * 32 + n_lane
                    value = weight_value(k, n) if k < input_c and n < output_c else 0
                    data.append(value & 0xFF)
    return data


def golden_pointwise(height, width, input_c, output_c):
    rows = height * width
    out = [0] * (rows * c32_groups(output_c) * 32)
    for row in range(rows):
        for n in range(output_c):
            acc = 0
            for k in range(input_c):
                acc += input_value(row, k) * weight_value(k, n)
            out[c32_offset(rows, n, row)] = clamp_i8(acc) & 0xFF
    return out


async def run_pointwise_case(dut, *, name, firmware, height, width, input_c, output_c, timeout_cycles):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    fw_path = firmware_path(__file__, f"sw/test/pointwise_conv/{firmware}")
    assert os.path.exists(fw_path), "Run `make -C sw/test/pointwise_conv` first."

    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await write_l2_bytes(dut, L2_INPUT, make_input_bytes(height, width, input_c))
    await write_l2_bytes(dut, L2_WEIGHT, make_weight_bytes(input_c, output_c))
    await release_fetch(dut, axi_master=axi_master)
    await wait_for_host_irq(
        dut,
        timeout_cycles=timeout_cycles,
        axi_master=axi_master,
        report_name=name,
    )

    expected = golden_pointwise(height, width, input_c, output_c)
    got = await read_l2_bytes(dut, L2_OUTPUT, len(expected))
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"{name} output byte {idx}: "
            f"got={to_i8(got_byte)} expected={to_i8(exp_byte)}"
        )


@cocotb.test()
async def test_pointwise_conv1x1_c32_requant(dut):
    await run_pointwise_case(
        dut,
        name="test_pointwise_conv1x1_c32_requant",
        firmware="pointwise_conv.bin",
        height=H,
        width=W,
        input_c=C,
        output_c=C,
        timeout_cycles=300000,
    )


@cocotb.test()
async def test_pointwise_conv1x1_c64_c128_requant(dut):
    await run_pointwise_case(
        dut,
        name="test_pointwise_conv1x1_c64_c128_requant",
        firmware="pointwise_conv_c64_c128.bin",
        height=16,
        width=16,
        input_c=64,
        output_c=128,
        timeout_cycles=2000000,
    )
