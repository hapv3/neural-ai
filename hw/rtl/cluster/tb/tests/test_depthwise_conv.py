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
ROWS = H * W
ACTIVATION_BYTES = ROWS * C
WEIGHT_BYTES = 3 * 3 * C


def to_i8(value):
    value &= 0xFF
    return value - 256 if value >= 128 else value


def clamp_i8(value):
    return max(-128, min(127, value))


def input_value(row, col, channel):
    return ((row * 5 + col * 11 + channel * 7 + 3) % 17) - 8


def weight_value(kh, kw, channel):
    return ((kh * 3 + kw * 5 + channel * 2 + 1) % 7) - 3


def make_input_bytes():
    data = []
    for row in range(H):
        for col in range(W):
            for channel in range(C):
                data.append(input_value(row, col, channel) & 0xFF)
    return data


def make_weight_bytes():
    data = []
    for kh in range(3):
        for kw in range(3):
            for channel in range(C):
                data.append(weight_value(kh, kw, channel) & 0xFF)
    return data


def golden_depthwise():
    out = []
    for oh in range(H):
        for ow in range(W):
            for channel in range(C):
                acc = 0
                for kh in range(3):
                    ih = oh + kh - 1
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(3):
                        iw = ow + kw - 1
                        if iw < 0 or iw >= W:
                            continue
                        acc += input_value(ih, iw, channel) * weight_value(kh, kw, channel)
                out.append(clamp_i8(acc) & 0xFF)
    return out


@cocotb.test()
async def test_depthwise_conv3x3s1p1_c32_requant(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    fw_path = firmware_path(__file__, "sw/test/depthwise_conv/depthwise_conv.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/depthwise_conv` first."

    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await write_l2_bytes(dut, L2_INPUT, make_input_bytes())
    await write_l2_bytes(dut, L2_WEIGHT, make_weight_bytes())
    await release_fetch(dut, axi_master=axi_master)
    await wait_for_host_irq(
        dut,
        timeout_cycles=500000,
        axi_master=axi_master,
        report_name="test_depthwise_conv3x3s1p1_c32_requant",
    )

    expected = golden_depthwise()
    got = await read_l2_bytes(dut, L2_OUTPUT, ACTIVATION_BYTES)
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"depthwise C32 output byte {idx}: "
            f"got={to_i8(got_byte)} expected={to_i8(exp_byte)}"
        )
