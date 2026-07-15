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
L2_WEIGHT = 0x80040000
L2_OUTPUT = 0x80050000

def to_i8(value):
    value &= 0xFF
    return value - 256 if value >= 128 else value


def clamp_i8(value):
    return max(-128, min(127, value))


def input_value(row, col, channel):
    return ((row * 5 + col * 11 + channel * 7 + 3) % 17) - 8


def weight_value(kh, kw, channel):
    return ((kh * 3 + kw * 5 + channel * 2 + 1) % 7) - 3


def c32_groups(channels):
    return (channels + 31) // 32


def c32_offset(height, width, channel, row, col):
    pixel = row * width + col
    return (((channel >> 5) * (height * width) + pixel) * 32) + (channel & 31)


def make_input_bytes(height, width, channels):
    data = [0] * (height * width * c32_groups(channels) * 32)
    for row in range(height):
        for col in range(width):
            for channel in range(channels):
                data[c32_offset(height, width, channel, row, col)] = (
                    input_value(row, col, channel) & 0xFF
                )
    return data


def make_weight_bytes(channels):
    data = []
    for group in range(c32_groups(channels)):
        for kh in range(3):
            for kw in range(3):
                for lane in range(32):
                    channel = group * 32 + lane
                    data.append(weight_value(kh, kw, channel) & 0xFF if channel < channels else 0)
    return data


def output_dim(size, stride):
    return ((size - 1) // stride) + 1


def golden_depthwise(height, width, channels, stride=1):
    out_h = output_dim(height, stride)
    out_w = output_dim(width, stride)
    out = [0] * (out_h * out_w * c32_groups(channels) * 32)
    for oh in range(out_h):
        for ow in range(out_w):
            for channel in range(channels):
                acc = 0
                for kh in range(3):
                    ih = (oh * stride) + kh - 1
                    if ih < 0 or ih >= height:
                        continue
                    for kw in range(3):
                        iw = (ow * stride) + kw - 1
                        if iw < 0 or iw >= width:
                            continue
                        acc += input_value(ih, iw, channel) * weight_value(kh, kw, channel)
                out[c32_offset(out_h, out_w, channel, oh, ow)] = clamp_i8(acc) & 0xFF
    return out


async def run_depthwise_case(dut, fw_name, height, width, channels, report_name, stride=1):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    fw_path = firmware_path(__file__, f"sw/test/depthwise_conv/{fw_name}")
    assert os.path.exists(fw_path), "Run `make -C sw/test/depthwise_conv` first."

    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await write_l2_bytes(dut, L2_INPUT, make_input_bytes(height, width, channels))
    await write_l2_bytes(dut, L2_WEIGHT, make_weight_bytes(channels))
    await release_fetch(dut, axi_master=axi_master)
    await wait_for_host_irq(
        dut,
        timeout_cycles=500000,
        axi_master=axi_master,
        report_name=report_name,
    )

    expected = golden_depthwise(height, width, channels, stride)
    got = await read_l2_bytes(dut, L2_OUTPUT, len(expected))
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"depthwise C32 output byte {idx}: "
            f"got={to_i8(got_byte)} expected={to_i8(exp_byte)}"
        )


@cocotb.test()
async def test_depthwise_conv3x3s1p1_c32_requant(dut):
    await run_depthwise_case(
        dut,
        "depthwise_conv.bin",
        height=48,
        width=48,
        channels=32,
        report_name="test_depthwise_conv3x3s1p1_c32_requant",
        stride=1,
    )


@cocotb.test()
async def test_depthwise_conv3x3s1p1_c64_requant(dut):
    await run_depthwise_case(
        dut,
        "depthwise_conv_c64.bin",
        height=24,
        width=24,
        channels=64,
        report_name="test_depthwise_conv3x3s1p1_c64_requant",
        stride=1,
    )


@cocotb.test()
async def test_depthwise_conv3x3s1p1_c64_48_requant(dut):
    await run_depthwise_case(
        dut,
        "depthwise_conv_c64_48.bin",
        height=48,
        width=48,
        channels=64,
        report_name="test_depthwise_conv3x3s1p1_c64_48_requant",
        stride=1,
    )


@cocotb.test()
async def test_depthwise_conv3x3s1p1_c96_48_requant(dut):
    await run_depthwise_case(
        dut,
        "depthwise_conv_c96_48.bin",
        height=48,
        width=48,
        channels=96,
        report_name="test_depthwise_conv3x3s1p1_c96_48_requant",
        stride=1,
    )


@cocotb.test()
async def test_depthwise_conv3x3s2p1_c32_requant(dut):
    await run_depthwise_case(
        dut,
        "depthwise_conv_s2.bin",
        height=48,
        width=48,
        channels=32,
        report_name="test_depthwise_conv3x3s2p1_c32_requant",
        stride=2,
    )
