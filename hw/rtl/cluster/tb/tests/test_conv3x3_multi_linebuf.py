import os
import struct

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
L2_WEIGHT_SPLIT = 0x80010000
L2_WEIGHT_FUSED = 0x80020000
L2_OUT_SPLIT = 0x80030000
L2_OUT_FUSED = 0x80040000
L2_STATS_SPLIT = 0x80050000
L2_STATS_FUSED = 0x80050100
L2_MODE = 0x80050200

H = 24
W = 24
IC = 64
OC = 64
C32 = 32
ROWS = H * W
STATS_WORDS = 12


def to_i8(value):
    value &= 0xFF
    return value - 256 if value >= 128 else value


def clamp_i8(value):
    return max(-128, min(127, value))


def c32_groups(channels):
    return (channels + 31) // 32


def c32_offset(height, width, channel, row, col):
    pixel = row * width + col
    return (((channel >> 5) * (height * width) + pixel) * C32) + (channel & 31)


def input_value(row, col, channel):
    return ((row * 3 + col * 5 + channel * 7 + 1) % 5) - 2


def weight_value(kh, kw, ic, oc):
    return ((kh * 11 + kw * 5 + ic * 3 + oc * 7 + 4) % 3) - 1


def make_input_bytes():
    data = [0] * (H * W * c32_groups(IC) * C32)
    for row in range(H):
        for col in range(W):
            for channel in range(IC):
                data[c32_offset(H, W, channel, row, col)] = input_value(row, col, channel) & 0xFF
    return data


def make_weight_split_bytes():
    data = []
    for ocg in range(c32_groups(OC)):
        for icg in range(c32_groups(IC)):
            for kh in range(3):
                for kw in range(3):
                    for lane in range(C32):
                        ic = icg * C32 + lane
                        for n_lane in range(C32):
                            oc = ocg * C32 + n_lane
                            data.append(weight_value(kh, kw, ic, oc) & 0xFF)
    return data


def make_weight_fused_bytes():
    data = []
    for ocg in range(c32_groups(OC)):
        for icg in range(c32_groups(IC)):
            for kh in range(3):
                for kw in range(3):
                    for lane in range(C32):
                        ic = icg * C32 + lane
                        for n_lane in range(C32):
                            oc = ocg * C32 + n_lane
                            data.append(weight_value(kh, kw, ic, oc) & 0xFF)
    return data


def golden_conv3x3():
    out = [0] * (H * W * c32_groups(OC) * C32)
    for oh in range(H):
        for ow in range(W):
            for oc in range(OC):
                acc = 0
                for kh in range(3):
                    ih = oh + kh - 1
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(3):
                        iw = ow + kw - 1
                        if iw < 0 or iw >= W:
                            continue
                        for ic in range(IC):
                            acc += input_value(ih, iw, ic) * weight_value(kh, kw, ic, oc)
                out[c32_offset(H, W, oc, oh, ow)] = clamp_i8(acc) & 0xFF
    return out


def parse_stats(raw):
    fields = struct.unpack("<" + "I" * STATS_WORDS, bytes(raw))
    keys = [
        "rows",
        "k_tiles",
        "prepare_cycles",
        "gemm_cycles",
        "total_cycles",
        "last_prepare_cycles",
        "last_gemm_cycles",
        "status",
        "prepare_idma_tiles",
        "prepare_idma_transfers",
        "prepare_spatz_tiles",
        "prepare_scalar_tiles",
    ]
    return dict(zip(keys, fields))


async def check_output(dut, name, addr, expected):
    got = await read_l2_bytes(dut, addr, len(expected))
    mismatches = []
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        if got_byte != exp_byte:
            mismatches.append((idx, to_i8(got_byte), to_i8(exp_byte)))
            if len(mismatches) == 8:
                break
    if mismatches:
        dut._log.warning("%s first mismatches: %s", name, mismatches)
        idx, got_byte, exp_byte = mismatches[0]
        assert got_byte == exp_byte, (
            f"{name} byte {idx}: got={got_byte} expected={exp_byte}"
        )


async def run_mode(dut, axi_master, mode):
    fw_path = firmware_path(__file__, "sw/test/conv3x3_multi_linebuf/conv3x3_multi_linebuf.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/conv3x3_multi_linebuf` first."

    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await write_l2_bytes(dut, L2_INPUT, make_input_bytes())
    await write_l2_bytes(dut, L2_WEIGHT_SPLIT, make_weight_split_bytes())
    await write_l2_bytes(dut, L2_WEIGHT_FUSED, make_weight_fused_bytes())
    await write_l2_bytes(dut, L2_MODE, list(int(mode).to_bytes(4, "little")))
    await release_fetch(dut, axi_master=axi_master)
    counters = await wait_for_host_irq(
        dut,
        timeout_cycles=3000000,
        axi_master=axi_master,
        report_name=f"test_conv3x3_c64_oc64_mode{mode}",
    )
    return counters


@cocotb.test()
async def test_conv3x3_c64_oc64_split_vs_fused_linebuf(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    expected = golden_conv3x3()
    split_pmu = await run_mode(dut, axi_master, 0)
    split_head = await read_l2_bytes(dut, L2_OUT_SPLIT, C32)
    split_stats = parse_stats(await read_l2_bytes(dut, L2_STATS_SPLIT, STATS_WORDS * 4))
    fused_stats = parse_stats(await read_l2_bytes(dut, L2_STATS_FUSED, STATS_WORDS * 4))
    dut._log.info("conv3x3 expected head: %s", [to_i8(v) for v in expected[:C32]])
    dut._log.info("conv3x3 split head:    %s", [to_i8(v) for v in split_head])
    dut._log.info("conv3x3 split stats: %s", split_stats)
    await check_output(dut, "split", L2_OUT_SPLIT, expected)

    fused_pmu = await run_mode(dut, axi_master, 1)
    fused_head = await read_l2_bytes(dut, L2_OUT_FUSED, C32)
    fused_stats = parse_stats(await read_l2_bytes(dut, L2_STATS_FUSED, STATS_WORDS * 4))
    dut._log.info("conv3x3 fused head:    %s", [to_i8(v) for v in fused_head])
    dut._log.info("conv3x3 fused stats: %s", fused_stats)
    await check_output(dut, "fused", L2_OUT_FUSED, expected)

    assert fused_pmu["cycle"] < split_pmu["cycle"], (
        f"fused path did not improve total_cycles: "
        f"split={split_pmu['cycle']} fused={fused_pmu['cycle']}"
    )
