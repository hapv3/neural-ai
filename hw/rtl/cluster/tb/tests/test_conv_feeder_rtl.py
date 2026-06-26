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


L2_CONV1_INPUT = 0x80000000
L2_CONV1_WEIGHT = 0x80002000
L2_CONV1_IM2COL0 = 0x80008000
L2_CONV1_IM2COL1 = 0x80009000
L2_CONV1_OUT = 0x80010000

L2_CONV3_INPUT = 0x80020000
L2_CONV3_WEIGHT = 0x80022000
L2_CONV3_OUT = 0x80030000

CONV1_H = 4
CONV1_W = 5
CONV1_C = 33
CONV3_H = 5
CONV3_W = 5
CONV3_C = 3
OC = 32
K_TILE = 32
STATUS_BASE = 0x10008000


def read_dtcm_word(dut, addr):
    index = (addr - STATUS_BASE) >> 2
    value = dut.u_npu_cluster.u_sram_d_tcm.mem[index].value
    return value.to_unsigned() if value.is_resolvable else 0


def to_u8(value):
    return value & 0xFF


def s32_to_bytes(value):
    return [(value >> shift) & 0xFF for shift in (0, 8, 16, 24)]


def bytes_to_s32(data, index):
    value = 0
    for byte_idx in range(4):
        value |= data[(index * 4) + byte_idx] << (byte_idx * 8)
    return value - 0x100000000 if value & 0x80000000 else value


def conv1_input_value(h, w, c):
    return ((h * 17 + w * 11 + c * 5) % 19) - 9


def conv1_weight_value(c, oc):
    return ((c * 7 + oc * 3) % 17) - 8


def conv3_input_value(h, w, c):
    return ((h * 13 + w * 7 + c * 5) % 11) - 5


def conv3_weight_value(kh, kw, c, oc):
    return ((kh * 19 + kw * 11 + c * 7 + oc * 3) % 15) - 7


def make_conv1_input():
    return [
        to_u8(conv1_input_value(h, w, c))
        for h in range(CONV1_H)
        for w in range(CONV1_W)
        for c in range(CONV1_C)
    ]


def make_conv1_weight_packed():
    packed = []
    k_total = CONV1_C
    k_blocks = (k_total + K_TILE - 1) // K_TILE
    for block in range(k_blocks):
        for lane in range(K_TILE):
            c = block * K_TILE + lane
            for oc in range(OC):
                value = conv1_weight_value(c, oc) if c < CONV1_C else 0
                packed.append(to_u8(value))
    return packed


def make_conv3_input():
    return [
        to_u8(conv3_input_value(h, w, c))
        for h in range(CONV3_H)
        for w in range(CONV3_W)
        for c in range(CONV3_C)
    ]


def make_conv3_weight_packed():
    packed = []
    k_total = 3 * 3 * CONV3_C
    for lane in range(K_TILE):
        if lane < k_total:
            spatial = lane // CONV3_C
            c = lane - spatial * CONV3_C
            kh = spatial // 3
            kw = spatial - kh * 3
        for oc in range(OC):
            value = conv3_weight_value(kh, kw, c, oc) if lane < k_total else 0
            packed.append(to_u8(value))
    return packed


def golden_conv1():
    out = []
    for h in range(CONV1_H):
        for w in range(CONV1_W):
            for oc in range(OC):
                acc = 0
                for c in range(CONV1_C):
                    acc += conv1_input_value(h, w, c) * conv1_weight_value(c, oc)
                out.extend(s32_to_bytes(acc & 0xFFFFFFFF))
    return out


def golden_conv1_im2col(block):
    out = []
    for h in range(CONV1_H):
        for w in range(CONV1_W):
            for lane in range(K_TILE):
                c = block * K_TILE + lane
                value = conv1_input_value(h, w, c) if c < CONV1_C else 0
                out.append(to_u8(value))
    return out


def golden_conv3():
    out = []
    for oh in range(CONV3_H):
        for ow in range(CONV3_W):
            for oc in range(OC):
                acc = 0
                for kh in range(3):
                    ih = oh + kh - 1
                    for kw in range(3):
                        iw = ow + kw - 1
                        for c in range(CONV3_C):
                            if 0 <= ih < CONV3_H and 0 <= iw < CONV3_W:
                                acc += conv3_input_value(ih, iw, c) * conv3_weight_value(kh, kw, c, oc)
                out.extend(s32_to_bytes(acc & 0xFFFFFFFF))
    return out


async def boot_and_run(dut, test_file):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)

    fw_path = firmware_path(test_file, "sw/test/conv_feeder_rtl/conv_feeder_rtl.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/conv_feeder_rtl` first."

    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await write_l2_bytes(dut, L2_CONV1_INPUT, make_conv1_input())
    await write_l2_bytes(dut, L2_CONV1_WEIGHT, make_conv1_weight_packed())
    await write_l2_bytes(dut, L2_CONV3_INPUT, make_conv3_input())
    await write_l2_bytes(dut, L2_CONV3_WEIGHT, make_conv3_weight_packed())
    await release_fetch(dut)
    try:
        await wait_for_host_irq(dut, timeout_cycles=450000)
    except AssertionError as exc:
        status = read_dtcm_word(dut, STATUS_BASE + 0x00)
        pass_count = read_dtcm_word(dut, STATUS_BASE + 0x04)
        phase = read_dtcm_word(dut, STATUS_BASE + 0x18)
        op = read_dtcm_word(dut, STATUS_BASE + 0x1C)
        raise AssertionError(
            f"{exc}: status=0x{status:08x} pass_count={pass_count} phase={phase} op={op}"
        ) from exc


@cocotb.test()
async def test_conv_feeder_rtl(dut):
    await boot_and_run(dut, __file__)

    for block, addr in [(0, L2_CONV1_IM2COL0), (1, L2_CONV1_IM2COL1)]:
        expected = golden_conv1_im2col(block)
        got = await read_l2_bytes(dut, addr, len(expected))
        for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
            assert got_byte == exp_byte, (
                f"conv1 im2col block={block} byte {idx}: "
                f"got=0x{got_byte:02x} expected=0x{exp_byte:02x}"
            )

    expected = golden_conv1()
    got = await read_l2_bytes(dut, L2_CONV1_OUT, len(expected))
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"conv1x1 K=33 byte {idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}; "
            f"got_word={bytes_to_s32(got, idx // 4)} expected_word={bytes_to_s32(expected, idx // 4)}"
        )

    expected = golden_conv3()
    got = await read_l2_bytes(dut, L2_CONV3_OUT, len(expected))
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"conv3x3 pad1 byte {idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}; "
            f"got_word={bytes_to_s32(got, idx // 4)} expected_word={bytes_to_s32(expected, idx // 4)}"
        )
