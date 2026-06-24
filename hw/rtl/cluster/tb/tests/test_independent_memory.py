import os

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
    read_l2_bytes,
    read_tcdm_word32,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


L2_SRC = 0x80000000
L2_DST = 0x80001000
L2_SRC_2D = 0x80002000
L2_DST_2D = 0x80003000
L2_SRC_3D = 0x80004000
L2_DST_3D = 0x80005000
TCDM_BANK_LOW = 0x10102000
TCDM_BANK_HIGH = 0x1017E000
DMA_BYTES = 512

DMA_2D_LEN = 8
DMA_2D_REPS = 5
DMA_2D_L2_STRIDE = 13
DMA_2D_TCDM_STRIDE = 16

DMA_2D_OUT_LEN = 6
DMA_2D_OUT_REPS = 6
DMA_2D_OUT_TCDM_STRIDE = 17
DMA_2D_OUT_L2_STRIDE = 11

DMA_3D_LEN = 4
DMA_3D_REPS2 = 3
DMA_3D_REPS3 = 2
DMA_3D_L2_STRIDE2 = 7
DMA_3D_TCDM_STRIDE2 = 8
DMA_3D_L2_STRIDE3 = 40
DMA_3D_TCDM_STRIDE3 = 32

DMA_3D_OUT_LEN = 5
DMA_3D_OUT_REPS2 = 2
DMA_3D_OUT_REPS3 = 3
DMA_3D_OUT_TCDM_STRIDE2 = 9
DMA_3D_OUT_L2_STRIDE2 = 8
DMA_3D_OUT_TCDM_STRIDE3 = 64
DMA_3D_OUT_L2_STRIDE3 = 48


def l2_pattern(index):
    return (index * 13 + 7) & 0xFF


def l2_2d_pattern(index):
    return (index * 17 + 3) & 0xFF


def l2_3d_pattern(index):
    return (index * 29 + 5) & 0xFF


def tcdm_dst_pattern(index):
    return 0xA0 ^ ((index * 5) & 0xFF)


def tcdm_src_2d_pattern(index):
    return (index * 19 + 0x31) & 0xFF


def tcdm_src_3d_pattern(index):
    return (index * 23 + 0x41) & 0xFF


def make_expected_2d_output():
    total = (DMA_2D_OUT_REPS - 1) * DMA_2D_OUT_L2_STRIDE + DMA_2D_OUT_LEN
    expected = [0x5A] * total
    for rep in range(DMA_2D_OUT_REPS):
        for col in range(DMA_2D_OUT_LEN):
            dst_index = rep * DMA_2D_OUT_L2_STRIDE + col
            src_index = rep * DMA_2D_OUT_TCDM_STRIDE + col
            expected[dst_index] = tcdm_src_2d_pattern(src_index)
    return expected


def make_expected_3d_output():
    total = (
        (DMA_3D_OUT_REPS3 - 1) * DMA_3D_OUT_L2_STRIDE3
        + (DMA_3D_OUT_REPS2 - 1) * DMA_3D_OUT_L2_STRIDE2
        + DMA_3D_OUT_LEN
    )
    expected = [0x6B] * total
    for rep3 in range(DMA_3D_OUT_REPS3):
        for rep2 in range(DMA_3D_OUT_REPS2):
            for col in range(DMA_3D_OUT_LEN):
                dst_index = rep3 * DMA_3D_OUT_L2_STRIDE3 + rep2 * DMA_3D_OUT_L2_STRIDE2 + col
                src_index = rep3 * DMA_3D_OUT_TCDM_STRIDE3 + rep2 * DMA_3D_OUT_TCDM_STRIDE2 + col
                expected[dst_index] = tcdm_src_3d_pattern(src_index)
    return expected


@cocotb.test()
async def test_independent_memory(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)

    fw_path = firmware_path(__file__, "sw/test/independent_memory/independent_memory.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/independent_memory` first."
    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)

    await write_l2_bytes(dut, L2_SRC, [l2_pattern(i) for i in range(DMA_BYTES)])
    l2_2d_fixture_len = (DMA_2D_REPS - 1) * DMA_2D_L2_STRIDE + DMA_2D_LEN
    l2_3d_fixture_len = (
        (DMA_3D_REPS3 - 1) * DMA_3D_L2_STRIDE3
        + (DMA_3D_REPS2 - 1) * DMA_3D_L2_STRIDE2
        + DMA_3D_LEN
    )
    await write_l2_bytes(dut, L2_SRC_2D, [l2_2d_pattern(i) for i in range(l2_2d_fixture_len)])
    await write_l2_bytes(dut, L2_SRC_3D, [l2_3d_pattern(i) for i in range(l2_3d_fixture_len)])
    await write_l2_bytes(dut, L2_DST_2D, [0x5A] * len(make_expected_2d_output()))
    await write_l2_bytes(dut, L2_DST_3D, [0x6B] * len(make_expected_3d_output()))
    await release_fetch(dut)

    await wait_for_host_irq(dut, timeout_cycles=100000)
    dut._log.info("memory suite completed through host irq")

    got = await read_l2_bytes(dut, L2_DST, DMA_BYTES)
    expected = [tcdm_dst_pattern(i) for i in range(DMA_BYTES)]
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, f"L2 DMA-out mismatch idx={idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"

    for bank in range(16):
        low = read_tcdm_word32(dut, TCDM_BANK_LOW + bank * 32)
        high = read_tcdm_word32(dut, TCDM_BANK_HIGH + bank * 32)
        assert low == (0x11000000 | bank), f"TCDM low bank {bank} got=0x{low:08x}"
        assert high == (0x22000000 | bank), f"TCDM high bank {bank} got=0x{high:08x}"

    got_2d = await read_l2_bytes(dut, L2_DST_2D, len(make_expected_2d_output()))
    for idx, (got_byte, exp_byte) in enumerate(zip(got_2d, make_expected_2d_output())):
        assert got_byte == exp_byte, f"L2 2D DMA-out mismatch idx={idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"

    got_3d = await read_l2_bytes(dut, L2_DST_3D, len(make_expected_3d_output()))
    for idx, (got_byte, exp_byte) in enumerate(zip(got_3d, make_expected_3d_output())):
        assert got_byte == exp_byte, f"L2 3D DMA-out mismatch idx={idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"
