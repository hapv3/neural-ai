import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from npu_test_utils import (
    firmware_path,
    hold_reset,
    load_firmware_tcm_backdoor,
    read_dtcm_word,
    read_l2_bytes,
    release_reset,
    wait_for_status,
    write_dtcm_word,
    write_l2_bytes,
)
from test_independent_memory import (
    DMA_2D_L2_STRIDE,
    DMA_2D_LEN,
    DMA_2D_REPS,
    DMA_3D_L2_STRIDE2,
    DMA_3D_L2_STRIDE3,
    DMA_3D_LEN,
    DMA_3D_REPS2,
    DMA_3D_REPS3,
    DMA_BYTES,
    L2_DST,
    L2_DST_2D,
    L2_DST_3D,
    L2_SRC,
    L2_SRC_2D,
    L2_SRC_3D,
    SIG_START_WORD,
    l2_2d_pattern,
    l2_3d_pattern,
    l2_pattern,
    make_expected_2d_output,
    make_expected_3d_output,
    tcdm_dst_pattern,
)


@cocotb.test()
async def test_dma_tcm_path(dut):
    """
    Legacy DMA/TCDM smoke gate for the current cluster.

    The old version drove an APB test environment that no longer exists. The
    current architecture boots Snitch from I-TCM, lets firmware program iDMA
    through native 32-bit MMIO, and verifies L2 <-> shared TCDM data movement.
    """
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    await hold_reset(dut)
    fw_path = firmware_path(__file__, "sw/test/independent_memory/independent_memory.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/independent_memory` first."
    load_firmware_tcm_backdoor(dut, fw_path)
    await release_reset(dut)

    for _ in range(100):
        if read_dtcm_word(dut, 6) == 1:
            break
        await ClockCycles(dut.clk_i, 1)
    assert read_dtcm_word(dut, 6) == 1, "firmware did not reach boot/start gate"

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
    write_dtcm_word(dut, SIG_START_WORD, 1)

    debug = await wait_for_status(dut, expected_pass_count=7, timeout_cycles=100000)
    dut._log.info(f"dma/tcm smoke passed: {debug}")

    got = await read_l2_bytes(dut, L2_DST, DMA_BYTES)
    expected = [tcdm_dst_pattern(i) for i in range(DMA_BYTES)]
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, f"L2 DMA-out mismatch idx={idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"
