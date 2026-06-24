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
    dut._log.info("dma/tcm smoke completed through host irq")

    got = await read_l2_bytes(dut, L2_DST, DMA_BYTES)
    expected = [tcdm_dst_pattern(i) for i in range(DMA_BYTES)]
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, f"L2 DMA-out mismatch idx={idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"
