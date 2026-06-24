import os

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import firmware_path, load_firmware_axi, release_fetch, reset_dut, wait_for_host_irq


@cocotb.test()
async def test_snitch_boot(dut):
    clock = Clock(dut.clk_i, 1, units="ns")
    cocotb.start_soon(clock.start())

    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    await reset_dut(dut)

    fw_path = firmware_path(__file__, "sw/test/boot/boot.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/boot` first."
    await load_firmware_axi(axi_master, fw_path)
    await release_fetch(dut)

    await wait_for_host_irq(dut, timeout_cycles=50000)

    dut._log.info("TEST PASSED: Firmware raised host IRQ")
