import os

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    NPU_DTCM_BASE,
    PASS_SIGNATURE,
    firmware_path,
    load_firmware_axi,
    read_dtcm_word,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
)


@cocotb.test()
async def test_pmu_basic(dut):
    clock = Clock(dut.clk_i, 1, units="ns")
    cocotb.start_soon(clock.start())

    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    await reset_dut(dut)

    fw_path = firmware_path(__file__, "sw/test/pmu/pmu.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/pmu` first."
    await load_firmware_axi(axi_master, fw_path)
    await release_fetch(dut, axi_master=axi_master)

    report = await wait_for_host_irq(
        dut,
        timeout_cycles=100000,
        axi_master=axi_master,
        report_name="test_pmu_basic",
    )

    status = read_dtcm_word(dut, NPU_DTCM_BASE)
    if status != PASS_SIGNATURE:
        fail_test = read_dtcm_word(dut, NPU_DTCM_BASE + 0x08)
        got = read_dtcm_word(dut, NPU_DTCM_BASE + 0x10)
        expected = read_dtcm_word(dut, NPU_DTCM_BASE + 0x14)
        raise AssertionError(
            f"PMU firmware failed: status=0x{status:08x} "
            f"test={fail_test} got=0x{got:08x} expected=0x{expected:08x}"
        )

    assert report["cycle"] > 0
    assert report["snitch_tcdm_req"] > 0
    assert report["tcdm_req"] > 0
    assert report["tcdm_read_req"] > 0
    assert report["tcdm_write_req"] > 0

    dut._log.info("TEST PASSED: host AXI-Lite PMU counters are programmable and non-zero")
