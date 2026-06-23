import cocotb

from systolic_independent_common import (
    boot_and_run_independent_systolic,
    check_independent_systolic_output,
)


@cocotb.test()
async def test_independent_systolic(dut):
    debug = await boot_and_run_independent_systolic(dut, __file__)
    dut._log.info(f"systolic suite passed: {debug}")
    await check_independent_systolic_output(dut)
