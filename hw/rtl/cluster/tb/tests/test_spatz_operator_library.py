import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
    read_tcdm_byte,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
)


DST_I8 = 0x10100100
RELU_I8 = 0x10100200
DST_REQUANT = 0x10100500
VL = 32


def as_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def check_tcdm_outputs(dut):
    for idx in range(VL):
        expected = idx - 16
        got = as_i8(read_tcdm_byte(dut, DST_I8 + idx))
        assert got == expected, f"copy_i8[{idx}] got={got} expected={expected}"

    for idx in range(VL):
        expected = 0 if idx < 12 else idx - 12
        got = as_i8(read_tcdm_byte(dut, RELU_I8 + idx))
        assert got == expected, f"relu_i8[{idx}] got={got} expected={expected}"

    for idx in range(VL):
        src = (idx - 16) * 37
        expected = max(-20, min(31, (src * 2) >> 3))
        got = as_i8(read_tcdm_byte(dut, DST_REQUANT + idx))
        assert got == expected, f"requant[{idx}] got={got} expected={expected}"


@cocotb.test()
async def test_spatz_operator_library(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    await reset_dut(dut)
    await load_firmware_axi(
        axi_master,
        firmware_path(__file__, "sw/test/spatz_ops/spatz_ops_test.bin"),
    )
    await release_fetch(dut)

    await wait_for_host_irq(dut, timeout_cycles=30000)
    check_tcdm_outputs(dut)
