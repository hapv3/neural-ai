import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

async def reset_dut(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, units="ns")
    dut.rst_ni.value = 1
    
    # Wait a few cycles
    for _ in range(5):
        await RisingEdge(dut.clk_i)

async def dump_arbiter(dut):
    for i in range(20):
        await RisingEdge(dut.clk_i)
        arb = dut.u_npu_cluster.u_itcm_arbiter
        dut._log.info(f"C{i} m0_req={arb.m0_req_i.value} m0_gnt={arb.m0_gnt_o.value} m1_req={arb.m1_req_i.value} m1_gnt={arb.m1_gnt_o.value} "
                      f"slv_req={arb.slv_req_o.value} slv_gnt={arb.slv_gnt_i.value} slv_rvalid={arb.slv_rvalid_i.value} "
                      f"m0_rval={arb.m0_rvalid_o.value} m1_rval={arb.m1_rvalid_o.value}")

@cocotb.test()
async def test_arbiter(dut):
    from cocotb.clock import Clock
    clock = Clock(dut.clk_i, 1, units="ns") # 1GHz
    cocotb.start_soon(clock.start())

    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)

    await reset_dut(dut)
    
    cocotb.start_soon(dump_arbiter(dut))

    # Try to write to I-TCM via AXI
    await axi_master.write(0x10000000, b'\x13\x01\x01\x01')
    dut._log.info("Write finished!")

