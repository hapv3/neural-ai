import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLiteMaster, AxiLiteBus
import os

async def load_firmware_axi(dut, axi_master, filename, base_addr=0x10000000):
    with open(filename, "rb") as f:
        firmware = f.read()
    
    # Just load first 32 bytes for debug
    for i in range(0, min(32, len(firmware)), 4):
        word = firmware[i:i+4]
        await axi_master.write(base_addr + i, word)

@cocotb.test()
async def test_debug(dut):
    from cocotb.clock import Clock
    clock = Clock(dut.clk_i, 1, units="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)
    
    dut.rst_ni.value = 0
    await Timer(20, units="ns")
    dut.rst_ni.value = 1
    for _ in range(5): await RisingEdge(dut.clk_i)

    firmware_path = os.path.join(os.path.dirname(__file__), "../../../../../sw/test/matmul/matmul.bin")
    await load_firmware_axi(dut, axi_master, firmware_path)

    for _ in range(5): await RisingEdge(dut.clk_i)
    
    val_256 = dut.u_npu_cluster.u_sram_i_tcm.mem[0].value
    dut._log.info(f"MEM[0] AFTER LOAD: {val_256}")
