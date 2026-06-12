import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge

from cluster_env import ClusterEnv
from cluster_item import ClusterTestItem

@cocotb.test()
async def test_dma_tcm_path(dut):
    """
    Test Phase 2.5: DMA to TCM Data Path Test.
    Sử dụng UVM-like structure.
    """
    # 1 GHz NPU Clock
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    
    # 100 MHz APB Clock
    apb_clock = Clock(dut.apb_clk_i, 10, unit="ns")
    cocotb.start_soon(apb_clock.start())

    # 1. Initialize Environment
    env = ClusterEnv(dut, dut.clk_i, dut.apb_clk_i)
    await env.reset()

    # 2. Create Test Sequence / Test Item

    SRC_EXT_MEM = 0x8000_0000
    DST_TCDM    = 0x100C_0000 # IFM Ping Buffer
    data1 = 0x112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF00
    data2 = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    
    item = ClusterTestItem(
        src_addr=SRC_EXT_MEM,
        dst_addr=DST_TCDM,
        length=64,
        payloads=[data1, data2]
    )

    # 3. Load Memory via Backdoor
    await env.driver.load_payloads(item)

    # 4. Execute DMA Transfer Sequence
    await env.driver.execute_dma(item)
    await env.driver.wait_for_dma_done()

    # Wait for data to settle in SRAM
    await Timer(50, units="ns")

    # 5. Check Scoreboard
    # DST_TCDM maps to TCDM word address 0x200
    # Bank 0 handles word 0x200 part 1, Bank 1 handles part 2.
    env.scoreboard.check_dma_result(item, tcdm_word_addr=0x200)

    dut._log.info("==================================================")
    dut._log.info("[TEST] PASS! APB MMIO -> DMA -> axi_sim_mem -> TCDM working!")
    dut._log.info("==================================================")
