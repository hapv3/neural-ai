import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLiteMaster, AxiLiteBus
import os

ITCM_BASE = 0x10000000
DTCM_BASE = 0x10008000
TCM_SIZE_BYTES = 32 * 1024

async def reset_dut(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, units="ns")
    dut.rst_ni.value = 1
    await Timer(20, units="ns")

async def init_axi_sim_mem(dut, base_addr, data_words):
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)
    
    for i, word in enumerate(data_words):
        # Write 4 bytes (little endian)
        for byte_idx in range(4):
            dut.backdoor_we_i.value = 1
            dut.backdoor_addr_i.value = base_addr + (i * 4) + byte_idx
            dut.backdoor_data_i.value = (word >> (byte_idx * 8)) & 0xFF
            await RisingEdge(dut.clk_i)
            
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)

async def load_firmware_axi(dut, axi_master, filename, base_addr=0x10000000):
    with open(filename, "rb") as f:
        firmware = f.read()

    print(f"Loading {len(firmware)} bytes from {filename} into I-TCM via AXI Lite...")
    
    if len(firmware) % 4 != 0:
        firmware += b'\x00' * (4 - (len(firmware) % 4))

    for i in range(0, len(firmware), 4):
        word = firmware[i:i+4]
        addr = base_addr + i
        if ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES:
            await axi_master.write(addr, word)
        elif DTCM_BASE <= addr < DTCM_BASE + TCM_SIZE_BYTES:
            dut.u_npu_cluster.u_sram_d_tcm.mem[(addr - DTCM_BASE) // 4].value = int.from_bytes(word, "little")
        else:
            raise AssertionError(f"Firmware byte outside I/D-TCM range at 0x{addr:08x}")
    
    print("Firmware loaded successfully via AXI.")

def read_dtcm_signature(dut):
    val_32 = dut.u_npu_cluster.u_sram_d_tcm.mem[0].value
    if not val_32.is_resolvable:
        return 0

    return val_32.integer & 0xFFFFFFFF

@cocotb.test()
async def test_snitch_boot(dut):
    # Start clock
    clock = Clock(dut.clk_i, 1, units="ns") # 1GHz
    cocotb.start_soon(clock.start())

    # Initialize AXI Lite Master, reset_active_level=False because rst_ni is active-low
    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)

    dut._log.info("STARTING TEST"); await reset_dut(dut)
    
    # Load firmware using AXI while core might be executing garbage
    fw_path = os.path.join(os.path.dirname(__file__), "../../../../../sw/test/boot/boot.bin")
    await load_firmware_axi(dut, axi_master, fw_path, 0x10000000)

    # Initialize AXI Sim Mem with test data so the DMA has data to read
    # TEST_LEN is 64 bytes (16 words). Snitch writes 0xCAFEBABE + i
    test_data = [0xCAFEBABE + i for i in range(16)]
    await init_axi_sim_mem(dut, 0x10100000, test_data)

    # Reset again so Snitch core boots cleanly from 0x1000_0000 with loaded firmware
    dut._log.info("STARTING TEST"); await reset_dut(dut)

    # 2. Release reset to Snitch? Wait, Snitch was running while we were loading!
    # If Snitch is running from address 0 while we load, it might fetch garbage.
    # Actually we should keep Snitch in reset while loading. But wait, `dut.rst_ni` resets the whole cluster including AXI!
    # So we can't keep Snitch in reset without resetting AXI.
    # In a real system, the host asserts a specific `fetch_enable` signal to the Snitch core!
    # Let's check `snitch_core.sv`. Does it have `fetch_enable`?
    # Right now, `snitch_core` might be fetching garbage until we finish loading.
    # BUT, the start address is `0x1000_0000`. The first instruction is usually a jump.
    # For simulation, we can just reset again? No, reset clears SRAM!
    # Let's hope the firmware loading doesn't crash Snitch. In fact, if we write the first word last, it might be safer.
    # For now, let's just see what happens.
    
    dut._log.info("Firmware loaded. Waiting for Snitch to finish test...")

    # 3. Poll D-TCM signature backdoor
    timeout_cycles = 1000
    for _ in range(timeout_cycles):
        val = read_dtcm_signature(dut)
        
        if val == 0xDEADBEEF:
            dut._log.info("TEST PASSED: Firmware reported success signature (0xDEADBEEF)!")
            return
        elif val == 0xBADBAD00:
            dst_val = dut.u_npu_cluster.u_sram_d_tcm.mem[4].value.integer & 0xFFFFFFFF
            src_val = dut.u_npu_cluster.u_sram_d_tcm.mem[5].value.integer & 0xFFFFFFFF
            idx_val = dut.u_npu_cluster.u_sram_d_tcm.mem[6].value.integer & 0xFFFFFFFF
            dut._log.error(f"TEST FAILED: Firmware reported failure signature (0xBADBAD00). Mismatch at idx {idx_val}: dst={dst_val:08x}, src={src_val:08x}")
            assert False, "Firmware test failed."
        elif val == 0xBADBAD01:
            dut._log.error("TEST FAILED: Firmware MMIO test failed (0xBADBAD01).")
            assert False, "MMIO test failed."
            
        await Timer(10, units="ns")

    dut._log.error("TEST TIMEOUT: Firmware did not complete in time.")
    assert False, "Timeout waiting for firmware signature."
