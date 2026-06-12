import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLiteMaster, AxiBus
import os

async def reset_dut(dut):
    dut.rst_ni.value = 0
    await Timer(20, units="ns")
    dut.rst_ni.value = 1
    await Timer(20, units="ns")

def load_firmware_backdoor(dut, filename, base_addr=0x10000000):
    with open(filename, "rb") as f:
        firmware = f.read()

    print(f"Loading {len(firmware)} bytes from {filename} into TCDM backdoor...")
    
    # 512KB TCDM is divided into 16 banks of 32KB each (8192 words of 32-bit)
    # The interconnect is 256-bit wide, so 1 bank stores 256-bit (32 bytes) per index?
    # Wait, cluster_sram_bank DATA_WIDTH=256. SIZE_BYTES=32768. 
    # Depth = 32768 / 32 = 1024 lines per bank.
    # The tcdm_interconnect maps bits [4:0] as byte offset, [8:5] as Bank Select, [18:9] as Word Address.
    
    # Let's write byte by byte into the memory model
    for i, byte in enumerate(firmware):
        addr = base_addr + i
        # Address decoding based on tcdm_interconnect logic
        # slv_addr[b] is computed, but we can just map the global address:
        offset = addr - base_addr
        bank = (offset >> 5) & 0xF   # bits [8:5]
        word_addr = (offset >> 9)    # bits [18:9]
        byte_in_word = offset & 0x1F # bits [4:0]

        # The mem array is a 256-bit word array.
        # We need to read-modify-write the 256-bit word in Python.
        mem_handle = dut.gen_sram_banks[bank].u_sram_bank.mem
        
        current_word = mem_handle[word_addr].value
        if not current_word.is_resolvable:
            current_word = 0
        else:
            current_word = current_word.integer
            
        # Clear the target byte
        mask = ~(0xFF << (byte_in_word * 8))
        current_word &= mask
        
        # Set the target byte
        current_word |= (byte << (byte_in_word * 8))
        
        mem_handle[word_addr].value = current_word

@cocotb.test()
async def test_snitch_boot(dut):
    # Start clock
    clock = Clock(dut.clk_i, 1, units="ns") # 1GHz
    cocotb.start_soon(clock.start())

    # Initialize AXI Lite Master
    axi_master = AxiLiteMaster(AxiBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni)

    await reset_dut(dut)

    # Load firmware backdoor
    fw_path = os.path.join(os.path.dirname(__file__), "../../../sw/boot_app/boot.bin")
    load_firmware_backdoor(dut, fw_path, 0x10000000)

    # Release reset completely and let Snitch run
    dut._log.info("Firmware loaded. Snitch is booting...")

    # Wait for the success signature in TCDM 0x10000FFC
    # Address 0x10000FFC:
    # offset = 0xFFC -> bank = (0xFFC >> 5) & 0xF = 0x7F & 0xF = 15 ? No.
    # Let's just poll it via AXI Lite interface! This proves the AXI port works too!
    
    timeout_cycles = 1000
    for _ in range(timeout_cycles):
        result = await axi_master.read(0x10000FFC, 4)
        val = int.from_bytes(result.data, byteorder='little')
        
        if val == 0xDEADBEEF:
            dut._log.info("TEST PASSED: Firmware reported success signature (0xDEADBEEF)!")
            return
        elif val == 0xBADBAD00:
            dut._log.error("TEST FAILED: Firmware reported failure signature (0xBADBAD00).")
            assert False, "Firmware test failed."
            
        await Timer(10, units="ns")

    dut._log.error("TEST TIMEOUT: Firmware did not complete in time.")
    assert False, "Timeout waiting for firmware signature."
