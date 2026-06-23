import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLiteMaster, AxiLiteBus
import os
import random
import struct
import numpy as np

async def reset_dut(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, units="ns")
    dut.rst_ni.value = 1
    
    # Monitor signals for a few cycles
    for _ in range(5):
        await RisingEdge(dut.clk_i)
        dut._log.info(f"Cycle: req={dut.u_npu_cluster.u_snitch_core.obi_i_req_o.value} "
                      f"addr={dut.u_npu_cluster.u_snitch_core.obi_i_addr_o.value}")
        if dut.u_npu_cluster.u_snitch_core.obi_i_rvalid_i.value == 1:
            dut._log.info(f"RDATA: {dut.u_npu_cluster.u_snitch_core.obi_i_rdata_i.value}")

    await Timer(20, units="ns")

async def write_axi_sim_mem(dut, base_addr, data_words):
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)
    
    for i, word in enumerate(data_words):
        for byte_idx in range(4):
            dut.backdoor_we_i.value = 1
            dut.backdoor_addr_i.value = base_addr + (i * 4) + byte_idx
            dut.backdoor_data_i.value = (word >> (byte_idx * 8)) & 0xFF
            await RisingEdge(dut.clk_i)
            
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)

async def read_axi_sim_mem(dut, base_addr, num_words):
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)
    
    words = []
    for i in range(num_words):
        word = 0
        for byte_idx in range(4):
            dut.backdoor_addr_i.value = base_addr + (i * 4) + byte_idx
            await Timer(1, units="ps") # Allow combinational read
            byte_val = dut.backdoor_rdata_o.value.integer
            word |= (byte_val << (byte_idx * 8))
        words.append(word)
        await RisingEdge(dut.clk_i)
        
    return words

async def load_firmware_axi(dut, axi_master, filename, base_addr=0x10000000):
    with open(filename, "rb") as f:
        firmware = f.read()

    print(f"Loading {len(firmware)} bytes from {filename} into I-TCM via AXI Lite...")
    
    if len(firmware) % 8 != 0:
        firmware += b'\x00' * (8 - (len(firmware) % 8))

    for i in range(0, len(firmware), 8):
        word = firmware[i:i+8]
        await axi_master.write(base_addr + i, word)
    
    print("Firmware loaded successfully via AXI.")

@cocotb.test()
async def test_matmul(dut):
    clock = Clock(dut.clk_i, 1, units="ns") # 1GHz
    cocotb.start_soon(clock.start())

    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)

    dut._log.info("STARTING RANDOMIZED MATMUL TEST")
    await reset_dut(dut)
    
    fw_path = os.path.join(os.path.dirname(__file__), "../../../../../sw/test/matmul/matmul.bin")
    await load_firmware_axi(dut, axi_master, fw_path, 0x10000000)

    dut._log.info("Resetting NPU to start firmware execution...")
    await reset_dut(dut)

    for test_idx in range(10):
        # Generate random dimension M (from 1 to 64 to keep sim times reasonable)
        dim_m = random.randint(1, 64)
        
        dut._log.info(f"--- TEST ITERATION {test_idx+1}/10: dim_m = {dim_m} ---")
        
        # Clear done_flag via AXI before starting
        await axi_master.write(0x10008008, struct.pack("<I", 0))

        # We need a 32x32 weight matrix (K=32, N=32). Values 0-5 to avoid overflow
        W = np.random.randint(0, 6, size=(32, 32), dtype=np.int32)
        
        # We need a Mx32 IFM matrix (M=dim_m, K=32). Values 0-5
        IFM = np.random.randint(0, 6, size=(dim_m, 32), dtype=np.int32)
        
        # Golden Model OFM (Mx32)
        OFM_golden = np.dot(IFM, W)
        
        # Flatten and convert to 32-bit words for memory writing
        # W is 32x32 = 1024 bytes. But we pack 4 bytes into 1 word.
        # W elements are 8-bit.
        W_flat = W.flatten()
        weights_data = []
        for i in range(0, len(W_flat), 4):
            word = (W_flat[i+3] << 24) | (W_flat[i+2] << 16) | (W_flat[i+1] << 8) | W_flat[i]
            weights_data.append(int(word))
            
        IFM_flat = IFM.flatten()
        ifm_data = []
        for i in range(0, len(IFM_flat), 4):
            word = (IFM_flat[i+3] << 24) | (IFM_flat[i+2] << 16) | (IFM_flat[i+1] << 8) | IFM_flat[i]
            ifm_data.append(int(word))

        await axi_master.write(0x10008000, struct.pack("<I", dim_m))
        await write_axi_sim_mem(dut, 0x80000000, weights_data)
        await write_axi_sim_mem(dut, 0x80001000, ifm_data)

        # Trigger start_flag
        await axi_master.write(0x10008004, struct.pack("<I", 1))

        # Wait for done_flag to be set by hardware
        dut._log.info("Waiting for done_flag from firmware...")
        while True:
            done_resp = await axi_master.read(0x10008008, 4)
            done_val = struct.unpack("<I", done_resp.data)[0]
            if done_val == 1:
                break
            await Timer(100, units="ns")

        # Check OFM at 0x8000_2000
        # OFM is dim_m x 32 elements of 32-bit words
        expected_words = dim_m * 32
        ofm_words = await read_axi_sim_mem(dut, 0x80002000, expected_words)

        OFM_golden_flat = OFM_golden.flatten()
        
        errors = 0
        for i in range(expected_words):
            hw_val = ofm_words[i]
            golden_val = int(OFM_golden_flat[i])
            if hw_val != golden_val:
                r = i // 32
                c = i % 32
                dut._log.error(f"Mismatch at OFM[{r},{c}]: Expected {golden_val}, Got {hw_val}")
                errors += 1
                if errors > 10:
                    break
        
        if errors == 0:
            dut._log.info(f"ITERATION {test_idx+1} PASSED!")
        else:
            assert False, f"MatMul Output Mismatch in Iteration {test_idx+1}."

    dut._log.info("ALL 10 RANDOMIZED MATMUL TESTS PASSED SUCCESSFULLY!")
