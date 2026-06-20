import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from npu_test_utils import (
    firmware_path,
    hold_reset,
    load_firmware_tcm_backdoor,
    read_dtcm_word,
    read_l2_bytes,
    read_tcdm_word32,
    release_reset,
    wait_for_status,
    write_dtcm_word,
    write_l2_bytes,
)


L2_SRC = 0x80000000
L2_DST = 0x80001000
TCDM_BANK_LOW = 0x10102000
TCDM_BANK_HIGH = 0x1017E000
DMA_BYTES = 512
SIG_START_WORD = 8


def l2_pattern(index):
    return (index * 13 + 7) & 0xFF


def tcdm_dst_pattern(index):
    return 0xA0 ^ ((index * 5) & 0xFF)


@cocotb.test()
async def test_independent_memory(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    await hold_reset(dut)
    fw_path = firmware_path(__file__, "sw/independent_memory_test/independent_memory.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/independent_memory_test` first."
    load_firmware_tcm_backdoor(dut, fw_path)
    await release_reset(dut)

    for _ in range(100):
        if read_dtcm_word(dut, 6) == 1:
            break
        await ClockCycles(dut.clk_i, 1)
    assert read_dtcm_word(dut, 6) == 1, "firmware did not reach boot/start gate"

    await write_l2_bytes(dut, L2_SRC, [l2_pattern(i) for i in range(DMA_BYTES)])
    write_dtcm_word(dut, SIG_START_WORD, 1)

    debug = await wait_for_status(dut, expected_pass_count=3, timeout_cycles=50000)
    dut._log.info(f"memory suite passed: {debug}")

    got = await read_l2_bytes(dut, L2_DST, DMA_BYTES)
    expected = [tcdm_dst_pattern(i) for i in range(DMA_BYTES)]
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, f"L2 DMA-out mismatch idx={idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"

    for bank in range(16):
        low = read_tcdm_word32(dut, TCDM_BANK_LOW + bank * 32)
        high = read_tcdm_word32(dut, TCDM_BANK_HIGH + bank * 32)
        assert low == (0x11000000 | bank), f"TCDM low bank {bank} got=0x{low:08x}"
        assert high == (0x22000000 | bank), f"TCDM high bank {bank} got=0x{high:08x}"
