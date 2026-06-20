import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLiteBus, AxiLiteMaster


PASS_SIGNATURE = 0xDEADBEEF
FAIL_SIGNATURE_MASK = 0xFFF00000
FAIL_SIGNATURE_PREFIX = 0xBAD00000
EXPECTED_PASS_COUNT = 3
TCDM_NUM_BANKS = 16
TCDM_BANK_WORDS = 1024
TCDM_WORD_BYTES = 32
SRC_I8 = 0x10100000
DST_I8 = 0x10100100
RELU_I8 = 0x10100200
DST_REQUANT = 0x10100500
VL = 32


async def reset_dut(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, unit="ns")
    dut.rst_ni.value = 1
    await Timer(20, unit="ns")


async def load_firmware_axi(dut, axi_master, filename, base_addr=0x10000000):
    with open(filename, "rb") as firmware_file:
        firmware = firmware_file.read()

    if len(firmware) % 8 != 0:
        firmware += b"\x00" * (8 - (len(firmware) % 8))

    for offset in range(0, len(firmware), 8):
        await axi_master.write(base_addr + offset, firmware[offset : offset + 8])


def read_dtcm_word(dut, word_index):
    mem_index = word_index // 8
    bit_offset = (word_index % 8) * 32
    val_256 = dut.u_npu_cluster.u_sram_d_tcm.mem[mem_index].value

    if not val_256.is_resolvable:
        return 0

    return (val_256.to_unsigned() >> bit_offset) & 0xFFFFFFFF


def read_tcdm_byte(dut, addr):
    bank_idx = (addr >> 5) % TCDM_NUM_BANKS
    word_index = ((addr >> 5) // TCDM_NUM_BANKS) & (TCDM_BANK_WORDS - 1)
    bit_offset = (addr % TCDM_WORD_BYTES) * 8
    bank_mem = dut.u_npu_cluster.gen_sram_banks[bank_idx].u_sram_bank.mem
    val_256 = bank_mem[word_index].value

    if not val_256.is_resolvable:
        return 0

    return (val_256.to_unsigned() >> bit_offset) & 0xFF


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

    fw_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../sw/spatz_ops_test/spatz_ops_test.bin",
    )
    assert os.path.exists(fw_path), (
        "Missing Spatz operator firmware. Run `make -C sw/spatz_ops_test` first."
    )

    await load_firmware_axi(dut, axi_master, fw_path)
    await reset_dut(dut)

    for _ in range(30000):
        status = read_dtcm_word(dut, 0)
        if status == PASS_SIGNATURE:
            pass_count = read_dtcm_word(dut, 1)
            assert pass_count == EXPECTED_PASS_COUNT
            check_tcdm_outputs(dut)
            return

        if (status & FAIL_SIGNATURE_MASK) == FAIL_SIGNATURE_PREFIX:
            debug = {
                "status": status,
                "fail_test": read_dtcm_word(dut, 2),
                "fail_index": read_dtcm_word(dut, 3),
                "got": read_dtcm_word(dut, 4),
                "expected": read_dtcm_word(dut, 5),
            }
            raise AssertionError(f"Spatz operator firmware failed: {debug}")

        await RisingEdge(dut.clk_i)

    raise AssertionError(
        f"Timeout waiting for Spatz operator firmware, status=0x{read_dtcm_word(dut, 0):08x}"
    )
