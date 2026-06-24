import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotbext.axi import AxiLiteBus, AxiLiteMaster


SIG_STATUS = 0
SIG_PASS_COUNT = 1
SIG_FAIL_TEST = 2
SIG_FAIL_INDEX = 3
SIG_FAIL_GOT = 4
SIG_FAIL_EXP = 5
PASS_SIGNATURE = 0xDEADBEEF
FAIL_SIGNATURE_MASK = 0xFFF00000
FAIL_SIGNATURE_PREFIX = 0xBAD00000
EXPECTED_PASS_COUNT = 7
ITCM_BASE = 0x10000000
DTCM_BASE = 0x10008000
TCM_SIZE_BYTES = 32 * 1024
TCDM_NUM_BANKS = 16
TCDM_BANK_WORDS = 1024
TCDM_WORD_BYTES = 32
DST_ADD = 0x10100200
DST_SUB = 0x10100300
DST_AND = 0x10100400
DST_OR = 0x10100500
DST_XOR = 0x10100600
DST_SLL = 0x10100700
DST_SRL = 0x10100800
VL = 16


async def reset_dut(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, unit="ns")
    dut.rst_ni.value = 1
    await Timer(20, unit="ns")


async def load_firmware_axi(dut, axi_master, filename, base_addr=0x10000000):
    with open(filename, "rb") as firmware_file:
        firmware = firmware_file.read()

    dut._log.info(f"Loading {len(firmware)} bytes from {filename} into I-TCM")

    if len(firmware) % 4 != 0:
        firmware += b"\x00" * (4 - (len(firmware) % 4))

    for offset in range(0, len(firmware), 4):
        addr = base_addr + offset
        word = firmware[offset : offset + 4]
        if ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES:
            await axi_master.write(addr, word)
        elif DTCM_BASE <= addr < DTCM_BASE + TCM_SIZE_BYTES:
            dut.u_npu_cluster.u_sram_d_tcm.mem[(addr - DTCM_BASE) // 4].value = int.from_bytes(word, "little")
        else:
            raise AssertionError(f"Firmware byte outside I/D-TCM range at 0x{addr:08x}")


def read_dtcm_word(dut, word_index):
    val_32 = dut.u_npu_cluster.u_sram_d_tcm.mem[word_index].value

    if not val_32.is_resolvable:
        return 0

    return val_32.to_unsigned() & 0xFFFFFFFF


def read_spatz_debug(dut):
    return {
        "status": read_dtcm_word(dut, SIG_STATUS),
        "pass_count": read_dtcm_word(dut, SIG_PASS_COUNT),
        "fail_test": read_dtcm_word(dut, SIG_FAIL_TEST),
        "fail_index": read_dtcm_word(dut, SIG_FAIL_INDEX),
        "got": read_dtcm_word(dut, SIG_FAIL_GOT),
        "expected": read_dtcm_word(dut, SIG_FAIL_EXP),
    }


def read_tcdm_word32(dut, addr):
    bank_idx = (addr >> 5) % TCDM_NUM_BANKS
    word_index = ((addr >> 5) // TCDM_NUM_BANKS) & (TCDM_BANK_WORDS - 1)
    bit_offset = (addr % TCDM_WORD_BYTES) * 8
    bank_mem = dut.u_npu_cluster.gen_sram_banks[bank_idx].u_sram_bank.mem
    val_256 = bank_mem[word_index].value

    if not val_256.is_resolvable:
        return 0

    return (val_256.to_unsigned() >> bit_offset) & 0xFFFFFFFF


def expected_vectors():
    src_a = [idx + 1 for idx in range(VL)]
    src_b = [idx + 16 for idx in range(VL)]

    return {
        "vadd.vv": (DST_ADD, [(a + b) & 0xFFFFFFFF for a, b in zip(src_a, src_b)]),
        "vsub.vv": (DST_SUB, [(a - b) & 0xFFFFFFFF for a, b in zip(src_a, src_b)]),
        "vand.vv": (DST_AND, [a & b for a, b in zip(src_a, src_b)]),
        "vor.vv": (DST_OR, [a | b for a, b in zip(src_a, src_b)]),
        "vxor.vv": (DST_XOR, [a ^ b for a, b in zip(src_a, src_b)]),
        "vsll.vi": (DST_SLL, [(a << 1) & 0xFFFFFFFF for a in src_a]),
        "vsrl.vi": (DST_SRL, [(b >> 1) & 0xFFFFFFFF for b in src_b]),
    }


def check_tcdm_outputs(dut):
    for opname, (base_addr, expected_words) in expected_vectors().items():
        for idx, expected in enumerate(expected_words):
            addr = base_addr + idx * 4
            got = read_tcdm_word32(dut, addr)
            assert got == expected, (
                f"{opname} output mismatch at lane {idx}: "
                f"addr=0x{addr:08x} got=0x{got:08x} expected=0x{expected:08x}"
            )


@cocotb.test()
async def test_spatz_vector_basic(dut):
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
        "../../../../../sw/test/spatz_vector/basic_mem_arith.bin",
    )
    assert os.path.exists(fw_path), (
        "Missing Spatz-vector firmware. Run `make -C sw/test/spatz_vector` first."
    )

    await load_firmware_axi(dut, axi_master, fw_path)
    await reset_dut(dut)

    dut._log.info(
        "Waiting for Spatz RVV firmware: vsetvli/vle32/vse32/vadd/vsub/logical/shift"
    )

    for _ in range(20000):
        status = read_dtcm_word(dut, SIG_STATUS)

        if status == PASS_SIGNATURE:
            debug = read_spatz_debug(dut)
            assert debug["pass_count"] == EXPECTED_PASS_COUNT, (
                f"Expected {EXPECTED_PASS_COUNT} RVV subtests, "
                f"got {debug['pass_count']}"
            )
            check_tcdm_outputs(dut)
            dut._log.info(f"Spatz RVV firmware passed: {debug}")
            return

        if (status & FAIL_SIGNATURE_MASK) == FAIL_SIGNATURE_PREFIX:
            debug = read_spatz_debug(dut)
            raise AssertionError(f"Spatz RVV firmware failed: {debug}")

        await RisingEdge(dut.clk_i)

    debug = read_spatz_debug(dut)
    raise AssertionError(f"Timeout waiting for Spatz RVV firmware: {debug}")
