import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
    read_dtcm_word,
    read_tcdm_byte,
    read_tcdm_word32,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
)


DST_ADD = 0x10100200
DST_SUB = 0x10100300
DST_AND = 0x10100400
DST_OR = 0x10100500
DST_XOR = 0x10100600
DST_SLL = 0x10100700
DST_SRL = 0x10100800
VL = 16

DST_STRIDED = 0x10104100
DST_VLUXEI = 0x10104500
DST_VSUXEI = 0x10104600
DST_VLOXEI = 0x10104A00
DST_VSOXEI = 0x10104B00
DST_STRIDE32 = 0x10104C00
DST_M8 = 0x10105000
STRIDED_INDEXED_VL = 16
STRIDED_INDEXED_VL_M8 = 96
SIG_STATUS = 0x10008000
SIG_PASS_COUNT = 0x10008004
SIG_FAIL_TEST = 0x10008008
SIG_FAIL_INDEX = 0x1000800C
SIG_FAIL_GOT = 0x10008010
SIG_FAIL_EXP = 0x10008014
PASS_SIGNATURE = 0xDEADBEEF


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


def check_byte(dut, addr, expected, opname, lane):
    got = read_tcdm_byte(dut, addr)
    assert got == expected, (
        f"{opname} output mismatch at lane {lane}: "
        f"addr=0x{addr:08x} got=0x{got:02x} expected=0x{expected:02x}"
    )


def check_strided_indexed_outputs(dut):
    for lane in range(STRIDED_INDEXED_VL):
        check_byte(
            dut,
            DST_STRIDED + lane * 2,
            (15 * lane + 7) & 0xFF,
            "vlse8.v/vsse8.v",
            lane,
        )
        check_byte(
            dut,
            DST_VLUXEI + lane,
            (45 * lane + 12) & 0xFF,
            "vluxei8.v",
            lane,
        )
        check_byte(
            dut,
            DST_STRIDE32 + lane * 32,
            (15 * lane + 7) & 0xFF,
            "vlse8.v/vsse8.v stride32",
            lane,
        )
        check_byte(
            dut,
            DST_VSUXEI + lane * 2,
            (45 * lane + 12) & 0xFF,
            "vsuxei8.v",
            lane,
        )
        check_byte(
            dut,
            DST_VLOXEI + lane,
            (28 * lane + 25) & 0xFF,
            "vloxei8.v",
            lane,
        )
        check_byte(
            dut,
            DST_VSOXEI + lane * 3,
            (28 * lane + 25) & 0xFF,
            "vsoxei8.v",
            lane,
        )
    for lane in range(STRIDED_INDEXED_VL_M8):
        check_byte(
            dut,
            DST_M8 + lane * 32,
            (33 * lane + 19) & 0xFF,
            "vlse8.v/vsse8.v m8 stride32",
            lane,
        )


async def boot_and_wait(dut, axi_master, firmware, description, timeout_cycles=50000):
    await reset_dut(dut)
    await load_firmware_axi(axi_master, firmware_path(__file__, firmware))
    await release_fetch(dut)
    dut._log.info("Waiting for Spatz RVV firmware IRQ: %s", description)
    await wait_for_host_irq(dut, timeout_cycles=timeout_cycles)
    status = read_dtcm_word(dut, SIG_STATUS)
    if status != PASS_SIGNATURE:
        raise AssertionError(
            f"{description} firmware failed: status=0x{status:08x} "
            f"pass_count={read_dtcm_word(dut, SIG_PASS_COUNT)} "
            f"fail_test={read_dtcm_word(dut, SIG_FAIL_TEST)} "
            f"fail_index={read_dtcm_word(dut, SIG_FAIL_INDEX)} "
            f"got=0x{read_dtcm_word(dut, SIG_FAIL_GOT):08x} "
            f"expected=0x{read_dtcm_word(dut, SIG_FAIL_EXP):08x}"
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

    await boot_and_wait(
        dut,
        axi_master,
        "sw/test/spatz_vector/basic_mem_arith.bin",
        "vsetvli/vle32/vse32/arithmetic",
    )
    check_tcdm_outputs(dut)

    await boot_and_wait(
        dut,
        axi_master,
        "sw/test/spatz_vector/strided_indexed.bin",
        "vlse/vsse and indexed load/store",
        timeout_cycles=100000,
    )
    check_strided_indexed_outputs(dut)
