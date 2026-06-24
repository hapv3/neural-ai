import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
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
    await load_firmware_axi(
        axi_master,
        firmware_path(__file__, "sw/test/spatz_vector/basic_mem_arith.bin"),
    )
    await release_fetch(dut)

    dut._log.info(
        "Waiting for Spatz RVV firmware IRQ: vsetvli/vle32/vse32/arithmetic"
    )
    await wait_for_host_irq(dut, timeout_cycles=30000)

    check_tcdm_outputs(dut)
