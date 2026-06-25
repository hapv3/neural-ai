import os

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import firmware_path, load_firmware_axi, read_tcdm_byte, release_fetch, reset_dut, wait_for_host_irq


NPU_DTCM_BASE = 0x10008000
MODE_E8 = 0
MODE_E16 = 1
MODE_E32 = 2

CASES = [
    {"mode": MODE_E8, "src": 0x10100120, "dst": 0x10100480, "length": 64, "salt": 1},
    {"mode": MODE_E16, "src": 0x10101140, "dst": 0x10101580, "length": 33, "salt": 2},
    {"mode": MODE_E32, "src": 0x10102260, "dst": 0x10102680, "length": 17, "salt": 3},
]


def lut_value(mode, input_value, salt):
    if mode == MODE_E8:
        return ((input_value * 7) + salt + 0x5A) & 0xFF
    if mode == MODE_E16:
        return ((input_value * 257) + 0x1234 + salt) & 0xFFFF
    return ((input_value * 0x01010101) ^ (0xDEADBEEF + salt)) & 0xFFFFFFFF


def input_value(index, salt):
    return ((index * 37) + 11 + salt) & 0xFF


def read_tcdm_le(dut, addr, width_bytes):
    value = 0
    for byte_idx in range(width_bytes):
        value |= read_tcdm_byte(dut, addr + byte_idx) << (byte_idx * 8)
    return value


def read_dtcm_word(dut, addr):
    index = (addr - NPU_DTCM_BASE) >> 2
    value = dut.u_npu_cluster.u_sram_d_tcm.mem[index].value
    return value.to_unsigned() if value.is_resolvable else 0


def debug_words(dut):
    afu = dut.u_npu_cluster.u_afu
    return {
        "status": read_dtcm_word(dut, NPU_DTCM_BASE + 0x00),
        "case": read_dtcm_word(dut, NPU_DTCM_BASE + 0x10),
        "index": read_dtcm_word(dut, NPU_DTCM_BASE + 0x14),
        "got": read_dtcm_word(dut, NPU_DTCM_BASE + 0x18),
        "expected": read_dtcm_word(dut, NPU_DTCM_BASE + 0x1C),
        "phase": read_dtcm_word(dut, NPU_DTCM_BASE + 0x20),
        "afu_done": int(afu.done_o.value),
        "core_state": int(afu.i_core.state_q.value),
        "core_elem": int(afu.i_core.elem_cnt_q.value),
        "backend_idle": int(afu.backend_idle.value),
        "re_active": int(afu.i_backend.re_active_q.value),
        "re_addr": int(afu.i_backend.re_addr_q.value),
        "re_end": int(afu.i_backend.re_end_addr_q.value),
        "read_outstanding": int(afu.i_backend.read_outstanding_q.value),
        "pending": int(afu.i_backend.pending_valid_q.value),
        "rfifo_cnt": int(afu.i_rfifo.cnt_q.value),
        "wfifo_cnt": int(afu.i_wfifo.cnt_q.value),
        "rfifo_empty": int(afu.rfifo_empty.value),
        "wfifo_empty": int(afu.wfifo_empty.value),
    }


@cocotb.test()
async def test_afu_basic(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    fw_path = firmware_path(__file__, "sw/test/afu/afu.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/afu` first."

    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await release_fetch(dut)
    try:
        await wait_for_host_irq(dut, timeout_cycles=120000)
    except AssertionError as exc:
        raise AssertionError(f"{exc}; AFU debug={debug_words(dut)}") from exc

    for case_idx, case in enumerate(CASES):
        width_bytes = 1 if case["mode"] == MODE_E8 else (2 if case["mode"] == MODE_E16 else 4)
        for index in range(case["length"]):
            expected = lut_value(case["mode"], input_value(index, case["salt"]), case["salt"])
            got = read_tcdm_le(dut, case["dst"] + index * width_bytes, width_bytes)
            assert got == expected, (
                f"AFU case={case_idx} idx={index} got=0x{got:08x} expected=0x{expected:08x}"
            )

    dut._log.info("AFU basic suite completed through host irq and exact TCDM output compare")
