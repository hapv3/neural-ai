import logging
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer


PASS_SIGNATURE = 0xDEADBEEF
FAIL_SIGNATURE_MASK = 0xFFF00000
FAIL_SIGNATURE_PREFIX = 0xBAD00000

ITCM_BASE = 0x10000000
DTCM_BASE = 0x10008000
TCM_SIZE_BYTES = 32 * 1024
TCM_WORD_BYTES = 32

L2_INPUT = 0x80000000
L2_WEIGHT0 = 0x80002000
L2_WEIGHT1 = 0x80002400
L2_OUTPUT = 0x80010000

INPUT_BYTES = 32 * 32 * 3
WEIGHT_BYTES = 32 * 32
OUTPUT_BYTES = 32 * 32 * 32
SIG_START_WORD = 5


async def assert_reset(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, unit="ns")


async def release_reset(dut):
    dut.rst_ni.value = 1
    await Timer(20, unit="ns")


def load_firmware_tcm(dut, filename, base_addr=ITCM_BASE):
    with open(filename, "rb") as firmware_file:
        firmware = firmware_file.read()

    if len(firmware) % TCM_WORD_BYTES != 0:
        firmware += b"\x00" * (TCM_WORD_BYTES - (len(firmware) % TCM_WORD_BYTES))

    for offset in range(0, len(firmware), TCM_WORD_BYTES):
        addr = base_addr + offset
        word = int.from_bytes(firmware[offset : offset + TCM_WORD_BYTES], "little")

        if ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES:
            word_index = (addr - ITCM_BASE) // TCM_WORD_BYTES
            dut.u_npu_cluster.u_sram_i_tcm.mem[word_index].value = word
        elif DTCM_BASE <= addr < DTCM_BASE + TCM_SIZE_BYTES:
            word_index = (addr - DTCM_BASE) // TCM_WORD_BYTES
            dut.u_npu_cluster.u_sram_d_tcm.mem[word_index].value = word
        else:
            raise AssertionError(f"Firmware byte outside I/D-TCM range at 0x{addr:08x}")


async def write_axi_sim_bytes(dut, base_addr, data):
    dut.backdoor_we_i.value = 0
    await ClockCycles(dut.clk_i, 1)

    for offset, byte_val in enumerate(data):
        dut.backdoor_we_i.value = 1
        dut.backdoor_addr_i.value = base_addr + offset
        dut.backdoor_data_i.value = byte_val & 0xFF
        await ClockCycles(dut.clk_i, 1)

    dut.backdoor_we_i.value = 0
    await ClockCycles(dut.clk_i, 1)


async def read_axi_sim_bytes(dut, base_addr, length):
    data = []
    dut.backdoor_we_i.value = 0
    await ClockCycles(dut.clk_i, 1)

    for offset in range(length):
        dut.backdoor_addr_i.value = base_addr + offset
        await Timer(1, unit="ps")
        data.append(dut.backdoor_rdata_o.value.integer & 0xFF)
        if (offset & 0x1F) == 0x1F:
            await ClockCycles(dut.clk_i, 1)

    return data


def read_dtcm_word(dut, word_index):
    mem_index = word_index // 8
    bit_offset = (word_index % 8) * 32
    val_256 = dut.u_npu_cluster.u_sram_d_tcm.mem[mem_index].value

    if not val_256.is_resolvable:
        return 0

    return (val_256.to_unsigned() >> bit_offset) & 0xFFFFFFFF


def write_dtcm_word(dut, word_index, value):
    mem_index = word_index // 8
    bit_offset = (word_index % 8) * 32
    val_256 = dut.u_npu_cluster.u_sram_d_tcm.mem[mem_index].value
    current = val_256.to_unsigned() if val_256.is_resolvable else 0
    current &= ~(0xFFFFFFFF << bit_offset)
    current |= (value & 0xFFFFFFFF) << bit_offset
    dut.u_npu_cluster.u_sram_d_tcm.mem[mem_index].value = current


def read_micro_debug(dut):
    return {
        "status": read_dtcm_word(dut, 0),
        "layer": read_dtcm_word(dut, 1),
        "code": read_dtcm_word(dut, 2),
        "op": read_dtcm_word(dut, 3),
        "event": read_dtcm_word(dut, 4),
    }


def safe_int(handle):
    value = handle.value
    return value.to_unsigned() if value.is_resolvable else None


def read_systolic_debug(dut):
    ctrl = dut.u_npu_cluster.u_systolic_controller
    return {
        "state": safe_int(ctrl.state_q),
        "req_cnt": safe_int(ctrl.req_cnt_q),
        "rsp_cnt": safe_int(ctrl.rsp_cnt_q),
        "drain_cnt": safe_int(ctrl.drain_cnt_q),
        "ofm_buf_valid": safe_int(ctrl.ofm_buf_valid_q),
        "ofm_valid": safe_int(dut.u_npu_cluster.sys_ofm_valid),
        "o_gnt": safe_int(dut.u_npu_cluster.sys_obi_o_gnt),
        "cfg_done": safe_int(dut.u_npu_cluster.cfg_sys_done),
    }


def to_u8(value):
    return value & 0xFF


def to_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def deterministic_fixture():
    input_hwc = [
        ((y * 3 + x * 5 + c * 7) % 7) - 3
        for y in range(32)
        for x in range(32)
        for c in range(3)
    ]
    weight0 = [
        ((k * 2 + n * 3) % 5) - 2
        for k in range(32)
        for n in range(32)
    ]
    weight1 = [
        ((k * 3 + n) % 5) - 2
        for k in range(32)
        for n in range(32)
    ]
    return input_hwc, weight0, weight1


def im2col3x3s1p1_c3_pad32(input_hwc):
    rows = []
    for y in range(32):
        for x in range(32):
            row = []
            for ky in range(-1, 2):
                iy = y + ky
                for kx in range(-1, 2):
                    ix = x + kx
                    for c in range(3):
                        if 0 <= iy < 32 and 0 <= ix < 32:
                            row.append(input_hwc[(iy * 32 + ix) * 3 + c])
                        else:
                            row.append(0)
            row.extend([0] * (32 - len(row)))
            rows.append(row)
    return rows


def matmul_mx32_32x32(lhs_rows, weight):
    out = []
    for row in lhs_rows:
        out_row = []
        for n in range(32):
            acc = 0
            for k in range(32):
                acc += row[k] * weight[k * 32 + n]
            out_row.append(acc)
        out.append(out_row)
    return out


def requant(values, min_val, max_val):
    return [max(min_val, min(max_val, value)) for value in values]


def golden_micro_yolo(input_hwc, weight0, weight1):
    im2col = im2col3x3s1p1_c3_pad32(input_hwc)
    conv0 = matmul_mx32_32x32(im2col, weight0)
    act0_flat = requant([value for row in conv0 for value in row], 0, 127)
    act0_rows = [act0_flat[idx : idx + 32] for idx in range(0, len(act0_flat), 32)]
    conv1 = matmul_mx32_32x32(act0_rows, weight1)
    out_flat = requant([value for row in conv1 for value in row], -128, 127)
    return [to_u8(value) for value in out_flat]


@cocotb.test()
async def test_micro_yolo_e2e(dut):
    logging.getLogger("cocotb.tb_npu_cluster.s_axi").setLevel(logging.WARNING)

    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    input_hwc, weight0, weight1 = deterministic_fixture()
    expected = golden_micro_yolo(input_hwc, weight0, weight1)

    fw_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../sw/micro_yolo_app/micro_yolo.bin",
    )
    assert os.path.exists(fw_path), "Missing firmware. Run `make -C sw/micro_yolo_app` first."

    await assert_reset(dut)
    load_firmware_tcm(dut, fw_path)
    await release_reset(dut)
    await write_axi_sim_bytes(dut, L2_INPUT, [to_u8(value) for value in input_hwc])
    await write_axi_sim_bytes(dut, L2_WEIGHT0, [to_u8(value) for value in weight0])
    await write_axi_sim_bytes(dut, L2_WEIGHT1, [to_u8(value) for value in weight1])
    write_dtcm_word(dut, SIG_START_WORD, 1)
    await ClockCycles(dut.clk_i, 1)

    last_debug = None
    for poll_idx in range(100000):
        status = read_dtcm_word(dut, 0)
        if status == PASS_SIGNATURE:
            got = await read_axi_sim_bytes(dut, L2_OUTPUT, OUTPUT_BYTES)
            for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
                assert got_byte == exp_byte, (
                    f"micro-YOLO output mismatch idx={idx}: "
                    f"got={to_i8(got_byte)} expected={to_i8(exp_byte)}"
                )
            return

        if (status & FAIL_SIGNATURE_MASK) == FAIL_SIGNATURE_PREFIX:
            raise AssertionError(f"micro-YOLO firmware failed: {read_micro_debug(dut)}")

        if poll_idx % 1000 == 0:
            debug = read_micro_debug(dut)
            dut._log.info(f"micro-YOLO progress: {debug}")
            last_debug = debug

        debug = last_debug if last_debug is not None else read_micro_debug(dut)
        if poll_idx > 7000 and debug["layer"] == 4 and debug["event"] == 1:
            raise AssertionError(
                f"micro-YOLO systolic Conv0 watchdog: graph={debug} "
                f"systolic={read_systolic_debug(dut)}"
            )

        await ClockCycles(dut.clk_i, 100)

    raise AssertionError(f"Timeout waiting for micro-YOLO firmware: {read_micro_debug(dut)}")
