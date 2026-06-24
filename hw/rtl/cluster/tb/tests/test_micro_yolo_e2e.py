import logging
import os

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    load_firmware_axi,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


L2_INPUT = 0x80000000
L2_WEIGHT0 = 0x80002000
L2_WEIGHT1 = 0x80002400
L2_OUTPUT = 0x80010000

INPUT_BYTES = 32 * 32 * 3
WEIGHT_BYTES = 32 * 32
OUTPUT_BYTES = 32 * 32 * 32


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

    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    input_hwc, weight0, weight1 = deterministic_fixture()
    expected = golden_micro_yolo(input_hwc, weight0, weight1)

    fw_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../sw/test/micro_yolo/micro_yolo.bin",
    )
    assert os.path.exists(fw_path), "Missing firmware. Run `make -C sw/test/micro_yolo` first."

    await reset_dut(dut)
    await write_l2_bytes(dut, L2_INPUT, [to_u8(value) for value in input_hwc])
    await write_l2_bytes(dut, L2_WEIGHT0, [to_u8(value) for value in weight0])
    await write_l2_bytes(dut, L2_WEIGHT1, [to_u8(value) for value in weight1])
    await load_firmware_axi(axi_master, fw_path)
    await release_fetch(dut)

    await wait_for_host_irq(dut, timeout_cycles=100000)

    got = await read_l2_bytes(dut, L2_OUTPUT, OUTPUT_BYTES)
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"micro-YOLO output mismatch idx={idx}: "
            f"got={to_i8(got_byte)} expected={to_i8(exp_byte)}"
        )
