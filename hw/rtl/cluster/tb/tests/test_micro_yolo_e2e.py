import logging
import math
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    PMU_COUNTER_NAMES,
    load_firmware_axi,
    read_dtcm_word,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


L2_INPUT = 0x80000000
L2_WEIGHT0 = 0x80008000
L2_SIG_LUT = 0x80009000
L2_WEIGHT1 = 0x8000A000
L2_WEIGHT2 = 0x8000D000
L2_WEIGHT3 = 0x80010000
L2_OUTPUT = 0x80020000

INPUT_H = 96
INPUT_W = 96
OUTPUT_H = 48
OUTPUT_W = 48
DOWN_H = 24
DOWN_W = 24
WEIGHT_BYTES = 32 * 32
C2F_WEIGHT_BYTES = 3 * 3 * 32 * 32
DOWN_WEIGHT_BYTES = C2F_WEIGHT_BYTES
HEAD_WEIGHT_BYTES = 2 * C2F_WEIGHT_BYTES
LUT_BYTES = 256
OUTPUT_BYTES = OUTPUT_H * OUTPUT_W * 32
DTCM_STATUS = 0x10008000
DTCM_FAIL_CODE = 0x10008004
DTCM_LAYER = 0x10008008
DTCM_OP = 0x1000800C
DTCM_EVENT = 0x10008010

OP_NAMES = {
    1: "DMA_IN",
    2: "DMA_OUT",
    5: "SPATZ_REQUANT",
    8: "CONV3x3s2_C3_LB_RQ",
    9: "LOGISTIC_LUT_I8",
    10: "MUL_I8",
    11: "CONV3x3s1_C32_LB",
    12: "CONV3x3s1_C32_LB_RQ",
    13: "ADD_I8",
    14: "CONV3x3s2_C32_LB_RQ",
    15: "MAXPOOL2D5x5s1p2_I8",
    16: "UPSAMPLE_NEAREST2X_I8",
    17: "CONV3x3s1_C32x2_LB_RQ_L2",
}


def pmu_direct_snapshot(dut):
    pmu = dut.u_npu_cluster.u_pmu
    raw_value = pmu.counter_q.value
    raw = raw_value.to_unsigned() if raw_value.is_resolvable else 0
    counters = {}
    for idx, name in enumerate(PMU_COUNTER_NAMES):
        counters[name] = (raw >> (idx * 64)) & ((1 << 64) - 1)
    return counters


def pmu_delta(end, start):
    return {
        name: end.get(name, 0) - start.get(name, 0)
        for name in PMU_COUNTER_NAMES
    }


def format_layer_pmu(layer, op, counters):
    cycles = counters.get("cycle", 0)

    def pct(name):
        return (100.0 * counters.get(name, 0) / cycles) if cycles else 0.0

    return (
        f"layer {layer}: {OP_NAMES.get(op, f'op{op}')} "
        f"cycles={cycles} "
        f"sys_compute={counters.get('sys_compute', 0)}({pct('sys_compute'):.1f}%) "
        f"ifm_req={counters.get('sys_ifm_req', 0)} "
        f"ofm_req={counters.get('sys_ofm_req', 0)} "
        f"ofm_stall={counters.get('sys_ofm_stall', 0)} "
        f"afu_done={counters.get('afu_done', 0)} "
        f"afu_stall={counters.get('afu_tcdm_stall', 0)} "
        f"idma_busy={counters.get('idma_busy', 0)} "
        f"tcdm_stall={counters.get('tcdm_stall', 0)}"
    )


async def monitor_step_pmu(dut, timeout_cycles):
    # Firmware updates a tiny D-TCM trace page at graph-layer entry/exit.
    # Sampling PMU on those trace transitions gives a per-layer performance
    # breakdown without needing Snitch-side printf or interrupt handlers.
    samples = []
    last_trace = None

    for _ in range(timeout_cycles):
        trace = (
            read_dtcm_word(dut, DTCM_LAYER),
            read_dtcm_word(dut, DTCM_OP),
            read_dtcm_word(dut, DTCM_EVENT),
        )
        if trace != last_trace:
            samples.append((trace, pmu_direct_snapshot(dut)))
            last_trace = trace
        irq_value = dut.irq_o.value
        if irq_value.is_resolvable and int(irq_value) == 1:
            break
        await RisingEdge(dut.clk_i)

    final_trace = (
        read_dtcm_word(dut, DTCM_LAYER),
        read_dtcm_word(dut, DTCM_OP),
        read_dtcm_word(dut, DTCM_EVENT),
    )
    final_sample = (final_trace, pmu_direct_snapshot(dut))
    if not samples or samples[-1][0] != final_trace:
        samples.append(final_sample)
    else:
        samples[-1] = final_sample

    layer_lines = []
    for idx in range(1, len(samples)):
        prev_trace, prev_pmu = samples[idx - 1]
        curr_trace, curr_pmu = samples[idx]
        layer, op, event = prev_trace
        if event != 1:
            continue
        layer_lines.append(format_layer_pmu(layer, op, pmu_delta(curr_pmu, prev_pmu)))
    return layer_lines


def to_u8(value):
    return value & 0xFF


def to_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def deterministic_fixture():
    # Keep all fixtures generated in Python so this test is self-contained.
    # The formulas intentionally stay small-valued: they exercise sign,
    # saturation, padding, and accumulation without making mismatches hard to
    # inspect as signed INT8 values.
    input_hwc = [
        ((y * 3 + x * 5 + c * 7) % 7) - 3
        for y in range(INPUT_H)
        for x in range(INPUT_W)
        for c in range(3)
    ]
    weight0 = [
        ((k * 2 + n * 3) % 5) - 2
        for k in range(32)
        for n in range(32)
    ]
    weight1 = [
        ((k * 5 + n * 7 + 3) % 9) - 4
        for k in range(3 * 3 * 32)
        for n in range(32)
    ]
    weight2 = [
        ((k * 11 + n * 13 + 5) % 11) - 5
        for k in range(3 * 3 * 32)
        for n in range(32)
    ]
    weight3 = [
        # The first contiguous C32 chunk is consumed by the upsample branch.
        # The second contiguous C32 chunk is consumed by the skip branch.
        # Firmware does not materialize concat; it runs these chunks separately.
        ((chunk * 17 + k * 7 + n * 5 + 1) % 13) - 6
        for chunk in range(2)
        for k in range(3 * 3 * 32)
        for n in range(32)
    ]
    sigmoid_lut = [
        max(0, min(127, int(round((1.0 / (1.0 + math.exp(-to_i8(index) / 16.0))) * 127.0))))
        for index in range(256)
    ]
    return input_hwc, weight0, weight1, weight2, weight3, sigmoid_lut


def conv3x3s2p1_c3_o32(input_hwc, weight):
    out = []
    for oy in range(OUTPUT_H):
        for ox in range(OUTPUT_W):
            out_row = []
            for n in range(32):
                acc = 0
                k = 0
                for ky in range(-1, 2):
                    iy = oy * 2 + ky
                    for kx in range(-1, 2):
                        ix = ox * 2 + kx
                        for c in range(3):
                            if 0 <= iy < INPUT_H and 0 <= ix < INPUT_W:
                                value = input_hwc[(iy * INPUT_W + ix) * 3 + c]
                            else:
                                value = 0
                            acc += value * weight[k * 32 + n]
                            k += 1
                out_row.append(acc)
            out.append(out_row)
    return out


def conv3x3_c32_o32(input_flat, input_h, input_w, output_h, output_w, stride, weight):
    # ROW32 logical layout: each spatial pixel stores one contiguous C32 vector.
    # Weight layout is K-major, OC-minor: [kh, kw, ic, oc32].
    out = []
    for oy in range(output_h):
        for ox in range(output_w):
            out_row = []
            for n in range(32):
                acc = 0
                k = 0
                for ky in range(-1, 2):
                    iy = oy * stride + ky
                    for kx in range(-1, 2):
                        ix = ox * stride + kx
                        for c in range(32):
                            if 0 <= iy < input_h and 0 <= ix < input_w:
                                value = input_flat[((iy * input_w + ix) * 32) + c]
                            else:
                                value = 0
                            acc += value * weight[k * 32 + n]
                            k += 1
                out_row.append(acc)
            out.append(out_row)
    return out


def conv3x3s1p1_c32_o32(input_flat, weight):
    return conv3x3_c32_o32(input_flat, OUTPUT_H, OUTPUT_W, OUTPUT_H, OUTPUT_W, 1, weight)


def conv3x3s2p1_c32_o32(input_flat, weight):
    return conv3x3_c32_o32(input_flat, OUTPUT_H, OUTPUT_W, DOWN_H, DOWN_W, 2, weight)


def maxpool2d_5x5s1p2_c32(input_flat):
    out = []
    for oh in range(DOWN_H):
        for ow in range(DOWN_W):
            out_pixel = [-128] * 32
            for kh in range(-2, 3):
                ih = oh + kh
                if ih < 0 or ih >= DOWN_H:
                    continue
                for kw in range(-2, 3):
                    iw = ow + kw
                    if iw < 0 or iw >= DOWN_W:
                        continue
                    base = ((ih * DOWN_W + iw) * 32)
                    for c in range(32):
                        out_pixel[c] = max(out_pixel[c], input_flat[base + c])
            out.extend(out_pixel)
    return out


def upsample_nearest2x_c32(input_flat):
    # C32 nearest-neighbor upsample mirrors spatz_upsample_nearest2x_c32_i8:
    # every 24x24x32 input pixel is copied to a 2x2 block in the 48x48 output.
    out = []
    for oh in range(OUTPUT_H):
        ih = oh // 2
        for ow in range(OUTPUT_W):
            iw = ow // 2
            base = ((ih * DOWN_W + iw) * 32)
            out.extend(input_flat[base:base + 32])
    return out


def requant(values, min_val, max_val):
    return [max(min_val, min(max_val, value)) for value in values]


def golden_micro_yolo(input_hwc, weight0, weight1, weight2, weight3, sigmoid_lut):
    # Golden follows the current hardware graph, not an older materialized
    # concat graph. The head identity is:
    #   Conv([upsample, skip], W) == Conv(upsample, W0) + Conv(skip, W1)
    # Requant/clamp happens only after the two C32 chunks are accumulated.
    conv0 = conv3x3s2p1_c3_o32(input_hwc, weight0)
    stem_flat = requant([value for row in conv0 for value in row], -128, 127)
    sig_flat = [to_i8(sigmoid_lut[to_u8(value)]) for value in stem_flat]
    out_flat = requant(
        [(stem * sig) >> 7 for stem, sig in zip(stem_flat, sig_flat)],
        -128,
        127,
    )
    c2f = conv3x3s1p1_c32_o32(out_flat, weight1)
    c2f_flat = requant([value for row in c2f for value in row], -128, 127)
    residual_flat = requant(
        [conv_value + residual_value for conv_value, residual_value in zip(c2f_flat, out_flat)],
        -128,
        127,
    )
    down = conv3x3s2p1_c32_o32(residual_flat, weight2)
    down_flat = requant([value for row in down for value in row], -128, 127)
    pool_flat = maxpool2d_5x5s1p2_c32(down_flat)
    upsample_flat = upsample_nearest2x_c32(pool_flat)
    head_up = conv3x3s1p1_c32_o32(upsample_flat, weight3[:C2F_WEIGHT_BYTES])
    head_skip = conv3x3s1p1_c32_o32(out_flat, weight3[C2F_WEIGHT_BYTES:])
    head_flat = requant(
        [
            up_value + skip_value
            for up_row, skip_row in zip(head_up, head_skip)
            for up_value, skip_value in zip(up_row, skip_row)
        ],
        -128,
        127,
    )
    return [to_u8(value) for value in head_flat]


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

    input_hwc, weight0, weight1, weight2, weight3, sigmoid_lut = deterministic_fixture()
    expected = golden_micro_yolo(input_hwc, weight0, weight1, weight2, weight3, sigmoid_lut)

    fw_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../sw/test/micro_yolo/micro_yolo.bin",
    )
    assert os.path.exists(fw_path), "Missing firmware. Run `make -C sw/test/micro_yolo` first."

    await reset_dut(dut)
    await write_l2_bytes(dut, L2_INPUT, [to_u8(value) for value in input_hwc])
    await write_l2_bytes(dut, L2_WEIGHT0, [to_u8(value) for value in weight0])
    await write_l2_bytes(dut, L2_SIG_LUT, sigmoid_lut)
    await write_l2_bytes(dut, L2_WEIGHT1, [to_u8(value) for value in weight1])
    await write_l2_bytes(dut, L2_WEIGHT2, [to_u8(value) for value in weight2])
    await write_l2_bytes(dut, L2_WEIGHT3, [to_u8(value) for value in weight3])
    await load_firmware_axi(axi_master, fw_path)
    await release_fetch(dut, axi_master=axi_master)
    step_pmu_task = cocotb.start_soon(monitor_step_pmu(dut, 1200000))

    try:
        await wait_for_host_irq(
            dut,
            timeout_cycles=1200000,
            axi_master=axi_master,
            report_name="test_micro_yolo_e2e",
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}: status=0x{read_dtcm_word(dut, DTCM_STATUS):08x} "
            f"fail=0x{read_dtcm_word(dut, DTCM_FAIL_CODE):08x} "
            f"layer={read_dtcm_word(dut, DTCM_LAYER)} "
            f"op={read_dtcm_word(dut, DTCM_OP)} "
            f"event={read_dtcm_word(dut, DTCM_EVENT)}"
        ) from exc
    step_pmu_lines = await step_pmu_task
    dut._log.info("test_micro_yolo_e2e per-layer PMU:\n  %s", "\n  ".join(step_pmu_lines))

    got = await read_l2_bytes(dut, L2_OUTPUT, OUTPUT_BYTES)
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"micro-YOLO output mismatch idx={idx}: "
            f"got={to_i8(got_byte)} expected={to_i8(exp_byte)}"
        )
