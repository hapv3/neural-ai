import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    PMU_COUNTER_NAMES,
    firmware_path,
    load_firmware_axi,
    read_dtcm_word,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


L2_INPUT = 0x80000000
L2_W_STEM = 0x80010000
L2_W_DW0 = 0x80011000
L2_W_PW0 = 0x80012000
L2_W_DW1 = 0x80013000
L2_W_PW1 = 0x80014000
L2_W_PW2 = 0x80015000
L2_W_DW2 = 0x80018000
L2_W_PW3 = 0x80019000
L2_W_VALIDATE = 0x8001C000
L2_W_CLASSIFIER = 0x80026000
L2_OUTPUT = 0x80030000

INPUT_H = 96
INPUT_W = 96
DTCM_TRACE_LAYER = 0x10008020
DTCM_TRACE_OP = 0x10008024
DTCM_TRACE_EVENT = 0x10008028

OP_NAMES = {
    1: "DMA_IN",
    2: "DMA_OUT",
    8: "CONV2D_RGB_LINEBUF_REQUANT",
    13: "ADD_I8",
    20: "CONV2D_POINTWISE_C32_REQUANT",
    21: "DEPTHWISE_CONV2D_C32_REQUANT",
    22: "DEPTHWISE_CONV2D_C32_DOWNSAMPLE_REQUANT",
    23: "GLOBAL_AVGPOOL_C32_REDUCE",
    24: "CLAMP_I8",
    25: "CONV2D_C32_MULTI_LINEBUF_REQUANT",
}


def pmu_direct_snapshot(dut):
    pmu = dut.u_npu_cluster.u_pmu
    counters = {}
    for idx, name in enumerate(PMU_COUNTER_NAMES):
        counter = getattr(getattr(pmu, f"gen_counters[{idx}]"), "u_counter")
        raw_value = counter.counter_q.value
        counters[name] = raw_value.to_unsigned() if raw_value.is_resolvable else 0
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
    samples = []
    last_trace = None

    for _ in range(timeout_cycles):
        trace = (
            read_dtcm_word(dut, DTCM_TRACE_LAYER),
            read_dtcm_word(dut, DTCM_TRACE_OP),
            read_dtcm_word(dut, DTCM_TRACE_EVENT),
        )
        if trace != last_trace:
            samples.append((trace, pmu_direct_snapshot(dut)))
            last_trace = trace

        irq_value = dut.irq_o.value
        if irq_value.is_resolvable and int(irq_value) == 1:
            break
        await RisingEdge(dut.clk_i)

    final_trace = (
        read_dtcm_word(dut, DTCM_TRACE_LAYER),
        read_dtcm_word(dut, DTCM_TRACE_OP),
        read_dtcm_word(dut, DTCM_TRACE_EVENT),
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


def to_i8(value):
    value &= 0xFF
    return value - 256 if value >= 128 else value


def clamp(value, lo=-128, hi=127):
    return max(lo, min(hi, value))


def trunc_div_i32(value, divisor):
    sign = -1 if value < 0 else 1
    return sign * (abs(value) // divisor)


def c32_groups(channels):
    return (channels + 31) // 32


def c32_offset(height, width, channel, row, col):
    pixel = row * width + col
    return (((channel >> 5) * (height * width) + pixel) * 32) + (channel & 31)


def input_value(row, col, channel):
    return ((row * 3 + col * 5 + channel * 7 + 1) % 5) - 2


def stem_weight_value(kh, kw, ic, oc):
    return ((kh * 2 + kw * 3 + ic * 5 + oc * 7 + 2) % 3) - 1


def depthwise_weight_value(tag, kh, kw, channel):
    return ((tag * 3 + kh * 5 + kw * 7 + channel * 2 + 1) % 3) - 1


def pointwise_weight_value(tag, ic, oc):
    return ((tag * 5 + ic * 3 + oc * 7 + 4) % 3) - 1


def conv_weight_value(tag, kh, kw, ic, oc):
    return ((tag * 11 + kh * 3 + kw * 5 + ic * 7 + oc * 2 + 6) % 3) - 1


def make_input_bytes():
    data = []
    for row in range(INPUT_H):
        for col in range(INPUT_W):
            for channel in range(3):
                data.append(input_value(row, col, channel) & 0xFF)
    return data


def make_stem_weight_bytes():
    data = []
    for k_lane in range(32):
        if k_lane < 27:
            spatial = k_lane // 3
            kh = spatial // 3
            kw = spatial % 3
            ic = k_lane % 3
        else:
            kh = kw = ic = -1
        for oc in range(32):
            value = stem_weight_value(kh, kw, ic, oc) if k_lane < 27 else 0
            data.append(value & 0xFF)
    return data


def make_depthwise_weight_bytes(tag, channels):
    data = []
    for group in range(c32_groups(channels)):
        for kh in range(3):
            for kw in range(3):
                for lane in range(32):
                    channel = group * 32 + lane
                    value = depthwise_weight_value(tag, kh, kw, channel) if channel < channels else 0
                    data.append(value & 0xFF)
    return data


def make_pointwise_weight_bytes(tag, input_c, output_c):
    data = []
    for ocg in range(c32_groups(output_c)):
        for icg in range(c32_groups(input_c)):
            for k_lane in range(32):
                ic = icg * 32 + k_lane
                for n_lane in range(32):
                    oc = ocg * 32 + n_lane
                    value = pointwise_weight_value(tag, ic, oc) if ic < input_c and oc < output_c else 0
                    data.append(value & 0xFF)
    return data


def make_conv3x3_weight_bytes(tag, input_c, output_c):
    data = []
    for ocg in range(c32_groups(output_c)):
        for icg in range(c32_groups(input_c)):
            for kh in range(3):
                for kw in range(3):
                    for k_lane in range(32):
                        ic = icg * 32 + k_lane
                        for n_lane in range(32):
                            oc = ocg * 32 + n_lane
                            value = conv_weight_value(tag, kh, kw, ic, oc) if ic < input_c and oc < output_c else 0
                            data.append(value & 0xFF)
    return data


def assert_micro_mobilenet_layout_contract():
    # RGB stem remains raw HWC C3. All tensors after the stem use ROW32 or
    # C32-blocked storage, and all middle-layer weights are already C32-padded.
    assert len(make_input_bytes()) == INPUT_H * INPUT_W * 3

    stem_weight = make_stem_weight_bytes()
    assert len(stem_weight) == 32 * 32
    assert all(value == 0 for value in stem_weight[27 * 32:])

    assert len(make_depthwise_weight_bytes(tag=0, channels=32)) == 3 * 3 * 32
    assert len(make_depthwise_weight_bytes(tag=2, channels=128)) == 4 * 3 * 3 * 32
    assert len(make_pointwise_weight_bytes(tag=1, input_c=32, output_c=64)) == 2 * 32 * 32
    assert len(make_pointwise_weight_bytes(tag=2, input_c=64, output_c=128)) == 4 * 2 * 32 * 32
    assert len(make_conv3x3_weight_bytes(tag=4, input_c=64, output_c=64)) == 2 * 2 * 3 * 3 * 32 * 32


def zeros_c32(height, width, channels):
    return [0] * (height * width * c32_groups(channels) * 32)


def get_c32(tensor, height, width, channel, row, col):
    return to_i8(tensor[c32_offset(height, width, channel, row, col)])


def set_c32(tensor, height, width, channel, row, col, value):
    tensor[c32_offset(height, width, channel, row, col)] = clamp(value) & 0xFF


def stem_conv():
    out_h = 48
    out_w = 48
    out = zeros_c32(out_h, out_w, 32)
    for oh in range(out_h):
        for ow in range(out_w):
            for oc in range(32):
                acc = 0
                for kh in range(3):
                    ih = oh * 2 + kh - 1
                    if ih < 0 or ih >= INPUT_H:
                        continue
                    for kw in range(3):
                        iw = ow * 2 + kw - 1
                        if iw < 0 or iw >= INPUT_W:
                            continue
                        for ic in range(3):
                            acc += input_value(ih, iw, ic) * stem_weight_value(kh, kw, ic, oc)
                set_c32(out, out_h, out_w, oc, oh, ow, acc)
    return out


def clamp_tensor(src, height, width, channels, lo, hi):
    out = zeros_c32(height, width, channels)
    for row in range(height):
        for col in range(width):
            for channel in range(channels):
                set_c32(out, height, width, channel, row, col,
                        clamp(get_c32(src, height, width, channel, row, col), lo, hi))
    return out


def add_tensor(lhs, rhs, height, width, channels):
    out = zeros_c32(height, width, channels)
    for row in range(height):
        for col in range(width):
            for channel in range(channels):
                value = get_c32(lhs, height, width, channel, row, col) + get_c32(rhs, height, width, channel, row, col)
                set_c32(out, height, width, channel, row, col, value)
    return out


def depthwise(src, height, width, channels, stride, tag):
    out_h = ((height - 1) // stride) + 1
    out_w = ((width - 1) // stride) + 1
    out = zeros_c32(out_h, out_w, channels)
    for oh in range(out_h):
        for ow in range(out_w):
            for channel in range(channels):
                acc = 0
                for kh in range(3):
                    ih = oh * stride + kh - 1
                    if ih < 0 or ih >= height:
                        continue
                    for kw in range(3):
                        iw = ow * stride + kw - 1
                        if iw < 0 or iw >= width:
                            continue
                        acc += get_c32(src, height, width, channel, ih, iw) * depthwise_weight_value(tag, kh, kw, channel)
                set_c32(out, out_h, out_w, channel, oh, ow, acc)
    return out


def pointwise(src, height, width, input_c, output_c, tag):
    out = zeros_c32(height, width, output_c)
    for row in range(height):
        for col in range(width):
            for oc in range(output_c):
                acc = 0
                for ic in range(input_c):
                    acc += get_c32(src, height, width, ic, row, col) * pointwise_weight_value(tag, ic, oc)
                set_c32(out, height, width, oc, row, col, acc)
    return out


def conv3x3(src, height, width, input_c, output_c, tag):
    out = zeros_c32(height, width, output_c)
    for oh in range(height):
        for ow in range(width):
            for oc in range(output_c):
                acc = 0
                for kh in range(3):
                    ih = oh + kh - 1
                    if ih < 0 or ih >= height:
                        continue
                    for kw in range(3):
                        iw = ow + kw - 1
                        if iw < 0 or iw >= width:
                            continue
                        for ic in range(input_c):
                            acc += get_c32(src, height, width, ic, ih, iw) * conv_weight_value(tag, kh, kw, ic, oc)
                set_c32(out, height, width, oc, oh, ow, acc)
    return out


def global_avgpool(src, height, width, channels):
    out = zeros_c32(1, 1, channels)
    pixels = height * width
    for channel in range(channels):
        total = 0
        for row in range(height):
            for col in range(width):
                total += get_c32(src, height, width, channel, row, col)
        set_c32(out, 1, 1, channel, 0, 0, trunc_div_i32(total, pixels))
    return out


def golden_micro_mobilenet():
    stem = stem_conv()
    stem_relu = clamp_tensor(stem, 48, 48, 32, 0, 6)
    dw0 = depthwise(stem_relu, 48, 48, 32, 1, tag=0)
    dw0_relu = clamp_tensor(dw0, 48, 48, 32, 0, 6)
    pw0 = pointwise(dw0_relu, 48, 48, 32, 32, tag=0)
    residual0 = add_tensor(pw0, stem_relu, 48, 48, 32)
    dw1 = depthwise(residual0, 48, 48, 32, 2, tag=1)
    pw1 = pointwise(dw1, 24, 24, 32, 64, tag=1)
    pw1_relu = clamp_tensor(pw1, 24, 24, 64, 0, 6)
    pw2 = pointwise(pw1_relu, 24, 24, 64, 128, tag=2)
    dw2 = depthwise(pw2, 24, 24, 128, 1, tag=2)
    pw3 = pointwise(dw2, 24, 24, 128, 64, tag=3)
    residual1 = add_tensor(pw3, pw1_relu, 24, 24, 64)
    validate = conv3x3(residual1, 24, 24, 64, 64, tag=4)
    pooled = global_avgpool(validate, 24, 24, 64)
    return pointwise(pooled, 1, 1, 64, 32, tag=5)


async def write_fixtures(dut):
    await write_l2_bytes(dut, L2_INPUT, make_input_bytes())
    await write_l2_bytes(dut, L2_W_STEM, make_stem_weight_bytes())
    await write_l2_bytes(dut, L2_W_DW0, make_depthwise_weight_bytes(tag=0, channels=32))
    await write_l2_bytes(dut, L2_W_PW0, make_pointwise_weight_bytes(tag=0, input_c=32, output_c=32))
    await write_l2_bytes(dut, L2_W_DW1, make_depthwise_weight_bytes(tag=1, channels=32))
    await write_l2_bytes(dut, L2_W_PW1, make_pointwise_weight_bytes(tag=1, input_c=32, output_c=64))
    await write_l2_bytes(dut, L2_W_PW2, make_pointwise_weight_bytes(tag=2, input_c=64, output_c=128))
    await write_l2_bytes(dut, L2_W_DW2, make_depthwise_weight_bytes(tag=2, channels=128))
    await write_l2_bytes(dut, L2_W_PW3, make_pointwise_weight_bytes(tag=3, input_c=128, output_c=64))
    await write_l2_bytes(dut, L2_W_VALIDATE, make_conv3x3_weight_bytes(tag=4, input_c=64, output_c=64))
    await write_l2_bytes(dut, L2_W_CLASSIFIER, make_pointwise_weight_bytes(tag=5, input_c=64, output_c=32))


@cocotb.test()
async def test_micro_mobilenet_e2e_native_ops(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    fw_path = firmware_path(__file__, "sw/test/micro_mobilenet/micro_mobilenet.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/micro_mobilenet` first."

    assert_micro_mobilenet_layout_contract()
    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await write_fixtures(dut)
    timeout_cycles = 6000000
    await release_fetch(dut, axi_master=axi_master)
    step_pmu_task = cocotb.start_soon(monitor_step_pmu(dut, timeout_cycles))
    await wait_for_host_irq(
        dut,
        timeout_cycles=timeout_cycles,
        axi_master=axi_master,
        report_name="test_micro_mobilenet_e2e_native_ops",
    )
    step_pmu_lines = await step_pmu_task
    dut._log.info("test_micro_mobilenet_e2e per-layer PMU:\n  %s", "\n  ".join(step_pmu_lines))

    expected = golden_micro_mobilenet()
    got = await read_l2_bytes(dut, L2_OUTPUT, 32)
    for channel in range(32):
        exp_byte = expected[c32_offset(1, 1, channel, 0, 0)]
        assert got[channel] == exp_byte, (
            f"micro_mobilenet output channel {channel}: "
            f"got={to_i8(got[channel])} expected={to_i8(exp_byte)}"
        )
