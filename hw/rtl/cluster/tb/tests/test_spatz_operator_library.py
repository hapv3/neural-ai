import inspect
import logging

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, with_timeout
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
    read_dtcm_word,
    read_tcdm_byte,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
)


PASS_SIGNATURE = 0xDEADBEEF
DTCM_STATUS = 0x10008000
DTCM_PASS_COUNT = 0x10008004
DTCM_FAIL_TEST = 0x10008008
DTCM_FAIL_INDEX = 0x1000800C
DTCM_FAIL_GOT = 0x10008010
DTCM_FAIL_EXP = 0x10008014
NPU_AFU_STATUS = 0x20003400

DST_I8 = 0x10100100
RELU_I8 = 0x10100200
DST_REQUANT = 0x10100500
ADD_DST = 0x10100800
MUL_DST = 0x10100B00
LOG_DST = 0x10100D00
LOG_LUT = 0x10100E00
POOL_DST = 0x10101200
UP_DST = 0x10101500
CONCAT_DST = 0x10101C00
LOG_FULL_DST = 0x1011A000
MUL_Q7_LHS = 0x10108000
MUL_Q7_RHS = 0x1011A000
MUL_Q7_DST = 0x1012C000
ADD_FULL_LHS = MUL_Q7_LHS
ADD_FULL_RHS = MUL_Q7_RHS
ADD_FULL_DST = MUL_Q7_DST
DFL_ROW32_SRC = 0x10155000
DFL_ROW32_DST = 0x10156000
DFL_EXP_LUT32 = 0x10157000
DFL_RECIP_LUT = 0x10158000
CLASS_ROW32_SRC = 0x10159000
CLASS_DST = 0x1015A000
GAP_SRC = 0x1015B000
GAP_DST = 0x1015D000
CLAMP_SRC = 0x10160000
CLAMP_DST = 0x10168000

VL = 32
LOG_FULL_H = 48
LOG_FULL_W = 48
LOG_FULL_C = 32
LOG_FULL_PIXELS = LOG_FULL_H * LOG_FULL_W
LOG_FULL_BYTES = LOG_FULL_PIXELS * LOG_FULL_C
POOL_H = 4
POOL_W = 5
POOL_C = 2
POOL_K = 5
POOL_PAD = 2
UP_H = 2
UP_W = 3
UP_C = 2
UP_SCALE = 2
CONCAT_H = 2
CONCAT_W = 3
CONCAT_PIXELS = CONCAT_H * CONCAT_W
DFL_FUSED_LOCATIONS = 64
DFL16_FUSED_RECORDS = 19
DFL_Q8_RECORDS = 37
CLASS_SIGMOID_LOCATIONS = 17
GAP_H = 7
GAP_W = 5
GAP_C = 65
GAP_PIXELS = GAP_H * GAP_W
GAP_GROUPS = (GAP_C + 31) // 32
CLAMP_H = 17
CLAMP_W = 13
CLAMP_C = 65
CLAMP_GROUPS = (CLAMP_C + 31) // 32
CLAMP_PHYS_C = CLAMP_GROUPS * 32
CLAMP_BYTES = CLAMP_H * CLAMP_W * CLAMP_PHYS_C
CLAMP_MIN = 0
CLAMP_MAX = 63
DFL_SIDES = 4
DFL_REG_MAX = 4
DFL16_REG_MAX = 16


def as_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def scale_add_value(value, scale, shift, double_round_shift):
    product = value * scale
    if shift == 0:
        return product
    double_round = (
        1 << (30 - double_round_shift)
        if shift > 31 - double_round_shift
        else 0
    )
    product += (1 << (shift - 1)) + (double_round if value >= 0 else -double_round)
    return product >> shift


def safe_signal_int(getter):
    try:
        value = getter().value
        return value.to_unsigned() if value.is_resolvable else None
    except Exception:
        return None


def safe_signal_paths_int(dut, *paths):
    for path in paths:
        candidates = []
        try:
            obj = dut
            for part in path.split("."):
                obj = getattr(obj, part)
            candidates.append(obj)
        except Exception:
            pass
        for extended in (False, True):
            try:
                candidates.append(dut._id(path, extended=extended))
            except Exception:
                pass
        for obj in candidates:
            try:
                value = obj.value
                if value.is_resolvable:
                    return value.to_unsigned()
            except Exception:
                pass
    return None


def fmt_opt_hex(value):
    return "None" if value is None else f"0x{value:08x}"


async def axi_read32_or_none(axi_master, addr):
    try:
        resp = await with_timeout(axi_master.read(addr, 4), 1000, "ns")
        data = resp.data if hasattr(resp, "data") else resp
        return int.from_bytes(bytes(data), "little")
    except Exception:
        return None


def c32_index(pixel, channel, pixels=CONCAT_PIXELS):
    return (((channel // 32) * pixels + pixel) * 32) + (channel % 32)


def c32_hw_index(height, width, h, w, channel):
    pixel = h * width + w
    return (((channel // 32) * (height * width) + pixel) * 32) + (channel % 32)


def pool_input_value(h, w, c):
    return ((h * 11 + w * 7 + c * 5) % 31) - 15


def expected_pool_value(oh, ow, c):
    best = -128
    for kh in range(POOL_K):
        ih = oh + kh - POOL_PAD
        if ih < 0 or ih >= POOL_H:
            continue
        for kw in range(POOL_K):
            iw = ow + kw - POOL_PAD
            if iw < 0 or iw >= POOL_W:
                continue
            best = max(best, pool_input_value(ih, iw, c))
    return best


def up_input_value(h, w, c):
    return h * 17 + w * 9 + c * 3 - 20


def read_tcdm_u16(dut, addr):
    lo = read_tcdm_byte(dut, addr)
    hi = read_tcdm_byte(dut, addr + 1)
    return lo | (hi << 8)


def dfl_exp_lut_value(index):
    if index == 0:
        return 32768
    neg_delta = 256 - index
    shift = neg_delta >> 4
    frac = neg_delta & 15
    base = 1 if shift >= 15 else 32768 >> shift
    next_value = base >> 1 if base > 1 else 1
    value = ((base * (16 - frac)) + (next_value * frac) + 8) >> 4
    return value or 1


def dfl_input_value(loc, channel):
    return ((loc * 37 + channel * 19 + 11) & 0xFF) - 128


def sigmoid_lut_value(index):
    return (index * 3 + 7) & 0xFF


def class_sigmoid_input_value(loc, channel):
    return ((loc * 29 + channel * 13 + 5) & 0xFF) - 128


def gap_input_value(h, w, channel):
    return ((h * 23 + w * 17 + channel * 11 + 9) & 0x7F) - 64


def trunc_div_i32(value, divisor):
    quotient = abs(value) // divisor
    return -quotient if value < 0 else quotient


def gap_expected_value(channel):
    total = sum(
        gap_input_value(h, w, channel)
        for h in range(GAP_H)
        for w in range(GAP_W)
    )
    return trunc_div_i32(total, GAP_PIXELS)


def clamp_input_value(h, w, channel):
    return ((h * 31 + w * 19 + channel * 7 + 5) & 0xFF) - 128


def clamp_expected_value(h, w, channel):
    return min(max(clamp_input_value(h, w, channel), CLAMP_MIN), CLAMP_MAX)


def dfl_delta_index(value, max_value):
    return (value - max_value) & 0xFF


def dfl_recip_lut_value(index):
    midpoint_q9 = 512 + (index * 2) + 1
    return ((1 << 37) + (midpoint_q9 >> 1)) // midpoint_q9


def dfl_recip_index_from_sum(total):
    shift = total.bit_length() - 1
    norm_q8 = (total << 8) >> shift
    if norm_q8 < 256:
        return 0
    if norm_q8 > 511:
        return 255
    return norm_q8 & 0xFF


def dfl_expected_fused_value(loc, side):
    base_channel = side * DFL_REG_MAX
    values = [dfl_input_value(loc, base_channel + bin_idx) for bin_idx in range(DFL_REG_MAX)]
    max_value = max(values)
    exp_values = [dfl_exp_lut_value(dfl_delta_index(value, max_value)) for value in values]
    total = sum(exp_values)
    weighted = sum(bin_idx * exp_values[bin_idx] for bin_idx in range(DFL_REG_MAX))
    shift = total.bit_length() - 1
    recip = dfl_recip_lut_value(dfl_recip_index_from_sum(total))
    total_shift = 28 + shift
    rounded = (((weighted << 8) * recip) + (1 << (total_shift - 1))) >> total_shift
    return min(rounded, 0xFFFF)


def dfl16_input_value(record, bin_idx):
    return ((record * 13 + bin_idx * 11 + 7) % 61) - 30


def dfl16_expected_fused_value(record):
    values = [dfl16_input_value(record, bin_idx) for bin_idx in range(DFL16_REG_MAX)]
    max_value = max(values)
    exp_values = [dfl_exp_lut_value(dfl_delta_index(value, max_value)) for value in values]
    total = sum(exp_values)
    weighted = sum(bin_idx * exp_values[bin_idx] for bin_idx in range(DFL16_REG_MAX))
    shift = total.bit_length() - 1
    recip = dfl_recip_lut_value(dfl_recip_index_from_sum(total))
    total_shift = 28 + shift
    rounded = (((weighted << 8) * recip) + (1 << (total_shift - 1))) >> total_shift
    return min(rounded, 0xFFFF)


def pack_u32_le(values):
    data = []
    for value in values:
        data.extend([(value >> shift) & 0xFF for shift in (0, 8, 16, 24)])
    return data


def check_status(dut, expected_pass_count):
    status = read_dtcm_word(dut, DTCM_STATUS)
    afu_core_state = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.i_core.state_q)
    afu_elem_cnt = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.i_core.elem_cnt_q)
    rfifo_cnt = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.i_rfifo.cnt_q)
    wfifo_cnt = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.i_wfifo.cnt_q)
    assert status == PASS_SIGNATURE, (
        f"firmware status=0x{status:08x} "
        f"test={read_dtcm_word(dut, DTCM_FAIL_TEST)} "
        f"index={read_dtcm_word(dut, DTCM_FAIL_INDEX)} "
        f"got=0x{read_dtcm_word(dut, DTCM_FAIL_GOT):08x} "
        f"exp=0x{read_dtcm_word(dut, DTCM_FAIL_EXP):08x} "
        f"afu_state={afu_core_state} afu_elem_cnt={afu_elem_cnt} "
        f"rfifo_cnt={rfifo_cnt} wfifo_cnt={wfifo_cnt}"
    )
    assert read_dtcm_word(dut, DTCM_PASS_COUNT) == expected_pass_count


def check_copy(dut):
    for idx in range(VL):
        expected = idx - 16
        got = as_i8(read_tcdm_byte(dut, DST_I8 + idx))
        assert got == expected, f"copy_i8[{idx}] got={got} expected={expected}"


def check_relu(dut):
    for idx in range(VL):
        expected = 0 if idx < 12 else idx - 12
        got = as_i8(read_tcdm_byte(dut, RELU_I8 + idx))
        assert got == expected, f"relu_i8[{idx}] got={got} expected={expected}"


def check_requant(dut):
    for idx in range(VL):
        src = (idx - 16) * 37
        expected = max(-20, min(31, (src * 2) >> 3))
        got = as_i8(read_tcdm_byte(dut, DST_REQUANT + idx))
        assert got == expected, f"requant[{idx}] got={got} expected={expected}"


def check_add(dut):
    for idx in range(VL):
        lhs = idx - 16
        rhs = (idx % 9) - 4
        expected = max(-12, min(18, lhs + rhs))
        got = as_i8(read_tcdm_byte(dut, ADD_DST + idx))
        assert got == expected, f"add_i8[{idx}] got={got} expected={expected}"


def check_mul(dut):
    for idx in range(VL):
        lhs = (idx % 9) - 4
        rhs = (idx % 7) - 3
        expected = max(-30, min(31, (lhs * rhs * 3) >> 2))
        got = as_i8(read_tcdm_byte(dut, MUL_DST + idx))
        assert got == expected, f"mul_i8[{idx}] got={got} expected={expected}"


def check_logistic(dut):
    for idx in range(VL):
        src_byte = (idx * 13 + 5) & 0xFF
        expected = as_i8((src_byte * 3 + 7) & 0xFF)
        got = as_i8(read_tcdm_byte(dut, LOG_DST + idx))
        assert got == expected, f"logistic_lut[{idx}] got={got} expected={expected}"


def logistic_full_input_byte(pixel, channel):
    return (pixel * 17 + channel * 13 + 5) & 0xFF


def logistic_full_lut_byte(index):
    return (index * 5 + 11) & 0x7F


def write_tcdm_bytes_aligned32(dut, base_addr, data):
    assert (base_addr & 31) == 0
    assert (len(data) & 31) == 0
    for offset in range(0, len(data), 32):
        addr = base_addr + offset
        bank_idx = (addr >> 5) % 16
        word_index = ((addr >> 5) // 16) & (1024 - 1)
        word = 0
        for byte_idx, byte_val in enumerate(data[offset : offset + 32]):
            word |= (byte_val & 0xFF) << (byte_idx * 8)
        dut.u_npu_cluster.gen_sram_banks[bank_idx].u_sram_bank.mem[word_index].value = word


async def preload_logistic_full_tcdm(dut):
    src = [
        logistic_full_input_byte(idx // LOG_FULL_C, idx % LOG_FULL_C)
        for idx in range(LOG_FULL_BYTES)
    ]
    write_tcdm_bytes_aligned32(dut, 0x10108000, src)
    await Timer(1, "ps")
    for idx in (0, 31, 32, 63, 64, 95, 96):
        got = read_tcdm_byte(dut, 0x10108000 + idx)
        expected = src[idx]
        assert got == expected, f"preload src[{idx}] got=0x{got:02x} expected=0x{expected:02x}"


def check_logistic_full(dut):
    for idx in range(LOG_FULL_BYTES):
        pixel = idx // LOG_FULL_C
        channel = idx % LOG_FULL_C
        src_byte = logistic_full_input_byte(pixel, channel)
        expected = as_i8(logistic_full_lut_byte(src_byte))
        got = as_i8(read_tcdm_byte(dut, LOG_FULL_DST + idx))
        assert got == expected, f"logistic_full[{idx}] got={got} expected={expected}"


async def preload_clamp_relu6_tcdm(dut):
    src = []
    for group in range(CLAMP_GROUPS):
        for pixel in range(CLAMP_H * CLAMP_W):
            h = pixel // CLAMP_W
            w = pixel % CLAMP_W
            for lane in range(32):
                channel = (group * 32) + lane
                src.append(clamp_input_value(h, w, channel) & 0xFF)

    write_tcdm_bytes_aligned32(dut, CLAMP_SRC, src)
    write_tcdm_bytes_aligned32(dut, CLAMP_DST, [0x55] * CLAMP_BYTES)
    await Timer(1, "ps")


def mul_q7_lhs_value(index):
    return as_i8((index * 19 + 7) & 0xFF)


def mul_q7_rhs_value(index):
    return as_i8((index * 23 + 11) & 0x7F)


async def preload_mul_q7_full_tcdm(dut):
    lhs = [(mul_q7_lhs_value(idx) & 0xFF) for idx in range(LOG_FULL_BYTES)]
    rhs = [(mul_q7_rhs_value(idx) & 0xFF) for idx in range(LOG_FULL_BYTES)]
    write_tcdm_bytes_aligned32(dut, MUL_Q7_LHS, lhs)
    write_tcdm_bytes_aligned32(dut, MUL_Q7_RHS, rhs)
    write_tcdm_bytes_aligned32(dut, MUL_Q7_DST, [0] * LOG_FULL_BYTES)
    await Timer(1, "ps")
    for idx in (0, 31, 32, 63, 64, 95, 96):
        got_lhs = as_i8(read_tcdm_byte(dut, MUL_Q7_LHS + idx))
        got_rhs = as_i8(read_tcdm_byte(dut, MUL_Q7_RHS + idx))
        assert got_lhs == mul_q7_lhs_value(idx)
        assert got_rhs == mul_q7_rhs_value(idx)


def check_mul_q7_full(dut):
    for idx in range(LOG_FULL_BYTES):
        lhs = mul_q7_lhs_value(idx)
        rhs = mul_q7_rhs_value(idx)
        expected = max(-128, min(127, (lhs * rhs) >> 7))
        got = as_i8(read_tcdm_byte(dut, MUL_Q7_DST + idx))
        assert got == expected, f"mul_q7_full[{idx}] got={got} expected={expected}"


def add_full_lhs_value(index):
    return as_i8((index * 29 + 17) & 0xFF)


def add_full_rhs_value(index):
    return as_i8((index * 31 + 23) & 0xFF)


async def preload_add_full_tcdm(dut):
    lhs = [(add_full_lhs_value(idx) & 0xFF) for idx in range(LOG_FULL_BYTES)]
    rhs = [(add_full_rhs_value(idx) & 0xFF) for idx in range(LOG_FULL_BYTES)]
    write_tcdm_bytes_aligned32(dut, ADD_FULL_LHS, lhs)
    write_tcdm_bytes_aligned32(dut, ADD_FULL_RHS, rhs)
    write_tcdm_bytes_aligned32(dut, ADD_FULL_DST, [0] * LOG_FULL_BYTES)
    await Timer(1, "ps")
    for idx in (0, 31, 32, 63, 64, 95, 96):
        assert as_i8(read_tcdm_byte(dut, ADD_FULL_LHS + idx)) == add_full_lhs_value(idx)
        assert as_i8(read_tcdm_byte(dut, ADD_FULL_RHS + idx)) == add_full_rhs_value(idx)


def check_add_full(dut):
    for idx in range(LOG_FULL_BYTES):
        lhs = add_full_lhs_value(idx)
        rhs = add_full_rhs_value(idx)
        expected = max(-128, min(127, lhs + rhs))
        got = as_i8(read_tcdm_byte(dut, ADD_FULL_DST + idx))
        assert got == expected, f"add_full[{idx}] got={got} expected={expected}"


def check_maxpool(dut):
    for h in range(POOL_H):
        for w in range(POOL_W):
            for c in range(POOL_C):
                idx = (h * POOL_W + w) * POOL_C + c
                expected = expected_pool_value(h, w, c)
                got = as_i8(read_tcdm_byte(dut, POOL_DST + idx))
                assert got == expected, f"maxpool[{idx}] got={got} expected={expected}"


def check_upsample(dut):
    up_out_w = UP_W * UP_SCALE
    for h in range(UP_H * UP_SCALE):
        for w in range(UP_W * UP_SCALE):
            for c in range(UP_C):
                idx = (h * up_out_w + w) * UP_C + c
                expected = up_input_value(h // UP_SCALE, w // UP_SCALE, c)
                got = as_i8(read_tcdm_byte(dut, UP_DST + idx))
                assert got == expected, f"upsample[{idx}] got={got} expected={expected}"


def check_concat(dut):
    for pixel in range(CONCAT_PIXELS):
        for channel in range(64):
            idx = c32_index(pixel, channel)
            expected = (
                pixel + channel - 20
                if channel < 32
                else 50 + pixel - (channel - 32)
            )
            got = as_i8(read_tcdm_byte(dut, CONCAT_DST + idx))
            assert got == expected, f"concat_c32[{idx}] got={got} expected={expected}"


async def preload_dfl_fused_tcdm(dut):
    src = []
    for loc in range(DFL_FUSED_LOCATIONS):
        for channel in range(32):
            value = dfl_input_value(loc, channel) if channel < 16 else loc + channel
            src.append(value & 0xFF)

    write_tcdm_bytes_aligned32(dut, DFL_ROW32_SRC, src)
    write_tcdm_bytes_aligned32(
        dut, DFL_EXP_LUT32, pack_u32_le(dfl_exp_lut_value(i) for i in range(256))
    )
    write_tcdm_bytes_aligned32(
        dut, DFL_RECIP_LUT, pack_u32_le(dfl_recip_lut_value(i) for i in range(256))
    )
    write_tcdm_bytes_aligned32(
        dut, DFL_ROW32_DST, [0] * (DFL_FUSED_LOCATIONS * DFL_SIDES * 2)
    )
    await Timer(1, "ps")


def check_dfl_fused(dut):
    for loc in range(DFL_FUSED_LOCATIONS):
        for side in range(DFL_SIDES):
            idx = loc * DFL_SIDES + side
            expected = dfl_expected_fused_value(loc, side)
            got = read_tcdm_u16(dut, DFL_ROW32_DST + idx * 2)
            assert got == expected, (
                f"dfl_fused_q8[{idx}] got={got} expected={expected}"
            )


async def preload_dfl16_fused_tcdm(dut):
    src = []
    for record in range(DFL16_FUSED_RECORDS):
        for channel in range(32):
            value = dfl16_input_value(record, channel) if channel < 16 else record + channel
            src.append(value & 0xFF)

    write_tcdm_bytes_aligned32(dut, DFL_ROW32_SRC, src)
    write_tcdm_bytes_aligned32(
        dut, DFL_EXP_LUT32, pack_u32_le(dfl_exp_lut_value(i) for i in range(256))
    )
    write_tcdm_bytes_aligned32(
        dut, DFL_RECIP_LUT, pack_u32_le(dfl_recip_lut_value(i) for i in range(256))
    )
    output_bytes = DFL16_FUSED_RECORDS * 2
    output_aligned_bytes = ((output_bytes + 31) // 32) * 32
    write_tcdm_bytes_aligned32(dut, DFL_ROW32_DST, [0] * output_aligned_bytes)
    await Timer(1, "ps")


def check_dfl16_fused(dut):
    for record in range(DFL16_FUSED_RECORDS):
        expected = dfl16_expected_fused_value(record)
        got = read_tcdm_u16(dut, DFL_ROW32_DST + record * 2)
        assert got == expected, (
            f"dfl16_fused_q8[{record}] got={got} expected={expected}"
        )


def check_dfl_q8(dut):
    for index in range(DFL_Q8_RECORDS):
        value = 0 if index == 0 else (3840 if index == 1 else (index * 379 + 127) % 4096)
        multiplier, shift = ((58267, 21) if index < 13 else
                             ((58267, 20) if index < 25 else (58039, 19)))
        expected = ((value * multiplier + (1 << (shift - 1))) >> shift) - 128
        expected = max(-128, min(127, expected))
        got = as_i8(read_tcdm_byte(dut, DFL_ROW32_DST + index))
        assert got == expected, f"dfl_q8[{index}] got={got} expected={expected}"


async def preload_class_sigmoid_tcdm(dut):
    src = []
    for loc in range(CLASS_SIGMOID_LOCATIONS):
        for channel in range(32):
            src.append(class_sigmoid_input_value(loc, channel) & 0xFF)

    write_tcdm_bytes_aligned32(dut, CLASS_ROW32_SRC, src)
    write_tcdm_bytes_aligned32(dut, LOG_LUT, [sigmoid_lut_value(i) for i in range(256)])
    class_dst_bytes = CLASS_SIGMOID_LOCATIONS * 16
    class_dst_aligned_bytes = ((class_dst_bytes + 31) // 32) * 32
    write_tcdm_bytes_aligned32(dut, CLASS_DST, [0] * class_dst_aligned_bytes)
    await Timer(1, "ps")


def check_class_sigmoid(dut):
    for loc in range(CLASS_SIGMOID_LOCATIONS):
        for cls in range(16):
            idx = loc * 16 + cls
            input_byte = class_sigmoid_input_value(loc, cls + 16) & 0xFF
            expected = as_i8(sigmoid_lut_value(input_byte))
            got = as_i8(read_tcdm_byte(dut, CLASS_DST + idx))
            assert got == expected, (
                f"class_sigmoid[{idx}] got={got} expected={expected}"
            )


async def preload_global_avgpool_tcdm(dut):
    src = [0] * (GAP_PIXELS * GAP_GROUPS * 32)
    for h in range(GAP_H):
        for w in range(GAP_W):
            for channel in range(GAP_C):
                idx = c32_hw_index(GAP_H, GAP_W, h, w, channel)
                src[idx] = gap_input_value(h, w, channel) & 0xFF

    write_tcdm_bytes_aligned32(dut, GAP_SRC, src)
    write_tcdm_bytes_aligned32(dut, GAP_DST, [0x5A] * (GAP_GROUPS * 32))
    await Timer(1, "ps")


def check_global_avgpool(dut):
    for channel in range(GAP_C):
        idx = ((channel // 32) * 32) + (channel % 32)
        expected = gap_expected_value(channel)
        got = as_i8(read_tcdm_byte(dut, GAP_DST + idx))
        assert got == expected, (
            f"global_avgpool[{idx}] got={got} expected={expected}"
        )


def check_clamp_relu6(dut):
    for h in range(CLAMP_H):
        for w in range(CLAMP_W):
            for channel in range(CLAMP_PHYS_C):
                idx = c32_hw_index(CLAMP_H, CLAMP_W, h, w, channel)
                expected = clamp_expected_value(h, w, channel)
                got = as_i8(read_tcdm_byte(dut, CLAMP_DST + idx))
                assert got == expected, (
                    f"clamp_relu6[{idx}] got={got} expected={expected}"
                )


def check_all(dut):
    check_copy(dut)
    check_relu(dut)
    check_requant(dut)
    check_add(dut)
    check_mul(dut)
    check_maxpool(dut)
    check_upsample(dut)
    check_concat(dut)


async def run_firmware_case(
    dut,
    fw_name,
    report_name,
    expected_pass_count,
    checker,
    timeout_cycles=500000,
    pre_release=None,
    firmware_dir="sw/test/spatz_ops",
):
    logging.getLogger("cocotb.tb_npu_cluster.s_axi").setLevel(logging.WARNING)

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
        firmware_path(__file__, f"{firmware_dir}/{fw_name}"),
    )
    if pre_release is not None:
        await pre_release(dut)
    await release_fetch(dut, axi_master=axi_master)

    try:
        await wait_for_host_irq(
            dut,
            timeout_cycles=timeout_cycles,
            axi_master=axi_master,
            report_name=report_name,
        )
    except AssertionError as exc:
        afu_core_state = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.i_core.state_q)
        afu_elem_cnt = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.i_core.elem_cnt_q)
        afu_core_done = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.core_done",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__core_done",
        )
        afu_done = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.done_o",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__done_o",
        )
        afu_backend_idle = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.backend_idle",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__backend_idle",
        )
        wfifo_empty = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.wfifo_empty",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__wfifo_empty",
        )
        wfifo_all_empty = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.wfifo_all_empty",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__wfifo_all_empty",
        )
        wfifo_push = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.wfifo_push",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__wfifo_push",
        )
        wfifo_pop = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.wfifo_pop",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__wfifo_pop",
        )
        wfifo_cnt = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.i_wfifo.cnt_q)
        backend_re_active = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.i_backend.re_active_q",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__i_backend__DOT__re_active_q",
        )
        backend_read_outstanding = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.i_backend.read_outstanding_q",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__i_backend__DOT__read_outstanding_q",
        )
        backend_pending_valid = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.i_backend.pending_valid_q",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__i_backend__DOT__pending_valid_q",
        )
        backend_obi_req = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.obi_m_req_o",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__obi_m_req_o",
        )
        backend_obi_gnt = safe_signal_paths_int(
            dut,
            "u_npu_cluster.u_afu.obi_m_gnt_i",
            "tb_npu_cluster__DOT__u_npu_cluster__DOT__u_afu__DOT__obi_m_gnt_i",
        )
        afu_mm_req = safe_signal_int(lambda: dut.u_npu_cluster.afu_mm_req)
        afu_mm_gnt = safe_signal_int(lambda: dut.u_npu_cluster.afu_mm_gnt)
        afu_mm_rvalid = safe_signal_int(lambda: dut.u_npu_cluster.afu_mm_rvalid)
        afu_mm_rdata = safe_signal_int(lambda: dut.u_npu_cluster.afu_mm_rdata)
        afu_s_req = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.obi_s_req_i)
        afu_s_rvalid = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.obi_s_rvalid_o)
        afu_s_rdata = safe_signal_int(lambda: dut.u_npu_cluster.u_afu.obi_s_rdata_o)
        frontend_s_req = safe_signal_int(
            lambda: dut.u_npu_cluster.u_afu.i_frontend.obi_s_req_i
        )
        frontend_s_rvalid = safe_signal_int(
            lambda: dut.u_npu_cluster.u_afu.i_frontend.obi_s_rvalid_o
        )
        frontend_s_rdata = safe_signal_int(
            lambda: dut.u_npu_cluster.u_afu.i_frontend.obi_s_rdata_o
        )
        afu_status_axi = await axi_read32_or_none(axi_master, NPU_AFU_STATUS)
        raise AssertionError(
            f"{exc}: status=0x{read_dtcm_word(dut, DTCM_STATUS):08x} "
            f"test={read_dtcm_word(dut, DTCM_FAIL_TEST)} "
            f"index={read_dtcm_word(dut, DTCM_FAIL_INDEX)} "
            f"got=0x{read_dtcm_word(dut, DTCM_FAIL_GOT):08x} "
            f"exp=0x{read_dtcm_word(dut, DTCM_FAIL_EXP):08x} "
            f"afu_state={afu_core_state} afu_elem_cnt={afu_elem_cnt} "
            f"afu_core_done={afu_core_done} afu_done={afu_done} "
            f"backend_idle={afu_backend_idle} wfifo_empty={wfifo_empty} "
            f"wfifo_all_empty={wfifo_all_empty} wfifo_push={wfifo_push} "
            f"wfifo_pop={wfifo_pop} wfifo_cnt={wfifo_cnt} "
            f"re_active={backend_re_active} read_outstanding={backend_read_outstanding} "
            f"pending_valid={backend_pending_valid} obi_req={backend_obi_req} obi_gnt={backend_obi_gnt} "
            f"afu_mm_req={afu_mm_req} afu_mm_gnt={afu_mm_gnt} "
            f"afu_mm_rvalid={afu_mm_rvalid} afu_mm_rdata={fmt_opt_hex(afu_mm_rdata)} "
            f"afu_status_axi={fmt_opt_hex(afu_status_axi)} "
            f"afu_s_req={afu_s_req} afu_s_rvalid={afu_s_rvalid} "
            f"afu_s_rdata={fmt_opt_hex(afu_s_rdata)} frontend_s_req={frontend_s_req} "
            f"frontend_s_rvalid={frontend_s_rvalid} frontend_s_rdata={fmt_opt_hex(frontend_s_rdata)}"
        ) from exc
    check_status(dut, expected_pass_count)
    check_result = checker(dut)
    if inspect.isawaitable(check_result):
        await check_result


@cocotb.test()
async def test_spatz_operator_library(dut):
    await run_firmware_case(
        dut, "spatz_ops_test.bin", "test_spatz_operator_library", 8, check_all
    )


@cocotb.test()
async def test_spatz_op_copy(dut):
    await run_firmware_case(dut, "spatz_ops_copy.bin", "test_spatz_op_copy", 1, check_copy)


@cocotb.test()
async def test_spatz_op_relu(dut):
    await run_firmware_case(dut, "spatz_ops_relu.bin", "test_spatz_op_relu", 1, check_relu)


@cocotb.test()
async def test_spatz_op_requant(dut):
    await run_firmware_case(
        dut, "spatz_ops_requant.bin", "test_spatz_op_requant", 1, check_requant
    )


@cocotb.test()
async def test_spatz_op_add(dut):
    await run_firmware_case(dut, "spatz_ops_add.bin", "test_spatz_op_add", 1, check_add)


@cocotb.test()
async def test_spatz_op_quantized_add(dut):
    def check_quantized_add(dut):
        for index in range(VL):
            lhs = as_i8(index * 17 + 3) - (-3)
            rhs = as_i8(index * 29 + 11) - 5
            value = scale_add_value(lhs, 1610612736, 20, 20)
            value += scale_add_value(rhs, 1073741824, 20, 20)
            value = scale_add_value(value, 1073741824, 41, 0) + 7
            expected = min(max(value, -100), 100)
            got = as_i8(read_tcdm_byte(dut, ADD_DST + index))
            assert got == expected, (
                f"quantized add mismatch at {index}: got={got} expected={expected}"
            )

    await run_firmware_case(
        dut,
        "spatz_ops_quant_add.bin",
        "test_spatz_op_quantized_add",
        1,
        check_quantized_add,
    )


@cocotb.test()
async def test_afu_op_add_full(dut):
    await run_firmware_case(
        dut,
        "afu_ops_add_full.bin",
        "test_afu_op_add_full",
        1,
        check_add_full,
        timeout_cycles=900000,
        pre_release=preload_add_full_tcdm,
        firmware_dir="sw/test/afu_ops",
    )


@cocotb.test()
async def test_spatz_op_mul(dut):
    await run_firmware_case(dut, "spatz_ops_mul.bin", "test_spatz_op_mul", 1, check_mul)


@cocotb.test()
async def test_afu_op_mul_q7_full(dut):
    await run_firmware_case(
        dut,
        "afu_ops_mul_q7_full.bin",
        "test_afu_op_mul_q7_full",
        1,
        check_mul_q7_full,
        timeout_cycles=900000,
        pre_release=preload_mul_q7_full_tcdm,
        firmware_dir="sw/test/afu_ops",
    )


@cocotb.test()
async def test_afu_op_logistic(dut):
    await run_firmware_case(
        dut,
        "afu_ops_logistic.bin",
        "test_afu_op_logistic",
        1,
        check_logistic,
        firmware_dir="sw/test/afu_ops",
    )


@cocotb.test()
async def test_afu_op_logistic_full(dut):
    await run_firmware_case(
        dut,
        "afu_ops_logistic_full.bin",
        "test_afu_op_logistic_full",
        1,
        check_logistic_full,
        timeout_cycles=180000,
        pre_release=preload_logistic_full_tcdm,
        firmware_dir="sw/test/afu_ops",
    )


@cocotb.test()
async def test_afu_op_clamp_relu6(dut):
    await run_firmware_case(
        dut,
        "afu_ops_clamp_relu6.bin",
        "test_afu_op_clamp_relu6",
        1,
        check_clamp_relu6,
        pre_release=preload_clamp_relu6_tcdm,
        firmware_dir="sw/test/afu_ops",
    )


@cocotb.test()
async def test_spatz_op_maxpool(dut):
    await run_firmware_case(
        dut, "spatz_ops_maxpool.bin", "test_spatz_op_maxpool", 1, check_maxpool
    )


@cocotb.test()
async def test_spatz_op_upsample(dut):
    await run_firmware_case(
        dut, "spatz_ops_upsample.bin", "test_spatz_op_upsample", 1, check_upsample
    )


@cocotb.test()
async def test_spatz_op_concat(dut):
    await run_firmware_case(
        dut, "spatz_ops_concat.bin", "test_spatz_op_concat", 1, check_concat
    )


@cocotb.test()
async def test_afu_op_dfl_fused(dut):
    await run_firmware_case(
        dut,
        "afu_ops_dfl_fused.bin",
        "test_afu_op_dfl_fused",
        1,
        check_dfl_fused,
        timeout_cycles=200000,
        pre_release=preload_dfl_fused_tcdm,
        firmware_dir="sw/test/afu_ops",
    )


@cocotb.test()
async def test_afu_op_dfl16_fused(dut):
    await run_firmware_case(
        dut,
        "afu_ops_dfl16_fused.bin",
        "test_afu_op_dfl16_fused",
        1,
        check_dfl16_fused,
        timeout_cycles=200000,
        pre_release=preload_dfl16_fused_tcdm,
        firmware_dir="sw/test/afu_ops",
    )


@cocotb.test()
async def test_spatz_op_dfl_q8(dut):
    await run_firmware_case(
        dut,
        "spatz_ops_dfl_q8.bin",
        "test_spatz_op_dfl_q8",
        1,
        check_dfl_q8,
    )


@cocotb.test()
async def test_spatz_op_dfl16_pack(dut):
    await run_firmware_case(
        dut,
        "spatz_ops_dfl16_pack.bin",
        "test_spatz_op_dfl16_pack",
        1,
        lambda _dut: None,
    )


@cocotb.test()
async def test_afu_op_class_sigmoid(dut):
    await run_firmware_case(
        dut,
        "afu_ops_class_sigmoid.bin",
        "test_afu_op_class_sigmoid",
        1,
        check_class_sigmoid,
        firmware_dir="sw/test/afu_ops",
        pre_release=preload_class_sigmoid_tcdm,
    )


@cocotb.test()
async def test_afu_op_global_avgpool(dut):
    await run_firmware_case(
        dut,
        "afu_ops_global_avgpool.bin",
        "test_afu_op_global_avgpool",
        1,
        check_global_avgpool,
        firmware_dir="sw/test/afu_ops",
        pre_release=preload_global_avgpool_tcdm,
    )
