import cocotb
from cocotb.clock import Clock
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

DST_I8 = 0x10100100
RELU_I8 = 0x10100200
DST_REQUANT = 0x10100500
ADD_DST = 0x10100800
MUL_DST = 0x10100B00
LOG_DST = 0x10100D00
POOL_DST = 0x10101200
UP_DST = 0x10101500
CONCAT_DST = 0x10101C00

VL = 32
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


def as_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def c32_index(pixel, channel, pixels=CONCAT_PIXELS):
    return (((channel // 32) * pixels + pixel) * 32) + (channel % 32)


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


def check_status(dut, expected_pass_count):
    status = read_dtcm_word(dut, DTCM_STATUS)
    assert status == PASS_SIGNATURE, (
        f"firmware status=0x{status:08x} "
        f"test={read_dtcm_word(dut, DTCM_FAIL_TEST)} "
        f"index={read_dtcm_word(dut, DTCM_FAIL_INDEX)} "
        f"got=0x{read_dtcm_word(dut, DTCM_FAIL_GOT):08x} "
        f"exp=0x{read_dtcm_word(dut, DTCM_FAIL_EXP):08x}"
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


def check_all(dut):
    check_copy(dut)
    check_relu(dut)
    check_requant(dut)
    check_add(dut)
    check_mul(dut)
    check_logistic(dut)
    check_maxpool(dut)
    check_upsample(dut)
    check_concat(dut)


async def run_firmware_case(dut, fw_name, report_name, expected_pass_count, checker):
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
        firmware_path(__file__, f"sw/test/spatz_ops/{fw_name}"),
    )
    await release_fetch(dut, axi_master=axi_master)

    await wait_for_host_irq(
        dut,
        timeout_cycles=500000,
        axi_master=axi_master,
        report_name=report_name,
    )
    check_status(dut, expected_pass_count)
    checker(dut)


@cocotb.test()
async def test_spatz_operator_library(dut):
    await run_firmware_case(
        dut, "spatz_ops_test.bin", "test_spatz_operator_library", 9, check_all
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
async def test_spatz_op_mul(dut):
    await run_firmware_case(dut, "spatz_ops_mul.bin", "test_spatz_op_mul", 1, check_mul)


@cocotb.test()
async def test_spatz_op_logistic(dut):
    await run_firmware_case(
        dut, "spatz_ops_logistic.bin", "test_spatz_op_logistic", 1, check_logistic
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
