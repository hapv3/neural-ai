import os

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)
from test_conv_feeder import (
    L2_CONV1_INPUT,
    L2_CONV1_OUT,
    L2_CONV1_WEIGHT,
    L2_CONV3_INPUT,
    L2_CONV3_OUT,
    L2_CONV3_WEIGHT,
    STATUS_BASE,
    bytes_to_s32,
    golden_conv1,
    golden_conv3,
    make_conv1_input,
    make_conv1_weight_packed,
    make_conv3_input,
    make_conv3_weight_packed,
)


L2_CONV1_STATS = 0x80018000
L2_CONV3_STATS = 0x80038000
L2_CONV1_C32_INPUT = 0x80040000
L2_CONV1_C32_WEIGHT = 0x80044000
L2_CONV1_C32_OUT = 0x80050000
L2_CONV1_C32_STATS = 0x80058000
L2_CONV1_C64_INPUT = 0x80060000
L2_CONV1_C64_WEIGHT = 0x80068000
L2_CONV1_C64_OUT = 0x80070000
L2_CONV1_C64_STATS = 0x80078000
L2_P3_BASE = 0x80080000
P3_CASE_STRIDE = 0x00010000
P3_H = 4
P3_W = 4
OC = 32
K_TILE = 32

P3_CASE_IC1 = 0
P3_CASE_IC3 = 1
P3_CASE_IC31 = 2
P3_CASE_OC64 = 3
P3_CASE_3X3_P0_C32 = 4
P3_CASE_3X3_P1_C32 = 5
P3_CASE_5X5_P2_C3 = 6
P3_CASE_7X7_P3_C1 = 7
P3_CASE_1X3_C3 = 8
P3_CASE_3X1_C3 = 9
P3_CASE_1X5_C3 = 10
P3_CASE_5X1_C3 = 11
P3_CASE_3X3_S2_C3 = 12
P3_CASE_3X3_C1 = 13
P3_CASE_3X3_C5 = 14
P3_CASE_REQUANT = 15

P3_CASES = {
    P3_CASE_IC1: ("conv1x1 IC1", 4, 4, 1, 4, 4, 1, 1, 1, 1, 0, 0, 32, 1, 1, 0, False),
    P3_CASE_IC3: ("conv1x1 IC3", 4, 4, 3, 4, 4, 1, 1, 1, 1, 0, 0, 32, 1, 1, 0, False),
    P3_CASE_IC31: ("conv1x1 IC31", 4, 4, 31, 4, 4, 1, 1, 1, 1, 0, 0, 32, 1, 1, 0, False),
    P3_CASE_OC64: ("conv1x1 IC33 OC64", 4, 4, 33, 4, 4, 1, 1, 1, 1, 0, 0, 64, 2, 2, 0, False),
    P3_CASE_3X3_P0_C32: ("conv3x3 pad0 IC32", 4, 4, 32, 2, 2, 3, 3, 1, 1, 0, 0, 32, 9, 9, 0, False),
    P3_CASE_3X3_P1_C32: ("conv3x3 pad1 IC32", 4, 4, 32, 4, 4, 3, 3, 1, 1, 1, 1, 32, 9, 9, 0, False),
    P3_CASE_5X5_P2_C3: ("conv5x5 pad2 IC3", 4, 4, 3, 4, 4, 5, 5, 1, 1, 2, 2, 32, 3, 0, 3, False),
    P3_CASE_7X7_P3_C1: ("conv7x7 pad3 IC1", 4, 4, 1, 4, 4, 7, 7, 1, 1, 3, 3, 32, 2, 0, 2, False),
    P3_CASE_1X3_C3: ("conv1x3 IC3", 4, 4, 3, 4, 4, 1, 3, 1, 1, 0, 1, 32, 1, 0, 1, False),
    P3_CASE_3X1_C3: ("conv3x1 IC3", 4, 4, 3, 4, 4, 3, 1, 1, 1, 1, 0, 32, 1, 0, 1, False),
    P3_CASE_1X5_C3: ("conv1x5 IC3", 4, 4, 3, 4, 4, 1, 5, 1, 1, 0, 2, 32, 1, 0, 1, False),
    P3_CASE_5X1_C3: ("conv5x1 IC3", 4, 4, 3, 4, 4, 5, 1, 1, 1, 2, 0, 32, 1, 0, 1, False),
    P3_CASE_3X3_S2_C3: ("conv3x3 stride2 IC3", 4, 4, 3, 2, 2, 3, 3, 2, 2, 1, 1, 32, 1, 0, 1, False),
    P3_CASE_3X3_C1: ("conv3x3 IC1 K9", 4, 4, 1, 4, 4, 3, 3, 1, 1, 1, 1, 32, 1, 0, 1, False),
    P3_CASE_3X3_C5: ("conv3x3 IC5 K45", 4, 4, 5, 4, 4, 3, 3, 1, 1, 1, 1, 32, 2, 0, 2, False),
    P3_CASE_REQUANT: ("conv1x1 IC64 requant", 4, 4, 64, 4, 4, 1, 1, 1, 1, 0, 0, 32, 2, 2, 0, True),
}

CONV_PERF_GROUP = int(os.environ.get("CONV_PERF_GROUP", "0"))
CONV_PERF_GROUP_ALL = 0
CONV_PERF_GROUP_POINTWISE = 1
CONV_PERF_GROUP_KERNELS = 2
CONV_PERF_GROUP_REQUANT = 3


def legacy_enabled():
    return CONV_PERF_GROUP in (CONV_PERF_GROUP_ALL, CONV_PERF_GROUP_POINTWISE)


def p3_case_enabled(case_id):
    if CONV_PERF_GROUP == CONV_PERF_GROUP_ALL:
        return True
    if CONV_PERF_GROUP == CONV_PERF_GROUP_POINTWISE:
        return case_id <= P3_CASE_OC64
    if CONV_PERF_GROUP == CONV_PERF_GROUP_KERNELS:
        return P3_CASE_3X3_P0_C32 <= case_id <= P3_CASE_3X3_C5
    if CONV_PERF_GROUP == CONV_PERF_GROUP_REQUANT:
        return case_id == P3_CASE_REQUANT
    return False


def read_dtcm_word(dut, addr):
    index = (addr - STATUS_BASE) >> 2
    value = dut.u_npu_cluster.u_sram_d_tcm.mem[index].value
    return value.to_unsigned() if value.is_resolvable else 0


def bytes_to_u32(data, word_index):
    value = 0
    for byte_index in range(4):
        value |= data[(word_index * 4) + byte_index] << (byte_index * 8)
    return value


def to_u8(value):
    return value & 0xFF


def s32_to_bytes(value):
    return [(value >> shift) & 0xFF for shift in (0, 8, 16, 24)]


def conv1_input_value(h, w, c):
    return ((h * 17 + w * 11 + c * 5) % 19) - 9


def conv1_weight_value(c, oc):
    return ((c * 7 + oc * 3) % 17) - 8


def p3_input_addr(case_id):
    return L2_P3_BASE + case_id * P3_CASE_STRIDE + 0x0000


def p3_weight_addr(case_id):
    return L2_P3_BASE + case_id * P3_CASE_STRIDE + 0x3000


def p3_out_addr(case_id):
    return L2_P3_BASE + case_id * P3_CASE_STRIDE + 0x6000


def p3_stats_addr(case_id):
    return L2_P3_BASE + case_id * P3_CASE_STRIDE + 0xE000


def p3_input_value(h, w, c):
    return ((h * 13 + w * 7 + c * 5) % 19) - 9


def p3_weight_value(kh, kw, c, oc):
    return ((kh * 17 + kw * 11 + c * 7 + oc * 3) % 17) - 8


def make_conv1_input_generic(input_c, input_h=P3_H, input_w=P3_W):
    return [
        to_u8(conv1_input_value(h, w, c))
        for h in range(input_h)
        for w in range(input_w)
        for c in range(input_c)
    ]


def make_conv1_weight_packed_generic(input_c):
    packed = []
    k_blocks = (input_c + K_TILE - 1) // K_TILE
    for block in range(k_blocks):
        for lane in range(K_TILE):
            c = block * K_TILE + lane
            for oc in range(OC):
                value = conv1_weight_value(c, oc) if c < input_c else 0
                packed.append(to_u8(value))
    return packed


def golden_conv1_generic(input_c, input_h=P3_H, input_w=P3_W):
    out = []
    for h in range(input_h):
        for w in range(input_w):
            for oc in range(OC):
                acc = 0
                for c in range(input_c):
                    acc += conv1_input_value(h, w, c) * conv1_weight_value(c, oc)
                out.extend(s32_to_bytes(acc & 0xFFFFFFFF))
    return out


def p3_qparams(channel):
    return {
        "bias": (channel - 16) * 3,
        "multiplier": (channel % 5) + 1,
        "shift": channel % 4,
        "zero_point": (channel % 7) - 3,
        "clamp_min": -50,
        "clamp_max": 60,
    }


def round_shift_right(value, shift):
    if shift == 0:
        return value
    offset = 1 << (shift - 1)
    if value >= 0:
        return (value + offset) >> shift
    return -(((-value) + offset) >> shift)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def p3_requant(acc, channel):
    params = p3_qparams(channel)
    biased = acc + params["bias"]
    scaled = biased * params["multiplier"]
    rounded = round_shift_right(scaled, params["shift"])
    with_zp = rounded + params["zero_point"]
    return clamp(with_zp, params["clamp_min"], params["clamp_max"]) & 0xFF


def make_p3_input(input_h, input_w, input_c):
    return [
        to_u8(p3_input_value(h, w, c))
        for h in range(input_h)
        for w in range(input_w)
        for c in range(input_c)
    ]


def make_p3_weight_packed(input_c, kernel_h, kernel_w, oc_count):
    packed = []
    k_total = input_c * kernel_h * kernel_w
    k_blocks = (k_total + K_TILE - 1) // K_TILE
    oc_tiles = (oc_count + OC - 1) // OC
    for oc_tile in range(oc_tiles):
        oc_base = oc_tile * OC
        for block in range(k_blocks):
            for lane in range(K_TILE):
                k_index = block * K_TILE + lane
                if k_index < k_total:
                    spatial = k_index // input_c
                    c = k_index - spatial * input_c
                    kh = spatial // kernel_w
                    kw = spatial - kh * kernel_w
                for oc_lane in range(OC):
                    oc = oc_base + oc_lane
                    value = p3_weight_value(kh, kw, c, oc) if k_index < k_total and oc < oc_count else 0
                    packed.append(to_u8(value))
    return packed


def golden_p3_case(case_id):
    (
        _name,
        input_h,
        input_w,
        input_c,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        oc_count,
        _k_tiles,
        _idma,
        _spatz,
        requant_output,
    ) = P3_CASES[case_id]
    out = []
    for oh in range(output_h):
        for ow in range(output_w):
            for oc in range(oc_count):
                acc = 0
                for kh in range(kernel_h):
                    ih = oh * stride_h + kh - pad_h
                    for kw in range(kernel_w):
                        iw = ow * stride_w + kw - pad_w
                        if 0 <= ih < input_h and 0 <= iw < input_w:
                            for c in range(input_c):
                                acc += p3_input_value(ih, iw, c) * p3_weight_value(kh, kw, c, oc)
                if requant_output:
                    out.append(p3_requant(acc, oc))
                else:
                    out.extend(s32_to_bytes(acc & 0xFFFFFFFF))
    return out


async def boot_and_run(dut, test_file):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)

    fw_path = firmware_path(test_file, "sw/test/conv_perf/conv_perf.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/conv_perf` first."

    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    if legacy_enabled():
        await write_l2_bytes(dut, L2_CONV1_INPUT, make_conv1_input())
        await write_l2_bytes(dut, L2_CONV1_WEIGHT, make_conv1_weight_packed())
        await write_l2_bytes(dut, L2_CONV3_INPUT, make_conv3_input())
        await write_l2_bytes(dut, L2_CONV3_WEIGHT, make_conv3_weight_packed())
        await write_l2_bytes(dut, L2_CONV1_C32_INPUT, make_conv1_input_generic(32))
        await write_l2_bytes(dut, L2_CONV1_C32_WEIGHT, make_conv1_weight_packed_generic(32))
        await write_l2_bytes(dut, L2_CONV1_C64_INPUT, make_conv1_input_generic(64))
        await write_l2_bytes(dut, L2_CONV1_C64_WEIGHT, make_conv1_weight_packed_generic(64))
    for case_id, case in P3_CASES.items():
        if not p3_case_enabled(case_id):
            continue
        (
            _name,
            input_h,
            input_w,
            input_c,
            _output_h,
            _output_w,
            kernel_h,
            kernel_w,
            _stride_h,
            _stride_w,
            _pad_h,
            _pad_w,
            oc_count,
            _k_tiles,
            _idma,
            _spatz,
            _requant_output,
        ) = case
        await write_l2_bytes(dut, p3_input_addr(case_id), make_p3_input(input_h, input_w, input_c))
        await write_l2_bytes(dut, p3_weight_addr(case_id), make_p3_weight_packed(input_c, kernel_h, kernel_w, oc_count))
    await release_fetch(dut)
    try:
        await wait_for_host_irq(dut, timeout_cycles=1600000)
    except AssertionError as exc:
        status = read_dtcm_word(dut, STATUS_BASE + 0x00)
        pass_count = read_dtcm_word(dut, STATUS_BASE + 0x04)
        phase = read_dtcm_word(dut, STATUS_BASE + 0x18)
        op = read_dtcm_word(dut, STATUS_BASE + 0x1C)
        raise AssertionError(
            f"{exc}: status=0x{status:08x} pass_count={pass_count} phase={phase} op={op}"
        ) from exc


async def check_output(dut, name, addr, expected):
    got = await read_l2_bytes(dut, addr, len(expected))
    for index, (got_byte, expected_byte) in enumerate(zip(got, expected)):
        assert got_byte == expected_byte, (
            f"{name} byte {index}: got=0x{got_byte:02x} expected=0x{expected_byte:02x}; "
            f"got_word={bytes_to_s32(got, index // 4)} expected_word={bytes_to_s32(expected, index // 4)}"
        )


async def check_stats(dut, name, addr, expected_rows, expected_k_tiles, expected_idma_tiles=None, expected_spatz_tiles=None):
    stats = await read_l2_bytes(dut, addr, 44)
    rows = bytes_to_u32(stats, 0)
    k_tiles = bytes_to_u32(stats, 1)
    prepare_cycles = bytes_to_u32(stats, 2)
    gemm_cycles = bytes_to_u32(stats, 3)
    total_cycles = bytes_to_u32(stats, 4)
    last_prepare_cycles = bytes_to_u32(stats, 5)
    last_gemm_cycles = bytes_to_u32(stats, 6)
    status = bytes_to_u32(stats, 7)
    idma_tiles = bytes_to_u32(stats, 8)
    spatz_tiles = bytes_to_u32(stats, 9)
    scalar_tiles = bytes_to_u32(stats, 10)

    assert status == 0, f"{name}: scheduler status=0x{status:08x}"
    assert rows == expected_rows, f"{name}: rows={rows} expected={expected_rows}"
    assert k_tiles == expected_k_tiles, f"{name}: k_tiles={k_tiles} expected={expected_k_tiles}"
    assert prepare_cycles > 0, f"{name}: prepare_cycles must be non-zero"
    assert gemm_cycles > 0, f"{name}: gemm_cycles must be non-zero"
    assert total_cycles >= prepare_cycles + gemm_cycles, (
        f"{name}: total_cycles={total_cycles} prepare={prepare_cycles} gemm={gemm_cycles}"
    )
    assert last_prepare_cycles > 0, f"{name}: last_prepare_cycles must be non-zero"
    assert last_gemm_cycles > 0, f"{name}: last_gemm_cycles must be non-zero"
    assert idma_tiles + spatz_tiles + scalar_tiles == expected_k_tiles, (
        f"{name}: backend tiles idma={idma_tiles} spatz={spatz_tiles} scalar={scalar_tiles}"
    )
    assert scalar_tiles == 0, f"{name}: scalar prepare backend must not be used"
    if expected_idma_tiles is not None:
        assert idma_tiles == expected_idma_tiles, f"{name}: idma_tiles={idma_tiles} expected={expected_idma_tiles}"
    if expected_spatz_tiles is not None:
        assert spatz_tiles == expected_spatz_tiles, f"{name}: spatz_tiles={spatz_tiles} expected={expected_spatz_tiles}"

    dut._log.info(
        "%s stats: rows=%d k_tiles=%d prepare=%d gemm=%d total=%d last_prepare=%d last_gemm=%d idma=%d spatz=%d scalar=%d",
        name,
        rows,
        k_tiles,
        prepare_cycles,
        gemm_cycles,
        total_cycles,
        last_prepare_cycles,
        last_gemm_cycles,
        idma_tiles,
        spatz_tiles,
        scalar_tiles,
    )


async def check_stats_pair(dut, name, addr, expected_rows, expected_k_tiles, expected_idma_tiles, expected_spatz_tiles):
    stats = await read_l2_bytes(dut, addr, 88)
    for tile in range(2):
        base = tile * 44
        rows = bytes_to_u32(stats[base:], 0)
        k_tiles = bytes_to_u32(stats[base:], 1)
        prepare_cycles = bytes_to_u32(stats[base:], 2)
        gemm_cycles = bytes_to_u32(stats[base:], 3)
        total_cycles = bytes_to_u32(stats[base:], 4)
        last_prepare_cycles = bytes_to_u32(stats[base:], 5)
        last_gemm_cycles = bytes_to_u32(stats[base:], 6)
        status = bytes_to_u32(stats[base:], 7)
        idma_tiles = bytes_to_u32(stats[base:], 8)
        spatz_tiles = bytes_to_u32(stats[base:], 9)
        scalar_tiles = bytes_to_u32(stats[base:], 10)
        assert status == 0, f"{name} tile {tile}: scheduler status=0x{status:08x}"
        assert rows == expected_rows, f"{name} tile {tile}: rows={rows} expected={expected_rows}"
        assert k_tiles == expected_k_tiles, f"{name} tile {tile}: k_tiles={k_tiles} expected={expected_k_tiles}"
        assert prepare_cycles > 0, f"{name} tile {tile}: prepare_cycles must be non-zero"
        assert gemm_cycles > 0, f"{name} tile {tile}: gemm_cycles must be non-zero"
        assert total_cycles >= prepare_cycles + gemm_cycles, (
            f"{name} tile {tile}: total_cycles={total_cycles} prepare={prepare_cycles} gemm={gemm_cycles}"
        )
        assert last_prepare_cycles > 0, f"{name} tile {tile}: last_prepare_cycles must be non-zero"
        assert last_gemm_cycles > 0, f"{name} tile {tile}: last_gemm_cycles must be non-zero"
        assert idma_tiles == expected_idma_tiles, f"{name} tile {tile}: idma={idma_tiles} expected={expected_idma_tiles}"
        assert spatz_tiles == expected_spatz_tiles, f"{name} tile {tile}: spatz={spatz_tiles} expected={expected_spatz_tiles}"
        assert scalar_tiles == 0, f"{name} tile {tile}: scalar prepare backend must not be used"
        dut._log.info(
            "%s tile %d stats: rows=%d k_tiles=%d prepare=%d gemm=%d total=%d idma=%d spatz=%d scalar=%d",
            name,
            tile,
            rows,
            k_tiles,
            prepare_cycles,
            gemm_cycles,
            total_cycles,
            idma_tiles,
            spatz_tiles,
            scalar_tiles,
        )


@cocotb.test()
async def test_conv_perf(dut):
    await boot_and_run(dut, __file__)

    if legacy_enabled():
        await check_output(dut, "conv1x1 packed", L2_CONV1_OUT, golden_conv1())
        await check_stats(
            dut,
            "conv1x1 packed",
            L2_CONV1_STATS,
            expected_rows=20,
            expected_k_tiles=2,
            expected_idma_tiles=2,
            expected_spatz_tiles=0,
        )

        await check_output(dut, "conv3x3 packed", L2_CONV3_OUT, golden_conv3())
        await check_stats(
            dut,
            "conv3x3 packed",
            L2_CONV3_STATS,
            expected_rows=25,
            expected_k_tiles=1,
            expected_idma_tiles=0,
            expected_spatz_tiles=1,
        )

        await check_output(dut, "conv1x1 IC32 packed", L2_CONV1_C32_OUT, golden_conv1_generic(32))
        await check_stats(
            dut,
            "conv1x1 IC32 packed",
            L2_CONV1_C32_STATS,
            expected_rows=P3_H * P3_W,
            expected_k_tiles=1,
            expected_idma_tiles=1,
            expected_spatz_tiles=0,
        )

        await check_output(dut, "conv1x1 IC64 packed", L2_CONV1_C64_OUT, golden_conv1_generic(64))
        await check_stats(
            dut,
            "conv1x1 IC64 packed",
            L2_CONV1_C64_STATS,
            expected_rows=P3_H * P3_W,
            expected_k_tiles=2,
            expected_idma_tiles=2,
            expected_spatz_tiles=0,
        )

    for case_id, case in P3_CASES.items():
        if not p3_case_enabled(case_id):
            continue
        (
            name,
            _input_h,
            _input_w,
            _input_c,
            output_h,
            output_w,
            _kernel_h,
            _kernel_w,
            _stride_h,
            _stride_w,
            _pad_h,
            _pad_w,
            oc_count,
            expected_k_tiles,
            expected_idma_tiles,
            expected_spatz_tiles,
            _requant_output,
        ) = case
        await check_output(dut, name, p3_out_addr(case_id), golden_p3_case(case_id))
        if oc_count == 64:
            await check_stats_pair(
                dut,
                name,
                p3_stats_addr(case_id),
                expected_rows=output_h * output_w,
                expected_k_tiles=expected_k_tiles,
                expected_idma_tiles=expected_idma_tiles,
                expected_spatz_tiles=expected_spatz_tiles,
            )
        else:
            await check_stats(
                dut,
                name,
                p3_stats_addr(case_id),
                expected_rows=output_h * output_w,
                expected_k_tiles=expected_k_tiles,
                expected_idma_tiles=expected_idma_tiles,
                expected_spatz_tiles=expected_spatz_tiles,
            )
