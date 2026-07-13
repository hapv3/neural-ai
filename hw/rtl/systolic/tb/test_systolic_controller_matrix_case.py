import os

import cocotb

from systolic_controller_case_utils import (
    ARRAY_DIM,
    BEAT_BYTES,
    IFM_ADDR,
    OFM_ADDR,
    WEIGHT_ADDR,
    clear_output_i32,
    clear_output_i8,
    disable_optional_modes,
    fill_weight_tiles,
    pack_u8,
    pool_expected_row,
    program_direct_gemm,
    program_linebuf,
    read_s32_row,
    read_u8_row,
    start_and_wait,
    start_clock_and_reset,
    valid_tap_count,
    write_bytes,
)
from systolic_controller_matrix import get_case


def input_pattern(input_h, input_w, input_c, mode):
    values = []
    for ih in range(input_h):
        for iw in range(input_w):
            for ch in range(input_c):
                if mode == "pool":
                    values.append((((ih * 17 + iw * 7 + ch * 5) % 113) - 56) & 0xFF)
                else:
                    values.append(1)
    return values


async def run_direct_case(dut, case):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)
    fill_weight_tiles(dut, WEIGHT_ADDR)
    for row in range(case.dim_m):
        dut.tcdm_mem[(IFM_ADDR >> 5) + row].value = pack_u8([1] * BEAT_BYTES)
    clear_output_i32(dut, OFM_ADDR, case.dim_m)

    await program_direct_gemm(dut, dim_m=case.dim_m)
    stats = await start_and_wait(dut, timeout_cycles=2000 + case.dim_m * 80)

    assert stats["weight"] == ARRAY_DIM
    assert stats["compute"] == case.dim_m
    for row in [0, case.dim_m // 2, case.dim_m - 1]:
        assert read_s32_row(dut, OFM_ADDR, row) == [ARRAY_DIM] * ARRAY_DIM


async def run_linebuf_case(dut, case):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    input_data = input_pattern(case.input_h, case.input_w, case.input_c, case.mode)
    write_bytes(dut, IFM_ADDR, input_data)

    if not case.pool:
        fill_weight_tiles(dut, WEIGHT_ADDR, k_tiles=case.k_tiles)
        clear_output_i32(dut, OFM_ADDR, case.spatial_m)
    else:
        clear_output_i8(dut, OFM_ADDR, case.spatial_m)

    dim_m = case.spatial_m
    if not (case.coalesce or case.kgen or case.pool):
        dim_m = case.spatial_m * case.kernel_h * case.kernel_w

    await program_linebuf(
        dut,
        input_base=IFM_ADDR,
        input_h=case.input_h,
        input_w=case.input_w,
        input_c=case.input_c,
        output_h=case.output_h,
        output_w=case.output_w,
        kernel_h=case.kernel_h,
        kernel_w=case.kernel_w,
        stride_h=case.stride_h,
        stride_w=case.stride_w,
        pad_h=case.pad_h,
        pad_w=case.pad_w,
        c_base=case.c_base,
        coalesce=case.coalesce,
        kgen=case.kgen,
        pool=case.pool,
        c32_fast=case.c32_fast,
        k_tiles=case.k_tiles,
        block_valid_bytes=case.block_valid_bytes,
        channel_offset=case.channel_offset,
        coalesce_k_bytes=case.coalesce_k_bytes,
        dim_m=dim_m,
    )
    stats = await start_and_wait(dut, timeout_cycles=3000 + case.spatial_m * case.k_tiles * 400)

    expected_compute = 0 if case.pool else (case.spatial_m * case.k_tiles if case.kgen else dim_m)
    assert stats["compute"] == expected_compute

    for oh in range(case.output_h):
        for ow in range(case.output_w):
            row = oh * case.output_w + ow
            if case.pool:
                expected = pool_expected_row(
                    input_data,
                    case.input_h,
                    case.input_w,
                    case.input_c,
                    oh,
                    ow,
                    case.kernel_h,
                    case.kernel_w,
                    stride_h=case.stride_h,
                    stride_w=case.stride_w,
                    pad_h=case.pad_h,
                    pad_w=case.pad_w,
                )
                assert read_u8_row(dut, OFM_ADDR, row) == expected
            else:
                taps = valid_tap_count(
                    oh,
                    ow,
                    case.input_h,
                    case.input_w,
                    case.kernel_h,
                    case.kernel_w,
                    case.stride_h,
                    case.stride_w,
                    case.pad_h,
                    case.pad_w,
                )
                expected_value = taps * case.expected_channels
                assert read_s32_row(dut, OFM_ADDR, row) == [expected_value] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_matrix_case(dut):
    case_id = os.environ.get("SYSTOLIC_CTRL_MATRIX_CASE", "direct_m_001")
    case = get_case(case_id)
    if case.mode == "direct":
        await run_direct_case(dut, case)
    else:
        await run_linebuf_case(dut, case)
