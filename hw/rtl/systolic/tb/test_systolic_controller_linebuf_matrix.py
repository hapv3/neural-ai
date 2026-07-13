import cocotb

from systolic_controller_case_utils import (
    ARRAY_DIM,
    BEAT_BYTES,
    IFM_ADDR,
    LB_C32_FAST,
    LB_COALESCE,
    LB_EN,
    LB_KGEN,
    OFM_ADDR,
    WEIGHT_ADDR,
    clear_output_i32,
    disable_optional_modes,
    fill_weight_tiles,
    pack_u8,
    program_linebuf,
    read_s32_row,
    start_and_wait,
    start_clock_and_reset,
    valid_tap_count,
    write_bytes,
)


async def run_linebuf_sum_case(
    dut, *, name, input_h, input_w, input_c, output_h, output_w,
    kernel_h, kernel_w, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
    c_base=0, coalesce=False, kgen=False, c32_fast=False, k_tiles=1,
    block_valid_bytes=0, channel_offset=0, coalesce_k_bytes=0,
    expected_channels=None, timeout_cycles=20000,
):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    spatial_m = output_h * output_w
    dim_m = spatial_m if coalesce else spatial_m * kernel_h * kernel_w
    if kgen:
        dim_m = spatial_m

    input_data = [1] * (input_h * input_w * input_c)
    write_bytes(dut, IFM_ADDR, input_data)
    fill_weight_tiles(dut, WEIGHT_ADDR, k_tiles=k_tiles)
    clear_output_i32(dut, OFM_ADDR, spatial_m)

    await program_linebuf(
        dut,
        input_base=IFM_ADDR,
        input_h=input_h,
        input_w=input_w,
        input_c=input_c,
        output_h=output_h,
        output_w=output_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        c_base=c_base,
        coalesce=coalesce,
        kgen=kgen,
        c32_fast=c32_fast,
        k_tiles=k_tiles,
        block_valid_bytes=block_valid_bytes,
        channel_offset=channel_offset,
        coalesce_k_bytes=coalesce_k_bytes,
        dim_m=dim_m,
    )
    stats = await start_and_wait(dut, timeout_cycles=timeout_cycles)

    expected_compute = spatial_m * k_tiles if kgen else dim_m
    assert stats["compute"] == expected_compute, (
        f"{name}: compute={stats['compute']} expected={expected_compute}"
    )

    channels = expected_channels
    if channels is None:
        channels = max(0, min(ARRAY_DIM, input_c - c_base))

    for oh in range(output_h):
        for ow in range(output_w):
            taps = valid_tap_count(oh, ow, input_h, input_w, kernel_h, kernel_w,
                                   stride_h, stride_w, pad_h, pad_w)
            if coalesce or kgen:
                expected_value = taps * channels
                row = oh * output_w + ow
                assert read_s32_row(dut, OFM_ADDR, row) == [expected_value] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_linebuf_bypass_1x1_c32_rect(dut):
    await run_linebuf_sum_case(
        dut,
        name="1x1 C32 rectangular bypass",
        input_h=4,
        input_w=3,
        input_c=32,
        output_h=4,
        output_w=3,
        kernel_h=1,
        kernel_w=1,
        expected_channels=32,
        timeout_cycles=3000,
    )
    for row in range(12):
        assert read_s32_row(dut, OFM_ADDR, row) == [32] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_linebuf_bypass_1x1_c33_cross_beat(dut):
    await run_linebuf_sum_case(
        dut,
        name="1x1 C33 c_base1 cross-beat bypass",
        input_h=2,
        input_w=3,
        input_c=33,
        output_h=2,
        output_w=3,
        kernel_h=1,
        kernel_w=1,
        c_base=1,
        expected_channels=32,
        timeout_cycles=3000,
    )
    for row in range(6):
        assert read_s32_row(dut, OFM_ADDR, row) == [32] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_linebuf_bypass_1x1_c33_tail(dut):
    await run_linebuf_sum_case(
        dut,
        name="1x1 C33 c_base32 tail bypass",
        input_h=2,
        input_w=2,
        input_c=33,
        output_h=2,
        output_w=2,
        kernel_h=1,
        kernel_w=1,
        c_base=32,
        expected_channels=1,
        timeout_cycles=3000,
    )
    for row in range(4):
        assert read_s32_row(dut, OFM_ADDR, row) == [1] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_linebuf_coalesce_3x3_c3_pad1(dut):
    await run_linebuf_sum_case(
        dut,
        name="coalesce 3x3 C3 pad1",
        input_h=2,
        input_w=4,
        input_c=3,
        output_h=2,
        output_w=4,
        kernel_h=3,
        kernel_w=3,
        pad_h=1,
        pad_w=1,
        coalesce=True,
        expected_channels=3,
        timeout_cycles=6000,
    )


@cocotb.test()
async def systolic_controller_linebuf_coalesce_5x5_c1_pad2(dut):
    await run_linebuf_sum_case(
        dut,
        name="coalesce 5x5 C1 pad2",
        input_h=3,
        input_w=3,
        input_c=1,
        output_h=3,
        output_w=3,
        kernel_h=5,
        kernel_w=5,
        pad_h=2,
        pad_w=2,
        coalesce=True,
        expected_channels=1,
        timeout_cycles=8000,
    )


@cocotb.test()
async def systolic_controller_linebuf_kgen_3x3_c32_c32fast(dut):
    await run_linebuf_sum_case(
        dut,
        name="KGEN 3x3 C32 C32-fast",
        input_h=4,
        input_w=4,
        input_c=32,
        output_h=4,
        output_w=4,
        kernel_h=3,
        kernel_w=3,
        pad_h=1,
        pad_w=1,
        coalesce=True,
        kgen=True,
        c32_fast=True,
        k_tiles=9,
        block_valid_bytes=BEAT_BYTES,
        channel_offset=0,
        coalesce_k_bytes=3 * 3 * BEAT_BYTES,
        expected_channels=32,
        timeout_cycles=40000,
    )


@cocotb.test()
async def systolic_controller_linebuf_kgen_3x3_stride2_c32fast(dut):
    await run_linebuf_sum_case(
        dut,
        name="KGEN 3x3 stride2 C32-fast",
        input_h=5,
        input_w=5,
        input_c=32,
        output_h=3,
        output_w=3,
        kernel_h=3,
        kernel_w=3,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        coalesce=True,
        kgen=True,
        c32_fast=True,
        k_tiles=9,
        block_valid_bytes=BEAT_BYTES,
        channel_offset=0,
        coalesce_k_bytes=3 * 3 * BEAT_BYTES,
        expected_channels=32,
        timeout_cycles=40000,
    )
