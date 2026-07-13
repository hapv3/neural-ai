import cocotb

from systolic_controller_case_utils import (
    IFM_ADDR,
    LB_EN,
    LB_POOL,
    OFM_ADDR,
    clear_output_i8,
    disable_optional_modes,
    pool_expected_row,
    program_linebuf,
    read_u8_row,
    start_and_wait,
    start_clock_and_reset,
    write_bytes,
)


@cocotb.test()
async def systolic_controller_linebuf_pool_3x3_c32_pad1(dut):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    input_h = 3
    input_w = 4
    input_c = 32
    output_h = 3
    output_w = 4
    kernel_h = 3
    kernel_w = 3
    pad_h = 1
    pad_w = 1
    spatial_m = output_h * output_w

    input_data = []
    for ih in range(input_h):
        for iw in range(input_w):
            for ch in range(input_c):
                value = ((ih * 17 + iw * 7 + ch * 3) % 101) - 50
                input_data.append(value & 0xFF)
    write_bytes(dut, IFM_ADDR, input_data)
    clear_output_i8(dut, OFM_ADDR, spatial_m)

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
        pad_h=pad_h,
        pad_w=pad_w,
        pool=True,
        dim_m=spatial_m,
    )
    stats = await start_and_wait(dut, timeout_cycles=10000)

    assert stats["compute"] == 0
    for oh in range(output_h):
        for ow in range(output_w):
            row = oh * output_w + ow
            expected = pool_expected_row(input_data, input_h, input_w, input_c,
                                         oh, ow, kernel_h, kernel_w,
                                         pad_h=pad_h, pad_w=pad_w)
            assert read_u8_row(dut, OFM_ADDR, row) == expected


@cocotb.test()
async def systolic_controller_linebuf_pool_5x5_c16_stride2(dut):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    input_h = 6
    input_w = 6
    input_c = 16
    output_h = 3
    output_w = 3
    kernel_h = 5
    kernel_w = 5
    stride_h = 2
    stride_w = 2
    pad_h = 2
    pad_w = 2
    spatial_m = output_h * output_w

    input_data = []
    for ih in range(input_h):
        for iw in range(input_w):
            for ch in range(input_c):
                value = ((ih * 11 + iw * 5 + ch * 9) % 113) - 56
                input_data.append(value & 0xFF)
    write_bytes(dut, IFM_ADDR, input_data)
    clear_output_i8(dut, OFM_ADDR, spatial_m)

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
        pool=True,
        dim_m=spatial_m,
    )
    stats = await start_and_wait(dut, timeout_cycles=20000)

    assert stats["compute"] == 0
    for oh in range(output_h):
        for ow in range(output_w):
            row = oh * output_w + ow
            expected = pool_expected_row(input_data, input_h, input_w, input_c,
                                         oh, ow, kernel_h, kernel_w,
                                         stride_h=stride_h, stride_w=stride_w,
                                         pad_h=pad_h, pad_w=pad_w)
            assert read_u8_row(dut, OFM_ADDR, row) == expected
