import cocotb

from systolic_controller_case_utils import (
    ARRAY_DIM,
    BEAT_BYTES,
    IFM_ADDR,
    OFM_ADDR,
    PSUM_ADDR,
    WEIGHT_ADDR,
    clear_output_i32,
    clear_output_i8,
    clamp_i8_to_u8,
    configure_identity_requant,
    disable_optional_modes,
    fill_weight_tiles,
    pack_u8,
    program_direct_gemm,
    read_s32_row,
    read_u8_row,
    start_and_wait,
    start_clock_and_reset,
    write_s32_rows,
)


@cocotb.test()
async def systolic_controller_direct_gemm_m1(dut):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    fill_weight_tiles(dut, WEIGHT_ADDR)
    dut.tcdm_mem[IFM_ADDR >> 5].value = pack_u8([1] * BEAT_BYTES)
    clear_output_i32(dut, OFM_ADDR, 1)

    await program_direct_gemm(dut, dim_m=1)
    stats = await start_and_wait(dut, timeout_cycles=1000)

    assert stats["weight"] == ARRAY_DIM
    assert stats["compute"] == 1
    assert read_s32_row(dut, OFM_ADDR, 0) == [ARRAY_DIM] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_direct_gemm_m17_fifo_pressure(dut):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    dim_m = 17
    fill_weight_tiles(dut, WEIGHT_ADDR)
    for row in range(dim_m):
        dut.tcdm_mem[(IFM_ADDR >> 5) + row].value = pack_u8([1] * BEAT_BYTES)
    clear_output_i32(dut, OFM_ADDR, dim_m)

    await program_direct_gemm(dut, dim_m=dim_m)
    stats = await start_and_wait(dut, timeout_cycles=3000)

    assert stats["weight"] == ARRAY_DIM
    assert stats["compute"] == dim_m
    for row in range(dim_m):
        assert read_s32_row(dut, OFM_ADDR, row) == [ARRAY_DIM] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_direct_accumulate_external_psum(dut):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    dim_m = 3
    fill_weight_tiles(dut, WEIGHT_ADDR)
    for row in range(dim_m):
        dut.tcdm_mem[(IFM_ADDR >> 5) + row].value = pack_u8([1] * BEAT_BYTES)
    write_s32_rows(dut, PSUM_ADDR, [[5] * ARRAY_DIM for _ in range(dim_m)])
    clear_output_i32(dut, OFM_ADDR, dim_m)

    await program_direct_gemm(dut, dim_m=dim_m, psum_addr=PSUM_ADDR, accum=True)
    stats = await start_and_wait(dut, timeout_cycles=3000)

    assert stats["compute"] == dim_m
    for row in range(dim_m):
        assert read_s32_row(dut, OFM_ADDR, row) == [ARRAY_DIM + 5] * ARRAY_DIM


@cocotb.test()
async def systolic_controller_direct_requant_shift_clamp(dut):
    await start_clock_and_reset(dut)
    await disable_optional_modes(dut)

    dim_m = 4
    fill_weight_tiles(dut, WEIGHT_ADDR)
    for row in range(dim_m):
        dut.tcdm_mem[(IFM_ADDR >> 5) + row].value = pack_u8([1] * BEAT_BYTES)
    clear_output_i8(dut, OFM_ADDR, dim_m)

    await configure_identity_requant(dut, shift=1, clamp_min=-20, clamp_max=20)
    await program_direct_gemm(dut, dim_m=dim_m)
    stats = await start_and_wait(dut, timeout_cycles=3000)

    assert stats["compute"] == dim_m
    expected = [clamp_i8_to_u8(ARRAY_DIM >> 1)] * ARRAY_DIM
    for row in range(dim_m):
        assert read_u8_row(dut, OFM_ADDR, row) == expected
