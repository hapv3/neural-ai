import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


LANES = 32


def pack_lanes(values, width):
    word = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        word |= (value & mask) << (lane * width)
    return word


def drive_defaults(dut):
    dut.job_start_i.value = 0
    dut.tile_advance_i.value = 0
    dut.tile_advance_overlap_i.value = 0
    dut.tile_start_i.value = 0
    dut.tile_start_add_rows_i.value = 0
    dut.depthwise_group_start_i.value = 0
    dut.depthwise_group_output_ptr_i.value = 0
    dut.drain_active_i.value = 0
    dut.compute_phase_i.value = 0
    dut.pool_mode_i.value = 0
    dut.depthwise_mode_i.value = 0
    dut.external_accum_enable_i.value = 0
    dut.accum_active_i.value = 0
    dut.requant_active_i.value = 0
    dut.psum_buf_active_i.value = 0
    dut.psum_buf_needs_external_i.value = 0
    dut.psum_buf_final_tile_i.value = 0
    dut.k_tile_idx_i.value = 0
    dut.ofm_base_ptr_i.value = 0x4000
    dut.psum_base_ptr_i.value = 0x8000
    dut.row_count_i.value = 1
    dut.spatial_row_count_i.value = 1
    dut.ofm_row_stride_bytes_i.value = 0
    dut.ofm_tile_cols_i.value = 0
    dut.psum_row_stride_bytes_i.value = 0
    dut.result_i.value = 0
    dut.result_valid_i.value = 0
    dut.depthwise_result_i.value = 0
    dut.depthwise_result_valid_i.value = 0
    dut.pool_result_i.value = 0
    dut.pool_result_valid_i.value = 0

    dut.requant_enable_i.value = 0
    dut.requant_bias_i.value = 0
    dut.requant_multiplier_i.value = pack_lanes([1] * LANES, 32)
    dut.requant_shift_i.value = 0
    dut.requant_zero_point_i.value = 0
    dut.requant_clamp_min_i.value = (-128) & 0xFFFFFFFF
    dut.requant_clamp_max_i.value = 127
    dut.binary_enable_i.value = 0
    dut.binary_mode_i.value = 0
    dut.binary_rhs_ptr_i.value = 0x1000
    dut.binary_rhs_row_stride_bytes_i.value = 0
    dut.binary_rhs_tile_cols_i.value = 0
    dut.binary_lhs_multiplier_i.value = 1
    dut.binary_lhs_shift_i.value = 0
    dut.binary_rhs_multiplier_i.value = 1
    dut.binary_rhs_shift_i.value = 0
    dut.binary_output_multiplier_i.value = 1
    dut.binary_output_shift_i.value = 0
    dut.binary_lhs_zero_point_i.value = 0
    dut.binary_rhs_zero_point_i.value = 0
    dut.binary_output_zero_point_i.value = 0
    dut.binary_clamp_min_i.value = (-128) & 0xFFFFFFFF
    dut.binary_clamp_max_i.value = 127
    dut.binary_double_round_shift_i.value = 0

    dut.obi_b_gnt_i.value = 0
    dut.obi_b_rvalid_i.value = 0
    dut.obi_b_rdata_i.value = 0
    dut.obi_o_gnt_i.value = 0
    dut.obi_o_rvalid_i.value = 0
    dut.obi_o_rdata_i.value = 0


async def reset(dut):
    dut.rst_ni.value = 0
    drive_defaults(dut)
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def pulse(dut, signal):
    await FallingEdge(dut.clk_i)
    signal.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    signal.value = 0


@cocotb.test()
async def output_drain_writes_raw_row_on_four_obi_ports(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    values = [0x10000 + lane for lane in range(LANES)]
    expected_row = pack_lanes(values, 32)
    await pulse(dut, dut.job_start_i)
    assert int(dut.remaining_rows_o.value) == 1

    dut.drain_active_i.value = 1
    dut.result_i.value = expected_row
    dut.result_valid_i.value = 1
    await Timer(1, unit="ps")
    assert dut.result_ready_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.result_valid_i.value = 0
    await Timer(1, unit="ps")

    assert int(dut.obi_o_req_o.value) == 0xF
    assert int(dut.obi_o_we_o.value) == 0xF
    addresses = int(dut.obi_o_addr_o.value)
    write_data = int(dut.obi_o_wdata_o.value)
    for port in range(4):
        assert (addresses >> (port * 32)) & 0xFFFFFFFF == 0x4000 + port * 32
        assert (write_data >> (port * 256)) & ((1 << 256) - 1) == (
            expected_row >> (port * 256)
        ) & ((1 << 256) - 1)

    dut.obi_o_gnt_i.value = 0xF
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.remaining_rows_o.value) == 0
    assert dut.ofm_empty_o.value


@cocotb.test()
async def output_drain_pool_writer_honors_row_stride(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.pool_mode_i.value = 1
    dut.spatial_row_count_i.value = 2
    dut.ofm_tile_cols_i.value = 1
    dut.ofm_row_stride_bytes_i.value = 96
    await pulse(dut, dut.job_start_i)
    dut.compute_phase_i.value = 1
    dut.obi_o_gnt_i.value = 1

    for index, expected_addr in enumerate((0x4000, 0x4060)):
        await FallingEdge(dut.clk_i)
        dut.pool_result_i.value = index + 1
        dut.pool_result_valid_i.value = 1
        await Timer(1, unit="ps")
        assert dut.pool_result_ready_o.value
        assert int(dut.obi_o_addr_o.value) & 0xFFFFFFFF == expected_addr
        await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.pool_result_valid_i.value = 0
    assert int(dut.remaining_rows_o.value) == 0


@cocotb.test()
async def output_drain_lifecycle_events_update_owned_count(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.row_count_i.value = 4
    await pulse(dut, dut.job_start_i)
    assert int(dut.remaining_rows_o.value) == 4

    dut.tile_advance_overlap_i.value = 1
    await pulse(dut, dut.tile_advance_i)
    dut.tile_advance_overlap_i.value = 0
    assert int(dut.remaining_rows_o.value) == 8

    await pulse(dut, dut.tile_start_i)
    assert int(dut.remaining_rows_o.value) == 4

    dut.tile_start_add_rows_i.value = 1
    await pulse(dut, dut.tile_start_i)
    dut.tile_start_add_rows_i.value = 0
    assert int(dut.remaining_rows_o.value) == 8
