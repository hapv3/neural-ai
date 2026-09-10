import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


IDLE = 0
LOAD_WEIGHTS = 1
COMPUTE = 2
WAIT_DRAIN = 3
DONE = 4
ARRAY_FLUSH_CYCLES = 63


def drive_defaults(dut):
    defaults = {
        "start_i": 0,
        "linebuf_enable_i": 0,
        "pool_mode_i": 0,
        "depthwise_mode_i": 0,
        "kgen_multi_i": 0,
        "tile_index_i": 0,
        "has_next_tile_i": 0,
        "psum_overlap_active_i": 0,
        "requant_enable_i": 0,
        "requant_config_invalid_i": 0,
        "binary_config_invalid_i": 0,
        "binary_enable_i": 0,
        "weight_load_done_i": 0,
        "weight_preload_done_i": 0,
        "input_feed_done_i": 0,
        "array_pipe_ready_i": 1,
        "linebuf_row_valid_i": 0,
        "linebuf_busy_i": 0,
        "linebuf_prefetch_busy_i": 0,
        "drain_remaining_i": 0,
        "ofm_empty_i": 1,
        "binary_operand_busy_i": 0,
        "depthwise_input_ready_i": 0,
        "depthwise_output_valid_i": 0,
        "quantized_output_valid_i": 0,
        "pool_input_ready_i": 0,
        "pool_output_valid_i": 0,
        "weight_base_ptr_i": 0x1000,
        "output_base_ptr_i": 0x8000,
        "input_c_i": 32,
        "input_h_i": 1,
        "input_row_stride_bytes_i": 32,
        "spatial_row_count_i": 1,
        "kernel_vectors_i": 1,
    }
    for name, value in defaults.items():
        getattr(dut, name).value = value


async def reset(dut):
    drive_defaults(dut)
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.state_o.value) == IDLE


async def clock_inputs(dut, **values):
    for name, value in values.items():
        getattr(dut, name).value = value
    await Timer(1, unit="ps")
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    for name in values:
        getattr(dut, name).value = 0


async def enter_compute(dut):
    dut.start_i.value = 1
    await Timer(1, unit="ps")
    assert dut.job_start_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0
    assert int(dut.state_o.value) == LOAD_WEIGHTS

    dut.weight_load_done_i.value = 1
    await Timer(1, unit="ps")
    assert dut.input_feed_start_o.value
    assert dut.drain_tile_start_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.weight_load_done_i.value = 0
    assert int(dut.state_o.value) == COMPUTE


@cocotb.test()
async def direct_job_serially_advances_a_ready_next_tile(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.linebuf_enable_i.value = 1
    dut.kgen_multi_i.value = 1
    dut.has_next_tile_i.value = 1
    dut.drain_remaining_i.value = 1
    await enter_compute(dut)

    await clock_inputs(dut, input_feed_done_i=1)
    assert int(dut.state_o.value) == WAIT_DRAIN
    assert dut.drain_service_o.value
    assert dut.input_preload_hold_o.value

    dut.weight_preload_done_i.value = 1
    dut.drain_remaining_i.value = 0
    await Timer(1, unit="ps")
    assert dut.k_tile_advance_o.value
    assert dut.drain_tile_advance_o.value
    assert not dut.drain_tile_advance_overlap_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    assert int(dut.state_o.value) == LOAD_WEIGHTS
    assert dut.load_service_o.value


@cocotb.test()
async def psum_overlap_waits_for_array_flush_before_direct_restart(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.linebuf_enable_i.value = 1
    dut.kgen_multi_i.value = 1
    dut.has_next_tile_i.value = 1
    dut.psum_overlap_active_i.value = 1
    dut.drain_remaining_i.value = 1
    await enter_compute(dut)

    await clock_inputs(dut, input_feed_done_i=1)
    assert int(dut.state_o.value) == WAIT_DRAIN
    dut.weight_preload_done_i.value = 1

    for _ in range(ARRAY_FLUSH_CYCLES):
        await Timer(1, unit="ps")
        assert not dut.k_tile_advance_o.value
        assert not dut.weight_preload_allow_o.value
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)

    await Timer(1, unit="ps")
    assert dut.weight_preload_allow_o.value
    assert dut.k_tile_advance_o.value
    assert dut.drain_tile_advance_overlap_o.value
    assert dut.input_feed_start_o.value
    assert dut.weight_preload_consume_o.value
    assert dut.linebuf_next_tile_o.value

    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.state_o.value) == COMPUTE


@cocotb.test()
async def depthwise_groups_advance_addresses_and_report_tail_lanes(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.linebuf_enable_i.value = 1
    dut.depthwise_mode_i.value = 1
    dut.input_c_i.value = 65
    dut.input_h_i.value = 2
    dut.input_row_stride_bytes_i.value = 100
    dut.spatial_row_count_i.value = 3
    dut.kernel_vectors_i.value = 3
    dut.linebuf_busy_i.value = 1

    dut.start_i.value = 1
    await Timer(1, unit="ps")
    assert dut.job_start_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0
    assert int(dut.state_o.value) == LOAD_WEIGHTS
    assert int(dut.depthwise_group_valid_bytes_o.value) == 32

    await clock_inputs(dut, weight_load_done_i=1)
    assert int(dut.state_o.value) == COMPUTE

    dut.linebuf_row_valid_i.value = 1
    dut.depthwise_input_ready_i.value = 1
    for tap in range(3):
        await Timer(1, unit="ps")
        assert int(dut.depthwise_tap_index_o.value) == tap
        assert bool(dut.depthwise_tap_is_last_o.value) == (tap == 2)
        assert dut.depthwise_input_valid_o.value
        assert dut.input_side_ready_o.value
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
    assert int(dut.depthwise_tap_index_o.value) == 0

    dut.linebuf_row_valid_i.value = 0
    dut.depthwise_input_ready_i.value = 0
    dut.linebuf_busy_i.value = 0
    await Timer(1, unit="ps")
    assert dut.weight_depthwise_group_start_o.value
    assert int(dut.weight_depthwise_group_ptr_o.value) == 0x1060
    assert dut.drain_depthwise_group_start_o.value
    assert int(dut.drain_depthwise_group_output_ptr_o.value) == 0x8060
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.depthwise_group_index_o.value) == 1
    assert int(dut.depthwise_group_input_offset_o.value) == 200
    assert int(dut.depthwise_group_valid_bytes_o.value) == 32

    dut.linebuf_busy_i.value = 1
    await clock_inputs(dut, weight_load_done_i=1)
    dut.linebuf_busy_i.value = 0
    await Timer(1, unit="ps")
    assert int(dut.weight_depthwise_group_ptr_o.value) == 0x10C0
    assert int(dut.drain_depthwise_group_output_ptr_o.value) == 0x80C0
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.depthwise_group_index_o.value) == 2
    assert int(dut.depthwise_group_input_offset_o.value) == 400
    assert int(dut.depthwise_group_valid_bytes_o.value) == 1

    dut.linebuf_busy_i.value = 1
    await clock_inputs(dut, weight_load_done_i=1)
    dut.linebuf_busy_i.value = 0
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.state_o.value) == DONE
    assert dut.done_o.value


@cocotb.test()
async def invalid_start_flushes_children_then_reports_done(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.requant_enable_i.value = 1
    dut.requant_config_invalid_i.value = 1
    dut.start_i.value = 1
    await Timer(1, unit="ps")
    assert dut.job_start_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    assert int(dut.state_o.value) == DONE
    assert dut.done_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.state_o.value) == IDLE
    assert not dut.done_o.value
