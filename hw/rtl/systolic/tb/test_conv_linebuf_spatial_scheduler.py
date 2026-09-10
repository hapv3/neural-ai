import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


ARRAY_DIM = 32


def drive_defaults(dut):
    defaults = {
        "start_i": 0,
        "next_tile_i": 0,
        "prefetch_i": 0,
        "dim_m_i": 1,
        "cfg_k_tiles_i": 1,
        "cfg_origin_base_i": 0x1000,
        "cfg_row_stride_bytes_i": 0x100,
        "cfg_ow_step_bytes_i": 32,
        "cfg_oh_step_bytes_i": 0x100,
        "cfg_input_h_i": 4,
        "cfg_input_w_i": 4,
        "cfg_output_w_i": 1,
        "cfg_kernel_h_i": 3,
        "cfg_kernel_w_i": 3,
        "cfg_stride_h_i": 1,
        "cfg_stride_w_i": 1,
        "cfg_pad_h_i": 1,
        "cfg_pad_w_i": 1,
        "cfg_coalesce_i": 0,
        "cfg_kgen_i": 0,
        "block_valid_bytes_i": 32,
        "coalesce_k_bytes_i": 32,
        "lane_kh_i": 0,
        "lane_kw_i": 0,
        "lane_ic_i": 0,
        "effective_c_base_i": 0,
        "channel_addr_offset_i": 0,
        "row_cache_reuse_i": 0,
        "row_ring_mode_i": 0,
        "fill_done_rows_i": 0,
        "pad_row_offset_i": 0,
        "row_cache_full_i": 1,
        "cached_c_base_i": 0,
        "row_store_main_cached_i": 0,
        "row_store_main_pending_i": 0,
        "fetch_main_request_accepted_i": 0,
        "fetch_main_next_phase_i": 0,
        "fetch_main_done_i": 0,
        "fetch_background_idle_i": 1,
        "formatter_row_i": 0,
        "formatter_valid_i": 0,
        "formatter_empty_i": 1,
        "bypass_row_i": 0,
        "bypass_row_valid_i": 0,
        "bypass_debug_state_i": 8,
        "row_ready_i": 1,
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


async def pulse_start(dut):
    dut.start_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.row_store_job_start_o.value) == 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0


@cocotb.test()
async def scheduler_rejects_invalid_configuration(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.dim_m_i.value = 0
    await pulse_start(dut)
    assert int(dut.busy_o.value) == 0
    assert int(dut.done_o.value) == 1
    assert int(dut.bypass_start_o.value) == 0

    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.done_o.value) == 0


@cocotb.test()
async def scheduler_preserves_bypass_backpressure_and_tail(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.dim_m_i.value = 2
    dut.cfg_output_w_i.value = 2
    dut.cfg_kernel_h_i.value = 1
    dut.cfg_kernel_w_i.value = 1
    dut.cfg_pad_h_i.value = 0
    dut.cfg_pad_w_i.value = 0
    dut.row_ready_i.value = 0
    dut.bypass_row_i.value = int.from_bytes(bytes(range(ARRAY_DIM)), "little")
    dut.bypass_row_valid_i.value = 1

    dut.start_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.bypass_start_o.value) == 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0

    assert int(dut.busy_o.value) == 1
    assert int(dut.row_valid_o.value) == 1
    assert int(dut.row_data_o.value) == int(dut.bypass_row_i.value)
    assert int(dut.emitted_vectors_o.value) == 0
    for _ in range(2):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        assert int(dut.emitted_vectors_o.value) == 0

    dut.row_ready_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.emitted_vectors_o.value) == 1
    assert int(dut.bypass_spatial_addr_o.value) == 0x1020
    assert int(dut.bypass_last_o.value) == 1

    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.emitted_vectors_o.value) == 2
    assert int(dut.debug_state_o.value) == 14

    dut.bypass_row_valid_i.value = 0
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.done_o.value) == 1
    assert int(dut.busy_o.value) == 0


@cocotb.test()
async def scheduler_walks_window_taps_and_holds_formatted_output(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.formatter_row_i.value = int.from_bytes(bytes([0x5A] * ARRAY_DIM), "little")
    dut.formatter_valid_i.value = 1
    dut.row_ready_i.value = 0
    await pulse_start(dut)

    # Full-cache mode skips fetch and performs the original three-cycle window load.
    for _ in range(8):
        await Timer(1, unit="ps")
        if int(dut.window_load_request_o.value):
            break
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
    assert int(dut.window_load_request_o.value) == 1

    for _ in range(16):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        if int(dut.row_valid_o.value):
            break
    assert int(dut.row_valid_o.value) == 1
    held_row = int(dut.row_data_o.value)
    assert held_row == int(dut.formatter_row_i.value)
    assert int(dut.formatter_advance_o.value) == 0

    for _ in range(3):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        assert int(dut.row_valid_o.value) == 1
        assert int(dut.row_data_o.value) == held_row

    dut.row_ready_i.value = 1
    dut.formatter_valid_i.value = 0
    for _ in range(32):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        if int(dut.done_o.value):
            break
    assert int(dut.done_o.value) == 1
    assert int(dut.emitted_vectors_o.value) == 9
