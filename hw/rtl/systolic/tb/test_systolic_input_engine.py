import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


ROW_MASK = (1 << 256) - 1


def drive_defaults(dut):
    dut.job_start_i.value = 0
    dut.feed_start_i.value = 0
    dut.feed_service_i.value = 0
    dut.drain_service_i.value = 0
    dut.linebuf_start_i.value = 0
    dut.linebuf_next_tile_i.value = 0
    dut.preload_service_i.value = 0
    dut.preload_has_next_i.value = 0
    dut.preload_hold_i.value = 0
    dut.linebuf_enable_i.value = 0
    dut.side_stream_mode_i.value = 0
    dut.array_pipe_ready_i.value = 1
    dut.side_ready_i.value = 0
    dut.ifm_base_ptr_i.value = 0x2000
    dut.row_count_i.value = 0

    dut.cfg_spatial_m_i.value = 0
    dut.cfg_k_tiles_i.value = 1
    dut.cfg_origin_base_i.value = 0
    dut.cfg_row_stride_bytes_i.value = 0
    dut.cfg_pixel_stride_bytes_i.value = 0
    dut.cfg_ow_step_bytes_i.value = 0
    dut.cfg_oh_step_bytes_i.value = 0
    dut.cfg_input_h_i.value = 0
    dut.cfg_input_w_i.value = 0
    dut.cfg_input_c_i.value = 0
    dut.cfg_output_w_i.value = 0
    dut.cfg_kernel_h_i.value = 0
    dut.cfg_kernel_w_i.value = 0
    dut.cfg_stride_h_i.value = 1
    dut.cfg_stride_w_i.value = 1
    dut.cfg_pad_h_i.value = 0
    dut.cfg_pad_w_i.value = 0
    dut.cfg_c_base_i.value = 0
    dut.cfg_lane_base_i.value = 0
    dut.cfg_coalesce_i.value = 0
    dut.cfg_kgen_i.value = 0
    dut.cfg_pool_i.value = 0
    dut.cfg_c32_fast_i.value = 0
    dut.cfg_depthwise_i.value = 0
    dut.cfg_block_valid_bytes_i.value = 32
    dut.cfg_channel_addr_offset_i.value = 0
    dut.cfg_coalesce_k_bytes_i.value = 0
    dut.cfg_k_seed_kh_i.value = 0
    dut.cfg_k_seed_kw_i.value = 0
    dut.cfg_k_seed_ic_i.value = 0

    dut.obi_gnt_i.value = 0
    dut.obi_rvalid_i.value = 0
    dut.obi_rdata_i.value = 0


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
async def direct_feed_holds_grant_and_array_backpressure(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.row_count_i.value = 3
    await pulse(dut, dut.job_start_i)
    await pulse(dut, dut.feed_start_i)
    dut.feed_service_i.value = 1

    for _ in range(2):
        await FallingEdge(dut.clk_i)
        await Timer(1, unit="ps")
        assert dut.obi_req_o.value
        assert int(dut.obi_addr_o.value) == 0x2000
        await RisingEdge(dut.clk_i)

    pending = []
    addresses = []
    rows = []
    saw_done = False
    for cycle in range(20):
        await FallingEdge(dut.clk_i)
        dut.array_pipe_ready_i.value = 0 if cycle in (2, 3) else 1
        dut.obi_gnt_i.value = 1
        if pending:
            dut.obi_rvalid_i.value = 1
            dut.obi_rdata_i.value = pending.pop(0)
        else:
            dut.obi_rvalid_i.value = 0
        await Timer(1, unit="ps")

        if dut.obi_req_o.value:
            address = int(dut.obi_addr_o.value)
            addresses.append(address)
            pending.append((0xA0 + ((address - 0x2000) // 32)) & ROW_MASK)
        if dut.compute_en_o.value:
            rows.append(int(dut.compute_data_o.value))
        saw_done |= bool(dut.feed_done_o.value)
        await RisingEdge(dut.clk_i)
        if saw_done:
            break

    await FallingEdge(dut.clk_i)
    assert addresses == [0x2000, 0x2020, 0x2040]
    assert rows == [0xA0, 0xA1, 0xA2]
    assert saw_done
    assert int(dut.request_count_o.value) == 0
    assert int(dut.response_count_o.value) == 0


@cocotb.test()
async def linebuffer_bypass_routes_obi_to_side_stream(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    base = 0x3000
    expected = [
        sum(((0x10 + lane) & 0xFF) << (8 * lane) for lane in range(32)),
        sum(((0x50 + lane) & 0xFF) << (8 * lane) for lane in range(32)),
    ]
    dut.linebuf_enable_i.value = 1
    dut.side_stream_mode_i.value = 1
    dut.side_ready_i.value = 1
    dut.feed_service_i.value = 1
    dut.cfg_spatial_m_i.value = 2
    dut.cfg_origin_base_i.value = base
    dut.cfg_row_stride_bytes_i.value = 64
    dut.cfg_pixel_stride_bytes_i.value = 32
    dut.cfg_ow_step_bytes_i.value = 32
    dut.cfg_oh_step_bytes_i.value = 64
    dut.cfg_input_h_i.value = 1
    dut.cfg_input_w_i.value = 2
    dut.cfg_input_c_i.value = 32
    dut.cfg_output_w_i.value = 2
    dut.cfg_kernel_h_i.value = 1
    dut.cfg_kernel_w_i.value = 1
    dut.cfg_c32_fast_i.value = 1

    await pulse(dut, dut.job_start_i)
    await pulse(dut, dut.linebuf_start_i)

    pending = []
    addresses = []
    rows = []
    for _ in range(40):
        await FallingEdge(dut.clk_i)
        dut.obi_gnt_i.value = 1
        if pending:
            address = pending.pop(0)
            dut.obi_rvalid_i.value = 1
            dut.obi_rdata_i.value = expected[(address - base) // 32]
        else:
            dut.obi_rvalid_i.value = 0
        await Timer(1, unit="ps")

        if dut.obi_req_o.value:
            address = int(dut.obi_addr_o.value)
            addresses.append(address)
            pending.append(address)
        if dut.side_valid_o.value and dut.side_ready_i.value:
            assert dut.linebuf_row_ready_o.value
            rows.append(int(dut.side_data_o.value))
        await RisingEdge(dut.clk_i)
        if len(rows) == 2:
            break

    await FallingEdge(dut.clk_i)
    assert addresses == [base, base + 32]
    assert rows == expected
    assert int(dut.bypass_vectors_o.value) == 2
