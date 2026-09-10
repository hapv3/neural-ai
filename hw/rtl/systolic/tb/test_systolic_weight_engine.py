import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


ARRAY_DIM = 32
ROW_MASK = (1 << 256) - 1


def drive_defaults(dut):
    dut.job_start_i.value = 0
    dut.load_service_i.value = 0
    dut.preload_service_i.value = 0
    dut.preload_allow_i.value = 0
    dut.preload_consume_i.value = 0
    dut.depthwise_group_start_i.value = 0
    dut.depthwise_mode_i.value = 0
    dut.array_pipe_ready_i.value = 1
    dut.weight_base_ptr_i.value = 0x1000
    dut.depthwise_group_weight_ptr_i.value = 0x4000
    dut.depthwise_tap_count_i.value = 3
    dut.depthwise_tap_index_i.value = 0
    dut.next_tile_index_i.value = 1
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


async def service_requests(dut, expected_count, response_base):
    addresses = []
    loaded = []
    pending = []
    saw_done = False

    for _ in range(expected_count * 4 + 8):
        await FallingEdge(dut.clk_i)
        dut.obi_gnt_i.value = 1
        if pending:
            dut.obi_rvalid_i.value = 1
            dut.obi_rdata_i.value = pending.pop(0)
        else:
            dut.obi_rvalid_i.value = 0
        await Timer(1, unit="ps")

        if dut.obi_req_o.value and dut.obi_gnt_i.value:
            addresses.append(int(dut.obi_addr_o.value))
            pending.append((response_base + len(addresses) - 1) & ROW_MASK)
        if dut.weight_load_en_o.value:
            loaded.append(int(dut.weight_data_o.value))
        saw_done |= bool(dut.load_done_o.value)

        await RisingEdge(dut.clk_i)
        if len(addresses) == expected_count and not pending and (
            saw_done or dut.preload_done_o.value
        ):
            break

    dut.obi_gnt_i.value = 0
    dut.obi_rvalid_i.value = 0
    return addresses, loaded, saw_done


@cocotb.test()
async def initial_load_streams_rows_in_reverse_address_order(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    await pulse(dut, dut.job_start_i)
    dut.load_service_i.value = 1
    addresses, loaded, saw_done = await service_requests(dut, ARRAY_DIM, 0xA000)

    assert addresses == [0x1000 + index * 32 for index in reversed(range(ARRAY_DIM))]
    assert loaded == [0xA000 + index for index in range(ARRAY_DIM)]
    assert saw_done


@cocotb.test()
async def preload_holds_fifo_until_array_pipeline_is_ready(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.next_tile_index_i.value = 2
    await pulse(dut, dut.job_start_i)
    dut.preload_service_i.value = 1
    dut.preload_allow_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.preload_allow_i.value = 0

    await FallingEdge(dut.clk_i)
    dut.array_pipe_ready_i.value = 0
    dut.obi_gnt_i.value = 1
    await Timer(1, unit="ps")
    assert dut.obi_req_o.value
    first_address = int(dut.obi_addr_o.value)
    await RisingEdge(dut.clk_i)

    await FallingEdge(dut.clk_i)
    dut.obi_gnt_i.value = 0
    dut.obi_rvalid_i.value = 1
    dut.obi_rdata_i.value = 0xB000
    await Timer(1, unit="ps")
    assert not dut.weight_load_en_o.value
    await RisingEdge(dut.clk_i)

    dut.obi_rvalid_i.value = 0
    dut.array_pipe_ready_i.value = 1
    addresses, loaded, _ = await service_requests(dut, ARRAY_DIM - 1, 0xB001)

    expected_last = 0x1000 + (2 << 10) + (ARRAY_DIM - 1) * 32
    assert first_address == expected_last
    assert addresses == [expected_last - index * 32 for index in range(1, ARRAY_DIM)]
    assert loaded == [0xB000 + index for index in range(ARRAY_DIM)]
    assert dut.preload_done_o.value

    await pulse(dut, dut.preload_consume_i)
    assert not dut.preload_done_o.value


@cocotb.test()
async def depthwise_load_stores_taps_in_response_order(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.depthwise_mode_i.value = 1
    dut.depthwise_tap_count_i.value = 3
    await pulse(dut, dut.job_start_i)
    dut.load_service_i.value = 1
    addresses, loaded, saw_done = await service_requests(dut, 3, 0xC000)

    assert addresses == [0x1000, 0x1020, 0x1040]
    assert loaded == []
    assert saw_done
    for index in range(3):
        dut.depthwise_tap_index_i.value = index
        await Timer(1, unit="ps")
        assert int(dut.depthwise_weight_o.value) == 0xC000 + index

    dut.load_service_i.value = 0
    dut.depthwise_group_weight_ptr_i.value = 0x4800
    await pulse(dut, dut.depthwise_group_start_i)
    dut.load_service_i.value = 1
    addresses, _, saw_done = await service_requests(dut, 3, 0xD000)
    assert addresses == [0x4800, 0x4820, 0x4840]
    assert saw_done
    dut.depthwise_tap_index_i.value = 2
    await Timer(1, unit="ps")
    assert int(dut.depthwise_weight_o.value) == 0xD002
