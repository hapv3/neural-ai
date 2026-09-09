import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge


ROW_SLOTS = 7
BANK_ADDR_WIDTH = 9
DATA_WIDTH = 256


def pack_lane(current, lane, width, value):
    mask = ((1 << width) - 1) << (lane * width)
    return (current & ~mask) | ((value << (lane * width)) & mask)


def drive_defaults(dut):
    defaults = {
        "job_start_i": 0,
        "job_full_mode_i": 0,
        "job_c_base_i": 0,
        "invalidate_i": 0,
        "cached_c_base_set_i": 0,
        "cached_c_base_i": 0,
        "alloc_main_valid_i": 0,
        "alloc_main_slot_i": 0,
        "alloc_main_ih_i": 0,
        "alloc_background_valid_i": 0,
        "alloc_background_slot_i": 0,
        "alloc_background_ih_i": 0,
        "beat_push_i": 0,
        "beat_push_slot_i": 0,
        "beat_push_last_for_row_i": 0,
        "beat_pop_i": 0,
        "beat_pop_slot_i": 0,
        "bank_write_req_i": 0,
        "bank_write_addr_i": 0,
        "bank_write_data_i": 0,
        "bank_read_req_i": 0,
        "bank_read_addr_i": 0,
        "query_main_slot_i": 0,
        "query_main_ih_i": 0,
        "query_background_slot_i": 0,
        "query_background_ih_i": 0,
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


async def pulse(dut, signal_name):
    getattr(dut, signal_name).value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    getattr(dut, signal_name).value = 0


@cocotb.test()
async def row_store_tracks_rows_and_preserves_sram_latency(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.job_full_mode_i.value = 1
    dut.job_c_base_i.value = 64
    await pulse(dut, "job_start_i")
    assert int(dut.row_cache_full_o.value) == 1
    assert int(dut.cached_c_base_o.value) == 64

    dut.query_main_slot_i.value = 2
    dut.query_main_ih_i.value = 5
    dut.alloc_main_slot_i.value = 2
    dut.alloc_main_ih_i.value = 5
    await pulse(dut, "alloc_main_valid_i")
    assert int(dut.query_main_pending_o.value) == 1
    assert int(dut.query_main_cached_o.value) == 0

    dut.beat_push_slot_i.value = 2
    dut.beat_push_last_for_row_i.value = 1
    await pulse(dut, "beat_push_i")
    assert int(dut.query_main_pending_o.value) == 1
    assert int(dut.query_main_cached_o.value) == 0

    dut.beat_pop_slot_i.value = 2
    await pulse(dut, "beat_pop_i")
    assert int(dut.query_main_pending_o.value) == 0
    assert int(dut.query_main_cached_o.value) == 1

    dut.query_main_slot_i.value = 1
    dut.query_main_ih_i.value = 7
    dut.query_background_slot_i.value = 3
    dut.query_background_ih_i.value = 9
    dut.alloc_main_slot_i.value = 1
    dut.alloc_main_ih_i.value = 7
    dut.alloc_background_slot_i.value = 3
    dut.alloc_background_ih_i.value = 9
    dut.alloc_main_valid_i.value = 1
    dut.alloc_background_valid_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.alloc_main_valid_i.value = 0
    dut.alloc_background_valid_i.value = 0
    assert int(dut.query_main_pending_o.value) == 1
    assert int(dut.query_background_pending_o.value) == 1

    await pulse(dut, "invalidate_i")
    assert int(dut.query_main_pending_o.value) == 0
    assert int(dut.query_background_pending_o.value) == 0

    dut.cached_c_base_i.value = 96
    await pulse(dut, "cached_c_base_set_i")
    assert int(dut.cached_c_base_o.value) == 96

    bank = 4
    address = 3
    data = int.from_bytes(bytes(range(DATA_WIDTH // 8)), "little")
    dut.bank_write_req_i.value = 1 << bank
    dut.bank_write_addr_i.value = pack_lane(0, bank, BANK_ADDR_WIDTH, address)
    dut.bank_write_data_i.value = pack_lane(0, bank, DATA_WIDTH, data)
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.bank_write_req_i.value = 0

    dut.bank_read_req_i.value = 1 << bank
    dut.bank_read_addr_i.value = pack_lane(0, bank, BANK_ADDR_WIDTH, address)
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.bank_read_req_i.value = 0
    read_data = (int(dut.bank_read_data_o.value) >> (bank * DATA_WIDTH)) & (
        (1 << DATA_WIDTH) - 1
    )
    assert read_data == data

    dut.job_full_mode_i.value = 0
    dut.job_c_base_i.value = 32
    await pulse(dut, "job_start_i")
    assert int(dut.row_cache_full_o.value) == 0
    assert int(dut.cached_c_base_o.value) == 32
