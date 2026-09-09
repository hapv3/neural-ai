import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


DATA_WIDTH = 256
BEAT_BYTES = DATA_WIDTH // 8
ROW_SLOTS = 7
BANKS = 14
BANK_ADDR_WIDTH = 9


def drive_defaults(dut):
    defaults = {
        "clear_count_i": 0,
        "reset_background_i": 0,
        "row_ring_mode_i": 0,
        "c32_blocked_mode_i": 0,
        "input_h_i": 8,
        "input_w_i": 1,
        "kernel_h_i": 1,
        "pixel_stride_bytes_i": BEAT_BYTES,
        "row_stride_bytes_i": 256,
        "channel_addr_offset_i": 0,
        "main_start_i": 0,
        "main_base_addr_i": 0,
        "main_row_slot_i": 0,
        "main_row_ih_i": 0,
        "main_valid_bytes_i": BEAT_BYTES,
        "main_row_ready_i": 0,
        "background_start_i": 0,
        "background_base_ih_i": 0,
        "background_row_base_addr_i": 0,
        "background_row_cached_i": 0,
        "background_row_pending_i": 0,
        "obi_gnt_i": 0,
        "obi_rvalid_i": 0,
        "obi_rdata_i": 0,
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


async def pulse(dut, name):
    getattr(dut, name).value = 1
    await Timer(1, unit="ps")
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    getattr(dut, name).value = 0


async def accept_request(dut, expected_address, expected_slot, last_for_row):
    dut.obi_gnt_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.obi_req_o.value) == 1
    assert int(dut.obi_addr_o.value) == expected_address
    assert int(dut.beat_push_o.value) == int(dut.row_ring_mode_i.value)
    if dut.beat_push_o.value:
        assert int(dut.beat_push_slot_o.value) == expected_slot
        assert int(dut.beat_push_last_for_row_o.value) == last_for_row
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.obi_gnt_i.value = 0


async def return_response(dut, data):
    dut.obi_rvalid_i.value = 1
    dut.obi_rdata_i.value = data
    await Timer(1, unit="ps")
    write_req = int(dut.bank_write_req_o.value)
    write_addr = int(dut.bank_write_addr_o.value)
    write_data = int(dut.bank_write_data_o.value)
    pop_slot = int(dut.beat_pop_slot_o.value)
    assert int(dut.beat_pop_o.value) == 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.obi_rvalid_i.value = 0
    dut.obi_rdata_i.value = 0
    return write_req, write_addr, write_data, pop_slot


def bank_lane(packed, bank, width):
    return (packed >> (bank * width)) & ((1 << width) - 1)


@cocotb.test()
async def fetch_engine_handles_main_background_and_crossing_responses(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    # A ring-mode foreground row owns request priority and marks row completion.
    dut.row_ring_mode_i.value = 1
    dut.input_w_i.value = 2
    dut.main_base_addr_i.value = 0x1000
    dut.main_row_slot_i.value = 2
    dut.main_row_ih_i.value = 9
    dut.main_valid_bytes_i.value = BEAT_BYTES
    dut.main_start_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.main_alloc_valid_o.value) == 1
    assert int(dut.main_alloc_slot_o.value) == 2
    assert int(dut.main_alloc_ih_o.value) == 9
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.main_start_i.value = 0

    await accept_request(dut, 0x1000, 2, 0)
    first_data = int.from_bytes(bytes(range(BEAT_BYTES)), "little")
    write_req, write_addr, write_data, pop_slot = await return_response(dut, first_data)
    first_bank = 2 * 2
    assert pop_slot == 2
    assert write_req == 1 << first_bank
    assert bank_lane(write_addr, first_bank, BANK_ADDR_WIDTH) == 0
    assert bank_lane(write_data, first_bank, DATA_WIDTH) == first_data

    await accept_request(dut, 0x1020, 2, 1)
    second_data = int.from_bytes(bytes(reversed(range(BEAT_BYTES))), "little")
    write_req, write_addr, write_data, pop_slot = await return_response(dut, second_data)
    second_bank = first_bank + 1
    assert pop_slot == 2
    assert write_req == 1 << second_bank
    assert bank_lane(write_addr, second_bank, BANK_ADDR_WIDTH) == 0
    assert bank_lane(write_data, second_bank, DATA_WIDTH) == second_data
    assert int(dut.fetch_beats_o.value) == 2
    assert int(dut.main_done_o.value) == 0
    dut.main_row_ready_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.main_done_o.value) == 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.main_row_ready_i.value = 0

    # A crossing pixel writes only after beat 1 and preserves byte ordering.
    dut.row_ring_mode_i.value = 0
    dut.input_w_i.value = 1
    dut.main_base_addr_i.value = 0x2018
    dut.main_row_slot_i.value = 3
    dut.main_valid_bytes_i.value = 16
    await pulse(dut, "main_start_i")
    await accept_request(dut, 0x2000, 3, 0)
    beat0 = int.from_bytes(bytes(range(32, 64)), "little")
    write_req, _, _, _ = await return_response(dut, beat0)
    assert write_req == 0

    await accept_request(dut, 0x2020, 3, 0)
    beat1 = int.from_bytes(bytes(range(64, 96)), "little")
    write_req, write_addr, write_data, pop_slot = await return_response(dut, beat1)
    crossing_bank = 2 * 3
    expected = int.from_bytes(bytes(range(56, 72)) + bytes(16), "little")
    assert pop_slot == 3
    assert write_req == 1 << crossing_bank
    assert bank_lane(write_addr, crossing_bank, BANK_ADDR_WIDTH) == 0
    assert bank_lane(write_data, crossing_bank, DATA_WIDTH) == expected
    await Timer(1, unit="ps")
    assert int(dut.main_done_o.value) == 1
    assert int(dut.fetch_beats_o.value) == 4
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    # Background scanning allocates its row, but a newly started foreground
    # row takes OBI priority while both request FSMs are active.
    dut.row_ring_mode_i.value = 1
    dut.input_w_i.value = 1
    dut.kernel_h_i.value = 1
    dut.main_valid_bytes_i.value = 8
    dut.background_base_ih_i.value = 5
    dut.background_row_base_addr_i.value = 0x3000
    await pulse(dut, "background_start_i")
    await Timer(1, unit="ps")
    assert int(dut.background_alloc_valid_o.value) == 1
    assert int(dut.background_alloc_slot_o.value) == 5
    assert int(dut.background_alloc_ih_o.value) == 5
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    dut.main_base_addr_i.value = 0x4000
    dut.main_row_slot_i.value = 1
    dut.main_row_ih_i.value = 1
    await pulse(dut, "main_start_i")
    dut.obi_gnt_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.obi_req_o.value) == 1
    assert int(dut.obi_addr_o.value) == 0x4000
    assert int(dut.main_request_accepted_o.value) == 1

    dut.clear_count_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.clear_count_i.value = 0
    dut.obi_gnt_i.value = 0
    assert int(dut.fetch_beats_o.value) == 0
