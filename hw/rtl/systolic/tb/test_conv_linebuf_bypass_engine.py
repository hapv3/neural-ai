import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


BEAT_BYTES = 32
ARRAY_DIM = 32


def drive_defaults(dut):
    defaults = {
        "start_i": 0,
        "last_i": 0,
        "spatial_addr_i": 0,
        "base_ih_i": 0,
        "base_iw_i": 0,
        "input_h_i": 4,
        "input_w_i": 4,
        "channel_addr_offset_i": 0,
        "valid_bytes_i": BEAT_BYTES,
        "lane_base_i": 0,
        "c32_blocked_mode_i": 0,
        "obi_gnt_i": 0,
        "obi_rvalid_i": 0,
        "obi_rdata_i": 0,
        "row_ready_i": 0,
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
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0


async def grant_request(dut, expected_address):
    dut.obi_gnt_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.obi_req_o.value) == 1
    assert int(dut.obi_addr_o.value) == expected_address
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.obi_gnt_i.value = 0


async def return_response(dut, data):
    dut.obi_rdata_i.value = data
    dut.obi_rvalid_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.obi_rvalid_i.value = 0


def packed_bytes(values):
    return int.from_bytes(bytes(values), "little")


def row_bytes(dut):
    value = int(dut.row_o.value)
    return [(value >> (lane * 8)) & 0xFF for lane in range(ARRAY_DIM)]


@cocotb.test()
async def bypass_engine_handles_padding_alignment_and_crossing(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    # Aligned full-width input follows the legacy PREP/REQ/WAIT/EMIT timing.
    dut.spatial_addr_i.value = 0x1000
    dut.c32_blocked_mode_i.value = 1
    await pulse_start(dut)
    assert int(dut.debug_state_o.value) == 9
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.debug_state_o.value) == 10
    await grant_request(dut, 0x1000)
    assert int(dut.debug_state_o.value) == 11
    first = list(range(BEAT_BYTES))
    await return_response(dut, packed_bytes(first))
    assert int(dut.debug_state_o.value) == 8
    assert int(dut.row_valid_o.value) == 1
    assert row_bytes(dut) == first

    # Backpressure holds the row. The next pixel is sampled only after accept.
    dut.spatial_addr_i.value = 0x2018
    dut.valid_bytes_i.value = 16
    dut.c32_blocked_mode_i.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        assert int(dut.row_valid_o.value) == 1
        assert row_bytes(dut) == first
    dut.row_ready_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.row_ready_i.value = 0
    assert int(dut.debug_state_o.value) == 9
    assert int(dut.emitted_vectors_o.value) == 1

    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.debug_state_o.value) == 10
    await grant_request(dut, 0x2000)
    beat0 = list(range(32, 64))
    await return_response(dut, packed_bytes(beat0))
    assert int(dut.debug_state_o.value) == 12
    await grant_request(dut, 0x2020)
    beat1 = list(range(64, 96))
    dut.last_i.value = 1
    await return_response(dut, packed_bytes(beat1))
    assert int(dut.debug_state_o.value) == 8
    assert row_bytes(dut)[:16] == list(range(56, 72))
    assert row_bytes(dut)[16:] == [0] * 16
    assert int(dut.fetch_beats_o.value) == 3
    dut.row_ready_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.debug_state_o.value) == 0
    assert int(dut.emitted_vectors_o.value) == 2

    # An out-of-bounds spatial location emits zeros without an OBI request.
    await reset(dut)
    dut.base_ih_i.value = -1
    dut.last_i.value = 1
    await pulse_start(dut)
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.debug_state_o.value) == 8
    assert int(dut.obi_req_o.value) == 0
    assert row_bytes(dut) == [0] * ARRAY_DIM
