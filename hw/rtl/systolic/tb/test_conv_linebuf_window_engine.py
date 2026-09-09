import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


DATA_WIDTH = 256
K_MAX = 5
BANKS = 14
BANK_ADDR_WIDTH = 9


def drive_defaults(dut):
    defaults = {
        "clear_i": 0,
        "load_request_i": 0,
        "load_capture_i": 0,
        "load_request_kw_i": 0,
        "load_capture_kw_i": 0,
        "slide_request_i": 0,
        "slide_from_iw_i": 0,
        "slide_commit_i": 0,
        "input_h_i": 8,
        "input_w_i": 8,
        "kernel_h_i": 3,
        "kernel_w_i": 3,
        "stride_w_i": 1,
        "base_ih_i": 0,
        "base_iw_i": 0,
        "row_ring_mode_i": 0,
        "row_cache_full_i": 0,
        "pad_vector_i": 0xA5,
        "bank_read_data_i": 0,
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


def packed_bank_data(values):
    packed = 0
    for bank, value in values.items():
        packed |= value << (bank * DATA_WIDTH)
    return packed


def packed_lane(value, lane, width):
    return (value >> (lane * width)) & ((1 << width) - 1)


def window_cell(dut, kh, kw):
    packed = int(dut.window_o.value)
    return (packed >> ((kh * K_MAX + kw) * DATA_WIDTH)) & ((1 << DATA_WIDTH) - 1)


async def pulse_capture(dut, kw, bank_values):
    dut.load_capture_kw_i.value = kw
    dut.bank_read_data_i.value = packed_bank_data(bank_values)
    dut.load_capture_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.load_capture_i.value = 0


@cocotb.test()
async def window_engine_loads_padding_and_slides(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    # Initial column requests select one parity bank per active kernel row.
    dut.base_ih_i.value = 1
    dut.base_iw_i.value = 2
    dut.load_request_kw_i.value = 0
    dut.load_request_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.bank_read_req_o.value) == (1 << 0) | (1 << 2) | (1 << 4)
    addresses = int(dut.bank_read_addr_o.value)
    assert packed_lane(addresses, 0, BANK_ADDR_WIDTH) == 1
    assert packed_lane(addresses, 2, BANK_ADDR_WIDTH) == 1
    assert packed_lane(addresses, 4, BANK_ADDR_WIDTH) == 1
    dut.load_request_i.value = 0

    # Capture all three columns into the owned window register.
    dut.base_iw_i.value = 0
    await pulse_capture(dut, 0, {0: 0x10, 2: 0x20, 4: 0x30})
    await pulse_capture(dut, 1, {1: 0x11, 3: 0x21, 5: 0x31})
    await pulse_capture(dut, 2, {0: 0x12, 2: 0x22, 4: 0x32})
    assert window_cell(dut, 0, 0) == 0x10
    assert window_cell(dut, 0, 1) == 0x11
    assert window_cell(dut, 0, 2) == 0x12
    assert window_cell(dut, 2, 2) == 0x32

    # A stride-1 slide reuses columns 1/2 and captures only the new column.
    dut.slide_from_iw_i.value = 0
    dut.slide_request_i.value = 1
    await Timer(1, unit="ps")
    assert int(dut.bank_read_req_o.value) == (1 << 1) | (1 << 3) | (1 << 5)
    dut.slide_request_i.value = 0
    dut.bank_read_data_i.value = packed_bank_data({1: 0x13, 3: 0x23, 5: 0x33})
    dut.slide_commit_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.slide_commit_i.value = 0
    assert window_cell(dut, 0, 0) == 0x11
    assert window_cell(dut, 0, 1) == 0x12
    assert window_cell(dut, 0, 2) == 0x13
    assert window_cell(dut, 2, 2) == 0x33

    # Out-of-bounds rows are padded during initial load.
    dut.clear_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.clear_i.value = 0
    dut.base_ih_i.value = -1
    dut.base_iw_i.value = 0
    await pulse_capture(dut, 0, {2: 0x40, 4: 0x50})
    assert window_cell(dut, 0, 0) == 0xA5
    assert window_cell(dut, 1, 0) == 0x40
    assert window_cell(dut, 2, 0) == 0x50
