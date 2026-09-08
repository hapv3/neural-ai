import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


ARRAY_DIM = 32
DATA_WIDTH = 256
K_MAX = 5


def pack_lanes(values, width):
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (value & mask) << (lane * width)
    return packed


def pack_window(taps):
    packed = 0
    mask = (1 << DATA_WIDTH) - 1
    for (kh, kw), value in taps.items():
        packed |= (value & mask) << (((kh * K_MAX) + kw) * DATA_WIDTH)
    return packed


def pack_bytes(values):
    return pack_lanes(values, 8)


def unpack_bytes(value):
    return [(value >> (lane * 8)) & 0xFF for lane in range(ARRAY_DIM)]


async def clock_edge(dut):
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ps")


async def reset(dut):
    dut.rst_ni.value = 0
    dut.flush_i.value = 0
    dut.advance_i.value = 0
    dut.window_i.value = 0
    dut.lane_kh_i.value = 0
    dut.lane_kw_i.value = 0
    dut.lane_ic_i.value = 0
    dut.tap_kh_i.value = 0
    dut.tap_kw_i.value = 0
    dut.kernel_h_i.value = 1
    dut.kernel_w_i.value = 1
    dut.c_base_i.value = 0
    dut.input_c_i.value = ARRAY_DIM
    dut.lane_base_i.value = 0
    dut.block_valid_bytes_i.value = ARRAY_DIM
    dut.coalesce_i.value = 0
    dut.kgen_i.value = 0
    dut.c32_kgen_fast_i.value = 0
    dut.valid_i.value = 0
    for _ in range(3):
        await clock_edge(dut)
    dut.rst_ni.value = 1
    await clock_edge(dut)


async def push_and_wait(dut):
    dut.advance_i.value = 1
    dut.valid_i.value = 1
    await clock_edge(dut)
    dut.valid_i.value = 0
    for _ in range(3):
        await clock_edge(dut)
    assert dut.valid_o.value


@cocotb.test()
async def direct_format_preserves_latency_and_backpressure(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    tap = [0xA0 + lane for lane in range(ARRAY_DIM)]
    dut.window_i.value = pack_window({(1, 2): pack_bytes(tap)})
    dut.tap_kh_i.value = 1
    dut.tap_kw_i.value = 2
    dut.kernel_h_i.value = 3
    dut.kernel_w_i.value = 3
    dut.lane_base_i.value = 3
    dut.block_valid_bytes_i.value = 4

    await push_and_wait(dut)
    expected = [0] * ARRAY_DIM
    expected[3:7] = tap[:4]
    assert unpack_bytes(int(dut.row_o.value)) == expected
    assert not dut.empty_o.value

    held_row = int(dut.row_o.value)
    dut.advance_i.value = 0
    dut.window_i.value = 0
    for _ in range(3):
        await clock_edge(dut)
        assert dut.valid_o.value
        assert int(dut.row_o.value) == held_row

    dut.advance_i.value = 1
    await clock_edge(dut)
    assert not dut.valid_o.value
    assert dut.empty_o.value


@cocotb.test()
async def coalesced_format_and_flush(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    taps = {}
    expected = [0] * ARRAY_DIM
    dst = 1
    for kh in range(2):
        for kw in range(2):
            values = [0x20 + (kh * 0x40) + (kw * 0x10) + lane for lane in range(ARRAY_DIM)]
            taps[(kh, kw)] = pack_bytes(values)
            expected[dst:dst + 3] = values[:3]
            dst += 3

    dut.window_i.value = pack_window(taps)
    dut.kernel_h_i.value = 2
    dut.kernel_w_i.value = 2
    dut.lane_base_i.value = 1
    dut.block_valid_bytes_i.value = 3
    dut.coalesce_i.value = 1

    await push_and_wait(dut)
    assert unpack_bytes(int(dut.row_o.value)) == expected

    dut.valid_i.value = 1
    await clock_edge(dut)
    dut.valid_i.value = 0
    dut.flush_i.value = 1
    await clock_edge(dut)
    dut.flush_i.value = 0
    assert dut.empty_o.value
    assert not dut.valid_o.value
