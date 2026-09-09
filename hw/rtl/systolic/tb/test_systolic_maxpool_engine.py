import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


LANES = 32


def pack_i8(values):
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xFF) << (lane * 8)
    return packed


def unpack_i8(value):
    result = []
    for lane in range(LANES):
        byte = (value >> (lane * 8)) & 0xFF
        result.append(byte if byte < 0x80 else byte - 0x100)
    return result


async def reset(dut):
    dut.rst_ni.value = 0
    dut.flush_i.value = 0
    dut.kernel_vectors_i.value = 4
    dut.in_data_i.value = 0
    dut.in_valid_i.value = 0
    dut.out_ready_i.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def push_row(dut, values):
    await FallingEdge(dut.clk_i)
    dut.in_data_i.value = pack_i8(values)
    dut.in_valid_i.value = 1
    await Timer(1, unit="ps")
    assert dut.in_ready_o.value
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.in_valid_i.value = 0


@cocotb.test()
async def maxpool_accumulates_signed_rows_and_holds_output(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    rows = [
        [lane - 80 for lane in range(LANES)],
        [50 - lane for lane in range(LANES)],
        [-10 + (lane % 7) for lane in range(LANES)],
        [20 if lane & 1 else -20 for lane in range(LANES)],
    ]
    expected = [max(row[lane] for row in rows) for lane in range(LANES)]

    for row in rows:
        await push_row(dut, row)

    assert dut.out_valid_o.value
    assert unpack_i8(int(dut.out_data_o.value)) == expected
    assert not dut.in_ready_o.value

    held = int(dut.out_data_o.value)
    for _ in range(3):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        assert dut.out_valid_o.value
        assert int(dut.out_data_o.value) == held

    dut.out_ready_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.out_ready_i.value = 0
    assert not dut.out_valid_o.value
    assert dut.in_ready_o.value


@cocotb.test()
async def maxpool_flush_discards_partial_window(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    await push_row(dut, [-100] * LANES)
    await push_row(dut, [100] * LANES)
    dut.flush_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.flush_i.value = 0

    for value in (-5, -4, -3, -2):
        await push_row(dut, [value] * LANES)

    assert dut.out_valid_o.value
    assert unpack_i8(int(dut.out_data_o.value)) == [-2] * LANES
