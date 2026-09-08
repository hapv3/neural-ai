import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


BEAT_BYTES = 32


async def reset(dut):
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.obi_gnt_i.value = 0
    dut.obi_rvalid_i.value = 0
    dut.obi_rdata_i.value = 0
    dut.out_ready_i.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def start_stream(dut, base, rows, stride=0, cols=0):
    await FallingEdge(dut.clk_i)
    dut.base_addr_i.value = base
    dut.row_count_i.value = rows
    dut.row_stride_bytes_i.value = stride
    dut.tile_cols_i.value = cols
    dut.start_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_i.value = 0


def expected_addresses(base, rows, stride, cols):
    result = []
    address = base
    col = 0
    for _ in range(rows):
        result.append(address)
        if stride and cols and col + 1 == cols:
            address += stride - (cols - 1) * BEAT_BYTES
            col = 0
        else:
            address += BEAT_BYTES
            col += 1
    return result


@cocotb.test()
async def binary_operand_stream_handles_outstanding_reads_and_backpressure(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    base = 0x10102000
    rows = 29
    stride = 224
    cols = 5
    addresses = expected_addresses(base, rows, stride, cols)
    expected_data = [((index + 1) * 0x01010101) & ((1 << 256) - 1)
                     for index in range(rows)]
    await start_stream(dut, base, rows, stride, cols)

    rng = random.Random(0xB10C)
    accepted_addresses = []
    response_queue = []
    received = []
    done_pulses = 0

    for cycle in range(500):
        await FallingEdge(dut.clk_i)
        dut.obi_gnt_i.value = rng.randrange(4) != 0
        dut.out_ready_i.value = rng.randrange(5) not in (0, 1)

        response_valid = bool(response_queue) and response_queue[0][0] <= cycle
        dut.obi_rvalid_i.value = response_valid
        dut.obi_rdata_i.value = response_queue[0][1] if response_valid else 0
        await Timer(1, unit="ps")

        request_fire = bool(dut.obi_req_o.value and dut.obi_gnt_i.value)
        output_fire = bool(dut.out_valid_o.value and dut.out_ready_i.value)
        if request_fire:
            request_index = len(accepted_addresses)
            accepted_addresses.append(int(dut.obi_addr_o.value))
            latency = rng.randint(1, 4)
            ready_cycle = cycle + latency
            if response_queue:
                ready_cycle = max(ready_cycle, response_queue[-1][0] + 1)
            response_queue.append((ready_cycle, expected_data[request_index]))
        if output_fire:
            received.append(int(dut.out_data_o.value))
        if dut.done_o.value:
            done_pulses += 1

        await RisingEdge(dut.clk_i)
        if response_valid:
            response_queue.pop(0)

        if len(received) == rows and not dut.busy_o.value:
            break
    else:
        raise AssertionError(
            f"timeout requests={len(accepted_addresses)} responses={len(received)} "
            f"reserved={int(dut.reserved_q.value)}"
        )

    await FallingEdge(dut.clk_i)
    done_pulses += int(dut.done_o.value)
    assert accepted_addresses == addresses
    assert received == expected_data
    assert done_pulses == 1
    assert not dut.obi_we_o.value
    assert int(dut.obi_be_o.value) == (1 << BEAT_BYTES) - 1


@cocotb.test()
async def binary_operand_stream_sustains_one_request_per_cycle(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)
    rows = 16
    base = 0x10104000
    await start_stream(dut, base, rows)

    previous_request = False
    accepted_cycles = []
    received = []

    for cycle in range(80):
        await FallingEdge(dut.clk_i)
        dut.obi_gnt_i.value = 1
        dut.out_ready_i.value = 1
        dut.obi_rvalid_i.value = previous_request
        dut.obi_rdata_i.value = len(received) + 1
        await Timer(1, unit="ps")
        request_fire = bool(dut.obi_req_o.value)
        if request_fire:
            accepted_cycles.append(cycle)
        if dut.out_valid_o.value and dut.out_ready_i.value:
            received.append(int(dut.out_data_o.value))
        await RisingEdge(dut.clk_i)
        previous_request = request_fire
        if len(received) == rows and not dut.busy_o.value:
            break
    else:
        raise AssertionError(
            f"timeout requests={len(accepted_cycles)}/{rows} "
            f"responses={len(received)}/{rows} busy={int(dut.busy_o.value)} "
            f"request_count={int(dut.request_count_q.value)} "
            f"response_count={int(dut.response_count_q.value)} "
            f"reserved={int(dut.reserved_q.value)}"
        )

    assert len(accepted_cycles) == rows
    assert all(second == first + 1 for first, second in zip(
        accepted_cycles, accepted_cycles[1:]))
    assert received == list(range(1, rows + 1))


@cocotb.test()
async def binary_operand_stream_completes_empty_job(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)
    await FallingEdge(dut.clk_i)
    dut.base_addr_i.value = 0x10100000
    dut.row_count_i.value = 0
    dut.row_stride_bytes_i.value = 0
    dut.tile_cols_i.value = 0
    dut.start_i.value = 1
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ps")
    assert dut.done_o.value
    assert not dut.busy_o.value
    assert not dut.obi_req_o.value
