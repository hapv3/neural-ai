import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


def drive_defaults(dut):
    dut.job_start_i.value = 0
    dut.advance_i.value = 0
    dut.kgen_multi_i.value = 1
    dut.c32_group_stationary_i.value = 0
    dut.generic_linear_k32_i.value = 0
    dut.k_tiles_i.value = 1
    dut.input_c_i.value = 32
    dut.kernel_h_i.value = 1
    dut.kernel_w_i.value = 1
    dut.initial_seed_ic_i.value = 0
    dut.initial_seed_kw_i.value = 0
    dut.initial_seed_kh_i.value = 0
    dut.channel_addr_offset_i.value = 0


async def reset(dut):
    drive_defaults(dut)
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)


async def pulse(dut, signal):
    signal.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    signal.value = 0


def current_state(dut):
    return (
        int(dut.tile_index_o.value),
        int(dut.seed_kh_o.value),
        int(dut.seed_kw_o.value),
        int(dut.seed_ic_o.value),
        int(dut.channel_offset_o.value),
    )


def next_state(dut):
    return (
        int(dut.next_seed_kh_o.value),
        int(dut.next_seed_kw_o.value),
        int(dut.next_seed_ic_o.value),
        int(dut.next_channel_offset_o.value),
    )


@cocotb.test()
async def c32_stationary_walks_spatial_taps_then_channel_groups(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.c32_group_stationary_i.value = 1
    dut.k_tiles_i.value = 12
    dut.input_c_i.value = 96
    dut.kernel_h_i.value = 2
    dut.kernel_w_i.value = 2
    dut.channel_addr_offset_i.value = 0x800
    await pulse(dut, dut.job_start_i)

    expected = [
        (0, 0, 0, 0, 0),
        (1, 0, 1, 0, 0),
        (2, 1, 0, 0, 0),
        (3, 1, 1, 0, 0),
        (4, 0, 0, 32, 0x800),
        (5, 0, 1, 32, 0x800),
        (6, 1, 0, 32, 0x800),
        (7, 1, 1, 32, 0x800),
        (8, 0, 0, 64, 0x1000),
        (9, 0, 1, 64, 0x1000),
        (10, 1, 0, 64, 0x1000),
        (11, 1, 1, 64, 0x1000),
    ]
    for index, state in enumerate(expected):
        assert current_state(dut) == state
        assert bool(dut.has_next_o.value) == (index != len(expected) - 1)
        if index != len(expected) - 1:
            await pulse(dut, dut.advance_i)

    await Timer(1, unit="ps")
    assert next_state(dut) == (0, 0, 0, 0)


@cocotb.test()
async def generic_linear_walk_preserves_channel_seed(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.generic_linear_k32_i.value = 1
    dut.k_tiles_i.value = 6
    dut.kernel_h_i.value = 2
    dut.kernel_w_i.value = 3
    dut.initial_seed_ic_i.value = 64
    await pulse(dut, dut.job_start_i)

    expected_taps = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for index, (kh, kw) in enumerate(expected_taps):
        assert current_state(dut) == (index, kh, kw, 64, 0)
        assert bool(dut.has_next_o.value) == (index != len(expected_taps) - 1)
        if index != len(expected_taps) - 1:
            await pulse(dut, dut.advance_i)

    await Timer(1, unit="ps")
    assert next_state(dut) == (0, 0, 64, 0)


@cocotb.test()
async def job_start_has_priority_and_restores_programmed_seed(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.c32_group_stationary_i.value = 1
    dut.k_tiles_i.value = 4
    dut.input_c_i.value = 64
    dut.initial_seed_ic_i.value = 32
    dut.initial_seed_kw_i.value = 2
    dut.initial_seed_kh_i.value = 1
    dut.channel_addr_offset_i.value = 0x400

    dut.advance_i.value = 1
    await pulse(dut, dut.job_start_i)
    dut.advance_i.value = 0

    assert current_state(dut) == (0, 1, 2, 32, 0)
    assert bool(dut.has_next_o.value)
