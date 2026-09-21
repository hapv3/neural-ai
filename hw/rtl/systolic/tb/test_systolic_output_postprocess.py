import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


LANES = 32


def pack_lanes(values, width):
    word = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        word |= (value & mask) << (lane * width)
    return word


def unpack_bytes(word):
    return [(word >> (lane * 8)) & 0xFF for lane in range(LANES)]


def drive_valid_config(dut):
    dut.requant_enable_i.value = 1
    dut.bias_i.value = 0
    dut.multiplier_i.value = pack_lanes([1] * LANES, 32)
    dut.shift_i.value = 0
    dut.zero_point_i.value = 0
    dut.clamp_min_i.value = (-128) & 0xFFFFFFFF
    dut.clamp_max_i.value = 127

    dut.binary_enable_i.value = 0
    dut.binary_active_i.value = 0
    dut.binary_mode_i.value = 0
    dut.binary_rhs_ptr_i.value = 0x1000
    dut.binary_rhs_row_stride_bytes_i.value = 0
    dut.binary_rhs_tile_cols_i.value = 0
    dut.row_count_i.value = 0
    dut.binary_lhs_multiplier_i.value = 1
    dut.binary_lhs_shift_i.value = 0
    dut.binary_rhs_multiplier_i.value = 1
    dut.binary_rhs_shift_i.value = 0
    dut.binary_output_multiplier_i.value = 1
    dut.binary_output_shift_i.value = 0
    dut.binary_lhs_zero_point_i.value = 0
    dut.binary_rhs_zero_point_i.value = 0
    dut.binary_output_zero_point_i.value = 0
    dut.binary_clamp_min_i.value = (-128) & 0xFFFFFFFF
    dut.binary_clamp_max_i.value = 127
    dut.binary_double_round_shift_i.value = 0
    dut.binary_forbidden_i.value = 0


async def reset(dut):
    dut.rst_ni.value = 0
    dut.flush_i.value = 0
    dut.job_start_i.value = 0
    dut.acc_i.value = 0
    dut.acc_valid_i.value = 0
    dut.obi_gnt_i.value = 0
    dut.obi_rvalid_i.value = 0
    dut.obi_rdata_i.value = 0
    dut.out_ready_i.value = 0
    drive_valid_config(dut)
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def push_acc(dut, values):
    await FallingEdge(dut.clk_i)
    dut.acc_i.value = pack_lanes(values, 32)
    dut.acc_valid_i.value = 1
    await Timer(1, unit="ps")
    while not dut.acc_ready_o.value:
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        await Timer(1, unit="ps")
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.acc_valid_i.value = 0


@cocotb.test()
async def postprocess_requant_path_preserves_backpressure(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    values = [lane - 16 for lane in range(LANES)]
    await push_acc(dut, values)

    for _ in range(12):
        await FallingEdge(dut.clk_i)
        if dut.out_valid_o.value:
            break
        await RisingEdge(dut.clk_i)
    else:
        raise AssertionError("requant output did not become valid")

    held_data = int(dut.packed_o.value)
    assert dut.debug_requant_out_valid_o.value
    assert not dut.debug_requant_out_ready_o.value
    for _ in range(2):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        assert dut.out_valid_o.value
        assert int(dut.packed_o.value) == held_data

    dut.out_ready_i.value = 1
    await RisingEdge(dut.clk_i)
    assert unpack_bytes(held_data) == [value & 0xFF for value in values]
    assert not dut.invalid_o.value
    assert not dut.obi_req_o.value


@cocotb.test()
async def postprocess_rejects_legacy_systolic_binary_path(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.binary_enable_i.value = 1
    dut.binary_active_i.value = 1
    dut.out_ready_i.value = 1
    await Timer(1, unit="ps")
    assert dut.binary_config_invalid_o.value
    assert not dut.binary_busy_o.value
    assert not dut.obi_req_o.value


@cocotb.test()
async def postprocess_rejects_invalid_configuration(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    dut.shift_i.value = pack_lanes([32] + [0] * (LANES - 1), 8)
    await Timer(1, unit="ps")
    assert dut.requant_config_invalid_o.value

    dut.shift_i.value = 0
    dut.binary_enable_i.value = 1
    dut.binary_rhs_ptr_i.value = 0x1001
    await Timer(1, unit="ps")
    assert dut.binary_config_invalid_o.value

    dut.binary_rhs_ptr_i.value = 0x1000
    dut.binary_forbidden_i.value = 1
    await Timer(1, unit="ps")
    assert dut.binary_config_invalid_o.value

    dut.binary_enable_i.value = 0
    await Timer(1, unit="ps")
    assert not dut.binary_config_invalid_o.value
