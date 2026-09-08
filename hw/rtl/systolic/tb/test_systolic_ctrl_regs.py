import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


REG_SYS_START = 0x0110
REG_BINARY_CTRL = 0x012C
REG_BINARY_RHS_PTR = 0x0130
REG_BINARY_RHS_ROW_STRIDE = 0x0134
REG_BINARY_RHS_TILE_COLS = 0x0138
REG_BINARY_LHS_MULT = 0x013C
REG_BINARY_LHS_SHIFT = 0x0140
REG_BINARY_RHS_MULT = 0x0144
REG_BINARY_RHS_SHIFT = 0x0148
REG_BINARY_OUTPUT_MULT = 0x014C
REG_BINARY_OUTPUT_SHIFT = 0x0150
REG_BINARY_ZERO_POINTS = 0x0154
REG_BINARY_CLAMP = 0x0158
REG_BINARY_DOUBLE_ROUND = 0x015C


def signed(value, width=32):
    value = int(value) & ((1 << width) - 1)
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


async def reset(dut):
    dut.rst_ni.value = 0
    dut.req_i.value = 0
    dut.addr_i.value = 0
    dut.we_i.value = 0
    dut.be_i.value = 0
    dut.wdata_i.value = 0
    dut.cfg_sys_done_i.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def mmio_write(dut, address, value):
    await FallingEdge(dut.clk_i)
    dut.req_i.value = 1
    dut.addr_i.value = address
    dut.we_i.value = 1
    dut.be_i.value = 0xF
    dut.wdata_i.value = value & 0xFFFFFFFF
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.req_i.value = 0
    dut.we_i.value = 0


async def mmio_read(dut, address):
    await FallingEdge(dut.clk_i)
    dut.req_i.value = 1
    dut.addr_i.value = address
    dut.we_i.value = 0
    dut.be_i.value = 0xF
    await RisingEdge(dut.clk_i)
    await Timer(1, unit="ps")
    assert dut.rvalid_o.value
    value = int(dut.rdata_o.value)
    await FallingEdge(dut.clk_i)
    dut.req_i.value = 0
    return value


async def program_binary(dut, values):
    for address, value in values.items():
        await mmio_write(dut, address, value)


def assert_active_config(dut, config):
    assert int(dut.cfg_binary_en_o.value) == config["enable"]
    assert int(dut.cfg_binary_mode_o.value) == config["mode"]
    assert int(dut.cfg_binary_rhs_ptr_o.value) == config["rhs_ptr"]
    assert int(dut.cfg_binary_rhs_row_stride_bytes_o.value) == config["rhs_stride"]
    assert int(dut.cfg_binary_rhs_tile_cols_o.value) == config["rhs_cols"]
    assert int(dut.cfg_binary_lhs_multiplier_o.value) == config["lhs_mult"]
    assert int(dut.cfg_binary_lhs_shift_o.value) == config["lhs_shift"]
    assert int(dut.cfg_binary_rhs_multiplier_o.value) == config["rhs_mult"]
    assert int(dut.cfg_binary_rhs_shift_o.value) == config["rhs_shift"]
    assert int(dut.cfg_binary_output_multiplier_o.value) == config["output_mult"]
    assert int(dut.cfg_binary_output_shift_o.value) == config["output_shift"]
    assert signed(dut.cfg_binary_lhs_zero_point_o.value) == config["lhs_zp"]
    assert signed(dut.cfg_binary_rhs_zero_point_o.value) == config["rhs_zp"]
    assert signed(dut.cfg_binary_output_zero_point_o.value) == config["output_zp"]
    assert signed(dut.cfg_binary_clamp_min_o.value) == config["clamp_min"]
    assert signed(dut.cfg_binary_clamp_max_o.value) == config["clamp_max"]
    assert int(dut.cfg_binary_double_round_shift_o.value) == config["double_round"]


def register_values(config):
    return {
        REG_BINARY_CTRL: config["enable"] | (config["mode"] << 1),
        REG_BINARY_RHS_PTR: config["rhs_ptr"],
        REG_BINARY_RHS_ROW_STRIDE: config["rhs_stride"],
        REG_BINARY_RHS_TILE_COLS: config["rhs_cols"],
        REG_BINARY_LHS_MULT: config["lhs_mult"],
        REG_BINARY_LHS_SHIFT: config["lhs_shift"],
        REG_BINARY_RHS_MULT: config["rhs_mult"],
        REG_BINARY_RHS_SHIFT: config["rhs_shift"],
        REG_BINARY_OUTPUT_MULT: config["output_mult"],
        REG_BINARY_OUTPUT_SHIFT: config["output_shift"],
        REG_BINARY_ZERO_POINTS: ((config["lhs_zp"] & 0xFF) |
                                 ((config["rhs_zp"] & 0xFF) << 8) |
                                 ((config["output_zp"] & 0xFF) << 16)),
        REG_BINARY_CLAMP: ((config["clamp_min"] & 0xFF) |
                           ((config["clamp_max"] & 0xFF) << 8)),
        REG_BINARY_DOUBLE_ROUND: config["double_round"],
    }


@cocotb.test()
async def binary_registers_are_shadowed_until_start(dut):
    cocotb.start_soon(Clock(dut.clk_i, 2, unit="ns").start())
    await reset(dut)

    defaults = {
        "enable": 0, "mode": 0, "rhs_ptr": 0, "rhs_stride": 0, "rhs_cols": 0,
        "lhs_mult": 1, "lhs_shift": 0, "rhs_mult": 1, "rhs_shift": 0,
        "output_mult": 1, "output_shift": 0, "lhs_zp": 0, "rhs_zp": 0,
        "output_zp": 0, "clamp_min": -128, "clamp_max": 127, "double_round": 0,
    }
    first = {
        "enable": 1, "mode": 2, "rhs_ptr": 0x10124000,
        "rhs_stride": 320, "rhs_cols": 7, "lhs_mult": 1234567,
        "lhs_shift": 17, "rhs_mult": 7654321, "rhs_shift": 23,
        "output_mult": 3456789, "output_shift": 19, "lhs_zp": -117,
        "rhs_zp": 93, "output_zp": -11, "clamp_min": -101,
        "clamp_max": 99, "double_round": 20,
    }
    second = dict(first)
    second.update({"mode": 1, "rhs_ptr": 0x1012A000, "lhs_zp": 12,
                   "rhs_zp": -9, "output_zp": 31, "double_round": 0})

    assert_active_config(dut, defaults)
    await program_binary(dut, register_values(first))
    assert_active_config(dut, defaults)
    assert await mmio_read(dut, REG_BINARY_CTRL) == 0

    await mmio_write(dut, REG_SYS_START, 1)
    assert_active_config(dut, first)
    assert await mmio_read(dut, REG_BINARY_CTRL) == 5
    assert await mmio_read(dut, REG_BINARY_ZERO_POINTS) == register_values(first)[REG_BINARY_ZERO_POINTS]

    await program_binary(dut, register_values(second))
    assert_active_config(dut, first)
    await mmio_write(dut, REG_SYS_START, 1)
    assert_active_config(dut, second)

