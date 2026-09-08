import cocotb

from systolic_controller_case_utils import (
    ARRAY_DIM,
    BEAT_BYTES,
    IFM_ADDR,
    WEIGHT_ADDR,
    configure_identity_requant,
    fill_weight_tiles,
    mem_index,
    mmio_write,
    pack_u8,
    program_direct_gemm,
    start_and_wait,
    start_clock_and_reset,
    unpack_u8,
    write_bytes,
)


REG_SYS_OFM_ROW_STRIDE = 0x0450
REG_SYS_OFM_TILE_COLS = 0x0454
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

MODE_ADD = 0
MODE_SUB = 1
MODE_MUL = 2


async def configure_binary(dut, mode, rhs_addr, row_stride, tile_cols):
    lhs_zero_point = -3
    rhs_zero_point = 5
    output_zero_point = 7
    await mmio_write(dut, REG_BINARY_RHS_PTR, rhs_addr)
    await mmio_write(dut, REG_BINARY_RHS_ROW_STRIDE, row_stride)
    await mmio_write(dut, REG_BINARY_RHS_TILE_COLS, tile_cols)
    await mmio_write(dut, REG_BINARY_LHS_MULT, 1)
    await mmio_write(dut, REG_BINARY_LHS_SHIFT, 0)
    await mmio_write(dut, REG_BINARY_RHS_MULT, 1)
    await mmio_write(dut, REG_BINARY_RHS_SHIFT, 0)
    await mmio_write(dut, REG_BINARY_OUTPUT_MULT, 1)
    await mmio_write(dut, REG_BINARY_OUTPUT_SHIFT, 0)
    await mmio_write(
        dut,
        REG_BINARY_ZERO_POINTS,
        ((lhs_zero_point & 0xFF) |
         ((rhs_zero_point & 0xFF) << 8) |
         ((output_zero_point & 0xFF) << 16)),
    )
    await mmio_write(dut, REG_BINARY_CLAMP, ((-100 & 0xFF) | ((100 & 0xFF) << 8)))
    await mmio_write(dut, REG_BINARY_DOUBLE_ROUND, 0)
    await mmio_write(dut, REG_BINARY_CTRL, 1 | (mode << 1))


def binary_reference(lhs, rhs, mode):
    lhs_centered = lhs - (-3)
    rhs_centered = rhs - 5
    if mode == MODE_ADD:
        result = lhs_centered + rhs_centered
    elif mode == MODE_SUB:
        result = lhs_centered - rhs_centered
    else:
        result = lhs_centered * rhs_centered
    return max(-100, min(100, result + 7)) & 0xFF


def write_strided_rows(dut, base, rows, row_stride, tile_cols):
    for row_index, row in enumerate(rows):
        logical_row = (row_index // tile_cols) * (row_stride // BEAT_BYTES)
        logical_row += row_index % tile_cols
        dut.tcdm_mem[mem_index(base) + logical_row].value = pack_u8(row)


def read_strided_row(dut, base, row_index, row_stride, tile_cols):
    logical_row = (row_index // tile_cols) * (row_stride // BEAT_BYTES)
    logical_row += row_index % tile_cols
    return unpack_u8(int(dut.tcdm_mem[mem_index(base) + logical_row].value))


@cocotb.test()
async def systolic_controller_fuses_general_binary_post_op(dut):
    await start_clock_and_reset(dut)

    dim_m = 4
    tile_cols = 2
    row_stride = 3 * BEAT_BYTES
    rhs_addr = 0x00012000
    output_bases = {
        MODE_ADD: 0x00014000,
        MODE_SUB: 0x00015000,
        MODE_MUL: 0x00016000,
    }
    ifm_values = [1, 2, -1, -2]
    rhs_rows = [
        [((lane + 3 * row) % 31) - 15 for lane in range(ARRAY_DIM)]
        for row in range(dim_m)
    ]

    fill_weight_tiles(dut, WEIGHT_ADDR, value=1)
    write_bytes(dut, IFM_ADDR, [value for value in ifm_values for _ in range(ARRAY_DIM)])
    write_strided_rows(dut, rhs_addr, rhs_rows, row_stride, tile_cols)
    await configure_identity_requant(dut)

    for mode, output_addr in output_bases.items():
        for beat in range(8):
            dut.tcdm_mem[mem_index(output_addr) + beat].value = pack_u8([0xA5] * ARRAY_DIM)
        await program_direct_gemm(
            dut, dim_m, weight_addr=WEIGHT_ADDR, ifm_addr=IFM_ADDR,
            ofm_addr=output_addr,
        )
        await mmio_write(dut, REG_SYS_OFM_ROW_STRIDE, row_stride)
        await mmio_write(dut, REG_SYS_OFM_TILE_COLS, tile_cols)
        await configure_binary(dut, mode, rhs_addr, row_stride, tile_cols)
        await start_and_wait(dut, timeout_cycles=4000)

        for row_index, ifm_value in enumerate(ifm_values):
            lhs = max(-128, min(127, ARRAY_DIM * ifm_value))
            expected = [
                binary_reference(lhs, rhs, mode) for rhs in rhs_rows[row_index]
            ]
            actual = read_strided_row(
                dut, output_addr, row_index, row_stride, tile_cols
            )
            assert actual == expected, (
                f"mode={mode} row={row_index}: {actual} != {expected}"
            )

        # The padding beat in each logical row must not be overwritten.
        for padding_beat in (2, 5):
            value = int(dut.tcdm_mem[mem_index(output_addr) + padding_beat].value)
            assert value == pack_u8([0xA5] * ARRAY_DIM)
