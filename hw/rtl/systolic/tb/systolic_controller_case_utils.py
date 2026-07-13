from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


REG_SYS_W_PTR = 0x0100
REG_SYS_I_PTR = 0x0104
REG_SYS_O_PTR = 0x0108
REG_SYS_DIM_M = 0x010C
REG_SYS_START = 0x0110
REG_SYS_DONE = 0x0114
REG_SYS_PSUM_PTR = 0x0118
REG_SYS_ACCUM_CTRL = 0x011C
REG_RQ_CTRL = 0x0120
REG_RQ_CMIN = 0x0124
REG_RQ_CMAX = 0x0128
REG_RQ_SHIFT_BASE = 0x0300
REG_SYS_OFM_ROW_STRIDE = 0x0450
REG_SYS_OFM_TILE_COLS = 0x0454
REG_SYS_PSUM_ROW_STRIDE = 0x0458
REG_LB_CTRL = 0x0400
REG_LB_INPUT_BASE = 0x0404
REG_LB_INPUT_H = 0x0408
REG_LB_INPUT_W = 0x040C
REG_LB_INPUT_C = 0x0410
REG_LB_OUTPUT_W = 0x0414
REG_LB_STRIDE = 0x0418
REG_LB_PAD = 0x041C
REG_LB_ROW_STRIDE = 0x0428
REG_LB_PIXEL_STRIDE = 0x042C
REG_LB_OW_STEP = 0x0430
REG_LB_OH_STEP = 0x0434
REG_LB_KERNEL = 0x0438
REG_LB_C_BASE = 0x043C
REG_LB_SPATIAL_M = 0x0440
REG_LB_LANE_BASE = 0x0444
REG_LB_K_TILES = 0x0448
REG_LB_K_SEED = 0x044C
REG_LB_PRECOMP0 = 0x045C
REG_LB_CHANNEL_OFFSET = 0x0460
REG_LB_COALESCE_K_BYTES = 0x0464

ARRAY_DIM = 32
BEAT_BYTES = 32
OFM_ROW_BYTES = ARRAY_DIM * 4
WEIGHT_ADDR = 0x00001000
IFM_ADDR = 0x00008000
OFM_ADDR = 0x00010000
PSUM_ADDR = 0x00018000

LB_EN = 0x1
LB_COALESCE = 0x2
LB_KGEN = 0x4
LB_POOL = 0x8
LB_C32_FAST = 0x10


def mem_index(addr):
    return addr >> 5


def pack_u8(values):
    word = 0
    for index, value in enumerate(values):
        word |= (value & 0xFF) << (index * 8)
    return word


def pack_s32(values):
    word = 0
    for index, value in enumerate(values):
        word |= (value & 0xFFFFFFFF) << (index * 32)
    return word


def unpack_u8(value):
    return [(value >> (index * 8)) & 0xFF for index in range(BEAT_BYTES)]


def unpack_s32(value):
    out = []
    for index in range(8):
        raw = (value >> (index * 32)) & 0xFFFFFFFF
        if raw & 0x80000000:
            raw -= 0x100000000
        out.append(raw)
    return out


def signed_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def clamp_i8_to_u8(value):
    value = max(-128, min(127, value))
    return value & 0xFF


async def start_clock_and_reset(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    import cocotb
    cocotb.start_soon(clock.start())
    dut.rst_ni.value = 0
    dut.ctrl_req_i.value = 0
    dut.ctrl_addr_i.value = 0
    dut.ctrl_we_i.value = 0
    dut.ctrl_be_i.value = 0
    dut.ctrl_wdata_i.value = 0
    await Timer(30, unit="ns")
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def mmio_write(dut, addr, data):
    dut.ctrl_addr_i.value = addr
    dut.ctrl_wdata_i.value = data
    dut.ctrl_be_i.value = 0xF
    dut.ctrl_we_i.value = 1
    dut.ctrl_req_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.ctrl_req_i.value = 0
    dut.ctrl_we_i.value = 0
    dut.ctrl_be_i.value = 0
    await RisingEdge(dut.clk_i)


def write_bytes(dut, base_addr, values):
    for beat_index in range((len(values) + BEAT_BYTES - 1) // BEAT_BYTES):
        chunk = values[beat_index * BEAT_BYTES:(beat_index + 1) * BEAT_BYTES]
        padded = chunk + [0] * (BEAT_BYTES - len(chunk))
        dut.tcdm_mem[mem_index(base_addr) + beat_index].value = pack_u8(padded)


def write_s32_rows(dut, base_addr, rows):
    for row_index, row in enumerate(rows):
        assert len(row) == ARRAY_DIM
        for beat in range(4):
            chunk = row[beat * 8:(beat + 1) * 8]
            dut.tcdm_mem[mem_index(base_addr) + row_index * 4 + beat].value = pack_s32(chunk)


def read_s32_row(dut, base_addr, row_index):
    out = []
    for beat in range(4):
        value = int(dut.tcdm_mem[mem_index(base_addr) + row_index * 4 + beat].value)
        out.extend(unpack_s32(value))
    return out


def read_u8_row(dut, base_addr, row_index):
    value = int(dut.tcdm_mem[mem_index(base_addr) + row_index].value)
    return unpack_u8(value)


def fill_weight_tiles(dut, weight_addr, k_tiles=1, value=1):
    beat = pack_u8([value] * BEAT_BYTES)
    for tile in range(k_tiles):
        for row in range(ARRAY_DIM):
            dut.tcdm_mem[mem_index(weight_addr) + tile * ARRAY_DIM + row].value = beat


def clear_output_i32(dut, base_addr, rows):
    for beat in range(rows * 4):
        dut.tcdm_mem[mem_index(base_addr) + beat].value = 0


def clear_output_i8(dut, base_addr, rows):
    for beat in range(rows):
        dut.tcdm_mem[mem_index(base_addr) + beat].value = 0


async def configure_identity_requant(dut, shift=0, clamp_min=-128, clamp_max=127):
    await mmio_write(dut, REG_RQ_CMIN, clamp_min & 0xFFFFFFFF)
    await mmio_write(dut, REG_RQ_CMAX, clamp_max & 0xFFFFFFFF)
    for ch in range(ARRAY_DIM):
        await mmio_write(dut, REG_RQ_SHIFT_BASE + ch * 4, shift)
    await mmio_write(dut, REG_RQ_CTRL, 1)


async def disable_optional_modes(dut):
    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_LB_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_OFM_ROW_STRIDE, 0)
    await mmio_write(dut, REG_SYS_OFM_TILE_COLS, 0)
    await mmio_write(dut, REG_SYS_PSUM_ROW_STRIDE, 0)


async def program_direct_gemm(dut, dim_m, weight_addr=WEIGHT_ADDR, ifm_addr=IFM_ADDR,
                              ofm_addr=OFM_ADDR, psum_addr=0, accum=False):
    await mmio_write(dut, REG_LB_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 1 if accum else 0)
    await mmio_write(dut, REG_SYS_W_PTR, weight_addr)
    await mmio_write(dut, REG_SYS_I_PTR, ifm_addr)
    await mmio_write(dut, REG_SYS_O_PTR, ofm_addr)
    await mmio_write(dut, REG_SYS_PSUM_PTR, psum_addr)
    await mmio_write(dut, REG_SYS_DIM_M, dim_m)
    await mmio_write(dut, REG_SYS_DONE, 0)


async def program_linebuf(dut, *, input_base, input_h, input_w, input_c,
                          output_h, output_w, kernel_h, kernel_w,
                          stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                          c_base=0, lane_base=0, coalesce=False, kgen=False,
                          pool=False, c32_fast=False, k_tiles=1,
                          k_seed_kh=0, k_seed_kw=0, k_seed_ic=0,
                          block_valid_bytes=0, channel_offset=0,
                          coalesce_k_bytes=0, dim_m=None,
                          weight_addr=WEIGHT_ADDR, ofm_addr=OFM_ADDR,
                          psum_addr=0, accum=False):
    row_stride = input_w * input_c
    pixel_stride = input_c
    spatial_m = output_h * output_w
    sys_dim_m = dim_m if dim_m is not None else spatial_m
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 1 if accum else 0)
    await mmio_write(dut, REG_SYS_W_PTR, weight_addr)
    await mmio_write(dut, REG_SYS_I_PTR, 0)
    await mmio_write(dut, REG_SYS_O_PTR, ofm_addr)
    await mmio_write(dut, REG_SYS_PSUM_PTR, psum_addr)
    await mmio_write(dut, REG_SYS_DIM_M, sys_dim_m)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_LB_INPUT_BASE, input_base - pad_h * row_stride)
    await mmio_write(dut, REG_LB_INPUT_H, input_h)
    await mmio_write(dut, REG_LB_INPUT_W, input_w)
    await mmio_write(dut, REG_LB_INPUT_C, input_c)
    await mmio_write(dut, REG_LB_OUTPUT_W, output_w)
    await mmio_write(dut, REG_LB_STRIDE, (stride_w << 16) | stride_h)
    await mmio_write(dut, REG_LB_PAD, (pad_w << 16) | pad_h)
    await mmio_write(dut, REG_LB_ROW_STRIDE, row_stride)
    await mmio_write(dut, REG_LB_PIXEL_STRIDE, pixel_stride)
    await mmio_write(dut, REG_LB_OW_STEP, stride_w * pixel_stride)
    await mmio_write(dut, REG_LB_OH_STEP, stride_h * row_stride)
    await mmio_write(dut, REG_LB_KERNEL, (kernel_w << 16) | kernel_h)
    await mmio_write(dut, REG_LB_C_BASE, c_base)
    await mmio_write(dut, REG_LB_SPATIAL_M, spatial_m)
    await mmio_write(dut, REG_LB_LANE_BASE, lane_base)
    await mmio_write(dut, REG_LB_K_TILES, k_tiles)
    await mmio_write(dut, REG_LB_K_SEED, (k_seed_kh << 24) | (k_seed_kw << 16) | k_seed_ic)
    await mmio_write(dut, REG_LB_PRECOMP0, block_valid_bytes)
    await mmio_write(dut, REG_LB_CHANNEL_OFFSET, channel_offset)
    await mmio_write(dut, REG_LB_COALESCE_K_BYTES, coalesce_k_bytes)
    ctrl = LB_EN
    if coalesce:
        ctrl |= LB_COALESCE
    if kgen:
        ctrl |= LB_KGEN
    if pool:
        ctrl |= LB_POOL
    if c32_fast:
        ctrl |= LB_C32_FAST
    await mmio_write(dut, REG_LB_CTRL, ctrl)


async def start_and_wait(dut, timeout_cycles=20000):
    await mmio_write(dut, REG_SYS_START, 1)
    compute = 0
    weight = 0
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk_i)
        if int(dut.perf_compute_en_o.value) == 1:
            compute += 1
        if int(dut.perf_weight_load_en_o.value) == 1:
            weight += 1
        if int(dut.cfg_sys_done_o.value) == 1:
            return {"compute": compute, "weight": weight}
    raise AssertionError("systolic_controller did not complete before timeout")


def valid_tap_count(oh, ow, input_h, input_w, kernel_h, kernel_w,
                    stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    count = 0
    for kh in range(kernel_h):
        ih = oh * stride_h + kh - pad_h
        for kw in range(kernel_w):
            iw = ow * stride_w + kw - pad_w
            if 0 <= ih < input_h and 0 <= iw < input_w:
                count += 1
    return count


def pool_expected_row(input_data, input_h, input_w, input_c, oh, ow,
                      kernel_h, kernel_w, stride_h=1, stride_w=1,
                      pad_h=0, pad_w=0):
    out = []
    for ch in range(ARRAY_DIM):
        if ch >= input_c:
            out.append(0)
            continue
        best = -128
        for kh in range(kernel_h):
            ih = oh * stride_h + kh - pad_h
            for kw in range(kernel_w):
                iw = ow * stride_w + kw - pad_w
                if 0 <= ih < input_h and 0 <= iw < input_w:
                    value = signed_i8(input_data[((ih * input_w) + iw) * input_c + ch])
                else:
                    value = -128
                best = max(best, value)
        out.append(best & 0xFF)
    return out
