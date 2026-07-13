import cocotb
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
REG_SYS_OFM_ROW_STRIDE = 0x0450
REG_SYS_OFM_TILE_COLS = 0x0454
REG_SYS_PSUM_ROW_STRIDE = 0x0458
REG_RQ_CTRL = 0x0120
REG_RQ_CMIN = 0x0124
REG_RQ_CMAX = 0x0128
REG_RQ_SHIFT_BASE = 0x0300
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

WEIGHT_ADDR = 0x00001000
IFM_ADDR = 0x00003000
OFM_ADDR = 0x00005000
DIM_M = 2
ARRAY_DIM = 32
BEAT_BYTES = 32


def pack_u8(values):
    word = 0
    for index, value in enumerate(values):
        word |= (value & 0xFF) << (index * 8)
    return word


def signed_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def unpack_s32_beat(value):
    words = []
    for index in range(8):
        raw = (value >> (index * 32)) & 0xFFFFFFFF
        if raw & 0x80000000:
            raw -= 0x100000000
        words.append(raw)
    return words


def unpack_u8_beat(value):
    return [(value >> (index * 8)) & 0xFF for index in range(BEAT_BYTES)]


def clamp_u8_signed(value):
    value = max(-128, min(127, value))
    return value & 0xFF


def write_bytes(dut, base_addr, values):
    assert (base_addr % BEAT_BYTES) == 0
    for beat_index in range((len(values) + BEAT_BYTES - 1) // BEAT_BYTES):
        chunk = values[beat_index * BEAT_BYTES:(beat_index + 1) * BEAT_BYTES]
        padded = chunk + [0] * (BEAT_BYTES - len(chunk))
        dut.tcdm_mem[mem_index(base_addr) + beat_index].value = pack_u8(padded)


def golden_channel_linebuf_rows(input_data, input_h, input_w, input_c,
                                output_h, output_w, kernel_h, kernel_w,
                                stride_h, stride_w, pad_h, pad_w, c_base):
    rows = []
    for oh in range(output_h):
        for ow in range(output_w):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    row = [0] * ARRAY_DIM
                    ih = oh * stride_h + kh - pad_h
                    iw = ow * stride_w + kw - pad_w
                    if 0 <= ih < input_h and 0 <= iw < input_w:
                        for lane in range(ARRAY_DIM):
                            channel = c_base + lane
                            if channel < input_c:
                                index = ((ih * input_w) + iw) * input_c + channel
                                row[lane] = signed_i8(input_data[index])
                    rows.append(row)
    return rows


async def reset(dut):
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


async def configure_identity_requant(dut, shift=0):
    await mmio_write(dut, REG_RQ_CMIN, 0xFFFFFF80)
    await mmio_write(dut, REG_RQ_CMAX, 0x0000007F)
    for ch in range(ARRAY_DIM):
        await mmio_write(dut, REG_RQ_SHIFT_BASE + ch * 4, shift)
    await mmio_write(dut, REG_RQ_CTRL, 1)


async def wait_controller_done(dut, timeout_cycles):
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk_i)
        if int(dut.cfg_sys_done_o.value) == 1:
            return
    raise AssertionError("systolic_controller did not complete before timeout")


def mem_index(addr):
    return addr >> 5


@cocotb.test()
async def systolic_controller_gemm32_strided_ofm(dut):
    """
    Scenario: unit-test OFM drain row-stride addressing without linebuffer.
    Target: four output vectors are written as a 2x2 tile inside a wider
    logical output row, leaving the skipped columns untouched.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    dim_m = 4
    full_output_w = 4
    tile_ow = 2
    row_stride_bytes = full_output_w * ARRAY_DIM * 4

    one_beat = pack_u8([1] * BEAT_BYTES)
    sentinel = pack_u8([0xA5] * BEAT_BYTES)
    for beat in range(ARRAY_DIM):
        dut.tcdm_mem[mem_index(WEIGHT_ADDR) + beat].value = one_beat
    for row in range(dim_m):
        dut.tcdm_mem[mem_index(IFM_ADDR) + row].value = one_beat
    for beat in range(32):
        dut.tcdm_mem[mem_index(OFM_ADDR) + beat].value = sentinel

    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_LB_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_OFM_ROW_STRIDE, row_stride_bytes)
    await mmio_write(dut, REG_SYS_OFM_TILE_COLS, tile_ow)
    await mmio_write(dut, REG_SYS_PSUM_ROW_STRIDE, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, IFM_ADDR)
    await mmio_write(dut, REG_SYS_O_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_PSUM_PTR, 0)
    await mmio_write(dut, REG_SYS_DIM_M, dim_m)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_SYS_START, 1)

    done_seen = False
    for _ in range(1200):
        await RisingEdge(dut.clk_i)
        if int(dut.cfg_sys_done_o.value) == 1:
            done_seen = True
            break

    assert done_seen, "strided OFM GEMM run did not complete before timeout"

    expected = [ARRAY_DIM] * ARRAY_DIM
    written_rows = [0, 1, full_output_w, full_output_w + 1]
    for logical_row in written_rows:
        got = []
        for beat in range(4):
            value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + logical_row * 4 + beat].value)
            got.extend(unpack_s32_beat(value))
        assert got == expected, f"strided OFM logical row {logical_row} mismatch"

    untouched_rows = [2, 3, 6, 7]
    for logical_row in untouched_rows:
        value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + logical_row * 4].value)
        assert value == sentinel, f"strided OFM overwrote skipped logical row {logical_row}"


@cocotb.test()
async def systolic_controller_shadow_preload_next_config(dut):
    """
    Scenario: preload the next systolic/requant config while the current GEMM is running.
    Target: active config remains stable until the next START; staged requant takes
    effect only for the following invocation.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    ofm_a_addr = OFM_ADDR
    ofm_b_addr = OFM_ADDR + 0x1000
    dim_a = 64
    dim_b = 1
    one_beat = pack_u8([1] * BEAT_BYTES)

    for beat in range(ARRAY_DIM):
        dut.tcdm_mem[mem_index(WEIGHT_ADDR) + beat].value = one_beat
    for row in range(dim_a):
        dut.tcdm_mem[mem_index(IFM_ADDR) + row].value = one_beat
    for beat in range(dim_a * 4):
        dut.tcdm_mem[mem_index(ofm_a_addr) + beat].value = 0
    dut.tcdm_mem[mem_index(ofm_b_addr)].value = 0

    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_LB_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_OFM_ROW_STRIDE, 0)
    await mmio_write(dut, REG_SYS_OFM_TILE_COLS, 0)
    await mmio_write(dut, REG_SYS_PSUM_ROW_STRIDE, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, IFM_ADDR)
    await mmio_write(dut, REG_SYS_O_PTR, ofm_a_addr)
    await mmio_write(dut, REG_SYS_PSUM_PTR, 0)
    await mmio_write(dut, REG_SYS_DIM_M, dim_a)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_SYS_START, 1)

    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, IFM_ADDR)
    await mmio_write(dut, REG_SYS_O_PTR, ofm_b_addr)
    await mmio_write(dut, REG_SYS_PSUM_PTR, 0)
    await mmio_write(dut, REG_SYS_DIM_M, dim_b)
    await configure_identity_requant(dut, shift=0)

    await wait_controller_done(dut, 2000)

    expected_i32 = [ARRAY_DIM] * ARRAY_DIM
    for row in range(dim_a):
        got = []
        for beat in range(4):
            value = int(dut.tcdm_mem[mem_index(ofm_a_addr) + row * 4 + beat].value)
            got.extend(unpack_s32_beat(value))
        assert got == expected_i32, f"shadow preload corrupted active INT32 row {row}"

    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_SYS_START, 1)
    await wait_controller_done(dut, 1000)

    got_i8 = unpack_u8_beat(int(dut.tcdm_mem[mem_index(ofm_b_addr)].value))
    assert got_i8 == [ARRAY_DIM] * ARRAY_DIM, "staged requant config did not commit on next START"


@cocotb.test()
async def systolic_controller_gemm32_all_ones(dut):
    """
    Scenario: unit-test the integrated systolic_controller + local array.
    Target: controller reads 32 weight beats and M IFM beats, drives the
    internal array, drains M full INT32 OFM rows through four OBI write ports,
    and reports completion without relying on npu_cluster wiring.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    one_beat = pack_u8([1] * BEAT_BYTES)
    for beat in range(ARRAY_DIM):
        dut.tcdm_mem[mem_index(WEIGHT_ADDR) + beat].value = one_beat
    for row in range(DIM_M):
        dut.tcdm_mem[mem_index(IFM_ADDR) + row].value = one_beat
    for beat in range(DIM_M * 4):
        dut.tcdm_mem[mem_index(OFM_ADDR) + beat].value = 0

    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_LB_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, IFM_ADDR)
    await mmio_write(dut, REG_SYS_O_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_PSUM_PTR, 0)
    await mmio_write(dut, REG_SYS_DIM_M, DIM_M)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_SYS_START, 1)

    weight_pulses = 0
    compute_pulses = 0
    done_seen = False
    for _ in range(600):
        await RisingEdge(dut.clk_i)
        if int(dut.perf_weight_load_en_o.value) == 1:
            weight_pulses += 1
        if int(dut.perf_compute_en_o.value) == 1:
            compute_pulses += 1
        if int(dut.cfg_sys_done_o.value) == 1:
            done_seen = True
            break

    assert done_seen, "systolic_controller did not complete before timeout"
    assert weight_pulses == ARRAY_DIM, f"weight_load pulses={weight_pulses}, expected={ARRAY_DIM}"
    assert compute_pulses == DIM_M, f"compute pulses={compute_pulses}, expected={DIM_M}"

    expected = [ARRAY_DIM] * ARRAY_DIM
    for row in range(DIM_M):
        got = []
        for beat in range(4):
            value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + row * 4 + beat].value)
            got.extend(unpack_s32_beat(value))
        assert got == expected, f"OFM row {row} mismatch: got={got}, expected={expected}"


@cocotb.test()
async def systolic_controller_channel_linebuf_1x1_c32(dut):
    """
    Scenario: configure the controller channel linebuffer for a 1x1 C=32
    packed tile. The packer emits one full IFM row per output pixel, the
    controller feeds those rows into the local array, and writeback stores
    one INT32 OFM row per spatial position.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    input_h = 4
    input_w = 3
    input_c = 32
    output_h = 4
    output_w = 3
    kernel_h = 1
    kernel_w = 1
    stride_h = 1
    stride_w = 1
    pad_h = 0
    pad_w = 0
    c_base = 0
    spatial_m = output_h * output_w
    dim_m = spatial_m * kernel_h * kernel_w

    input_data = [1] * (input_h * input_w * input_c)
    write_bytes(dut, IFM_ADDR, input_data)

    one_beat = pack_u8([1] * BEAT_BYTES)
    for beat in range(ARRAY_DIM):
        dut.tcdm_mem[mem_index(WEIGHT_ADDR) + beat].value = one_beat
    for beat in range(dim_m * 4):
        dut.tcdm_mem[mem_index(OFM_ADDR) + beat].value = 0

    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, IFM_ADDR)
    await mmio_write(dut, REG_SYS_O_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_PSUM_PTR, 0)
    await mmio_write(dut, REG_SYS_DIM_M, dim_m)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_LB_INPUT_BASE, IFM_ADDR - pad_h * input_w * input_c)
    await mmio_write(dut, REG_LB_INPUT_H, input_h)
    await mmio_write(dut, REG_LB_INPUT_W, input_w)
    await mmio_write(dut, REG_LB_INPUT_C, input_c)
    await mmio_write(dut, REG_LB_OUTPUT_W, output_w)
    await mmio_write(dut, REG_LB_STRIDE, (stride_w << 16) | stride_h)
    await mmio_write(dut, REG_LB_PAD, (pad_w << 16) | pad_h)
    await mmio_write(dut, REG_LB_ROW_STRIDE, input_w * input_c)
    await mmio_write(dut, REG_LB_PIXEL_STRIDE, input_c)
    await mmio_write(dut, REG_LB_OW_STEP, stride_w * input_c)
    await mmio_write(dut, REG_LB_OH_STEP, stride_h * input_w * input_c)
    await mmio_write(dut, REG_LB_KERNEL, (kernel_w << 16) | kernel_h)
    await mmio_write(dut, REG_LB_C_BASE, c_base)
    await mmio_write(dut, REG_LB_SPATIAL_M, spatial_m)
    await mmio_write(dut, REG_LB_LANE_BASE, 0)
    await mmio_write(dut, REG_LB_CTRL, 1)
    await mmio_write(dut, REG_SYS_START, 1)

    compute_pulses = 0
    done_seen = False
    for _ in range(1200):
        await RisingEdge(dut.clk_i)
        if int(dut.perf_compute_en_o.value) == 1:
            compute_pulses += 1
        if int(dut.cfg_sys_done_o.value) == 1:
            done_seen = True
            break

    assert done_seen, "channel line-buffer 1x1 run did not complete before timeout"
    assert compute_pulses == dim_m, f"compute pulses={compute_pulses}, expected={dim_m}"

    expected = [ARRAY_DIM] * ARRAY_DIM
    for row in range(dim_m):
        got = []
        for beat in range(4):
            value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + row * 4 + beat].value)
            got.extend(unpack_s32_beat(value))
        assert got == expected, f"line-buffer 1x1 OFM row {row} mismatch: got={got}, expected={expected}"


@cocotb.test()
async def systolic_controller_channel_linebuf_coalesced_3x3_c3(dut):
    """
    Scenario: configure coalesced line-buffer mode for a small Conv2D K tile
    where KH*KW*IC fits in one 32-lane systolic row.
    Target: controller receives exactly one IFM row per output pixel and the
    array accumulates all valid 3x3/C3 lanes against one packed weight tile.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    input_h = 2
    input_w = 4
    input_c = 3
    output_h = 2
    output_w = 4
    kernel_h = 3
    kernel_w = 3
    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1
    spatial_m = output_h * output_w

    input_data = [1] * (input_h * input_w * input_c)
    write_bytes(dut, IFM_ADDR, input_data)

    one_beat = pack_u8([1] * BEAT_BYTES)
    for beat in range(ARRAY_DIM):
        dut.tcdm_mem[mem_index(WEIGHT_ADDR) + beat].value = one_beat
    for beat in range(spatial_m * 4):
        dut.tcdm_mem[mem_index(OFM_ADDR) + beat].value = 0

    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, 0)
    await mmio_write(dut, REG_SYS_O_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_PSUM_PTR, 0)
    await mmio_write(dut, REG_SYS_DIM_M, spatial_m)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_LB_INPUT_BASE, IFM_ADDR - pad_h * input_w * input_c)
    await mmio_write(dut, REG_LB_INPUT_H, input_h)
    await mmio_write(dut, REG_LB_INPUT_W, input_w)
    await mmio_write(dut, REG_LB_INPUT_C, input_c)
    await mmio_write(dut, REG_LB_OUTPUT_W, output_w)
    await mmio_write(dut, REG_LB_STRIDE, (stride_w << 16) | stride_h)
    await mmio_write(dut, REG_LB_PAD, (pad_w << 16) | pad_h)
    await mmio_write(dut, REG_LB_ROW_STRIDE, input_w * input_c)
    await mmio_write(dut, REG_LB_PIXEL_STRIDE, input_c)
    await mmio_write(dut, REG_LB_OW_STEP, stride_w * input_c)
    await mmio_write(dut, REG_LB_OH_STEP, stride_h * input_w * input_c)
    await mmio_write(dut, REG_LB_KERNEL, (kernel_w << 16) | kernel_h)
    await mmio_write(dut, REG_LB_C_BASE, 0)
    await mmio_write(dut, REG_LB_SPATIAL_M, spatial_m)
    await mmio_write(dut, REG_LB_LANE_BASE, 0)
    await mmio_write(dut, REG_LB_CTRL, 0x3)
    await mmio_write(dut, REG_SYS_START, 1)

    compute_pulses = 0
    done_seen = False
    for _ in range(3000):
        await RisingEdge(dut.clk_i)
        if int(dut.perf_compute_en_o.value) == 1:
            compute_pulses += 1
        if int(dut.cfg_sys_done_o.value) == 1:
            done_seen = True
            break

    assert done_seen, "coalesced line-buffer 3x3 run did not complete before timeout"
    assert compute_pulses == spatial_m, f"compute pulses={compute_pulses}, expected={spatial_m}"

    for oh in range(output_h):
        for ow in range(output_w):
            valid_taps = 0
            for kh in range(kernel_h):
                ih = oh * stride_h + kh - pad_h
                for kw in range(kernel_w):
                    iw = ow * stride_w + kw - pad_w
                    if 0 <= ih < input_h and 0 <= iw < input_w:
                        valid_taps += 1
            expected = [valid_taps * input_c] * ARRAY_DIM
            row = oh * output_w + ow
            got = []
            for beat in range(4):
                value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + row * 4 + beat].value)
                got.extend(unpack_s32_beat(value))
            assert got == expected, f"coalesced line-buffer OFM row {row} mismatch: got={got}, expected={expected}"


@cocotb.test()
async def systolic_controller_channel_linebuf_kgen_3x3_c32(dut):
    """
    Scenario: run Conv2D 3x3/C32 as nine internal K tiles with one MMIO start.
    Target: RTL advances the {kh,kw,ic} seed by 32 lanes per tile, reloads the
    next weight tile, accumulates against the previous output, and reports one
    final done pulse for the whole layer tile.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    input_h = 4
    input_w = 4
    input_c = 32
    output_h = 4
    output_w = 4
    kernel_h = 3
    kernel_w = 3
    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1
    spatial_m = output_h * output_w
    k_tiles = kernel_h * kernel_w

    input_data = [1] * (input_h * input_w * input_c)
    write_bytes(dut, IFM_ADDR, input_data)

    one_beat = pack_u8([1] * BEAT_BYTES)
    for tile in range(k_tiles):
        for beat in range(ARRAY_DIM):
            dut.tcdm_mem[mem_index(WEIGHT_ADDR) + tile * ARRAY_DIM + beat].value = one_beat
    for beat in range(spatial_m * 4):
        dut.tcdm_mem[mem_index(OFM_ADDR) + beat].value = 0

    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, 0)
    await mmio_write(dut, REG_SYS_O_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_PSUM_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_DIM_M, spatial_m)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_LB_INPUT_BASE, IFM_ADDR - pad_h * input_w * input_c)
    await mmio_write(dut, REG_LB_INPUT_H, input_h)
    await mmio_write(dut, REG_LB_INPUT_W, input_w)
    await mmio_write(dut, REG_LB_INPUT_C, input_c)
    await mmio_write(dut, REG_LB_OUTPUT_W, output_w)
    await mmio_write(dut, REG_LB_STRIDE, (stride_w << 16) | stride_h)
    await mmio_write(dut, REG_LB_PAD, (pad_w << 16) | pad_h)
    await mmio_write(dut, REG_LB_ROW_STRIDE, input_w * input_c)
    await mmio_write(dut, REG_LB_PIXEL_STRIDE, input_c)
    await mmio_write(dut, REG_LB_OW_STEP, stride_w * input_c)
    await mmio_write(dut, REG_LB_OH_STEP, stride_h * input_w * input_c)
    await mmio_write(dut, REG_LB_KERNEL, (kernel_w << 16) | kernel_h)
    await mmio_write(dut, REG_LB_C_BASE, 0)
    await mmio_write(dut, REG_LB_SPATIAL_M, spatial_m)
    await mmio_write(dut, REG_LB_LANE_BASE, 0)
    await mmio_write(dut, REG_LB_K_TILES, k_tiles)
    await mmio_write(dut, REG_LB_K_SEED, 0)
    await mmio_write(dut, REG_LB_CTRL, 0x7)
    await mmio_write(dut, REG_SYS_START, 1)

    compute_pulses = 0
    done_seen = False
    for _ in range(40000):
        await RisingEdge(dut.clk_i)
        if int(dut.perf_compute_en_o.value) == 1:
            compute_pulses += 1
        if int(dut.cfg_sys_done_o.value) == 1:
            done_seen = True
            break

    if not done_seen:
        dut._log.warning(
            "kgen timeout: state=%s ktile=%s seed=(%s,%s,%s) req=%s rsp=%s drain=%s linebuf_state=%s done=%s emitted=%s ow=%s kh=%s kw=%s valid=%s ready=%s ofm_empty=%s",
            dut.dut.state_q.value,
            dut.dut.k_tile_idx_q.value,
            dut.dut.k_seed_kh_q.value,
            dut.dut.k_seed_kw_q.value,
            dut.dut.k_seed_ic_q.value,
            dut.dut.req_cnt_q.value,
            dut.dut.rsp_cnt_q.value,
            dut.dut.drain_cnt_q.value,
            dut.dut.i_conv_channel_linebuf_packer.state_q.value,
            dut.dut.linebuf_done.value,
            dut.dut.linebuf_emitted_vectors.value,
            dut.dut.i_conv_channel_linebuf_packer.ow_q.value,
            dut.dut.i_conv_channel_linebuf_packer.kh_q.value,
            dut.dut.i_conv_channel_linebuf_packer.kw_q.value,
            dut.dut.linebuf_row_valid.value,
            dut.dut.linebuf_row_ready.value,
            dut.dut.ofm_fifo_empty.value,
        )
    assert done_seen, "kgen line-buffer 3x3/C32 run did not complete before timeout"
    assert compute_pulses == spatial_m * k_tiles, (
        f"compute pulses={compute_pulses}, expected={spatial_m * k_tiles}"
    )

    for oh in range(output_h):
        for ow in range(output_w):
            valid_taps = 0
            for kh in range(kernel_h):
                ih = oh * stride_h + kh - pad_h
                for kw in range(kernel_w):
                    iw = ow * stride_w + kw - pad_w
                    if 0 <= ih < input_h and 0 <= iw < input_w:
                        valid_taps += 1
            expected = [valid_taps * input_c] * ARRAY_DIM
            row = oh * output_w + ow
            got = []
            for beat in range(4):
                value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + row * 4 + beat].value)
                got.extend(unpack_s32_beat(value))
            assert got == expected, f"kgen line-buffer OFM row {row} mismatch: got={got}, expected={expected}"


@cocotb.test()
async def systolic_controller_channel_linebuf_kgen_3x3_c32_requant(dut):
    """
    Scenario: run Conv2D 3x3/C32 as nine internal K tiles with final-only
    requant enabled. The controller must keep partial sums internal until the
    last K tile, then emit one packed int8 row per spatial output.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    input_h = 4
    input_w = 4
    input_c = 32
    output_h = 4
    output_w = 4
    kernel_h = 3
    kernel_w = 3
    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1
    spatial_m = output_h * output_w
    k_tiles = kernel_h * kernel_w

    input_data = [1] * (input_h * input_w * input_c)
    write_bytes(dut, IFM_ADDR, input_data)

    one_beat = pack_u8([1] * BEAT_BYTES)
    for tile in range(k_tiles):
        for beat in range(ARRAY_DIM):
            dut.tcdm_mem[mem_index(WEIGHT_ADDR) + tile * ARRAY_DIM + beat].value = one_beat
    for beat in range(spatial_m):
        dut.tcdm_mem[mem_index(OFM_ADDR) + beat].value = 0

    await configure_identity_requant(dut, shift=2)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, 0)
    await mmio_write(dut, REG_SYS_O_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_PSUM_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_DIM_M, spatial_m)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_LB_INPUT_BASE, IFM_ADDR - pad_h * input_w * input_c)
    await mmio_write(dut, REG_LB_INPUT_H, input_h)
    await mmio_write(dut, REG_LB_INPUT_W, input_w)
    await mmio_write(dut, REG_LB_INPUT_C, input_c)
    await mmio_write(dut, REG_LB_OUTPUT_W, output_w)
    await mmio_write(dut, REG_LB_STRIDE, (stride_w << 16) | stride_h)
    await mmio_write(dut, REG_LB_PAD, (pad_w << 16) | pad_h)
    await mmio_write(dut, REG_LB_ROW_STRIDE, input_w * input_c)
    await mmio_write(dut, REG_LB_PIXEL_STRIDE, input_c)
    await mmio_write(dut, REG_LB_OW_STEP, stride_w * input_c)
    await mmio_write(dut, REG_LB_OH_STEP, stride_h * input_w * input_c)
    await mmio_write(dut, REG_LB_KERNEL, (kernel_w << 16) | kernel_h)
    await mmio_write(dut, REG_LB_C_BASE, 0)
    await mmio_write(dut, REG_LB_SPATIAL_M, spatial_m)
    await mmio_write(dut, REG_LB_LANE_BASE, 0)
    await mmio_write(dut, REG_LB_K_TILES, k_tiles)
    await mmio_write(dut, REG_LB_K_SEED, 0)
    await mmio_write(dut, REG_LB_CTRL, 0x7)
    await mmio_write(dut, REG_SYS_START, 1)

    compute_pulses = 0
    done_seen = False
    for _ in range(40000):
        await RisingEdge(dut.clk_i)
        if int(dut.perf_compute_en_o.value) == 1:
            compute_pulses += 1
        if int(dut.cfg_sys_done_o.value) == 1:
            done_seen = True
            break

    if not done_seen:
        dut._log.warning(
            "requant kgen timeout: state=%s drain=%s ktile=%s has_next=%s req=%s rsp=%s drain_cnt=%s "
            "ofm_empty=%s psum_empty=%s psum_buf_active=%s psum_buf_drain=%s req_valid=%s req_ready=%s "
            "rq_out_valid=%s rq_out_ready=%s accum_sent=%s o_req=%s linebuf_state=%s linebuf_done=%s emitted=%s",
            dut.dut.state_q.value,
            dut.dut.drain_state_q.value,
            dut.dut.k_tile_idx_q.value,
            dut.dut.linebuf_has_next_k_tile.value,
            dut.dut.req_cnt_q.value,
            dut.dut.rsp_cnt_q.value,
            dut.dut.drain_cnt_q.value,
            dut.dut.ofm_fifo_empty.value,
            dut.dut.psum_fifo_empty.value,
            dut.dut.psum_buf_active.value,
            dut.dut.psum_buf_drain_entry.value,
            dut.dut.requant_in_valid.value,
            dut.dut.requant_in_ready.value,
            dut.dut.requant_out_valid.value,
            dut.dut.requant_out_ready.value,
            dut.dut.accum_requant_sent_q.value,
            dut.dut.obi_o_req_o.value,
            dut.dut.i_conv_channel_linebuf_packer.state_q.value,
            dut.dut.linebuf_done.value,
            dut.dut.linebuf_emitted_vectors.value,
        )
    assert done_seen, "requant kgen line-buffer 3x3/C32 run did not complete before timeout"
    assert compute_pulses == spatial_m * k_tiles, (
        f"compute pulses={compute_pulses}, expected={spatial_m * k_tiles}"
    )

    for oh in range(output_h):
        for ow in range(output_w):
            valid_taps = 0
            for kh in range(kernel_h):
                ih = oh * stride_h + kh - pad_h
                for kw in range(kernel_w):
                    iw = ow * stride_w + kw - pad_w
                    if 0 <= ih < input_h and 0 <= iw < input_w:
                        valid_taps += 1
            expected_byte = clamp_u8_signed((valid_taps * input_c) >> 2)
            row = oh * output_w + ow
            value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + row].value)
            got = unpack_u8_beat(value)
            expected = [expected_byte] * ARRAY_DIM
            assert got == expected, (
                f"requant kgen line-buffer OFM row {row} mismatch: got={got}, expected={expected}"
            )
