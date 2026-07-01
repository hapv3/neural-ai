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
REG_RQ_CTRL = 0x0120
REG_LB_CTRL = 0x0400
REG_LB_INPUT_BASE = 0x0404
REG_LB_INPUT_H = 0x0408
REG_LB_INPUT_W = 0x040C
REG_LB_INPUT_C = 0x0410
REG_LB_OUTPUT_W = 0x0414
REG_LB_STRIDE = 0x0418
REG_LB_PAD = 0x041C
REG_LB_TILE_BASE = 0x0420
REG_LB_LANE_VALID = 0x0424
REG_LB_LANE_BASE = 0x0500

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


def write_bytes(dut, base_addr, values):
    assert (base_addr % BEAT_BYTES) == 0
    for beat_index in range((len(values) + BEAT_BYTES - 1) // BEAT_BYTES):
        chunk = values[beat_index * BEAT_BYTES:(beat_index + 1) * BEAT_BYTES]
        padded = chunk + [0] * (BEAT_BYTES - len(chunk))
        dut.tcdm_mem[mem_index(base_addr) + beat_index].value = pack_u8(padded)


def make_lane_desc(input_c):
    lane_valid = 0
    lanes = []
    for kh in range(3):
        for kw in range(3):
            for channel in range(input_c):
                if len(lanes) < ARRAY_DIM:
                    lane_valid |= 1 << len(lanes)
                    lanes.append((kh, kw, channel))
    return lane_valid, lanes


def golden_conv_rows(input_data, input_h, input_w, input_c, output_h, output_w, stride_h, stride_w, pad_h, pad_w):
    _, lanes = make_lane_desc(input_c)
    rows = []
    for oh in range(output_h):
        for ow in range(output_w):
            row = [0] * ARRAY_DIM
            for lane, (kh, kw, channel) in enumerate(lanes):
                ih = oh * stride_h + kh - pad_h
                iw = ow * stride_w + kw - pad_w
                if 0 <= ih < input_h and 0 <= iw < input_w:
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


def mem_index(addr):
    return addr >> 5


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
async def systolic_controller_conv3x3_linebuf_rgb_stride2(dut):
    """
    Scenario: configure the controller line-buffer path for Conv3x3 RGB
    pad1/stride2, feed rows directly into the integrated systolic array, and
    write full INT32 OFM rows.
    Target: verifies MMIO line-buffer descriptors, TCDM reads through the
    shared input port, line-buffer row packing, array compute, and O-TCDM
    writeback against a Python golden model.
    """
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    input_h = 5
    input_w = 5
    input_c = 3
    output_h = 3
    output_w = 3
    stride_h = 2
    stride_w = 2
    pad_h = 1
    pad_w = 1
    dim_m = output_h * output_w

    input_data = [((index * 3 + 1) & 0x7F) for index in range(input_h * input_w * input_c)]
    write_bytes(dut, IFM_ADDR, input_data)

    one_beat = pack_u8([1] * BEAT_BYTES)
    for beat in range(ARRAY_DIM):
        dut.tcdm_mem[mem_index(WEIGHT_ADDR) + beat].value = one_beat
    for beat in range(dim_m * 4):
        dut.tcdm_mem[mem_index(OFM_ADDR) + beat].value = 0

    lane_valid, lanes = make_lane_desc(input_c)
    await mmio_write(dut, REG_RQ_CTRL, 0)
    await mmio_write(dut, REG_SYS_ACCUM_CTRL, 0)
    await mmio_write(dut, REG_SYS_W_PTR, WEIGHT_ADDR)
    await mmio_write(dut, REG_SYS_I_PTR, IFM_ADDR)
    await mmio_write(dut, REG_SYS_O_PTR, OFM_ADDR)
    await mmio_write(dut, REG_SYS_PSUM_PTR, 0)
    await mmio_write(dut, REG_SYS_DIM_M, dim_m)
    await mmio_write(dut, REG_SYS_DONE, 0)
    await mmio_write(dut, REG_LB_INPUT_BASE, IFM_ADDR)
    await mmio_write(dut, REG_LB_INPUT_H, input_h)
    await mmio_write(dut, REG_LB_INPUT_W, input_w)
    await mmio_write(dut, REG_LB_INPUT_C, input_c)
    await mmio_write(dut, REG_LB_OUTPUT_W, output_w)
    await mmio_write(dut, REG_LB_STRIDE, (stride_w << 16) | stride_h)
    await mmio_write(dut, REG_LB_PAD, (pad_w << 16) | pad_h)
    await mmio_write(dut, REG_LB_TILE_BASE, 0)
    await mmio_write(dut, REG_LB_LANE_VALID, lane_valid)
    for lane, (kh, kw, channel) in enumerate(lanes):
        await mmio_write(dut, REG_LB_LANE_BASE + lane * 4, (kh << 24) | (kw << 16) | channel)
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

    assert done_seen, "line-buffer conv did not complete before timeout"
    assert compute_pulses == dim_m, f"compute pulses={compute_pulses}, expected={dim_m}"

    golden_rows = golden_conv_rows(
        input_data,
        input_h,
        input_w,
        input_c,
        output_h,
        output_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    )
    expected_sums = [sum(row) for row in golden_rows]
    for row, expected_sum in enumerate(expected_sums):
        got = []
        for beat in range(4):
            value = int(dut.tcdm_mem[mem_index(OFM_ADDR) + row * 4 + beat].value)
            got.extend(unpack_s32_beat(value))
        expected = [expected_sum] * ARRAY_DIM
        assert got == expected, f"line-buffer conv OFM row {row} mismatch: got={got}, expected={expected}"
