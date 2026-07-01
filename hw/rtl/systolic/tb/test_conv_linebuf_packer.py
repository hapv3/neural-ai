import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


ARRAY_DIM = 32
BEAT_BYTES = 32
INPUT_BASE = 0x1000


def input_value(height, width, channel):
    return (height * 37 + width * 11 + channel * 5 + 3) & 0xFF


def build_input(input_h, input_w, input_c):
    return [
        input_value(height, width, channel)
        for height in range(input_h)
        for width in range(input_w)
        for channel in range(input_c)
    ]


def read_beat(memory, addr):
    base = addr & ~(BEAT_BYTES - 1)
    value = 0
    for byte_index in range(BEAT_BYTES):
        value |= memory.get(base + byte_index, 0) << (byte_index * 8)
    return value


def pack_lane_u8(values):
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xFF) << (lane * 8)
    return packed


def pack_lane_u16(values):
    packed = 0
    for lane, value in enumerate(values):
        packed |= (value & 0xFFFF) << (lane * 16)
    return packed


def unpack_row(value):
    return [(value >> (lane * 8)) & 0xFF for lane in range(ARRAY_DIM)]


def make_lane_desc(kernel_h, kernel_w, input_c):
    lane_valid = 0
    lane_kh = [0] * ARRAY_DIM
    lane_kw = [0] * ARRAY_DIM
    lane_ic = [0] * ARRAY_DIM
    lane = 0
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for channel in range(input_c):
                if lane < ARRAY_DIM:
                    lane_valid |= 1 << lane
                    lane_kh[lane] = kh
                    lane_kw[lane] = kw
                    lane_ic[lane] = channel
                    lane += 1
    return lane_valid, lane_kh, lane_kw, lane_ic


def golden_rows(input_data, input_h, input_w, input_c, output_h, output_w, stride_h, stride_w, pad_h, pad_w):
    rows = []
    lane_valid, lane_kh, lane_kw, lane_ic = make_lane_desc(3, 3, input_c)
    for oh in range(output_h):
        for ow in range(output_w):
            row = [0] * ARRAY_DIM
            for lane in range(ARRAY_DIM):
                if not ((lane_valid >> lane) & 1):
                    continue
                ih = oh * stride_h + lane_kh[lane] - pad_h
                iw = ow * stride_w + lane_kw[lane] - pad_w
                channel = lane_ic[lane]
                if 0 <= ih < input_h and 0 <= iw < input_w and channel < input_c:
                    index = ((ih * input_w) + iw) * input_c + channel
                    row[lane] = input_data[index]
            rows.append(row)
    return rows


async def reset(dut):
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    dut.dim_m_i.value = 0
    dut.cfg_input_base_i.value = INPUT_BASE
    dut.cfg_input_h_i.value = 0
    dut.cfg_input_w_i.value = 0
    dut.cfg_input_c_i.value = 0
    dut.cfg_output_w_i.value = 0
    dut.cfg_stride_h_i.value = 1
    dut.cfg_stride_w_i.value = 1
    dut.cfg_pad_h_i.value = 0
    dut.cfg_pad_w_i.value = 0
    dut.cfg_tile_oh_base_i.value = 0
    dut.cfg_tile_ow_base_i.value = 0
    dut.cfg_lane_valid_i.value = 0
    dut.cfg_lane_kh_i.value = 0
    dut.cfg_lane_kw_i.value = 0
    dut.cfg_lane_ic_i.value = 0
    dut.obi_gnt_i.value = 1
    dut.obi_rvalid_i.value = 0
    dut.obi_rdata_i.value = 0
    dut.row_ready_i.value = 1
    await Timer(30, unit="ns")
    dut.rst_ni.value = 1
    await RisingEdge(dut.clk_i)


async def memory_responder(dut, memory):
    pending_req = 0
    pending_addr = 0
    while True:
        await RisingEdge(dut.clk_i)
        dut.obi_rvalid_i.value = pending_req
        dut.obi_rdata_i.value = read_beat(memory, pending_addr)
        pending_req = int(dut.obi_req_o.value)
        pending_addr = int(dut.obi_addr_o.value)


def conv_output_dim(input_size, pad, stride):
    extent = input_size + 2 * pad - 3
    if extent < 0:
        return 0
    return (extent // stride) + 1


async def run_case(
    dut,
    *,
    name,
    input_h,
    input_w,
    input_c,
    output_h,
    output_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    stall_ready,
    log_result=True,
):
    input_data = build_input(input_h, input_w, input_c)
    memory = {}
    for offset, value in enumerate(input_data):
        memory[INPUT_BASE + offset] = value

    await reset(dut)
    responder = cocotb.start_soon(memory_responder(dut, memory))

    try:
        lane_valid, lane_kh, lane_kw, lane_ic = make_lane_desc(3, 3, input_c)
        expected = golden_rows(
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

        dut.dim_m_i.value = output_h * output_w
        dut.cfg_input_base_i.value = INPUT_BASE
        dut.cfg_input_h_i.value = input_h
        dut.cfg_input_w_i.value = input_w
        dut.cfg_input_c_i.value = input_c
        dut.cfg_output_w_i.value = output_w
        dut.cfg_stride_h_i.value = stride_h
        dut.cfg_stride_w_i.value = stride_w
        dut.cfg_pad_h_i.value = pad_h
        dut.cfg_pad_w_i.value = pad_w
        dut.cfg_tile_oh_base_i.value = 0
        dut.cfg_tile_ow_base_i.value = 0
        dut.cfg_lane_valid_i.value = lane_valid
        dut.cfg_lane_kh_i.value = pack_lane_u8(lane_kh)
        dut.cfg_lane_kw_i.value = pack_lane_u8(lane_kw)
        dut.cfg_lane_ic_i.value = pack_lane_u16(lane_ic)

        await RisingEdge(dut.clk_i)
        dut.start_i.value = 1
        await RisingEdge(dut.clk_i)
        dut.start_i.value = 0

        got = []
        valid_cycles = []
        cycle = 0
        stall_count = 0
        held_row = None
        scan_h = ((output_h - 1) * stride_h + 3) if output_h else 0
        scan_w = ((output_w - 1) * stride_w + 3) if output_w else 0
        timeout_cycles = max(2000, (scan_h * scan_w * 12) + (len(expected) * 8) + 200)
        while len(got) < len(expected):
            want_stall = stall_ready and (len(got) % 5 == 2) and (stall_count < 2)
            dut.row_ready_i.value = 0 if want_stall else 1
            await RisingEdge(dut.clk_i)
            cycle += 1
            if int(dut.row_valid_o.value) == 1:
                row = unpack_row(int(dut.row_data_o.value))
                if want_stall:
                    if held_row is None:
                        held_row = row
                    else:
                        assert row == held_row, f"{name}: row_data changed while row_ready=0"
                    stall_count += 1
                    continue

                held_row = None
                stall_count = 0
                got.append(row)
                valid_cycles.append(cycle)

            assert cycle < timeout_cycles, f"{name}: timeout, got {len(got)}/{len(expected)} rows"

        for index, (got_row, expected_row) in enumerate(zip(got, expected)):
            assert got_row == expected_row, (
                f"{name}: row {index} mismatch\n"
                f"got     ={got_row}\n"
                f"expected={expected_row}"
            )

        for _ in range(20):
            await RisingEdge(dut.clk_i)
            if int(dut.done_o.value) == 1:
                break
        else:
            raise AssertionError(f"{name}: done_o did not assert")

        gaps = [b - a for a, b in zip(valid_cycles, valid_cycles[1:])]
        max_gap = max(gaps) if gaps else 0
        if log_result:
            dut._log.info(
                "%s: rows=%d cache_hits=%d cache_misses=%d max_valid_gap=%d",
                name,
                len(got),
                int(dut.cache_hits_o.value),
                int(dut.cache_misses_o.value),
                max_gap,
            )
        return {
            "rows": len(got),
            "cache_hits": int(dut.cache_hits_o.value),
            "cache_misses": int(dut.cache_misses_o.value),
            "max_valid_gap": max_gap,
        }
    finally:
        responder.cancel()


@cocotb.test()
async def conv_linebuf_3x3_rgb_pad1_stride1(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="3x3 RGB pad1 stride1",
        input_h=4,
        input_w=4,
        input_c=3,
        output_h=4,
        output_w=4,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        stall_ready=False,
    )


@cocotb.test()
async def conv_linebuf_3x3_rgb_pad1_stride2(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="3x3 RGB pad1 stride2",
        input_h=5,
        input_w=5,
        input_c=3,
        output_h=3,
        output_w=3,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        stall_ready=False,
    )


@cocotb.test()
async def conv_linebuf_3x3_rgb_pad1_stride2_backpressure(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="3x3 RGB pad1 stride2 backpressure",
        input_h=5,
        input_w=5,
        input_c=3,
        output_h=3,
        output_w=3,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        stall_ready=True,
    )


@cocotb.test()
async def conv_linebuf_3x3_rgb_pad0_stride1(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="3x3 RGB pad0 stride1",
        input_h=5,
        input_w=5,
        input_c=3,
        output_h=3,
        output_w=3,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        stall_ready=False,
    )


@cocotb.test()
async def conv_linebuf_3x3_c1_pad1_stride1(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="3x3 C1 pad1 stride1",
        input_h=6,
        input_w=5,
        input_c=1,
        output_h=6,
        output_w=5,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        stall_ready=False,
    )


@cocotb.test()
async def conv_linebuf_3x3_rgb_wide_crosses_256b_beats(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="3x3 RGB wide crosses 256b beats",
        input_h=5,
        input_w=17,
        input_c=3,
        output_h=5,
        output_w=17,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        stall_ready=True,
    )


@cocotb.test()
async def conv_linebuf_3x3_shape_setting_sweep(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())

    cases = []
    for input_c in (1, 3):
        for input_h in range(1, 9):
            for input_w in range(1, 10):
                for stride_h in (1, 2):
                    for stride_w in (1, 2):
                        for pad_h in range(0, 3):
                            for pad_w in range(0, 3):
                                output_h = conv_output_dim(input_h, pad_h, stride_h)
                                output_w = conv_output_dim(input_w, pad_w, stride_w)
                                if output_h == 0 or output_w == 0:
                                    continue
                                cases.append(
                                    (
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
                                )

    total_rows = 0
    total_hits = 0
    total_misses = 0
    max_gap = 0
    for case_index, (
        input_h,
        input_w,
        input_c,
        output_h,
        output_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    ) in enumerate(cases):
        stats = await run_case(
            dut,
            name=(
                f"sweep[{case_index}] "
                f"ih={input_h} iw={input_w} ic={input_c} "
                f"oh={output_h} ow={output_w} "
                f"sh={stride_h} sw={stride_w} ph={pad_h} pw={pad_w}"
            ),
            input_h=input_h,
            input_w=input_w,
            input_c=input_c,
            output_h=output_h,
            output_w=output_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            stall_ready=(case_index % 17) == 0,
            log_result=False,
        )
        total_rows += stats["rows"]
        total_hits += stats["cache_hits"]
        total_misses += stats["cache_misses"]
        max_gap = max(max_gap, stats["max_valid_gap"])

    dut._log.info(
        "shape sweep: cases=%d rows=%d cache_hits=%d cache_misses=%d max_valid_gap=%d",
        len(cases),
        total_rows,
        total_hits,
        total_misses,
        max_gap,
    )
