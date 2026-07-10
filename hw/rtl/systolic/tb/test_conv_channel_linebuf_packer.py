import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


ARRAY_DIM = 32
BEAT_BYTES = 32
INPUT_BASE = 0x2000


def input_value(height, width, channel):
    return (height * 19 + width * 7 + channel * 3 + 5) & 0xFF


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


def unpack_row(value):
    return [(value >> (lane * 8)) & 0xFF for lane in range(ARRAY_DIM)]


def output_dim(input_size, kernel_size, pad, stride):
    extent = input_size + 2 * pad - kernel_size
    if extent < 0:
        return 0
    return (extent // stride) + 1


def golden_vectors(input_data, *, input_h, input_w, input_c, output_h, output_w,
                   kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, c_base, lane_base=0):
    vectors = []
    for oh in range(output_h):
        for ow in range(output_w):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    ih = oh * stride_h + kh - pad_h
                    iw = ow * stride_w + kw - pad_w
                    row = [0] * ARRAY_DIM
                    if 0 <= ih < input_h and 0 <= iw < input_w:
                        for lane in range(ARRAY_DIM - lane_base):
                            channel = c_base + lane
                            if channel < input_c:
                                index = ((ih * input_w) + iw) * input_c + channel
                                row[lane_base + lane] = input_data[index]
                    vectors.append(row)
    return vectors


def golden_coalesced_vectors(input_data, *, input_h, input_w, input_c, output_h, output_w,
                             kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                             c_base, lane_base=0):
    vectors = []
    for oh in range(output_h):
        for ow in range(output_w):
            row = [0] * ARRAY_DIM
            lane_base_cur = lane_base
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    ih = oh * stride_h + kh - pad_h
                    iw = ow * stride_w + kw - pad_w
                    for channel in range(c_base, input_c):
                        if lane_base_cur >= ARRAY_DIM:
                            break
                        if 0 <= ih < input_h and 0 <= iw < input_w:
                            index = ((ih * input_w) + iw) * input_c + channel
                            row[lane_base_cur] = input_data[index]
                        lane_base_cur += 1
                    if lane_base_cur >= ARRAY_DIM:
                        break
                if lane_base_cur >= ARRAY_DIM:
                    break
            vectors.append(row)
    return vectors


async def reset(dut):
    dut.rst_ni.value = 0
    dut.start_i.value = 0
    if hasattr(dut, "next_tile_i"):
        dut.next_tile_i.value = 0
    dut.dim_m_i.value = 0
    if hasattr(dut, "cfg_k_tiles_i"):
        dut.cfg_k_tiles_i.value = 1
    dut.cfg_origin_base_i.value = 0
    dut.cfg_row_stride_bytes_i.value = 0
    dut.cfg_pixel_stride_bytes_i.value = 0
    dut.cfg_ow_step_bytes_i.value = 0
    dut.cfg_oh_step_bytes_i.value = 0
    dut.cfg_input_h_i.value = 0
    dut.cfg_input_w_i.value = 0
    dut.cfg_input_c_i.value = 0
    dut.cfg_output_w_i.value = 0
    dut.cfg_kernel_h_i.value = 0
    dut.cfg_kernel_w_i.value = 0
    dut.cfg_stride_h_i.value = 1
    dut.cfg_stride_w_i.value = 1
    dut.cfg_pad_h_i.value = 0
    dut.cfg_pad_w_i.value = 0
    dut.cfg_c_base_i.value = 0
    dut.cfg_lane_base_i.value = 0
    dut.cfg_coalesce_i.value = 0
    dut.cfg_kgen_i.value = 0
    dut.cfg_k_seed_kh_i.value = 0
    dut.cfg_k_seed_kw_i.value = 0
    dut.cfg_k_seed_ic_i.value = 0
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


async def run_case(dut, *, name, input_h, input_w, input_c, kernel_h, kernel_w,
                   stride_h, stride_w, pad_h, pad_w, c_base, stall_ready,
                   lane_base=0, coalesce=False, expect_bypass=False, log_result=True,
                   input_base=INPUT_BASE, input_data_override=None, memory_override=None,
                   row_stride_override=None, pixel_stride_override=None):
    output_h = output_dim(input_h, kernel_h, pad_h, stride_h)
    output_w = output_dim(input_w, kernel_w, pad_w, stride_w)
    assert output_h > 0 and output_w > 0

    input_data = input_data_override if input_data_override is not None else build_input(input_h, input_w, input_c)
    memory = {} if memory_override is None else dict(memory_override)
    if memory_override is None:
        for offset, value in enumerate(input_data):
            memory[input_base + offset] = value

    await reset(dut)
    responder = cocotb.start_soon(memory_responder(dut, memory))
    try:
        row_stride = row_stride_override if row_stride_override is not None else input_w * input_c
        pixel_stride = pixel_stride_override if pixel_stride_override is not None else input_c
        origin_base = (input_base - pad_h * row_stride) & 0xFFFFFFFF
        golden_fn = golden_coalesced_vectors if coalesce else golden_vectors
        expected = golden_fn(
            input_data,
            input_h=input_h,
            input_w=input_w,
            input_c=input_c,
            output_h=output_h,
            output_w=output_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            c_base=c_base,
            lane_base=lane_base,
        )

        dut.dim_m_i.value = output_h * output_w
        dut.cfg_origin_base_i.value = origin_base
        dut.cfg_row_stride_bytes_i.value = row_stride
        dut.cfg_pixel_stride_bytes_i.value = pixel_stride
        dut.cfg_ow_step_bytes_i.value = stride_w * pixel_stride
        dut.cfg_oh_step_bytes_i.value = stride_h * row_stride
        dut.cfg_input_h_i.value = input_h
        dut.cfg_input_w_i.value = input_w
        dut.cfg_input_c_i.value = input_c
        dut.cfg_output_w_i.value = output_w
        dut.cfg_kernel_h_i.value = kernel_h
        dut.cfg_kernel_w_i.value = kernel_w
        dut.cfg_stride_h_i.value = stride_h
        dut.cfg_stride_w_i.value = stride_w
        dut.cfg_pad_h_i.value = pad_h
        dut.cfg_pad_w_i.value = pad_w
        dut.cfg_c_base_i.value = c_base
        dut.cfg_lane_base_i.value = lane_base
        dut.cfg_coalesce_i.value = 1 if coalesce else 0

        await RisingEdge(dut.clk_i)
        dut.start_i.value = 1
        await RisingEdge(dut.clk_i)
        dut.start_i.value = 0

        got = []
        cycle = 0
        stall_count = 0
        held_row = None
        timeout_cycles = max(3000, len(expected) * 20 + 500)
        while len(got) < len(expected):
            want_stall = stall_ready and (len(got) % 7 == 3) and (stall_count < 2)
            dut.row_ready_i.value = 0 if want_stall else 1
            await RisingEdge(dut.clk_i)
            cycle += 1
            if int(dut.row_valid_o.value) == 1:
                row = unpack_row(int(dut.row_data_o.value))
                if want_stall:
                    if held_row is None:
                        held_row = row
                    else:
                        assert row == held_row, f"{name}: row changed under backpressure"
                    stall_count += 1
                    continue
                held_row = None
                stall_count = 0
                got.append(row)
            assert cycle < timeout_cycles, f"{name}: timeout got {len(got)}/{len(expected)}"

        for index, (got_row, expected_row) in enumerate(zip(got, expected)):
            assert got_row == expected_row, (
                f"{name}: vector {index} mismatch\n"
                f"got     ={got_row}\n"
                f"expected={expected_row}"
            )

        for _ in range(20):
            await RisingEdge(dut.clk_i)
            if int(dut.done_o.value) == 1:
                break
        else:
            raise AssertionError(f"{name}: done_o did not assert")

        bypass_vectors = int(dut.bypass_vectors_o.value)
        if expect_bypass:
            assert bypass_vectors == len(got), (
                f"{name}: expected all vectors through 1x1 bypass, "
                f"bypass={bypass_vectors} total={len(got)}"
            )
        else:
            assert bypass_vectors == 0, f"{name}: unexpected bypass vectors {bypass_vectors}"

        if log_result:
            dut._log.info(
                "%s: vectors=%d fetch_beats=%d bypass_vectors=%d",
                name,
                len(got),
                int(dut.fetch_beats_o.value),
                bypass_vectors,
            )
    finally:
        responder.cancel()


async def run_rejected_case(dut, *, name, input_h, input_w, input_c,
                            kernel_h, kernel_w, stride_h, stride_w,
                            pad_h, pad_w, c_base):
    await reset(dut)

    row_stride = input_w * input_c
    pixel_stride = input_c
    output_h = max(1, output_dim(input_h, kernel_h, pad_h, stride_h))
    output_w = max(1, output_dim(input_w, kernel_w, pad_w, stride_w))

    dut.dim_m_i.value = output_h * output_w
    dut.cfg_origin_base_i.value = (INPUT_BASE - pad_h * row_stride) & 0xFFFFFFFF
    dut.cfg_row_stride_bytes_i.value = row_stride
    dut.cfg_pixel_stride_bytes_i.value = pixel_stride
    dut.cfg_ow_step_bytes_i.value = stride_w * pixel_stride
    dut.cfg_oh_step_bytes_i.value = stride_h * row_stride
    dut.cfg_input_h_i.value = input_h
    dut.cfg_input_w_i.value = input_w
    dut.cfg_input_c_i.value = input_c
    dut.cfg_output_w_i.value = output_w
    dut.cfg_kernel_h_i.value = kernel_h
    dut.cfg_kernel_w_i.value = kernel_w
    dut.cfg_stride_h_i.value = stride_h
    dut.cfg_stride_w_i.value = stride_w
    dut.cfg_pad_h_i.value = pad_h
    dut.cfg_pad_w_i.value = pad_w
    dut.cfg_c_base_i.value = c_base
    dut.cfg_lane_base_i.value = 0

    await RisingEdge(dut.clk_i)
    dut.start_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.start_i.value = 0

    for _ in range(8):
        await RisingEdge(dut.clk_i)
        assert int(dut.row_valid_o.value) == 0, f"{name}: emitted row for rejected config"
        if int(dut.done_o.value) == 1:
            assert int(dut.busy_o.value) == 0, f"{name}: busy after reject"
            dut._log.info("%s: rejected as expected", name)
            return

    raise AssertionError(f"{name}: did not reject unsupported config")


@cocotb.test()
async def conv_channel_linebuf_c48_3x3_cbase0(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C48 3x3 cbase0",
        input_h=4,
        input_w=5,
        input_c=48,
        kernel_h=3,
        kernel_w=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        c_base=0,
        stall_ready=False,
    )


@cocotb.test()
async def conv_channel_linebuf_c48_3x3_tail_stride2(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C48 3x3 tail stride2",
        input_h=5,
        input_w=5,
        input_c=48,
        kernel_h=3,
        kernel_w=3,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        c_base=32,
        stall_ready=True,
    )


@cocotb.test()
async def conv_channel_linebuf_c32_blocked_second_block(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())

    input_h = 4
    input_w = 6
    c_block = ARRAY_DIM
    block_span = input_h * input_w * c_block
    block0_base = INPUT_BASE
    block1_base = INPUT_BASE + block_span

    block1_data = [
        input_value(height, width, c_block + lane)
        for height in range(input_h)
        for width in range(input_w)
        for lane in range(c_block)
    ]
    memory = {}
    for offset in range(block_span):
        memory[block0_base + offset] = (0xA0 + offset) & 0xFF
    for offset, value in enumerate(block1_data):
        memory[block1_base + offset] = value

    await run_case(
        dut,
        name="C32-blocked second block 3x3",
        input_h=input_h,
        input_w=input_w,
        input_c=c_block,
        kernel_h=3,
        kernel_w=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        c_base=0,
        stall_ready=True,
        input_base=block1_base,
        input_data_override=block1_data,
        memory_override=memory,
        row_stride_override=input_w * c_block,
        pixel_stride_override=c_block,
    )


@cocotb.test()
async def conv_channel_linebuf_c80_crosses_256b_beats(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C80 3x3 cbase32 crosses 256b beats",
        input_h=4,
        input_w=4,
        input_c=80,
        kernel_h=3,
        kernel_w=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        c_base=32,
        stall_ready=True,
    )


@cocotb.test()
async def conv_channel_linebuf_c64_5x5(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C64 5x5 cbase32",
        input_h=6,
        input_w=6,
        input_c=64,
        kernel_h=5,
        kernel_w=5,
        stride_h=1,
        stride_w=1,
        pad_h=2,
        pad_w=2,
        c_base=32,
        stall_ready=False,
    )


@cocotb.test()
async def conv_channel_linebuf_c33_1x1_tail(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C33 1x1 tail",
        input_h=5,
        input_w=7,
        input_c=33,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        c_base=32,
        stall_ready=True,
        expect_bypass=True,
    )


@cocotb.test()
async def conv_channel_linebuf_c80_1x1_stride2_crossing(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C80 1x1 stride2 cbase32 bypass",
        input_h=7,
        input_w=9,
        input_c=80,
        kernel_h=1,
        kernel_w=1,
        stride_h=2,
        stride_w=2,
        pad_h=0,
        pad_w=0,
        c_base=32,
        stall_ready=True,
        expect_bypass=True,
    )


@cocotb.test()
async def conv_channel_linebuf_c33_1x1_width640_boundary(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C33 1x1 width640 boundary",
        input_h=1,
        input_w=640,
        input_c=33,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        c_base=32,
        stall_ready=False,
        expect_bypass=True,
    )


@cocotb.test()
async def conv_channel_linebuf_c40_1x1_padded_uses_ring(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C40 1x1 pad1 ring fallback",
        input_h=4,
        input_w=5,
        input_c=40,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        c_base=32,
        stall_ready=False,
        expect_bypass=False,
    )


@cocotb.test()
async def conv_channel_linebuf_lane_base_shift(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="C8 1x1 lane_base5",
        input_h=2,
        input_w=4,
        input_c=8,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        c_base=0,
        lane_base=5,
        stall_ready=False,
        expect_bypass=True,
    )


@cocotb.test()
async def conv_channel_linebuf_coalesced_3x3_c3(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await run_case(
        dut,
        name="coalesced 3x3 C3 K27",
        input_h=4,
        input_w=4,
        input_c=3,
        kernel_h=3,
        kernel_w=3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        c_base=0,
        lane_base=0,
        coalesce=True,
        stall_ready=True,
        expect_bypass=False,
    )


@cocotb.test()
async def conv_channel_linebuf_kernel_1_to_5_sweep(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())

    cases = 0
    total_vectors = 0
    total_fetch_beats = 0
    total_bypass_vectors = 0
    for kernel_h in range(1, 6):
        for kernel_w in range(1, 6):
            input_h = 7 + (kernel_h & 1)
            input_w = 8 + (kernel_w & 1)
            input_c = 48 if ((kernel_h + kernel_w) & 1) == 0 else 80
            stride_h = 2 if kernel_h in (1, 4, 7) else 1
            stride_w = 2 if kernel_w in (1, 5, 9) else 1
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            c_base = 32 if input_c > 48 or (kernel_h * kernel_w) & 1 else 0
            expect_bypass = (
                kernel_h == 1 and kernel_w == 1 and pad_h == 0 and pad_w == 0
            )

            await run_case(
                dut,
                name=f"sweep {kernel_h}x{kernel_w} C{input_c} cb{c_base}",
                input_h=input_h,
                input_w=input_w,
                input_c=input_c,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                c_base=c_base,
                stall_ready=((kernel_h + kernel_w) % 5 == 0),
                expect_bypass=expect_bypass,
                log_result=False,
            )
            cases += 1
            total_vectors += int(dut.emitted_vectors_o.value)
            total_fetch_beats += int(dut.fetch_beats_o.value)
            total_bypass_vectors += int(dut.bypass_vectors_o.value)

    dut._log.info(
        "kernel sweep: cases=%d vectors=%d fetch_beats=%d bypass_vectors=%d",
        cases,
        total_vectors,
        total_fetch_beats,
        total_bypass_vectors,
    )


@cocotb.test()
async def conv_channel_linebuf_rejects_over_limit_configs(dut):
    clock = Clock(dut.clk_i, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await run_rejected_case(
        dut,
        name="reject 7x7 with K_MAX5",
        input_h=8,
        input_w=8,
        input_c=48,
        kernel_h=7,
        kernel_w=7,
        stride_h=1,
        stride_w=1,
        pad_h=3,
        pad_w=3,
        c_base=0,
    )
    await run_rejected_case(
        dut,
        name="reject 9x9 with K_MAX5",
        input_h=10,
        input_w=10,
        input_c=80,
        kernel_h=9,
        kernel_w=9,
        stride_h=1,
        stride_w=1,
        pad_h=4,
        pad_w=4,
        c_base=32,
    )
    await run_rejected_case(
        dut,
        name="reject input_w641 with MAX_INPUT_W640",
        input_h=1,
        input_w=641,
        input_c=33,
        kernel_h=1,
        kernel_w=1,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        c_base=32,
    )
