import cocotb
from cocotb.triggers import Timer


ARRAY_DIM = 32


def packed_lane(value, lane, width):
    return (int(value) >> (lane * width)) & ((1 << width) - 1)


def drive_defaults(dut):
    defaults = {
        "cfg_k_tiles_i": 1,
        "cfg_row_stride_bytes_i": 128,
        "cfg_input_h_i": 8,
        "cfg_input_c_i": 32,
        "cfg_kernel_h_i": 3,
        "cfg_kernel_w_i": 3,
        "cfg_stride_h_i": 1,
        "cfg_stride_w_i": 1,
        "cfg_pad_h_i": 0,
        "cfg_c_base_i": 0,
        "cfg_lane_base_i": 0,
        "cfg_coalesce_i": 0,
        "cfg_kgen_i": 0,
        "cfg_pool_i": 0,
        "cfg_c32_fast_i": 0,
        "cfg_depthwise_i": 0,
        "cfg_block_valid_bytes_i": 0,
        "cfg_channel_addr_offset_i": 0,
        "cfg_coalesce_k_bytes_i": 0,
        "cfg_k_seed_kh_i": 0,
        "cfg_k_seed_kw_i": 0,
        "cfg_k_seed_ic_i": 0,
        "row_cache_full_i": 0,
        "cached_c_base_i": 0,
    }
    for name, value in defaults.items():
        getattr(dut, name).value = value


async def settle():
    await Timer(1, unit="ns")


@cocotb.test()
async def conv_linebuf_config_decoder_modes(dut):
    drive_defaults(dut)
    dut.cfg_input_c_i.value = 33
    dut.cfg_c_base_i.value = 1
    await settle()
    assert int(dut.block_valid_bytes_o.value) == 32
    assert int(dut.channel_addr_offset_o.value) == 1
    assert int(dut.coalesce_k_bytes_o.value) == 288
    assert int(dut.effective_c_base_o.value) == 1
    assert int(dut.c32_blocked_mode_o.value) == 0

    dut.cfg_c_base_i.value = 32
    await settle()
    assert int(dut.block_valid_bytes_o.value) == 1
    assert int(dut.channel_addr_offset_o.value) == 32

    drive_defaults(dut)
    dut.cfg_input_h_i.value = 3
    dut.cfg_input_c_i.value = 96
    dut.cfg_k_tiles_i.value = 9
    dut.cfg_coalesce_i.value = 1
    dut.cfg_kgen_i.value = 1
    dut.cfg_c32_fast_i.value = 1
    dut.cfg_block_valid_bytes_i.value = 32
    dut.cfg_channel_addr_offset_i.value = 0x400
    dut.cfg_coalesce_k_bytes_i.value = 0x480
    dut.cfg_k_seed_kh_i.value = 2
    dut.cfg_k_seed_kw_i.value = 1
    dut.cfg_k_seed_ic_i.value = 64
    await settle()
    assert int(dut.block_valid_bytes_o.value) == 32
    assert int(dut.coalesce_k_bytes_o.value) == 0x480
    assert int(dut.channel_addr_offset_o.value) == 0x400
    assert int(dut.effective_c_base_o.value) == 64
    assert int(dut.c32_blocked_mode_o.value) == 1
    assert int(dut.c32_kgen_fast_o.value) == 1
    assert int(dut.row_cache_full_mode_o.value) == 1
    for lane in range(ARRAY_DIM):
        assert packed_lane(dut.lane_kh_o.value, lane, 8) == 2
        assert packed_lane(dut.lane_kw_o.value, lane, 8) == 1
        assert packed_lane(dut.lane_ic_o.value, lane, 16) == 64 + lane

    dut.row_cache_full_i.value = 1
    dut.cached_c_base_i.value = 64
    await settle()
    assert int(dut.row_cache_reuse_o.value) == 1
    assert int(dut.row_ring_mode_o.value) == 0
    assert int(dut.fill_done_rows_o.value) == 3

    drive_defaults(dut)
    dut.cfg_input_c_i.value = 3
    dut.cfg_kernel_w_i.value = 3
    dut.cfg_kgen_i.value = 1
    dut.cfg_k_seed_kh_i.value = 1
    dut.cfg_k_seed_kw_i.value = 1
    dut.cfg_k_seed_ic_i.value = 2
    await settle()
    for lane in range(ARRAY_DIM):
        ic_index = 2 + lane
        ic_wrap, expected_ic = divmod(ic_index, 3)
        kh_wrap, expected_kw = divmod(1 + ic_wrap, 3)
        assert packed_lane(dut.lane_kh_o.value, lane, 8) == 1 + kh_wrap
        assert packed_lane(dut.lane_kw_o.value, lane, 8) == expected_kw
        assert packed_lane(dut.lane_ic_o.value, lane, 16) == expected_ic

    drive_defaults(dut)
    dut.cfg_depthwise_i.value = 1
    dut.cfg_kernel_h_i.value = 5
    dut.cfg_kernel_w_i.value = 5
    dut.cfg_stride_h_i.value = 2
    dut.cfg_stride_w_i.value = 2
    dut.cfg_pool_i.value = 1
    dut.cfg_pad_h_i.value = 2
    await settle()
    assert int(dut.row_ring_mode_o.value) == 1
    assert int(dut.fill_done_rows_o.value) == 5
    assert int(dut.pad_row_offset_o.value) == 256
    assert int(dut.pad_vector_o.value) == int.from_bytes(bytes([0x80] * 32), "little")
