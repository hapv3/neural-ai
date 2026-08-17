import base64
import logging
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.utils import get_sim_time
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    NPU_CMD_DONE_COUNT,
    NPU_CMD_FAIL_CODE,
    NPU_CMD_FAIL_PTR,
    NPU_CMD_STATUS,
    NPU_CMD_STATUS_FAIL,
    NPU_CMD_STATUS_PASS,
    _axi_read32,
    format_pmu_report,
    load_firmware_elf_axi,
    program_command_queue,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


INPUT_BASE = 0x80000000
OUTPUT_BASE = 0x80001000
INPUT2_BASE = 0x80002000
INVOCATION_BASE = 0x80040000
MODEL_BASE = 0x80041000
BINDING_TABLE_BASE = 0x80044000
WEIGHT_SCRATCH_OFFSET = 0x00010000
IFM_SCRATCH_OFFSET = 0x00020000
OFM_SCRATCH_OFFSET = 0x00030000

NAI_MODEL_MAGIC = 0x4D49414E
NAI_INVOCATION_MAGIC = 0x5649414E
NPU_CMD_FAIL_BAD_MODEL = 0xBADCD00B
NPU_CMD_FAIL_BAD_BINDING = 0xBADCD00C
NPU_CMD_FAIL_V2_OPERATION = 0xBADCD012


def _ref(region, index=0, offset=0):
    return struct.pack("<HHI", region, index, offset)


def _command_header(command_type, size, layer=0, tile=0):
    return struct.pack("<HHIII", command_type, size, 0, layer, tile)


def _dma_1d(source, destination, length, direction, tile):
    command = _command_header(2, 64, tile=tile)
    command += source + destination
    command += struct.pack("<II6I", length, direction, 0, 0, 0, 0, 0, 0)
    assert len(command) == 64
    return command


def _dma_2d(source, destination, length, source_stride, destination_stride,
            repetitions, direction, tile):
    command = _command_header(3, 64, tile=tile)
    command += source + destination
    command += struct.pack(
        "<5I3I",
        length,
        source_stride,
        destination_stride,
        repetitions,
        direction,
        0,
        0,
        0,
    )
    assert len(command) == 64
    return command


def _dma_3d(source, destination, length, source_stride_2,
            destination_stride_2, repetitions_2, source_stride_3,
            destination_stride_3, repetitions_3, direction, tile):
    command = _command_header(4, 64, tile=tile)
    command += source + destination
    command += struct.pack(
        "<8I",
        length,
        source_stride_2,
        destination_stride_2,
        repetitions_2,
        source_stride_3,
        destination_stride_3,
        repetitions_3,
        direction,
    )
    assert len(command) == 64
    return command


def _binding(direction, index, data_type=1, dimensions=(1, 1, 1, 32)):
    element_bytes = 1 if data_type == 1 else 4
    byte_size = element_bytes
    for dimension in dimensions:
        byte_size *= dimension
    descriptor = struct.pack(
        "<6HI4IIIiI4I",
        direction,
        index,
        data_type,
        1,
        4,
        0,
        index,
        *dimensions,
        byte_size,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert len(descriptor) == 64
    return descriptor


def _gemm(weights, ifm, ofm, dim_m, tile):
    command = _command_header(6, 96, tile=tile)
    command += weights + ifm + _ref(6) + ofm
    command += struct.pack("<4I8I", dim_m, 128, 128, 0, *([0] * 8))
    assert len(command) == 96
    return command


def _pointwise_c32(weights, ifm, partial, ofm, rows, input_groups, output_groups,
                   qparam_block, input_group_stride, output_group_stride, tile):
    command = _command_header(10, 96, tile=tile)
    command += weights + ifm + partial + ofm
    command += struct.pack("<6I6I", rows, input_groups, output_groups, qparam_block,
                           input_group_stride, output_group_stride, *([0] * 6))
    assert len(command) == 96
    return command


def _depthwise_c32(weights, ifm, ofm, input_h, input_w, output_h, output_w,
                   channels, stride_h, stride_w, pad_h, pad_w, qparam_block, tile):
    command = _command_header(11, 96, tile=tile)
    command += weights + ifm + ofm
    command += struct.pack(
        "<10I4I",
        input_h,
        input_w,
        output_h,
        output_w,
        channels,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        qparam_block,
        *([0] * 4),
    )
    assert len(command) == 96
    return command


def _afu_binary(lhs, rhs, ofm, length, mode, tile):
    command = _command_header(13, 64, tile=tile)
    command += lhs + rhs + ofm
    command += struct.pack("<2I4I", length, mode, 0, 0, 0, 0)
    assert len(command) == 64
    return command


def _linebuf_job(k_tiles, tile):
    linebuf = struct.pack(
        "<I8H4I14H4I",
        0,
        3, 3, 32, 2, 1, 1, 1, 1,
        96, 32, 32, 96,
        3, 3, 0, 0, 1, 1, 0, 1, 0, 1, 32, 0, 0, 0,
        k_tiles, 4, 0, 27,
    )
    gemm = struct.pack("<9I", 0, 0, 0, 0, 4, 0, 64, 2, 512)
    command = _command_header(9, 160, tile=tile) + linebuf + gemm
    command += struct.pack("<2I", 4, k_tiles) + bytes(20)
    assert len(command) == 160
    return command


def _copy_layout(source, destination, mode, dimensions, tile):
    command = _command_header(18, 96, tile=tile)
    command += source + destination
    source_layout, destination_layout = {
        1: (1, 2),
        2: (2, 1),
        3: (1, 3),
        4: (3, 1),
    }[mode]
    channels = dimensions[3]
    compact_stride = channels
    native_stride = ((channels + 31) // 32) * 32
    command += struct.pack(
        "<4H4I3I7I",
        mode,
        source_layout,
        destination_layout,
        1,
        *dimensions,
        channels,
        compact_stride if mode == 1 else native_stride,
        native_stride if mode == 1 else compact_stride,
        *([0] * 7),
    )
    assert len(command) == 96
    return command


def _afu_dfl16(source, destination, scratch, exp_lut, recip_lut, tile):
    command = _command_header(28, 96, tile=tile)
    command += source + destination + scratch + exp_lut + recip_lut
    command += struct.pack("<2I", 2100, 0)
    command += struct.pack("<iIiii3I", 51003, 8, -128, -128, 127, 0, 0, 0)
    assert len(command) == 96
    return command


def _package(commands, constants, bindings, command_count, required_tcdm_bytes,
             input_count, output_count, qparams=b""):
    payloads = [commands, constants, b"", bindings, qparams]
    section_types = [1, 2, 3, 4, 5]
    element_counts = [command_count + 1, len(constants), 0,
                      input_count + output_count, len(qparams) // 32]
    section_table_offset = 64
    offset = section_table_offset + 5 * 32
    sections = []
    for section_type, element_count, payload in zip(section_types, element_counts, payloads):
        assert offset % 32 == 0 and len(payload) % 32 == 0
        sections.append(
            struct.pack(
                "<8I",
                section_type,
                0,
                offset,
                len(payload),
                32,
                element_count,
                0,
                0,
            )
        )
        offset += len(payload)

    header = struct.pack(
        "<IHH11I3I",
        NAI_MODEL_MAGIC,
        1,
        1,
        1,
        0,
        offset,
        5,
        section_table_offset,
        224,
        command_count,
        required_tcdm_bytes,
        32,
        input_count,
        output_count,
        0,
        0,
        0,
    )
    model = header + b"".join(sections) + b"".join(payloads)
    assert len(header) == 64 and len(model) == offset
    return model


def build_model():
    commands = b"".join(
        [
            _dma_1d(_ref(3), _ref(6), 32, 0, 0),
            _dma_1d(_ref(6), _ref(4), 32, 1, 1),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0) + _binding(2, 0)
    return _package(commands, b"", bindings, 2, 32, 1, 1)


def build_unaligned_dma_model(dimension):
    binding_bytes = 192
    if dimension == 1:
        commands = [
            _dma_1d(_ref(3, offset=3), _ref(6, offset=5), 37, 0, 0),
            _dma_1d(_ref(6, offset=5), _ref(4, offset=7), 37, 1, 1),
        ]
        copies = [(3, 7, 37)]
    elif dimension == 2:
        commands = [
            _dma_2d(_ref(3, offset=3), _ref(6),
                    3, 3, 32, 11, 0, 0),
            _dma_2d(_ref(6), _ref(4, offset=5),
                    3, 32, 3, 11, 1, 1),
        ]
        copies = [(3 + row * 3, 5 + row * 3, 3) for row in range(11)]
    elif dimension == 3:
        commands = [
            _dma_3d(_ref(3, offset=3), _ref(6),
                    31, 31, 32, 2, 67, 96, 2, 0, 0),
            _dma_3d(_ref(6), _ref(4, offset=7),
                    31, 32, 31, 2, 96, 67, 2, 1, 1),
        ]
        copies = [
            (3 + plane * 67 + row * 31,
             7 + plane * 67 + row * 31,
             31)
            for plane in range(2)
            for row in range(2)
        ]
    else:
        raise ValueError(f"unsupported DMA dimension {dimension}")
    commands.append(_command_header(0, 32, tile=2).ljust(32, b"\x00"))
    bindings = (
        _binding(1, 0, dimensions=(1, 1, 1, binding_bytes))
        + _binding(2, 0, dimensions=(1, 1, 1, binding_bytes))
    )
    return (
        _package(b"".join(commands), b"", bindings, 2, 384, 1, 1),
        copies,
        binding_bytes,
    )


def build_layout_model():
    dimensions = (1, 2, 2, 33)
    commands = b"".join(
        [
            _copy_layout(_ref(3), _ref(6), 1, dimensions, 0),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0, dimensions=dimensions) + _binding(2, 0, dimensions=dimensions)
    return _package(commands, b"", bindings, 2, 256, 1, 1)


def build_c32_layout_model():
    dimensions = (1, 2, 2, 33)
    commands = b"".join(
        [
            _copy_layout(_ref(3), _ref(6), 3, dimensions, 0),
            _copy_layout(_ref(6), _ref(4), 4, dimensions, 1),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0, dimensions=dimensions) + _binding(2, 0, dimensions=dimensions)
    return _package(commands, b"", bindings, 2, 256, 1, 1)


def build_afu_add_model():
    dimensions = (1, 2, 2, 32)
    tensor_bytes = 2 * 2 * 32
    commands = b"".join(
        [
            _copy_layout(_ref(3, 0), _ref(6), 3, dimensions, 0),
            _copy_layout(_ref(3, 1), _ref(6, offset=0x200), 3, dimensions, 1),
            _afu_binary(
                _ref(6), _ref(6, offset=0x200), _ref(6, offset=0x400),
                tensor_bytes, 1, 2,
            ),
            _copy_layout(_ref(6, offset=0x400), _ref(4), 4, dimensions, 3),
            _command_header(0, 32, tile=4).ljust(32, b"\x00"),
        ]
    )
    bindings = (
        _binding(1, 0, dimensions=dimensions)
        + _binding(1, 1, dimensions=dimensions)
        + _binding(2, 0, dimensions=dimensions)
    )
    return _package(commands, b"", bindings, 4, 0x480, 2, 1)


def _dfl_exp_lut_value(index):
    if index == 0:
        return 32768
    neg_delta = 256 - index
    shift = neg_delta >> 4
    fraction = neg_delta & 15
    base = 1 if shift >= 15 else 32768 >> shift
    next_value = base >> 1 if base > 1 else 1
    return (((base * (16 - fraction)) + (next_value * fraction) + 8) >> 4) or 1


def _dfl_recip_lut_value(index):
    midpoint_q9 = 513 + index * 2
    return ((1 << 37) + (midpoint_q9 >> 1)) // midpoint_q9


def _dfl16_q8(values):
    maximum = max(values)
    exponentials = [
        _dfl_exp_lut_value((value - maximum) & 0xFF) for value in values
    ]
    total = sum(exponentials)
    weighted = sum(index * value for index, value in enumerate(exponentials))
    shift = total.bit_length() - 1
    normalized = (total << 8) >> shift
    reciprocal_index = min(max(normalized, 256), 511) & 0xFF
    reciprocal = _dfl_recip_lut_value(reciprocal_index)
    total_shift = 28 + shift
    return min(
        (((weighted << 8) * reciprocal) + (1 << (total_shift - 1))) >> total_shift,
        0xFFFF,
    )


def build_afu_dfl16_model():
    locations = 2100
    source_bytes = 144 * locations
    output_bytes = 4 * locations
    source_offset = 0
    destination_offset = source_bytes
    scratch_offset = (destination_offset + output_bytes + 31) & ~31
    required_tcdm = scratch_offset + 34 * 32
    constants = b"".join(
        struct.pack("<I", _dfl_exp_lut_value(index)) for index in range(256)
    ) + b"".join(
        struct.pack("<I", _dfl_recip_lut_value(index)) for index in range(256)
    )
    commands = b"".join(
        [
            _dma_1d(_ref(3), _ref(6, offset=source_offset), source_bytes, 0, 0),
            _afu_dfl16(
                _ref(6, offset=source_offset),
                _ref(6, offset=destination_offset),
                _ref(6, offset=scratch_offset),
                _ref(1),
                _ref(1, offset=1024),
                1,
            ),
            _dma_1d(_ref(6, offset=destination_offset), _ref(4), output_bytes, 1, 2),
            _command_header(0, 32, tile=3).ljust(32, b"\x00"),
        ]
    )
    bindings = (
        _binding(1, 0, dimensions=(1, 1, 144, locations))
        + _binding(2, 0, dimensions=(1, 1, 4, locations))
    )
    return _package(commands, constants, bindings, 3, required_tcdm, 1, 1)


def build_row32_layout_model(channels, height, width):
    dimensions = (1, height, width, channels)
    commands = b"".join(
        [
            _copy_layout(_ref(3), _ref(6), 1, dimensions, 0),
            _copy_layout(_ref(6), _ref(4), 2, dimensions, 1),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    compact_bytes = height * width * channels
    bindings = _binding(1, 0, dimensions=dimensions) + _binding(2, 0, dimensions=dimensions)
    return _package(
        commands,
        b"",
        bindings,
        2,
        ((channels + 31) // 32) * height * width * 32,
        1,
        1,
    ), compact_bytes


def build_pointwise_c32_model(height=2, width=2, channels=64, output_channels=None,
                              clamp_min=-128, clamp_max=127, qparam_specs=None,
                              qparam_block_delta=0):
    rows = height * width
    if output_channels is None:
        output_channels = channels
    input_groups = (channels + 31) // 32
    output_groups = (output_channels + 31) // 32
    weights = bytes([1]) * (output_groups * input_groups * 32 * 32)
    if qparam_specs is None:
        qparam_specs = [(0, 1 << 30, 30, 0, clamp_min, clamp_max)] * (output_groups * 32)
    assert len(qparam_specs) == output_groups * 32
    qparams = b"".join(
        struct.pack("<iiIiiiII", *spec, 0, 0) for spec in qparam_specs
    )
    input_dimensions = (1, height, width, channels)
    output_dimensions = (1, height, width, output_channels)
    commands = [_copy_layout(_ref(3), _ref(6), 3, input_dimensions, 0)]
    for output_group in range(output_groups):
        commands.append(
            _command_header(5, 32, tile=output_group * 2 + 1) +
            struct.pack("<4I", output_group * 32, 32, output_group, 0)
        )
        commands.append(
            _pointwise_c32(
                _ref(1, offset=output_group * input_groups * 32 * 32),
                _ref(6),
                _ref(6, offset=0x1000) if input_groups > 1 else _ref(0),
                _ref(6, offset=0x2000 + output_group * rows * 32),
                rows, input_groups, 1, output_group + qparam_block_delta,
                rows * 32, rows * 32, output_group * 2 + 2
            )
        )
    commands.extend([
        _copy_layout(_ref(6, offset=0x2000), _ref(4), 4, output_dimensions, 3),
        _command_header(0, 32, tile=len(commands)).ljust(32, b"\x00"),
    ])
    commands = b"".join(commands)
    bindings = _binding(1, 0, dimensions=input_dimensions) + _binding(2, 0, dimensions=output_dimensions)
    return _package(commands, weights, bindings, 2 + output_groups * 2, 0x2200, 1, 1, qparams)


def build_depthwise_c32_model(height=3, width=3, channels=32, stride=1):
    groups = (channels + 31) // 32
    weights = bytearray(groups * 3 * 3 * 32)
    for group in range(groups):
        for lane in range(32):
            channel = group * 32 + lane
            if channel < channels:
                weights[((group * 9 + 4) * 32) + lane] = 1
    qparams = b"".join(
        struct.pack("<iiIiiiII", 0, 1 << 30, 30, 0, -128, 127, 0, 0)
        for _ in range(groups * 32)
    )
    output_height = ((height - 1) // stride) + 1
    output_width = ((width - 1) // stride) + 1
    input_dimensions = (1, height, width, channels)
    output_dimensions = (1, output_height, output_width, channels)
    input_pixels = height * width
    output_pixels = output_height * output_width
    commands = [_copy_layout(_ref(3), _ref(6), 3, input_dimensions, 0)]
    for group in range(groups):
        valid_channels = min(32, channels - group * 32)
        commands.append(
            _command_header(5, 32, tile=group * 2 + 1) +
            struct.pack("<4I", group * 32, 32, group, 0)
        )
        commands.append(
            _depthwise_c32(
                _ref(1, offset=group * 9 * 32),
                _ref(6, offset=group * input_pixels * 32),
                _ref(6, offset=0x1000 + group * output_pixels * 32),
                height, width, output_height, output_width, valid_channels,
                stride, stride, 1, 1, group, group * 2 + 2,
            )
        )
    commands.extend([
        _copy_layout(_ref(6, offset=0x1000), _ref(4), 4, output_dimensions,
                     len(commands)),
        _command_header(0, 32, tile=len(commands)).ljust(32, b"\x00"),
    ])
    commands = b"".join(commands)
    bindings = _binding(1, 0, dimensions=input_dimensions) + _binding(2, 0, dimensions=output_dimensions)
    command_count = 2 + groups * 2
    return _package(commands, bytes(weights), bindings, command_count, 0x2200, 1, 1, qparams)


def build_pointwise_depthwise_chain_model(height=2, width=2):
    channels = 32
    rows = height * width
    input_dimensions = (1, height, width, channels)
    output_dimensions = input_dimensions
    pointwise_weights = bytes([1]) * (32 * 32)
    depthwise_weights = bytearray(9 * 32)
    depthwise_weights[4 * 32:5 * 32] = bytes([1]) * 32
    constants = pointwise_weights + bytes(depthwise_weights)
    qparam = struct.pack("<iiIiiiII", 0, 1 << 30, 30, 0, -128, 127, 0, 0)
    qparams = qparam * 64
    commands = [
        _copy_layout(_ref(3), _ref(6), 3, input_dimensions, 0),
        _command_header(5, 32, tile=1) + struct.pack("<4I", 0, 32, 0, 0),
        _pointwise_c32(
            _ref(1), _ref(6), _ref(0), _ref(6, offset=0x2000),
            rows, 1, 1, 0, rows * 32, rows * 32, 2,
        ),
        _command_header(5, 32, tile=3) + struct.pack("<4I", 32, 32, 1, 0),
        _depthwise_c32(
            _ref(1, offset=len(pointwise_weights)), _ref(6, offset=0x2000),
            _ref(6, offset=0x3000), height, width, height, width,
            channels, 1, 1, 1, 1, 1, 4,
        ),
        _copy_layout(_ref(6, offset=0x3000), _ref(4), 4, output_dimensions, 5),
        _command_header(0, 32, tile=6).ljust(32, b"\x00"),
    ]
    bindings = _binding(1, 0, dimensions=input_dimensions) + _binding(
        2, 0, dimensions=output_dimensions
    )
    return _package(
        b"".join(commands), constants, bindings, 6, 0x3100, 1, 1, qparams
    )


def build_gemm_model(dimensions):
    weights = bytes(1 if row == column else 0 for row in range(32) for column in range(32))
    commands = [
        _dma_1d(_ref(1), _ref(6, offset=WEIGHT_SCRATCH_OFFSET), len(weights), 0, 0)
    ]
    bindings = []
    for index, dim_m in enumerate(dimensions):
        commands.extend(
            [
                _dma_1d(_ref(3, index), _ref(6, offset=IFM_SCRATCH_OFFSET), dim_m * 32, 0, index * 3 + 1),
                _gemm(
                    _ref(6, offset=WEIGHT_SCRATCH_OFFSET),
                    _ref(6, offset=IFM_SCRATCH_OFFSET),
                    _ref(6, offset=OFM_SCRATCH_OFFSET),
                    dim_m,
                    index * 3 + 2,
                ),
                _dma_1d(
                    _ref(6, offset=OFM_SCRATCH_OFFSET),
                    _ref(4, index),
                    dim_m * 32 * 4,
                    1,
                    index * 3 + 3,
                ),
            ]
        )
        bindings.append(_binding(1, index, dimensions=(1, 1, dim_m, 32)))
    for index, dim_m in enumerate(dimensions):
        bindings.append(_binding(2, index, data_type=4, dimensions=(1, 1, dim_m, 32)))
    commands.append(_command_header(0, 32, tile=len(commands)).ljust(32, b"\x00"))
    required_tcdm_bytes = OFM_SCRATCH_OFFSET + max(dimensions) * 32 * 4
    return _package(
        b"".join(commands),
        weights,
        b"".join(bindings),
        len(commands) - 1,
        required_tcdm_bytes,
        len(dimensions),
        len(dimensions),
    )


def build_invocation(model):
    binding_addresses = struct.pack(
        "<HHIIIHHIII",
        1,
        0,
        INPUT_BASE,
        32,
        0,
        2,
        0,
        OUTPUT_BASE,
        32,
        0,
    )
    invocation = struct.pack(
        "<IHH6I8I",
        NAI_INVOCATION_MAGIC,
        1,
        0,
        64,
        MODEL_BASE,
        len(model),
        BINDING_TABLE_BASE,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert len(invocation) == 64 and len(binding_addresses) == 32
    return invocation, binding_addresses


def build_invocation_with_bindings(
    model,
    bindings,
    model_base=MODEL_BASE,
    binding_table_base=BINDING_TABLE_BASE,
):
    binding_addresses = b"".join(
        struct.pack("<HHIII", direction, index, base, byte_size, 0)
        for direction, index, base, byte_size in bindings
    )
    invocation = struct.pack(
        "<IHH6I8I",
        NAI_INVOCATION_MAGIC,
        1,
        0,
        64,
        model_base,
        len(model),
        binding_table_base,
        len(bindings),
        0,
        *([0] * 8),
    )
    return invocation, binding_addresses


def _signed_input(dim_m, seed):
    values = [((row * 3 + column + seed) % 15) - 7 for row in range(dim_m) for column in range(32)]
    return bytes(value & 0xFF for value in values), values


def _mm_clamp(value, lower=-128, upper=127):
    return max(lower, min(upper, value))


def _mm_input_value(row, column, channel):
    return ((row * 3 + column * 5 + channel * 7 + 1) % 5) - 2


def _mm_stem_weight(kh, kw, input_channel, output_channel):
    return (
        (
            kh * 2
            + kw * 3
            + input_channel * 5
            + output_channel * 7
            + 2
        )
        % 3
    ) - 1


def _mm_depthwise_weight(tag, kh, kw, channel):
    return ((tag * 3 + kh * 5 + kw * 7 + channel * 2 + 1) % 3) - 1


def _mm_pointwise_weight(tag, input_channel, output_channel):
    if tag == 5:
        return 1 if input_channel == output_channel else 0
    return (
        (tag * 5 + input_channel * 3 + output_channel * 7 + 4) % 3
    ) - 1


def _mm_conv_weight(tag, kh, kw, input_channel, output_channel):
    return (
        (
            tag * 11
            + kh * 3
            + kw * 5
            + input_channel * 7
            + output_channel * 2
            + 6
        )
        % 3
    ) - 1


def _mm_conv2d(
    source,
    height,
    width,
    input_channels,
    output_channels,
    kernel,
    stride,
    weight,
    activation=None,
):
    output_height = (height + stride - 1) // stride
    output_width = (width + stride - 1) // stride
    total_pad_h = max(
        0, (output_height - 1) * stride + kernel - height
    )
    total_pad_w = max(
        0, (output_width - 1) * stride + kernel - width
    )
    pad_top = total_pad_h // 2
    pad_left = total_pad_w // 2
    output = [0] * (output_height * output_width * output_channels)
    for output_y in range(output_height):
        for output_x in range(output_width):
            output_pixel = (
                output_y * output_width + output_x
            ) * output_channels
            for output_channel in range(output_channels):
                accumulator = 0
                for kh in range(kernel):
                    input_y = output_y * stride + kh - pad_top
                    if input_y < 0 or input_y >= height:
                        continue
                    for kw in range(kernel):
                        input_x = output_x * stride + kw - pad_left
                        if input_x < 0 or input_x >= width:
                            continue
                        input_pixel = (
                            input_y * width + input_x
                        ) * input_channels
                        for input_channel in range(input_channels):
                            accumulator += (
                                source[input_pixel + input_channel]
                                * weight(
                                    kh,
                                    kw,
                                    input_channel,
                                    output_channel,
                                )
                            )
                if activation is None:
                    output[output_pixel + output_channel] = _mm_clamp(
                        accumulator
                    )
                else:
                    output[output_pixel + output_channel] = _mm_clamp(
                        accumulator, *activation
                    )
    return output, output_height, output_width


def _mm_depthwise(
    source,
    height,
    width,
    channels,
    stride,
    tag,
    activation=None,
):
    output_height = (height + stride - 1) // stride
    output_width = (width + stride - 1) // stride
    total_pad_h = max(0, (output_height - 1) * stride + 3 - height)
    total_pad_w = max(0, (output_width - 1) * stride + 3 - width)
    pad_top = total_pad_h // 2
    pad_left = total_pad_w // 2
    output = [0] * (output_height * output_width * channels)
    for output_y in range(output_height):
        for output_x in range(output_width):
            output_pixel = (output_y * output_width + output_x) * channels
            for channel in range(channels):
                accumulator = 0
                for kh in range(3):
                    input_y = output_y * stride + kh - pad_top
                    if input_y < 0 or input_y >= height:
                        continue
                    for kw in range(3):
                        input_x = output_x * stride + kw - pad_left
                        if input_x < 0 or input_x >= width:
                            continue
                        input_pixel = (input_y * width + input_x) * channels
                        accumulator += (
                            source[input_pixel + channel]
                            * _mm_depthwise_weight(tag, kh, kw, channel)
                        )
                if activation is None:
                    output[output_pixel + channel] = _mm_clamp(accumulator)
                else:
                    output[output_pixel + channel] = _mm_clamp(
                        accumulator, *activation
                    )
    return output, output_height, output_width


def _mm_add(lhs, rhs):
    return [_mm_clamp(left + right) for left, right in zip(lhs, rhs)]


def _mm_global_avgpool(source, height, width, channels):
    pixels = height * width
    output = []
    for channel in range(channels):
        total = sum(
            source[pixel * channels + channel] for pixel in range(pixels)
        )
        sign = -1 if total < 0 else 1
        output.append(sign * (abs(total) // pixels))
    return output


def _micro_mobilenet_input_values():
    return [
        _mm_input_value(row, column, channel)
        for row in range(96)
        for column in range(96)
        for channel in range(3)
    ]


def _micro_mobilenet_stage10_values(input_values):
    stem, _, _ = _mm_conv2d(
        input_values,
        96,
        96,
        3,
        32,
        3,
        2,
        _mm_stem_weight,
        activation=(0, 6),
    )
    dw0, _, _ = _mm_depthwise(
        stem, 48, 48, 32, 1, tag=0, activation=(0, 6)
    )
    pw0, _, _ = _mm_conv2d(
        dw0,
        48,
        48,
        32,
        32,
        1,
        1,
        lambda _kh, _kw, ic, oc: _mm_pointwise_weight(0, ic, oc),
    )
    residual0 = _mm_add(pw0, stem)
    dw1, _, _ = _mm_depthwise(
        residual0, 48, 48, 32, 2, tag=1
    )
    pw1, _, _ = _mm_conv2d(
        dw1,
        24,
        24,
        32,
        64,
        1,
        1,
        lambda _kh, _kw, ic, oc: _mm_pointwise_weight(1, ic, oc),
        activation=(0, 6),
    )
    pw2, _, _ = _mm_conv2d(
        pw1,
        24,
        24,
        64,
        128,
        1,
        1,
        lambda _kh, _kw, ic, oc: _mm_pointwise_weight(2, ic, oc),
    )
    dw2, _, _ = _mm_depthwise(pw2, 24, 24, 128, 1, tag=2)
    pw3, _, _ = _mm_conv2d(
        dw2,
        24,
        24,
        128,
        64,
        1,
        1,
        lambda _kh, _kw, ic, oc: _mm_pointwise_weight(3, ic, oc),
    )
    residual1 = _mm_add(pw3, pw1)
    return residual1


def _micro_mobilenet_input_and_expected():
    input_values = _micro_mobilenet_input_values()
    residual1 = _micro_mobilenet_stage10_values(input_values)
    validate, _, _ = _mm_conv2d(
        residual1,
        24,
        24,
        64,
        64,
        3,
        1,
        lambda kh, kw, ic, oc: _mm_conv_weight(
            4, kh, kw, ic, oc
        ),
    )
    pooled = _mm_global_avgpool(validate, 24, 24, 64)
    classifier, _, _ = _mm_conv2d(
        pooled,
        1,
        1,
        64,
        32,
        1,
        1,
        lambda _kh, _kw, ic, oc: _mm_pointwise_weight(5, ic, oc),
    )
    return (
        bytes(value & 0xFF for value in input_values),
        bytes(value & 0xFF for value in classifier),
    )


def _micro_mobilenet_stage1_input_and_expected():
    input_values = _micro_mobilenet_input_values()
    output, output_height, output_width = _mm_conv2d(
        input_values,
        96,
        96,
        3,
        32,
        3,
        2,
        _mm_stem_weight,
        activation=(0, 6),
    )
    assert (output_height, output_width) == (48, 48)
    return (
        bytes(value & 0xFF for value in input_values),
        bytes(value & 0xFF for value in output),
    )


def _micro_mobilenet_stage10_input_and_expected():
    input_values = _micro_mobilenet_input_values()
    output = _micro_mobilenet_stage10_values(input_values)
    return (
        bytes(value & 0xFF for value in input_values),
        bytes(value & 0xFF for value in output),
    )


def _compile_fully_connected_model():
    default_root = Path(__file__).resolve().parents[4] / "neural-compiler"
    compiler_root = Path(os.environ.get("NEURAL_COMPILER_ROOT", default_root)).resolve()
    extension_modules = list((compiler_root / "ethosu").glob("regor*.so"))
    if not extension_modules:
        raise RuntimeError(
            f"Neural compiler Python extension is not built under {compiler_root}/ethosu"
        )

    fixture = Path(__file__).with_name("fully_connected_k33_n34.tflite.b64")
    model_data = base64.b64decode(fixture.read_text(encoding="ascii"))
    with tempfile.TemporaryDirectory(prefix="neural-ai-compiled-model-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        input_path = temporary_path / "fully_connected_k33_n34.tflite"
        output_path = temporary_path / "output"
        input_path.write_bytes(model_data)
        command = [
            sys.executable,
            "-c",
            "import sys; from ethosu.vela.vela import main; raise SystemExit(main(sys.argv[1:]))",
            "--accelerator-config=neural-ai",
            "--output-format=nai",
            f"--output-dir={output_path}",
            str(input_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(compiler_root), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        result = subprocess.run(
            command,
            cwd=compiler_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Neural compiler failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
            )
        package = (output_path / "fully_connected_k33_n34.nai").read_bytes()
    assert package[:4] == b"NAIM"
    return package


def _compile_pointwise_conv_model():
    default_root = Path(__file__).resolve().parents[4] / "neural-compiler"
    compiler_root = Path(os.environ.get("NEURAL_COMPILER_ROOT", default_root)).resolve()
    extension_modules = list((compiler_root / "ethosu").glob("regor*.so"))
    if not extension_modules:
        raise RuntimeError(
            f"Neural compiler Python extension is not built under {compiler_root}/ethosu"
        )

    fixture = Path(__file__).with_name("pointwise_conv_h2w3_k33_n34.tflite.b64")
    model_data = base64.b64decode(fixture.read_text(encoding="ascii"))
    with tempfile.TemporaryDirectory(prefix="neural-ai-compiled-conv-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        input_path = temporary_path / "pointwise_conv_h2w3_k33_n34.tflite"
        output_path = temporary_path / "output"
        input_path.write_bytes(model_data)
        command = [
            sys.executable,
            "-c",
            "import sys; from ethosu.vela.vela import main; raise SystemExit(main(sys.argv[1:]))",
            "--accelerator-config=neural-ai",
            "--output-format=nai",
            f"--output-dir={output_path}",
            str(input_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(compiler_root), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        result = subprocess.run(
            command,
            cwd=compiler_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Neural compiler failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
            )
        package = (output_path / "pointwise_conv_h2w3_k33_n34.nai").read_bytes()
    assert package[:4] == b"NAIM"
    return package


def _compile_tflite_fixture_model(
    fixture_stem, temporary_prefix, compressed=False
):
    default_root = Path(__file__).resolve().parents[4] / "neural-compiler"
    compiler_root = Path(os.environ.get("NEURAL_COMPILER_ROOT", default_root)).resolve()
    extension_modules = list((compiler_root / "ethosu").glob("regor*.so"))
    if not extension_modules:
        raise RuntimeError(
            f"Neural compiler Python extension is not built under {compiler_root}/ethosu"
        )

    suffix = ".tflite.zlib.b64" if compressed else ".tflite.b64"
    fixture = Path(__file__).with_name(f"{fixture_stem}{suffix}")
    model_data = base64.b64decode(fixture.read_text(encoding="ascii"))
    if compressed:
        model_data = zlib.decompress(model_data)
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary_dir:
        temporary_path = Path(temporary_dir)
        input_path = temporary_path / f"{fixture_stem}.tflite"
        output_path = temporary_path / "output"
        input_path.write_bytes(model_data)
        command = [
            sys.executable,
            "-c",
            "import sys; from ethosu.vela.vela import main; raise SystemExit(main(sys.argv[1:]))",
            "--accelerator-config=neural-ai",
            "--output-format=nai",
            f"--output-dir={output_path}",
            str(input_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(compiler_root), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        result = subprocess.run(
            command,
            cwd=compiler_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Neural compiler failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
            )
        package = (output_path / f"{fixture_stem}.nai").read_bytes()
    assert package[:4] == b"NAIM"
    return package


def _compile_generic_k3_conv_model():
    return _compile_tflite_fixture_model(
        "generic_k3_conv_h4w4_k32_n32", "neural-ai-compiled-k3-"
    )


def _compile_high_c16_slice_model():
    return _compile_tflite_fixture_model(
        "c32_high_c16_slice_h2w3", "neural-ai-compiled-high-c16-slice-"
    )


def _compile_rgb_k3_conv_model():
    return _compile_tflite_fixture_model(
        "rgb_k3_conv_h7w7_k3_n32", "neural-ai-compiled-rgb-"
    )


def _compile_depthwise_k3_conv_model():
    return _compile_tflite_fixture_model(
        "depthwise_k3_conv_h3w3_c33_s2", "neural-ai-compiled-depthwise-"
    )


def _compile_afu_add_model():
    return _compile_tflite_fixture_model(
        "add_h2w2_c32", "neural-ai-compiled-add-"
    )


def _compile_quantized_add_model():
    return _compile_tflite_fixture_model(
        "add_quant_h2w2_c32", "neural-ai-compiled-quant-add-"
    )


def _compile_sigmoid_model():
    return _compile_tflite_fixture_model(
        "sigmoid_h2w3_c33", "neural-ai-compiled-sigmoid-"
    )


def _compile_silu_model():
    return _compile_tflite_fixture_model(
        "silu_h2w3_c33", "neural-ai-compiled-silu-"
    )


def _compile_relu6_model():
    return _compile_tflite_fixture_model(
        "relu6_h2w3_c33", "neural-ai-compiled-relu6-"
    )


def _compile_global_avgpool_model():
    return _compile_tflite_fixture_model(
        "global_avgpool_h2w3_c33", "neural-ai-compiled-global-avgpool-"
    )


def _compile_resize_nearest_model():
    return _compile_tflite_fixture_model(
        "resize_nearest_h2w3_c32_2x", "neural-ai-compiled-resize-nearest-"
    )


def _compile_yolov8_resize_c256_model():
    return _compile_tflite_fixture_model(
        "yolov8_resize_h10w10_c256_2x", "neural-ai-compiled-yolov8-resize-"
    )


def _compile_maxpool_model():
    return _compile_tflite_fixture_model(
        "maxpool_h24w24_c32_k5s1p2", "neural-ai-compiled-maxpool-"
    )


def _compile_yolov8_maxpool_c128_model():
    return _compile_tflite_fixture_model(
        "yolov8_maxpool_h10w10_c128", "neural-ai-compiled-yolov8-maxpool-"
    )


def _compile_yolov8_head_transpose_model():
    return _compile_tflite_fixture_model(
        "yolov8_head_transpose_h10w10_c144",
        "neural-ai-compiled-yolov8-head-transpose-",
    )


def _compile_concat_model():
    return _compile_tflite_fixture_model(
        "concat_h24w24_c32_c32", "neural-ai-compiled-concat-"
    )


def _compile_public_reshape_model():
    return _compile_tflite_fixture_model(
        "reshape_1x4x32_to_1x2x2x32", "neural-ai-compiled-reshape-"
    )


def _compile_pointwise_depthwise_chain_model():
    return _compile_tflite_fixture_model(
        "pointwise_depthwise_chain", "neural-ai-compiled-chain-"
    )


def _compile_micro_mobilenet_full_model():
    return _compile_tflite_fixture_model(
        "micro_mobilenet_full",
        "neural-ai-compiled-micro-mobilenet-",
        compressed=True,
    )


def _compile_micro_mobilenet_stage1_model():
    return _compile_tflite_fixture_model(
        "micro_mobilenet_stage1",
        "neural-ai-compiled-micro-mobilenet-stage1-",
        compressed=True,
    )


def _compile_micro_mobilenet_stage10_model():
    return _compile_tflite_fixture_model(
        "micro_mobilenet_stage10",
        "neural-ai-compiled-micro-mobilenet-stage10-",
        compressed=True,
    )


def _load_compiler_striped_pointwise_package(width):
    fixture = Path(__file__).with_name(f"pointwise_m{width}.nai.b64")
    package = base64.b64decode(fixture.read_text(encoding="ascii"))
    assert package[:4] == b"NAIM"
    return package


def _pointwise_striped_expected(input_values, width):
    expected = bytearray()
    for pixel in range(width):
        channels = input_values[pixel * 32:(pixel + 1) * 32]
        value = sum(channel - 256 if channel >= 128 else channel for channel in channels)
        value = max(-128, min(127, value)) & 0xFF
        expected.extend([value] * 32)
    return bytes(expected)


async def _run_compiler_striped_pointwise(dut, axi_master, width):
    model = _load_compiler_striped_pointwise_package(width)
    input_values = [(index * 7 + 3) & 0xFF for index in range(width * 32)]
    input_data = bytes(input_values)
    expected = _pointwise_striped_expected(input_values, width)
    input_base = 0x80000000
    output_base = 0x80004000
    runtime_bindings = [
        (1, 0, input_base, len(input_data)),
        (2, 0, output_base, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, input_base, input_data)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=500000)

    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, output_base, len(expected))) == expected

    # The compiler must coalesce oversized M into one full-row pointwise
    # command; the runtime performs the <=256-row decomposition internally.
    section_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_offset + 8)[0]
    offset = command_offset
    pointwise_rows = []
    while offset < len(model):
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 10:
            pointwise_rows.append(struct.unpack_from("<I", model, offset + 48)[0])
        if command_type == 0:
            break
        offset += command_size
    assert pointwise_rows and sum(pointwise_rows) == width
    assert pointwise_rows == [width]



async def _load_and_run(
    dut,
    axi_master,
    invocation,
    timeout_cycles=600000,
    measure_pmu=False,
    invocation_base=INVOCATION_BASE,
):
    await load_firmware_elf_axi(
        dut,
        axi_master,
        Path(__file__).resolve().parents[3] / "sw/runtime/neural_ai/neural_ai.elf",
    )
    await program_command_queue(axi_master, invocation_base, len(invocation))
    await release_fetch(dut, axi_master=axi_master)
    try:
        counters = await wait_for_host_irq(
            dut,
            timeout_cycles=timeout_cycles,
            axi_master=axi_master,
            report_name="test_compiler_runtime",
        )
    except AssertionError as error:
        status = await _axi_read32(axi_master, NPU_CMD_STATUS)
        fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
        fail_pointer = await _axi_read32(axi_master, NPU_CMD_FAIL_PTR)
        done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
        pc_signal = dut.u_npu_cluster.u_snitch_core.i_snitch.pc_q
        program_counter = (
            pc_signal.value.to_unsigned() if pc_signal.value.is_resolvable else 0
        )
        raise AssertionError(
            f"{error}; status={status} fail=0x{fail_code:08x} "
            f"pointer=0x{fail_pointer:08x} done={done_count} "
            f"pc=0x{program_counter:08x}"
        ) from error
    if measure_pmu:
        return counters
    return None


@cocotb.test()
async def test_compiler_runtime_dma_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_data = bytes((index * 7 + 3) & 0xFF for index in range(32))
    model = build_model()
    invocation, binding_addresses = build_invocation(model)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(32))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == struct.unpack_from(
        "<I", model, 32
    )[0]
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, 32)) == input_data


@cocotb.test()
async def test_compiler_runtime_unaligned_raw_dma_1d_2d_3d(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    for dimension in (1, 2, 3):
        model, copies, binding_bytes = build_unaligned_dma_model(dimension)
        input_data = bytes(
            (index * 17 + dimension * 11) & 0xFF
            for index in range(binding_bytes)
        )
        expected = bytearray(binding_bytes)
        for source, destination, length in copies:
            expected[destination:destination + length] = (
                input_data[source:source + length]
            )
        runtime_bindings = [
            (1, 0, INPUT_BASE, binding_bytes),
            (2, 0, OUTPUT_BASE, binding_bytes),
        ]
        invocation, binding_addresses = build_invocation_with_bindings(
            model, runtime_bindings
        )

        await reset_dut(dut)
        await write_l2_bytes(dut, INPUT_BASE, input_data)
        await write_l2_bytes(dut, OUTPUT_BASE, bytes(binding_bytes))
        await write_l2_bytes(dut, MODEL_BASE, model)
        await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
        await write_l2_bytes(dut, INVOCATION_BASE, invocation)
        await _load_and_run(dut, axi_master, invocation)

        status = await _axi_read32(axi_master, NPU_CMD_STATUS)
        if status != NPU_CMD_STATUS_PASS:
            fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
            fail_pointer = await _axi_read32(axi_master, NPU_CMD_FAIL_PTR)
            done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
            raise AssertionError(
                f"unaligned DMA{dimension}D failed: status={status} "
                f"fail=0x{fail_code:08x} pointer=0x{fail_pointer:08x} "
                f"done={done_count}"
            )
        assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
        assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
        assert bytes(await read_l2_bytes(
            dut, OUTPUT_BASE, binding_bytes
        )) == bytes(expected)


@cocotb.test()
async def test_compiler_runtime_layout_round_trip(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_data = bytes((index * 11 + 5) & 0xFF for index in range(132))
    model = build_layout_model()
    invocation, binding_addresses = build_invocation(model)
    binding_addresses = bytearray(binding_addresses)
    struct.pack_into("<I", binding_addresses, 8, len(input_data))
    struct.pack_into("<I", binding_addresses, 24, len(input_data))
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(input_data)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 1
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(input_data))) == input_data


@cocotb.test()
async def test_compiler_runtime_c32_layout_round_trip(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_data = bytes((index * 13 + 7) & 0xFF for index in range(132))
    model = build_c32_layout_model()
    invocation, binding_addresses = build_invocation(model)
    binding_addresses = bytearray(binding_addresses)
    struct.pack_into("<I", binding_addresses, 8, len(input_data))
    struct.pack_into("<I", binding_addresses, 24, len(input_data))
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(input_data)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(input_data))) == input_data


@cocotb.test()
async def test_compiler_runtime_afu_add_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    count = 2 * 2 * 32
    lhs_values = [120 if index % 7 == 0 else ((index * 3) % 41) - 20
                  for index in range(count)]
    rhs_values = [20 if index % 7 == 0 else ((index * 5) % 31) - 15
                  for index in range(count)]
    expected = bytes(
        max(-128, min(127, lhs + rhs)) & 0xFF
        for lhs, rhs in zip(lhs_values, rhs_values)
    )
    model = build_afu_add_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, count),
        (1, 1, INPUT2_BASE, count),
        (2, 0, OUTPUT_BASE, count),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, bytes(value & 0xFF for value in lhs_values))
    await write_l2_bytes(dut, INPUT2_BASE, bytes(value & 0xFF for value in rhs_values))
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(count))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 4
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, count)) == expected


@cocotb.test()
async def test_compiler_runtime_afu_dfl16_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    locations = 2100
    input_base = 0x80100000
    output_base = 0x80150000
    input_values = bytearray(144 * locations)
    expected = bytearray(4 * locations)
    for channel in range(144):
        for location in range(locations):
            value = ((location * 13 + channel * 11 + 7) % 61) - 30
            input_values[channel * locations + location] = value & 0xFF
    for side in range(4):
        for location in range(locations):
            values = [
                ((location * 13 + (side * 16 + bin_index) * 11 + 7) % 61) - 30
                for bin_index in range(16)
            ]
            q8 = _dfl16_q8(values)
            value = ((q8 * 51003 + (1 << 7)) >> 8) - 128
            expected[side * locations + location] = max(-128, min(127, value)) & 0xFF

    model = build_afu_dfl16_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    dfl_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 28:
            assert command_size == 96
            assert struct.unpack_from("<I", model, offset + 56)[0] == locations
            dfl_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert dfl_commands == 1

    runtime_bindings = [
        (1, 0, input_base, len(input_values)),
        (2, 0, output_base, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, input_base, input_values)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=1000000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns
    status = await _axi_read32(axi_master, NPU_CMD_STATUS)
    fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
    done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
    assert status == NPU_CMD_STATUS_PASS, (
        f"DFL16 runtime failed: status=0x{status:08x} "
        f"code=0x{fail_code:08x} done={done_count}"
    )
    assert fail_code == 0
    assert done_count == 3
    assert bytes(await read_l2_bytes(dut, output_base, len(expected))) == expected
    cocotb.log.info(
        "DFL16 runtime=%s ns PMU cycles=%d", elapsed_ns, counters["cycle"]
    )


async def _run_row32_case(dut, axi_master, channels, height, width,
                          input_base, output_base):
    model, compact_bytes = build_row32_layout_model(channels, height, width)
    input_data = bytes((index * 17 + channels) & 0xFF for index in range(compact_bytes))
    runtime_bindings = [
        (1, 0, input_base, compact_bytes),
        (2, 0, output_base, compact_bytes),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await reset_dut(dut)
    await write_l2_bytes(dut, input_base, input_data)
    await write_l2_bytes(dut, output_base, bytes(compact_bytes))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    start_ns = get_sim_time("ns")
    await _load_and_run(dut, axi_master, invocation)
    elapsed_ns = get_sim_time("ns") - start_ns
    status = await _axi_read32(axi_master, NPU_CMD_STATUS)
    if status != NPU_CMD_STATUS_PASS:
        fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
        done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
        raise AssertionError(
            f"ROW32 C={channels} H={height} W={width} dispatch failed: "
            f"status={status} "
            f"fail=0x{fail_code:08x} done={done_count}"
        )
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert bytes(await read_l2_bytes(dut, output_base, compact_bytes)) == input_data
    return elapsed_ns


@cocotb.test()
async def test_compiler_runtime_unaligned_row32_c3_c31(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await _run_row32_case(dut, axi_master, 3, 1, 11,
                          0x80002003, 0x80003007)
    await _run_row32_case(dut, axi_master, 31, 1, 5,
                          0x80002005, 0x8000300b)


@cocotb.test()
async def test_compiler_runtime_row32_channel_boundary_matrix(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    logging.getLogger("cocotb.tb_npu_cluster.s_axi").setLevel(logging.WARNING)
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    for channels in (3, 31, 32, 33, 63, 64, 65):
        elapsed_ns = await _run_row32_case(
            dut,
            axi_master,
            channels,
            2,
            3,
            0x80002000,
            0x80003000,
        )
        dut._log.info(
            "ROW32 C=%d H=2 W=3 runtime completion: %.3f ns",
            channels,
            elapsed_ns,
        )


@cocotb.test()
async def test_compiler_runtime_pointwise_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels, output_channels = 1, 1, 64, 32
    input_values = [((pixel + channel) % 5) - 2
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for pixel in range(height * width):
        value = sum(input_values[pixel * channels:(pixel + 1) * channels]) & 0xFF
        expected.extend([value] * output_channels)
    model = build_pointwise_c32_model(height, width, channels, output_channels)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=200000)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 4
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_runtime_pointwise_c32_relu6_clamp(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_values = [-1] * 4 + [2] * 28
    input_data = bytes(value & 0xFF for value in input_values)
    expected_value = min(max(sum(input_values), 0), 6)
    expected = bytes([expected_value] * 32)
    model = build_pointwise_c32_model(1, 1, 32, 32, clamp_min=0, clamp_max=6)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=200000)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == expected


@cocotb.test()
async def test_compiler_runtime_pointwise_per_channel_requant(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    qparam_specs = []
    expected_values = []
    for channel in range(32):
        bias = channel * 5 - 75
        multiplier = 1 << (30 if channel % 2 == 0 else 29)
        shift = 30
        zero_point = -(channel // 2)
        qparam_specs.append((bias, multiplier, shift, zero_point, -128, 127))
        accumulator = 32 + bias
        product = accumulator * multiplier
        magnitude = abs(product)
        rounded = (magnitude + (1 << (shift - 1))) >> shift
        scaled = rounded if product >= 0 else -rounded
        expected_values.append(max(-128, min(127, scaled + zero_point)) & 0xFF)

    model = build_pointwise_c32_model(1, 1, 32, 32, qparam_specs=qparam_specs)
    input_data = bytes([1] * 32)
    expected = bytes(expected_values)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == expected


@cocotb.test()
async def test_compiler_runtime_rejects_mismatched_qparam_block(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    model = build_pointwise_c32_model(
        1, 1, 32, 32, qparam_block_delta=1
    )
    input_data = bytes([1] * 32)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, 32),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(32))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_FAIL
    assert await _axi_read32(
        axi_master, NPU_CMD_FAIL_CODE
    ) == NPU_CMD_FAIL_V2_OPERATION
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, 32)) == bytes(32)


@cocotb.test()
async def test_compiler_runtime_depthwise_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 3, 3, 32
    input_values = [((pixel + channel * 3) % 9) - 4
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    model = build_depthwise_c32_model(height, width, channels)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(input_data)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(input_data)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 4
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(input_data))) == input_data


@cocotb.test()
async def test_compiler_runtime_depthwise_c65_stride2_tail(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels, stride = 3, 3, 65, 2
    output_height = ((height - 1) // stride) + 1
    output_width = ((width - 1) // stride) + 1
    input_values = [((pixel * 7 + channel * 3) % 19) - 9
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for output_y in range(output_height):
        for output_x in range(output_width):
            input_pixel = (output_y * stride) * width + output_x * stride
            expected.extend(
                value & 0xFF
                for value in input_values[input_pixel * channels:(input_pixel + 1) * channels]
            )

    model = build_depthwise_c32_model(height, width, channels, stride)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=400000)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 8
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_runtime_depthwise_c96_stride2_groups(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels, stride = 3, 3, 96, 2
    output_height = ((height - 1) // stride) + 1
    output_width = ((width - 1) // stride) + 1
    input_values = [((pixel * 11 + channel * 5) % 23) - 11
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for output_y in range(output_height):
        for output_x in range(output_width):
            input_pixel = (output_y * stride) * width + output_x * stride
            expected.extend(
                value & 0xFF
                for value in input_values[input_pixel * channels:(input_pixel + 1) * channels]
            )

    model = build_depthwise_c32_model(height, width, channels, stride)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=400000)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 8
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_runtime_pointwise_depthwise_chain(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 2, 2, 32
    input_values = [((pixel * 3 + channel) % 5) - 2
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for pixel in range(height * width):
        value = sum(input_values[pixel * channels:(pixel + 1) * channels])
        expected.extend([max(-128, min(127, value)) & 0xFF] * channels)

    model = build_pointwise_depthwise_chain_model(height, width)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=300000)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 6
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_pointwise_depthwise_chain(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 2, 2, 32
    input_values = [((pixel * 5 + channel * 2) % 7) - 3
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for pixel in range(height * width):
        value = sum(input_values[pixel * channels:(pixel + 1) * channels])
        expected.extend([max(-128, min(127, value)) & 0xFF] * channels)

    model = _compile_pointwise_depthwise_chain_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=300000)

    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_micro_mobilenet_full_graph(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_base = 0x80000000
    output_base = 0x80008000
    invocation_base = 0x80010000
    binding_table_base = 0x80011000
    model_base = 0x80100000
    native_mobilenet_cycles = 347_992
    native_yolo_cycles = 388_146

    input_data, expected = _micro_mobilenet_input_and_expected()
    model = _compile_micro_mobilenet_full_model()
    assert len(input_data) == 96 * 96 * 3
    assert len(expected) == 32
    assert struct.unpack_from("<I", model, 16)[0] == len(model)
    command_count = struct.unpack_from("<I", model, 32)[0]
    required_tcdm = struct.unpack_from("<I", model, 36)[0]
    assert required_tcdm <= 0x7F000
    assert struct.unpack_from("<2I", model, 44) == (1, 1)

    section_table = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table + 12)[0]
    command_types = {}
    offset = command_offset
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        assert command_size >= 32
        command_types[command_type] = command_types.get(command_type, 0) + 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert command_types.get(12, 0) == 0
    assert command_types.get(13, 0) == 2
    assert command_types.get(14, 0) == 1
    assert command_types.get(9, 0) > 0
    assert command_types.get(10, 0) > 0
    assert command_types.get(11, 0) > 0

    runtime_bindings = [
        (1, 0, input_base, len(input_data)),
        (2, 0, output_base, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model,
        runtime_bindings,
        model_base=model_base,
        binding_table_base=binding_table_base,
    )
    await write_l2_bytes(dut, input_base, input_data)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, model_base, model)
    await write_l2_bytes(dut, binding_table_base, binding_addresses)
    await write_l2_bytes(dut, invocation_base, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut,
        axi_master,
        invocation,
        timeout_cycles=3_000_000,
        measure_pmu=True,
        invocation_base=invocation_base,
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    actual = bytes(await read_l2_bytes(dut, output_base, len(expected)))
    assert actual == expected, (
        f"Micro-MobileNet mismatch: actual={list(actual)} "
        f"expected={list(expected)}"
    )
    assert counters["cycle"] > 0
    cocotb.log.warning(
        "compiler Micro-MobileNet full graph package=%d bytes commands=%d "
        "TCDM=%d runtime=%s ns PMU cycles=%d "
        "vs native MobileNet=%d (%.3fx) and native YOLO=%d (%.3fx)",
        len(model),
        command_count,
        required_tcdm,
        elapsed_ns,
        counters["cycle"],
        native_mobilenet_cycles,
        counters["cycle"] / native_mobilenet_cycles,
        native_yolo_cycles,
        counters["cycle"] / native_yolo_cycles,
    )
    cocotb.log.warning("%s", format_pmu_report(counters))


@cocotb.test()
async def test_compiler_generated_micro_mobilenet_stage1(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_base = 0x80000000
    output_base = 0x80010000
    invocation_base = 0x80030000
    binding_table_base = 0x80031000
    model_base = 0x80040000
    input_data, expected = _micro_mobilenet_stage1_input_and_expected()
    model = _compile_micro_mobilenet_stage1_model()
    command_count = struct.unpack_from("<I", model, 32)[0]
    invocation, binding_addresses = build_invocation_with_bindings(
        model,
        [
            (1, 0, input_base, len(input_data)),
            (2, 0, output_base, len(expected)),
        ],
        model_base=model_base,
        binding_table_base=binding_table_base,
    )
    await write_l2_bytes(dut, input_base, input_data)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, model_base, model)
    await write_l2_bytes(dut, binding_table_base, binding_addresses)
    await write_l2_bytes(dut, invocation_base, invocation)

    counters = await _load_and_run(
        dut,
        axi_master,
        invocation,
        timeout_cycles=1_500_000,
        measure_pmu=True,
        invocation_base=invocation_base,
    )

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    actual = bytes(await read_l2_bytes(dut, output_base, len(expected)))
    if actual != expected:
        mismatch = next(
            index
            for index, (actual_value, expected_value) in enumerate(
                zip(actual, expected)
            )
            if actual_value != expected_value
        )
        raise AssertionError(
            "Micro-MobileNet stage 1 mismatch at byte "
            f"{mismatch}: actual={actual[mismatch]} "
            f"expected={expected[mismatch]}"
        )
    cocotb.log.warning(
        "compiler Micro-MobileNet stage 1\n%s", format_pmu_report(counters)
    )


@cocotb.test()
async def test_compiler_generated_micro_mobilenet_stage10(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_base = 0x80000000
    output_base = 0x80010000
    invocation_base = 0x80030000
    binding_table_base = 0x80031000
    model_base = 0x80040000
    input_data, expected = _micro_mobilenet_stage10_input_and_expected()
    model = _compile_micro_mobilenet_stage10_model()
    command_count = struct.unpack_from("<I", model, 32)[0]
    invocation, binding_addresses = build_invocation_with_bindings(
        model,
        [
            (1, 0, input_base, len(input_data)),
            (2, 0, output_base, len(expected)),
        ],
        model_base=model_base,
        binding_table_base=binding_table_base,
    )
    await write_l2_bytes(dut, input_base, input_data)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, model_base, model)
    await write_l2_bytes(dut, binding_table_base, binding_addresses)
    await write_l2_bytes(dut, invocation_base, invocation)

    counters = await _load_and_run(
        dut,
        axi_master,
        invocation,
        timeout_cycles=3_000_000,
        measure_pmu=True,
        invocation_base=invocation_base,
    )

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    actual = bytes(await read_l2_bytes(dut, output_base, len(expected)))
    if actual != expected:
        mismatch = next(
            index
            for index, (actual_value, expected_value) in enumerate(
                zip(actual, expected)
            )
            if actual_value != expected_value
        )
        raise AssertionError(
            "Micro-MobileNet stage 10 mismatch at byte "
            f"{mismatch}: actual={actual[mismatch]} "
            f"expected={expected[mismatch]}"
        )
    assert counters["cycle"] > 0
    cocotb.log.warning(
        "compiler Micro-MobileNet stage 10\n%s", format_pmu_report(counters)
    )


@cocotb.test()
async def test_compiler_runtime_gemm_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    dimensions = (1, 31, 32, 33, 256)
    input_bases = (0x80000000, 0x80001000, 0x80002000, 0x80004000, 0x80006000)
    output_bases = (0x80010000, 0x80011000, 0x80015000, 0x80019000, 0x80020000)
    model = build_gemm_model(dimensions)
    runtime_bindings = []
    expected_outputs = []
    for index, (dim_m, base) in enumerate(zip(dimensions, input_bases)):
        input_data, values = _signed_input(dim_m, index)
        await write_l2_bytes(dut, base, input_data)
        runtime_bindings.append((1, index, base, len(input_data)))
        expected_outputs.append(b"".join(struct.pack("<i", value) for value in values))
    for index, (base, output) in enumerate(zip(output_bases, expected_outputs)):
        runtime_bindings.append((2, index, base, len(output)))

    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 16
    for base, expected in zip(output_bases, expected_outputs):
        assert bytes(await read_l2_bytes(dut, base, len(expected))) == expected


@cocotb.test()
async def test_compiler_generated_fully_connected_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    model = _compile_fully_connected_model()
    input_values = [(index % 5) - 2 for index in range(33)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytes([sum(input_values) & 0xFF] * 34)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    await _load_and_run(dut, axi_master, invocation)

    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == expected


@cocotb.test()
async def test_compiler_generated_pointwise_m257_stripes(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)
    await _run_compiler_striped_pointwise(dut, axi_master, 257)


@cocotb.test()
async def test_compiler_generated_pointwise_m511_stripes(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)
    await _run_compiler_striped_pointwise(dut, axi_master, 511)


@cocotb.test()
async def test_compiler_generated_pointwise_conv_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels, output_channels = 2, 3, 33, 34
    input_values = [((pixel * 3 + channel) % 9) - 4
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for pixel in range(height * width):
        value = sum(input_values[pixel * channels:(pixel + 1) * channels]) & 0xFF
        expected.extend([value] * output_channels)
    model = _compile_pointwise_conv_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_afu_add_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    count = 2 * 2 * 32
    lhs_values = [
        120 if index % 7 == 0 else ((index * 3) % 41) - 20
        for index in range(count)
    ]
    rhs_values = [
        20 if index % 7 == 0 else ((index * 5) % 31) - 15
        for index in range(count)
    ]
    expected = bytes(
        max(-128, min(127, lhs + rhs)) & 0xFF
        for lhs, rhs in zip(lhs_values, rhs_values)
    )
    model = _compile_afu_add_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, count),
        (1, 1, INPUT2_BASE, count),
        (2, 0, OUTPUT_BASE, count),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(
        dut, INPUT_BASE, bytes(value & 0xFF for value in lhs_values)
    )
    await write_l2_bytes(
        dut, INPUT2_BASE, bytes(value & 0xFF for value in rhs_values)
    )
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(count))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, count)) == expected


@cocotb.test()
async def test_compiler_generated_quantized_add_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    count = 2 * 2 * 32
    lhs_values = [((index * 17 + 3) & 0xFF) - 128 for index in range(count)]
    rhs_values = [((index * 29 + 11) & 0xFF) - 128 for index in range(count)]
    model = _compile_quantized_add_model()
    commands_offset = struct.unpack_from("<I", model, 64 + 8)[0]
    commands_size = struct.unpack_from("<I", model, 64 + 12)[0]
    command_types = []
    offset = commands_offset
    while offset < commands_offset + commands_size:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        command_types.append(command_type)
        offset += command_size
    assert 16 in command_types

    # Read the exact compiler-emitted parameters so this independent reference
    # also freezes command field interpretation across the two repositories.
    spatz_offset = next(
        pos for pos in range(commands_offset, commands_offset + commands_size, 32)
        if struct.unpack_from("<H", model, pos)[0] == 16
    )
    # The command layout contains alternating signed scale / unsigned shift;
    # unpack explicitly to avoid relying on Python integer wraparound.
    lhs_scale = struct.unpack_from("<i", model, spatz_offset + 44)[0]
    lhs_shift = struct.unpack_from("<I", model, spatz_offset + 48)[0]
    rhs_scale = struct.unpack_from("<i", model, spatz_offset + 52)[0]
    rhs_shift = struct.unpack_from("<I", model, spatz_offset + 56)[0]
    output_scale = struct.unpack_from("<i", model, spatz_offset + 60)[0]
    output_shift = struct.unpack_from("<I", model, spatz_offset + 64)[0]
    lhs_zp, rhs_zp, output_zp, clamp_min, clamp_max = struct.unpack_from(
        "<5i", model, spatz_offset + 68
    )
    double_round_shift = struct.unpack_from("<I", model, spatz_offset + 88)[0]
    def scale_value(value, scale, shift, double_shift):
        if shift == 0:
            return value * scale
        correction = 1 << (30 - double_shift) if shift > 31 - double_shift else 0
        rounded = value * scale + (1 << (shift - 1))
        rounded += correction if value >= 0 else -correction
        return rounded >> shift

    expected = bytearray()
    for lhs, rhs in zip(lhs_values, rhs_values):
        value = scale_value(lhs - lhs_zp, lhs_scale, lhs_shift, double_round_shift)
        value += scale_value(rhs - rhs_zp, rhs_scale, rhs_shift, double_round_shift)
        value = scale_value(value, output_scale, output_shift, 0) + output_zp
        expected.append(min(max(value, clamp_min), clamp_max) & 0xFF)

    runtime_bindings = [
        (1, 0, INPUT_BASE, count),
        (1, 1, INPUT2_BASE, count),
        (2, 0, OUTPUT_BASE, count),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, bytes(value & 0xFF for value in lhs_values))
    await write_l2_bytes(dut, INPUT2_BASE, bytes(value & 0xFF for value in rhs_values))
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(count))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    await _load_and_run(dut, axi_master, invocation)

    status = await _axi_read32(axi_master, NPU_CMD_STATUS)
    fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
    fail_ptr = await _axi_read32(axi_master, NPU_CMD_FAIL_PTR)
    done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
    assert status == NPU_CMD_STATUS_PASS, (
        f"runtime failed: status=0x{status:08x} code=0x{fail_code:08x} "
        f"ptr=0x{fail_ptr:08x} done={done_count} types={command_types}"
    )
    assert fail_code == 0
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, count)) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_sigmoid_afu_lut_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    count = 2 * 3 * 33
    input_data = bytes(range(count))
    expected = bytearray()
    for raw in input_data:
        signed = raw if raw < 128 else raw - 256
        real = max(-8.0, min(8.0, 0.125 * signed))
        logistic = 1.0 / (1.0 + math.exp(-real))
        quantized = max(-128, min(127, math.floor(-128.0 + 256.0 * logistic + 0.5)))
        expected.append(quantized & 0xFF)

    model = _compile_sigmoid_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    lut_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 12:
            assert command_size == 64
            assert struct.unpack_from("<H", model, offset + 16)[0] == 6
            assert struct.unpack_from("<H", model, offset + 24)[0] == 6
            assert struct.unpack_from("<H", model, offset + 32)[0] == 1
            assert struct.unpack_from("<I", model, offset + 40)[0] == 384
            lut_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert lut_commands == 1

    runtime_bindings = [
        (1, 0, INPUT_BASE, count),
        (2, 0, OUTPUT_BASE, count),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(count))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=300000)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, count)) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_silu_afu_lut_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    count = 2 * 3 * 33
    input_data = bytes((index * 131 + 17) & 0xFF for index in range(count))
    expected = bytearray()
    for raw in input_data:
        signed = raw if raw < 128 else raw - 256
        real = max(-8.0, min(8.0, 0.125 * signed))
        logistic = 1.0 / (1.0 + math.exp(-real))
        quantized_logistic = max(
            -128, min(127, math.floor(-128.0 + 256.0 * logistic + 0.5))
        )
        product = signed * (quantized_logistic + 128)
        rounded = (
            (product + 64) // 128
            if product >= 0
            else -((-product + 64) // 128)
        )
        expected.append(max(-128, min(127, rounded)) & 0xFF)

    model = _compile_silu_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    command_types = []
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        assert command_size >= 32
        command_types.append(command_type)
        if command_type == 12:
            assert command_size == 64
            assert struct.unpack_from("<H", model, offset + 16)[0] == 6
            assert struct.unpack_from("<H", model, offset + 24)[0] == 6
            assert struct.unpack_from("<H", model, offset + 32)[0] == 1
            assert struct.unpack_from("<I", model, offset + 40)[0] == 384
        offset += command_size
    assert offset == command_offset + command_bytes
    assert command_types.count(12) == 1
    assert 13 not in command_types
    assert 17 not in command_types

    runtime_bindings = [
        (1, 0, INPUT_BASE, count),
        (2, 0, OUTPUT_BASE, count),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(count))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=300000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, count)) == bytes(expected)
    assert counters["cycle"] > 0
    cocotb.log.warning(
        "compiler SiLU LUT package=%d bytes commands=%d runtime=%s ns "
        "PMU cycles=%d; diagnostic ratios vs Micro-MobileNet=%0.3fx "
        "and Micro-YOLO=%0.3fx",
        len(model),
        command_count,
        elapsed_ns,
        counters["cycle"],
        counters["cycle"] / 347_992,
        counters["cycle"] / 388_146,
    )
    cocotb.log.warning("%s", format_pmu_report(counters))


@cocotb.test()
async def test_compiler_generated_relu6_afu_lut_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    count = 2 * 3 * 33
    input_data = bytes(range(count))
    expected = bytearray()
    for raw in input_data:
        signed = raw if raw < 128 else raw - 256
        real = 0.25 * (signed + 3)
        clamped = max(0.0, min(6.0, real))
        quantized = max(-128, min(127, math.floor(-3.0 + clamped / 0.25 + 0.5)))
        expected.append(quantized & 0xFF)

    model = _compile_relu6_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    lut_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 12:
            assert command_size == 64
            assert struct.unpack_from("<H", model, offset + 16)[0] == 6
            assert struct.unpack_from("<H", model, offset + 24)[0] == 6
            assert struct.unpack_from("<H", model, offset + 32)[0] == 1
            assert struct.unpack_from("<I", model, offset + 40)[0] == 384
            lut_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert lut_commands == 1

    runtime_bindings = [
        (1, 0, INPUT_BASE, count),
        (2, 0, OUTPUT_BASE, count),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(count))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation, timeout_cycles=300000)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, count)) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_global_avgpool_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 2, 3, 33
    deltas = (-5, -3, -1, 1, 3, 5)
    input_values = []
    expected_values = []
    for pixel in range(height * width):
        for channel in range(channels):
            base = (channel % 41) - 20
            input_values.append(base + deltas[pixel])
    for channel in range(channels):
        expected_values.append((channel % 41) - 20)
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytes(value & 0xFF for value in expected_values)
    model = _compile_global_avgpool_model()

    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    avgpool_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 14:
            assert command_size == 64
            assert struct.unpack_from("<3I", model, offset + 32) == (
                height,
                width,
                channels,
            )
            avgpool_commands += 1
        offset += command_size
    assert avgpool_commands == 1

    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=300000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == expected
    assert counters["afu_done"] > 0
    cocotb.log.info(
        "compiler global AvgPool C33 runtime=%s ns PMU cycles=%d "
        "AFU done=%d AFU TCDM req=%d",
        elapsed_ns,
        counters["cycle"],
        counters["afu_done"],
        counters["afu_tcdm_req"],
    )


@cocotb.test()
async def test_compiler_generated_resize_nearest_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 2, 3, 32
    input_data = bytes((index * 29 + 7) & 0xFF for index in range(height * width * channels))
    expected = bytearray()
    for output_y in range(height * 2):
        input_y = output_y // 2
        for output_x in range(width * 2):
            input_x = output_x // 2
            pixel = (input_y * width + input_x) * channels
            expected.extend(input_data[pixel : pixel + channels])
    model = _compile_resize_nearest_model()

    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    upsample_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 20:
            assert command_size == 64
            assert struct.unpack_from("<H", model, offset + 16)[0] == 6
            assert struct.unpack_from("<H", model, offset + 24)[0] == 6
            assert struct.unpack_from("<5I", model, offset + 32) == (
                height,
                width,
                channels,
                2,
                2,
            )
            upsample_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert upsample_commands == 1

    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=300000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)
    cocotb.log.info(
        "compiler nearest 2x C32 runtime=%s ns PMU cycles=%d",
        elapsed_ns,
        counters["cycle"],
    )


@cocotb.test()
async def test_compiler_generated_yolov8_resize_c256_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_h, input_w, channels = 10, 10, 256
    output_h, output_w = input_h * 2, input_w * 2
    output_base = 0x80010000
    input_data = bytes(
        (index * 37 + 11) & 0xFF
        for index in range(input_h * input_w * channels)
    )
    expected = bytearray()
    for output_y in range(output_h):
        for output_x in range(output_w):
            source = ((output_y // 2) * input_w + output_x // 2) * channels
            expected.extend(input_data[source : source + channels])

    model = _compile_yolov8_resize_c256_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    upsample_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 20:
            assert command_size == 64
            assert struct.unpack_from("<5I", model, offset + 32) == (
                input_h,
                input_w,
                channels,
                2,
                2,
            )
            upsample_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert upsample_commands == 1

    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, output_base, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=600000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, output_base, len(expected))) == bytes(expected)
    cocotb.log.info(
        "compiler YOLO nearest 2x C256 runtime=%s ns PMU cycles=%d",
        elapsed_ns,
        counters["cycle"],
    )


@cocotb.test()
async def test_compiler_generated_maxpool_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 24, 24, 32
    maxpool_output_base = 0x80010000
    input_values = [
        ((index * 37 + 11) & 0xFF) - 256
        if ((index * 37 + 11) & 0xFF) >= 128
        else ((index * 37 + 11) & 0xFF)
        for index in range(height * width * channels)
    ]
    expected_values = []
    for output_y in range(height):
        for output_x in range(width):
            for channel in range(channels):
                value = -128
                for kernel_y in range(-2, 3):
                    input_y = output_y + kernel_y
                    if input_y < 0 or input_y >= height:
                        continue
                    for kernel_x in range(-2, 3):
                        input_x = output_x + kernel_x
                        if input_x < 0 or input_x >= width:
                            continue
                        index = (input_y * width + input_x) * channels + channel
                        value = max(value, input_values[index])
                expected_values.append(value)
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytes(value & 0xFF for value in expected_values)
    model = _compile_maxpool_model()

    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    maxpool_commands = 0
    layout_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 19:
            assert command_size == 96
            assert struct.unpack_from("<H", model, offset + 16)[0] == 6
            assert struct.unpack_from("<H", model, offset + 24)[0] == 6
            assert struct.unpack_from("<9I", model, offset + 32) == (
                height,
                width,
                channels,
                5,
                5,
                1,
                1,
                2,
                2,
            )
            maxpool_commands += 1
        if command_type == 18:
            layout_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert maxpool_commands == 1
    assert layout_commands == 2

    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, maxpool_output_base, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, maxpool_output_base, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=300000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, maxpool_output_base, len(expected))) == expected
    cocotb.log.info(
        "compiler MaxPool K5/S1/P2 C32 runtime=%s ns PMU cycles=%d",
        elapsed_ns,
        counters["cycle"],
    )


@cocotb.test()
async def test_compiler_generated_yolov8_maxpool_c128_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 10, 10, 128
    output_base = 0x80010000
    input_values = [
        (index * 37 + 11) & 0xFF for index in range(height * width * channels)
    ]
    expected = bytearray()
    for output_y in range(height):
        for output_x in range(width):
            for channel in range(channels):
                value = -128
                for kernel_y in range(-2, 3):
                    input_y = output_y + kernel_y
                    if input_y < 0 or input_y >= height:
                        continue
                    for kernel_x in range(-2, 3):
                        input_x = output_x + kernel_x
                        if input_x < 0 or input_x >= width:
                            continue
                        raw = input_values[
                            (input_y * width + input_x) * channels + channel
                        ]
                        signed = raw if raw < 128 else raw - 256
                        value = max(value, signed)
                expected.append(value & 0xFF)

    model = _compile_yolov8_maxpool_c128_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    maxpool_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 19:
            assert command_size == 96
            assert struct.unpack_from("<9I", model, offset + 32) == (
                height,
                width,
                channels,
                5,
                5,
                1,
                1,
                2,
                2,
            )
            maxpool_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert maxpool_commands == 1

    input_data = bytes(input_values)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, output_base, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=300000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, output_base, len(expected))) == bytes(expected)
    cocotb.log.info(
        "compiler YOLO MaxPool K5/S1/P2 C128 runtime=%s ns PMU cycles=%d",
        elapsed_ns,
        counters["cycle"],
    )


@cocotb.test()
async def test_compiler_generated_yolov8_head_transpose_c144_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 10, 10, 144
    pixels = height * width
    output_base = 0x80010000
    input_data = bytes(
        (index * 37 + 11) & 0xFF for index in range(pixels * channels)
    )
    expected = bytes(
        input_data[pixel * channels + channel]
        for channel in range(channels)
        for pixel in range(pixels)
    )

    model = _compile_yolov8_head_transpose_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    head_pack_commands = 0
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 18 and struct.unpack_from("<H", model, offset + 32)[0] == 5:
            assert command_size == 96
            assert struct.unpack_from("<4I", model, offset + 40) == (
                1,
                height,
                width,
                channels,
            )
            assert struct.unpack_from("<3I", model, offset + 56) == (
                channels,
                pixels * 32,
                pixels,
            )
            head_pack_commands += 1
        offset += command_size
    assert offset == command_offset + command_bytes
    assert head_pack_commands == 1

    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, output_base, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, output_base, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=300000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    status = await _axi_read32(axi_master, NPU_CMD_STATUS)
    fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
    fail_pointer = await _axi_read32(axi_master, NPU_CMD_FAIL_PTR)
    done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
    assert status == NPU_CMD_STATUS_PASS, (
        f"runtime failed: status=0x{status:08x} code=0x{fail_code:08x} "
        f"pointer=0x{fail_pointer:08x} done={done_count}"
    )
    assert fail_code == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert done_count == command_count
    assert bytes(await read_l2_bytes(dut, output_base, len(expected))) == expected
    cocotb.log.info(
        "compiler YOLO C144 head transpose runtime=%s ns PMU cycles=%d",
        elapsed_ns,
        counters["cycle"],
    )


@cocotb.test()
async def test_compiler_generated_concat_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, input_channels = 24, 24, 32
    input_bytes = height * width * input_channels
    output_bytes = input_bytes * 2
    rhs_base = 0x80010000
    concat_output_base = 0x80020000
    lhs = bytes((index * 29 + 7) & 0xFF for index in range(input_bytes))
    rhs = bytes((index * 43 + 19) & 0xFF for index in range(input_bytes))
    expected = bytearray()
    for pixel in range(height * width):
        offset = pixel * input_channels
        expected.extend(lhs[offset : offset + input_channels])
        expected.extend(rhs[offset : offset + input_channels])

    model = _compile_concat_model()
    section_table_offset = struct.unpack_from("<I", model, 24)[0]
    command_offset = struct.unpack_from("<I", model, section_table_offset + 8)[0]
    command_bytes = struct.unpack_from("<I", model, section_table_offset + 12)[0]
    offset = command_offset
    layout_commands = 0
    local_dma_commands = []
    while offset < command_offset + command_bytes:
        command_type, command_size = struct.unpack_from("<HH", model, offset)
        if command_type == 18:
            layout_commands += 1
        if command_type == 2:
            source_region = struct.unpack_from("<H", model, offset + 16)[0]
            destination_region = struct.unpack_from("<H", model, offset + 24)[0]
            destination_offset = struct.unpack_from("<I", model, offset + 28)[0]
            length, direction = struct.unpack_from("<II", model, offset + 32)
            if direction == 2:
                assert source_region == 6
                assert destination_region == 6
                assert length == input_bytes
                local_dma_commands.append(destination_offset)
        offset += command_size
    assert offset == command_offset + command_bytes
    assert layout_commands == 3
    assert len(local_dma_commands) == 2
    assert abs(local_dma_commands[1] - local_dma_commands[0]) == input_bytes

    runtime_bindings = [
        (1, 0, INPUT_BASE, input_bytes),
        (1, 1, rhs_base, input_bytes),
        (2, 0, concat_output_base, output_bytes),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, lhs)
    await write_l2_bytes(dut, rhs_base, rhs)
    await write_l2_bytes(dut, concat_output_base, bytes(output_bytes))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    start_ns = get_sim_time("ns")
    counters = await _load_and_run(
        dut, axi_master, invocation, timeout_cycles=300000, measure_pmu=True
    )
    elapsed_ns = get_sim_time("ns") - start_ns

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, concat_output_base, output_bytes)) == bytes(expected)
    cocotb.log.warning(
        "compiler Concat C32+C32 runtime=%s ns PMU cycles=%d",
        elapsed_ns,
        counters["cycle"],
    )


@cocotb.test()
async def test_compiler_generated_public_reshape_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    tensor_bytes = 4 * 32
    input_data = bytes((index * 29 + 7) & 0xFF for index in range(tensor_bytes))
    model = _compile_public_reshape_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, tensor_bytes),
        (2, 0, OUTPUT_BASE, tensor_bytes),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(
        model, runtime_bindings
    )
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(tensor_bytes))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert command_count == 2
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, tensor_bytes)) == input_data


@cocotb.test()
async def test_compiler_generated_high_c16_slice_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 2, 3, 32
    input_data = bytearray(height * width * channels)
    for pixel in range(height * width):
        input_data[pixel * channels] = 1
    expected = bytes([32] * (height * width * channels))

    model = _compile_high_c16_slice_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, bytes(input_data))
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == expected


@cocotb.test()
async def test_compiler_generated_generic_k3_conv_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels, output_channels = 4, 4, 32, 32
    input_values = [((pixel * 3 + channel) % 9) - 4
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for output_y in range(height):
        for output_x in range(width):
            value = 0
            for kernel_y in range(3):
                input_y = output_y + kernel_y - 1
                for kernel_x in range(3):
                    input_x = output_x + kernel_x - 1
                    if 0 <= input_y < height and 0 <= input_x < width:
                        pixel = input_y * width + input_x
                        value += sum(input_values[pixel * channels:(pixel + 1) * channels])
            value = max(-128, min(127, value)) & 0xFF
            expected.extend([value] * output_channels)

    model = _compile_generic_k3_conv_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_rgb_k3_conv_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels, output_height, output_width = 7, 7, 3, 4, 4
    input_values = [((pixel * 5 + channel) % 11) - 5
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for output_y in range(output_height):
        for output_x in range(output_width):
            value = 0
            for kernel_y in range(3):
                input_y = output_y * 2 + kernel_y - 1
                for kernel_x in range(3):
                    input_x = output_x * 2 + kernel_x - 1
                    if 0 <= input_y < height and 0 <= input_x < width:
                        pixel = input_y * width + input_x
                        value += sum(input_values[pixel * channels:(pixel + 1) * channels])
            value = max(-128, min(127, value)) & 0xFF
            expected.extend([value] * 32)

    model = _compile_rgb_k3_conv_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_generated_depthwise_k3_conv_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels, output_height, output_width = 3, 3, 33, 2, 2
    input_values = [((pixel * 5 + channel * 3) % 13) - 6
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for output_y in range(output_height):
        for output_x in range(output_width):
            channel_values = [0] * channels
            for kernel_y in range(3):
                input_y = output_y * 2 + kernel_y - 1
                for kernel_x in range(3):
                    input_x = output_x * 2 + kernel_x - 1
                    if 0 <= input_y < height and 0 <= input_x < width:
                        pixel = input_y * width + input_x
                        for channel in range(channels):
                            channel_values[channel] += input_values[pixel * channels + channel]
            expected.extend(max(-128, min(127, value)) & 0xFF for value in channel_values)

    model = _compile_depthwise_k3_conv_model()
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_runtime_rejects_invalid_bindings(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    model = build_model()
    invocation, original_addresses = build_invocation(model)
    invalid_records = []
    misaligned = bytearray(original_addresses)
    struct.pack_into("<I", misaligned, 20, OUTPUT_BASE + 4)
    invalid_records.append(misaligned)
    undersized = bytearray(original_addresses)
    struct.pack_into("<I", undersized, 24, 16)
    invalid_records.append(undersized)

    for binding_addresses in invalid_records:
        await reset_dut(dut)
        await write_l2_bytes(dut, MODEL_BASE, model)
        await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
        await write_l2_bytes(dut, INVOCATION_BASE, invocation)
        await _load_and_run(dut, axi_master, invocation)
        assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_FAIL
        assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == NPU_CMD_FAIL_BAD_BINDING
        assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 0


@cocotb.test()
async def test_compiler_runtime_rejects_malformed_linebuf_job(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    qparams = b"".join(
        struct.pack("<iiIiiiII", 0, 1 << 30, 30, 0, -128, 127, 0, 0)
        for _ in range(32)
    )
    commands = b"".join(
        [
            _command_header(5, 32, tile=0) + struct.pack("<4I", 0, 32, 0, 0),
            _linebuf_job(8, 1),  # K3*C32 requires nine K tiles.
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    model = _package(commands, b"", b"", 2, 0x1000, 0, 0, qparams)
    invocation, binding_addresses = build_invocation_with_bindings(model, [])
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_FAIL
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 1
