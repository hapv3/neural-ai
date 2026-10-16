"""Resident multi-group pointwise outputs, streamed LUT tiles and retained L2."""

import struct

import numpy as np

from grouped_linebuffer_checkpoints import validate_grouped_coverage
from yolo_c2f_checkpoints import _MemoryLabels, checkpoint_indices


OUTPUT_ROLES = ("resumed_output_conv", "resumed_output_activation")
_TOKEN_STRIDE = 1 << 32
_PARTIAL_STRIPE_ROWS = 256  # npu_runtime_ops.c runtime_pointwise_c32, not total M.


def plan_resident_pointwise_checkpoints(commands, parent, shapes, local_bytes, l2_bytes):
    """Require fresh RQ/LUT setup, complete outputs and all completed L2 stores."""
    source, output = shapes["parent"], shapes[OUTPUT_ROLES[0]]
    if (any(len(s) != 4 or s[0] != 1 or any(type(v) is not int or v <= 0 for v in s)
            for s in (source, output, shapes[OUTPUT_ROLES[1]])) or source[:3] != output[:3] or
            output != shapes[OUTPUT_ROLES[1]] or not 64 <= source[3] <= 320 or source[3] % 32 or
            not 32 <= output[3] <= 256 or output[3] % 32):
        raise ValueError("Unsupported resident pointwise output shapes")
    rows, input_groups, output_groups = source[1] * source[2], source[3] // 32, output[3] // 32
    if (parent["rows"], parent["channels"], parent.get("group_stride"),
            parent["tile_cols"], parent["row_stride"]) != (rows, source[3], rows * 32, source[2], source[2] * 32):
        raise ValueError("Resident pointwise parent is not complete contiguous C32 groups")
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    tracker.write(6, checkpoint_indices(parent, local_bytes).ravel(), _TOKEN_STRIDE + np.arange(rows * source[3]))
    checks = []
    group_rows = [0] * output_groups
    qparam = None
    qparam_layer = None
    qparam_base = None
    conv_layer = None
    lut_layer = None
    live_lut = None
    base = None
    activation_cells = np.zeros((rows, output[3]), dtype=np.bool_)

    for index, command in enumerate(commands[parent["command"]:], parent["command"] + 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (2, 3, 4, 24, 25, 26):
            tracker.dma(command)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid resident output DMA wait")
            tracker.wait(direction)
        elif kind == 1:
            tracker.wait(0)
            tracker.wait(1)
        elif kind == 5:
            qparam = struct.unpack_from("<3I", command, 16)
            qparam_layer = layer
        elif kind == 10:
            src_region, src_index, src = struct.unpack_from("<HHI", command, 24)
            psum_region, psum_index, psum = struct.unpack_from("<HHI", command, 32)
            dst_region, dst_index, dst = struct.unpack_from("<HHI", command, 40)
            count, inputs, outputs, group, input_stride, output_stride = struct.unpack_from("<6I", command, 48)
            if conv_layer is None:
                conv_layer = layer
            if (flags or live_lut is not None or layer != conv_layer or layer != qparam_layer or
                    (src_region, src_index, psum_region, psum_index, dst_region, dst_index) != (6, 0, 6, 0, 6, 0) or
                    inputs != input_groups or outputs != 1 or group >= output_groups or not count or
                    (input_stride, output_stride) != (rows * 32, rows * 32) or
                    any(address % 32 for address in (src, psum, dst)) or
                    src < parent["address"] or (src - parent["address"]) % 32):
                raise ValueError("Resident pointwise requires fresh matching RQ and supported C32 geometry")
            if qparam_base is None:
                qparam_base = qparam[0] - group * 32
            if qparam != (qparam_base + group * 32, 32, group):
                raise ValueError("Stale resident pointwise RQ selector")
            first = (src - parent["address"]) // 32
            if first != group_rows[group] or first + count > rows:
                raise ValueError("Duplicate/out-of-range resident pointwise rows")
            indices = checkpoint_indices({**parent, "address": src, "rows": count}, local_bytes)
            if not np.array_equal(tracker.read(6, indices.ravel()),
                                  _TOKEN_STRIDE + np.arange(first * source[3], (first + count) * source[3])):
                raise ValueError("Stale resident pointwise parent input")
            if base is None:
                base = dst - group * rows * 32 - first * 32
            if dst != base + group * rows * 32 + first * 32:
                raise ValueError("Noncontiguous resident pointwise output groups")
            partial_bytes = min(count, _PARTIAL_STRIPE_ROWS) * 128
            for address, length in ((psum, partial_bytes), (dst, count * 32)):
                if (address < parent["address"] + input_groups * rows * 32 and
                        parent["address"] < address + length):
                    raise ValueError("Resident pointwise workspace overlaps parent")
            if psum < base + output_groups * rows * 32 and base < psum + partial_bytes:
                raise ValueError("Resident pointwise partial overlaps output groups")
            tracker.write(6, np.arange(psum, psum + partial_bytes), -1)
            check = dict(command=index, tensor_role=OUTPUT_ROLES[0], address=dst, rows=count, channels=32,
                         tile_cols=output[2], row_stride=output[2] * 32,
                         reference_first_row=first, reference_first_channel=group * 32)
            row = np.arange(first, first + count)[:, None]
            channel = np.arange(group * 32, (group + 1) * 32)[None, :]
            tracker.write(6, checkpoint_indices(check, local_bytes).ravel(),
                          (2 * _TOKEN_STRIDE + row * output[3] + channel).ravel())
            group_rows[group] += count
            checks.append(check)
        elif kind == 12:
            src_region, src_index, src = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, dst = struct.unpack_from("<HHI", command, 24)
            lut = struct.unpack_from("<HHI", command, 32)
            length = struct.unpack_from("<I", command, 40)[0]
            if lut_layer is None:
                lut_layer = layer
            if (group_rows != [rows] * output_groups or flags & ~4 or layer != lut_layer or
                    (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or
                    lut[:2] != (1, 0) or not length or length % 32 or src % 32 or dst % 32 or
                    src < base or src + length > base + rows * output_groups * 32 or
                    (flags & 4 and lut != live_lut) or (live_lut is not None and lut != live_lut)):
                raise ValueError("Resident output LUT requires completed groups and explicit first configuration")
            live_lut = lut
            spans, expected = [], []
            position = (src - base) // 32
            remaining = length // 32
            while remaining:
                group, first = divmod(position, rows)
                count = min(remaining, rows - first)
                if activation_cells[first:first + count, group * 32:(group + 1) * 32].any():
                    raise ValueError("Duplicate resident output activation cells")
                activation_cells[first:first + count, group * 32:(group + 1) * 32] = True
                spans.append(dict(rows=count, channels=32, reference_first_row=first,
                                  reference_first_channel=group * 32))
                expected.append((np.arange(first, first + count)[:, None] * output[3] +
                                 np.arange(group * 32, (group + 1) * 32)[None, :]).ravel())
                position += count
                remaining -= count
            values = np.concatenate(expected)
            if not np.array_equal(tracker.read(6, np.arange(src, src + length)), 2 * _TOKEN_STRIDE + values):
                raise ValueError("Stale resident output LUT source bytes")
            tracker.write(6, np.arange(dst, dst + length), 3 * _TOKEN_STRIDE + values)
            check = dict(command=index, tensor_role=OUTPUT_ROLES[1], address=dst, rows=length // 32, channels=32,
                         tile_cols=output[2], row_stride=output[2] * 32,
                         reference_first_row=spans[0]["reference_first_row"],
                         reference_first_channel=spans[0]["reference_first_channel"])
            if len(spans) > 1:
                check["reference_spans"] = spans
            checks.append(check)
        else:
            raise ValueError(f"Unsupported command {kind} before retained resident output")
        if activation_cells.all():
            for address in np.flatnonzero(tracker.memory[5] == 3 * _TOKEN_STRIDE):
                retained = dict(address=int(address), rows=rows, channels=output[3], tile_cols=output[2],
                                row_stride=output[2] * 32, group_stride=rows * 32,
                                reference_first_row=0, reference_first_channel=0)
                try:
                    indices = checkpoint_indices(retained, l2_bytes)
                except ValueError:
                    continue
                if np.array_equal(tracker.memory[5][indices].ravel(),
                                  3 * _TOKEN_STRIDE + np.arange(rows * output[3])):
                    tracker.read(5, indices.ravel())
                    validate_grouped_coverage(checks, shapes, OUTPUT_ROLES)
                    return checks, index, retained
    raise ValueError("Incomplete resident pointwise/LUT/L2 output checkpoints")
