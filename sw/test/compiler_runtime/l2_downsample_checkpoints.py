"""Grouped K3/S2 linebuffer accumulation from a verified native-C32 L2 tensor."""

import struct

import numpy as np

from grouped_linebuffer_checkpoints import validate_grouped_coverage
from yolo_c2f_checkpoints import _MemoryLabels, checkpoint_indices


L2_DOWNSAMPLE_ROLES = ("l2_downsample_conv", "l2_downsample_activation")
_TOKEN_STRIDE = 1 << 32


def resolve_l2_downsample_source(interpreter, parent_id):
    """Read explicit source padding; do not guess SAME/VALID padding offsets."""
    ops = [op for op in interpreter._get_ops_details() if op["op_name"] != "DELEGATE"]
    pads = [op for op in ops if op["op_name"] == "PAD" and int(op["inputs"][0]) == parent_id]
    if len(pads) != 1:
        raise ValueError("L2 downsample requires one explicit source PAD")
    pad = pads[0]
    padding = interpreter.get_tensor(int(pad["inputs"][1]))
    if (padding.shape != (4, 2) or not np.issubdtype(padding.dtype, np.integer) or
            padding.min() < 0 or padding[[0, 3]].any()):
        raise ValueError("Unsupported L2 downsample source padding")
    convs = [op for op in ops if op["op_name"] == "CONV_2D" and
             int(op["inputs"][0]) == int(pad["outputs"][0])]
    if len(convs) != 1:
        raise ValueError("L2 downsample requires one padded Conv consumer")
    conv = convs[0]
    weights = interpreter.get_tensor(int(conv["inputs"][1]))
    if weights.ndim != 4 or weights.shape[1:3] != (3, 3):
        raise ValueError("L2 downsample source is not K3")
    return int(conv["outputs"][0]), list(map(int, padding[1:3].ravel()))


def plan_l2_downsample_checkpoints(commands, parent, ready, replay_first, shapes, padding, local_bytes, l2_bytes):
    """Map final requant outputs only after every input-group partial is joined.

    Setup is scanned only inside the replayed prefix. Tensor labels verify each
    staged non-padding input pixel, pSUM continuity and complete group coverage;
    independent numerical oracles verify the actual Conv/LUT results.
    """
    source, output = shapes["parent"], shapes[L2_DOWNSAMPLE_ROLES[0]]
    if (any(len(s) != 4 or s[0] != 1 or any(type(v) is not int or v <= 0 for v in s)
            for s in (source, output, shapes[L2_DOWNSAMPLE_ROLES[1]])) or
            output != shapes[L2_DOWNSAMPLE_ROLES[1]] or
            not 64 <= source[3] <= 256 or source[3] % 32 or not 32 <= output[3] <= 256 or output[3] % 32 or
            len(padding) != 4 or any(type(v) is not int or v < 0 for v in padding) or
            output[1:3] != [(source[1] + padding[0] + padding[1] - 3) // 2 + 1,
                            (source[2] + padding[2] + padding[3] - 3) // 2 + 1]):
        raise ValueError("Unsupported L2 downsample shapes/padding")
    source_rows, rows = source[1] * source[2], output[1] * output[2]
    inputs, groups = source[3] // 32, output[3] // 32
    if (not 0 <= replay_first < ready <= len(commands) or
            (parent["rows"], parent["channels"], parent["group_stride"], parent["row_stride"], parent["tile_cols"]) !=
            (source_rows, source[3], source_rows * 32, source[2] * 32, source[2])):
        raise ValueError("L2 downsample requires a complete drained parent")
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    parent_indices = checkpoint_indices(parent, l2_bytes)
    parent_labels = _TOKEN_STRIDE + np.arange(source_rows * source[3])
    tracker.write(5, parent_indices.ravel(), parent_labels)
    qparam = None
    qparam_layer = None
    for command in commands[replay_first:ready]:
        if struct.unpack_from("<H", command)[0] == 5:
            qparam_layer = struct.unpack_from("<I", command, 8)[0]
            qparam = struct.unpack_from("<3I", command, 16)
    qparam_base = None
    layer_id = None
    group_rows = [0] * groups
    base = None
    active = None
    checks = []

    for index, command in enumerate(commands[ready:], ready + 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (2, 3, 4, 24, 25, 26):
            tracker.dma(command)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid L2 downsample DMA wait")
            tracker.wait(direction)
        elif kind == 1:
            tracker.wait(0)
            tracker.wait(1)
        elif kind == 5:
            qparam = struct.unpack_from("<3I", command, 16)
            qparam_layer = layer
        elif kind == 9:
            frame, height, width, channels, cols, sh, sw, ph, pw = struct.unpack_from("<I8H", command, 16)
            pitch, pixel_stride, ow_step, oh_step = struct.unpack_from("<4I", command, 36)
            kh, kw = struct.unpack_from("<2H", command, 52)
            weight, _, psum, address, count, mode, output_pitch, output_cols, psum_pitch = struct.unpack_from(
                "<9I", command, 96)
            if layer_id is None:
                layer_id = layer
            if (flags or layer != layer_id or layer != qparam_layer or qparam is None or
                    qparam[1] != 32 or qparam[2] >= groups or not count or count % output[2] or
                    (channels, cols, sh, sw, ph, pw, kh, kw) != (32, output[2], 2, 2, 0, 0, 3, 3) or
                    (pixel_stride, ow_step, oh_step) != (32, 64, pitch * 2) or
                    (output_pitch, output_cols, psum_pitch) != (output[2] * 32, output[2], output[2] * 128) or
                    height != (count // output[2] - 1) * 2 + 3 or width != (output[2] - 1) * 2 + 3 or
                    struct.unpack_from("<2H", command, 56) != (0, 0) or
                    struct.unpack_from("<H", command, 72)[0] != 32 or
                    struct.unpack_from("<2I", command, 80) != (9, count) or
                    pitch < width * 32 or any(v % 32 for v in (frame, weight, psum, address))):
                raise ValueError("Unsupported L2 downsample geometry or missing replayed RQ setup")
            group = qparam[2]
            if qparam_base is None:
                qparam_base = qparam[0] - group * 32
            if qparam[0] != qparam_base + group * 32:
                raise ValueError("Stale L2 downsample RQ group")
            first = group_rows[group]
            if first + count > rows:
                raise ValueError("Duplicate/out-of-range L2 downsample rows")
            if base is None:
                base = address - group * rows * 32 - first * 32
            if address != base + group * rows * 32 + first * 32:
                raise ValueError("Noncontiguous L2 downsample output rows/groups")
            input_group = 0 if active is None else active[-1]
            expected_mode = 1 if input_group == 0 else 2 if input_group == inputs - 1 else 3
            stripe = (group, first, count, psum, address, weight - input_group * 9 * 32 * 32)
            if mode != expected_mode or (active is not None and active[:-1] != stripe):
                raise ValueError("Missing/reordered L2 downsample partial accumulation group")
            yy = np.arange(height) + first // output[2] * 2 - padding[0]
            xx = np.arange(width) - padding[2]
            valid = (yy[:, None] >= 0) & (yy[:, None] < source[1]) & (xx[None, :] >= 0) & (xx[None, :] < source[2])
            frame_indices = (frame + np.arange(height)[:, None, None] * pitch +
                             np.arange(width)[None, :, None] * 32 + np.arange(32)[None, None, :])
            frame_labels = tracker.read(6, frame_indices.ravel()).reshape(height, width, 32)
            logical_rows = yy[:, None] * source[2] + xx[None, :]
            expected = (_TOKEN_STRIDE + logical_rows[valid, None] * source[3] +
                        input_group * 32 + np.arange(32)[None, :])
            if not np.array_equal(frame_labels[valid], expected):
                raise ValueError("L2 downsample staged input pixels/channels are stale or misaligned")
            psum_indices = np.arange(psum, psum + count * 128)
            carry = 4 * _TOKEN_STRIDE + (group * rows + first) * 128 + np.arange(count * 128)
            if active is not None and not np.array_equal(tracker.read(6, psum_indices), carry):
                raise ValueError("L2 downsample partial sums were overwritten")
            frame_end = frame + (height - 1) * pitch + width * 32
            if (max(psum, frame) < min(psum + count * 128, frame_end) or
                    max(address, frame) < min(address + count * 32, frame_end) or
                    max(psum, base) < min(psum + count * 128, base + groups * rows * 32)):
                raise ValueError("L2 downsample workspace overlaps live input/output groups")
            if input_group + 1 < inputs:
                tracker.write(6, psum_indices, carry)
                active = (*stripe, input_group + 1)
            else:
                check = dict(command=index, tensor_role=L2_DOWNSAMPLE_ROLES[0], address=address,
                             rows=count, channels=32, tile_cols=output[2], row_stride=output[2] * 32,
                             reference_first_row=first, reference_first_channel=group * 32)
                logical = (np.arange(first, first + count)[:, None] * output[3] +
                           np.arange(group * 32, (group + 1) * 32)[None, :])
                tracker.write(6, checkpoint_indices(check, local_bytes).ravel(), (2 * _TOKEN_STRIDE + logical).ravel())
                checks.append(check)
                group_rows[group] += count
                active = None
        elif kind == 12:
            src_region, src_index, src = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, dst = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if (flags or active is not None or group_rows != [rows] * groups or
                    (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or
                    src != base or length != rows * output[3]):
                raise ValueError("L2 downsample LUT requires every finalized group and fresh configuration")
            geometry = dict(rows=rows, channels=output[3], tile_cols=output[2], row_stride=output[2] * 32,
                            group_stride=rows * 32, reference_first_row=0, reference_first_channel=0)
            source_indices = checkpoint_indices({**geometry, "address": src}, local_bytes)
            if not np.array_equal(tracker.read(6, source_indices.ravel()),
                                  2 * _TOKEN_STRIDE + np.arange(rows * output[3])):
                raise ValueError("Stale L2 downsample LUT source")
            tracker.write(6, checkpoint_indices({**geometry, "address": dst}, local_bytes).ravel(),
                          3 * _TOKEN_STRIDE + np.arange(rows * output[3]))
            checks.append(dict(command=index, tensor_role=L2_DOWNSAMPLE_ROLES[1], address=dst, **geometry))
            if not np.array_equal(tracker.read(5, parent_indices.ravel()), parent_labels):
                raise ValueError("L2 downsample overwrote its retained L2 parent")
            validate_grouped_coverage(checks, shapes, L2_DOWNSAMPLE_ROLES)
            return checks, index
        else:
            raise ValueError(f"Unsupported command {kind} before complete L2 downsample")
    raise ValueError("Incomplete L2 downsample checkpoints")
