"""Bounded multi-output-group linebuffer and whole-tensor LUT checkpoints."""

import struct

import numpy as np

from yolo_c2f_checkpoints import _MemoryLabels, checkpoint_indices


DOWNSAMPLE_ROLES = ("downsample_conv", "downsample_activation")
RESUMED_POINTWISE_ROLES = ("resumed_pointwise_conv", "resumed_pointwise_activation")
_TOKEN_STRIDE = 1 << 32


def reference_rectangles(check):
    """Describe logical rectangles in physical sample order, including wrap."""
    if "reference_spans" not in check:
        return [check]
    spans = check["reference_spans"]
    keys = {"rows", "channels", "reference_first_row", "reference_first_channel"}
    if (not isinstance(spans, list) or not spans or len(spans) > 32 or
            any(not isinstance(s, dict) or set(s) != keys or
                any(type(v) is not int for v in s.values()) or
                s["rows"] <= 0 or s["channels"] != check["channels"] for s in spans) or
            sum(s["rows"] for s in spans) != check["rows"] or
            any(check.get(k, 0) != spans[0][k] for k in ("reference_first_row", "reference_first_channel"))):
        raise ValueError("Invalid reference spans for physical checkpoint")
    return spans


def reference_tensor_slice(tensor, check):
    """Slice NHWC oracles in physical sample order, without flattening C."""
    channels = tensor.shape[-1]
    tensor = tensor.reshape(-1, channels)
    pieces = []
    for rectangle in reference_rectangles(check):
        row, column = rectangle["reference_first_row"], rectangle.get("reference_first_channel", 0)
        rows, cols = rectangle["rows"], rectangle["channels"]
        if (any(type(v) is not int for v in (row, column, rows, cols)) or min(row, column) < 0 or
                min(rows, cols) <= 0 or row + rows > tensor.shape[0] or column + cols > channels):
            raise ValueError("Reference checkpoint rectangle exceeds tensor")
        pieces.append(tensor[row:row + rows, column:column + cols])
    return np.concatenate(pieces, axis=0).copy()


def validate_grouped_coverage(checks, shapes, roles=DOWNSAMPLE_ROLES):
    coverage = {}
    for role in roles:
        shape = shapes[role]
        if len(shape) != 4 or shape[0] != 1 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Invalid grouped checkpoint tensor shape")
        mask = np.zeros((shape[1] * shape[2], shape[3]), dtype=np.bool_)
        rectangles = (r for c in checks if c["tensor_role"] == role for r in reference_rectangles(c))
        for check in rectangles:
            row, col = check["reference_first_row"], check.get("reference_first_channel", 0)
            rows, channels = check["rows"], check["channels"]
            if (any(type(v) is not int for v in (row, col, rows, channels)) or
                    min(row, col) < 0 or min(rows, channels) <= 0 or
                    row + rows > mask.shape[0] or col + channels > mask.shape[1] or
                    mask[row:row + rows, col:col + channels].any()):
                raise ValueError("Overlapping/out-of-range grouped checkpoint cells")
            mask[row:row + rows, col:col + channels] = True
        if not mask.all():
            raise ValueError("Incomplete grouped checkpoint coverage")
        coverage[role] = dict(rows=mask.shape[0], channels=mask.shape[1], bytes=int(mask.size))
    return coverage


def plan_downsample_checkpoints(commands, parent_checks, shapes, local_bytes, l2_bytes):
    """Map the next K3/S2 Conv using explicit RQ-load output-group selectors.

    Each group has independent raster rows; all groups must finish before a
    whole-tensor LUT can read them. Numerical oracles are source-runtime data.
    """
    parent = shapes["block_output_activation"]
    output = shapes["downsample_conv"]
    if (any(len(s) != 4 or s[0] != 1 or any(type(v) is not int or v <= 0 for v in s)
            for s in (parent, output, shapes["downsample_activation"])) or
            output != shapes["downsample_activation"] or not 32 < output[3] <= 64 or
            parent[3] != 32 or output[1:3] != [(parent[1] + 1) // 2, (parent[2] + 1) // 2]):
        raise ValueError("Unsupported grouped downsample tensor shapes")
    seeds = [c for c in parent_checks if c["tensor_role"] == "block_output_activation"]
    if not seeds:
        raise ValueError("Grouped downsample has no verified parent")
    begin = max(c["command"] for c in seeds)
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    for check in seeds:
        indices = checkpoint_indices(check, local_bytes)
        first = check["reference_first_row"] * parent[3]
        tracker.write(6, indices.ravel(), _TOKEN_STRIDE + np.arange(first, first + indices.size))
    rows = output[1] * output[2]
    groups = (output[3] + 31) // 32
    group_rows = [0] * groups
    qparams = {}
    qparam_base = None
    conv_layer = None
    base = None
    pending = None
    checks = []

    def complete(index, check, submit=None):
        indices = checkpoint_indices(check, local_bytes)
        row = np.arange(check["reference_first_row"], check["reference_first_row"] + check["rows"])[:, None]
        channel = np.arange(check["reference_first_channel"],
                            check["reference_first_channel"] + check["channels"])[None, :]
        tracker.write(6, indices.ravel(), (2 * _TOKEN_STRIDE + row * output[3] + channel).ravel())
        checks.append({**check, "command": index, **({"submit_command": submit} if submit is not None else {})})

    for index, command in enumerate(commands, 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind == 5:
            qparams[layer] = struct.unpack_from("<3I", command, 16)
        if index <= begin:
            continue
        if kind in (2, 3, 4, 24, 25, 26):
            tracker.dma(command)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid grouped downsample DMA wait")
            tracker.wait(direction)
        elif kind == 1:
            tracker.wait(0)
            tracker.wait(1)
        elif kind in (9, 29):
            if conv_layer is None:
                conv_layer = layer
            address, count, accum, pitch, cols = struct.unpack_from("<5I", command, 108)
            if (layer != conv_layer or pending is not None or flags or accum or not count or
                    cols != output[2] or pitch != cols * 32 or
                    struct.unpack_from("<2H", command, 28) != (2, 2) or layer not in qparams):
                raise ValueError("Unsupported grouped downsample linebuffer job")
            first_qparam, qparam_count, group = qparams[layer]
            if group >= groups or qparam_count != min(32, output[3] - group * 32):
                raise ValueError("Invalid grouped downsample RQ selector")
            if qparam_base is None:
                qparam_base = first_qparam - group * 32
            if first_qparam != qparam_base + group * 32:
                raise ValueError("Stale grouped downsample RQ group")
            first_row = group_rows[group]
            if base is None:
                base = address - group * rows * 32 - first_row * 32
            if count + first_row > rows or address != base + group * rows * 32 + first_row * 32:
                raise ValueError("Grouped downsample output rows are not contiguous")
            tracker.write(6, np.arange(address, address + count * 32), -1)
            check = dict(tensor_role="downsample_conv", address=address, rows=count,
                         channels=min(32, output[3] - group * 32), tile_cols=cols, row_stride=pitch,
                         reference_first_row=first_row, reference_first_channel=group * 32)
            group_rows[group] += count
            if kind == 29:
                tracker.compute = np.arange(address, address + count * 32)
                pending = (check, index)
            else:
                complete(index, check)
        elif kind == 30:
            if pending is None:
                raise ValueError("Grouped downsample wait has no pending Conv")
            tracker.compute = None
            check, submit = pending
            complete(index, check, submit)
            pending = None
        elif kind == 12:
            src_region, src_index, source = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if (pending is not None or group_rows != [rows] * groups or flags & ~4 or
                    (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or
                    source != base or length != rows * groups * 32):
                raise ValueError("Grouped downsample LUT requires all completed output groups")
            shape = dict(rows=rows, channels=output[3], tile_cols=output[2], row_stride=output[2] * 32,
                         group_stride=rows * 32, reference_first_row=0, reference_first_channel=0)
            src_indices = checkpoint_indices({**shape, "address": source}, local_bytes)
            if not np.array_equal(tracker.read(6, src_indices.ravel()),
                                  2 * _TOKEN_STRIDE + np.arange(rows * output[3])):
                raise ValueError("Missing/stale grouped downsample LUT input")
            tracker.write(6, np.arange(destination, destination + length), -1)
            check = dict(command=index, tensor_role="downsample_activation", address=destination, **shape)
            checks.append(check)
            validate_grouped_coverage(checks, shapes)
            return checks, index
        elif kind != 5:
            raise ValueError(f"Unsupported command {kind} in grouped downsample mapping")
    raise ValueError("Incomplete grouped downsample checkpoints")


def plan_grouped_pointwise_checkpoints(commands, parent, shapes, local_bytes, l2_bytes):
    """Resume at explicit RQ loads, then verify resident pointwise groups/LUT."""
    source_shape = shapes["downsample_activation"]
    output = shapes["resumed_pointwise_conv"]
    if (any(len(s) != 4 or s[0] != 1 or any(type(v) is not int or v <= 0 for v in s)
            for s in (source_shape, output, shapes["resumed_pointwise_activation"])) or
            source_shape[:3] != output[:3] or output != shapes["resumed_pointwise_activation"] or
            not 32 < source_shape[3] <= 256 or not 32 < output[3] <= 256 or
            parent["rows"] != source_shape[1] * source_shape[2] or parent["channels"] != source_shape[3] or
            parent.get("group_stride") != parent["rows"] * 32 or
            (parent["tile_cols"], parent["row_stride"]) != (source_shape[2], source_shape[2] * 32)):
        raise ValueError("Unsupported resumed grouped pointwise shapes")
    rows = parent["rows"]
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    indices = checkpoint_indices(parent, local_bytes)
    tracker.write(6, indices.ravel(), _TOKEN_STRIDE + np.arange(indices.size))
    qparams = {}
    qparam_layer = None
    group_rows = [0] * ((output[3] + 31) // 32)
    checks = []
    layer_id = None
    base = None
    qparam_base = None
    for index, command in enumerate(commands[parent["command"]:], parent["command"] + 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind == 5:
            qparams[layer] = struct.unpack_from("<3I", command, 16)
            qparam_layer = layer
        elif kind in (2, 3, 4, 24, 25, 26):
            tracker.dma(command)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid resumed grouped pointwise DMA wait")
            tracker.wait(direction)
        elif kind == 10:
            if layer_id is None:
                layer_id = layer
            src_region, src_index, source = struct.unpack_from("<HHI", command, 24)
            partial_region, partial_index, partial = struct.unpack_from("<HHI", command, 32)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 40)
            count, input_groups, output_groups, group, input_stride, output_stride = struct.unpack_from("<6I", command, 48)
            if (layer != layer_id or flags or layer != qparam_layer or
                    (src_region, src_index, partial_region, partial_index, dst_region, dst_index) != (6, 0, 6, 0, 6, 0) or
                    input_groups != (source_shape[3] + 31) // 32 or output_groups != 1 or
                    group >= len(group_rows) or input_stride != rows * 32 or output_stride != rows * 32 or
                    source < parent["address"] or (source - parent["address"]) % 32 or not count or
                    any(address % 32 for address in (source, partial, destination))):
                raise ValueError("Resumed grouped pointwise needs explicit matching RQ loads and C32 layout")
            first_qparam, qparam_count, selected = qparams[layer]
            if qparam_base is None:
                qparam_base = first_qparam - group * 32
            if (selected != group or qparam_count != min(32, output[3] - group * 32) or
                    first_qparam != qparam_base + group * 32):
                raise ValueError("Stale resumed grouped pointwise RQ selector")
            first_row = (source - parent["address"]) // 32
            if first_row != group_rows[group] or count + first_row > rows:
                raise ValueError("Duplicate/out-of-range resumed pointwise rows")
            source_indices = checkpoint_indices({**parent, "address": source, "rows": count}, local_bytes)
            if not np.array_equal(tracker.read(6, source_indices.ravel()), _TOKEN_STRIDE +
                    np.arange(first_row * source_shape[3], (first_row + count) * source_shape[3])):
                raise ValueError("Stale resumed grouped pointwise input")
            if base is None:
                base = destination - group * rows * 32 - first_row * 32
            if destination != base + group * rows * 32 + first_row * 32:
                raise ValueError("Noncontiguous resumed grouped pointwise output")
            partial_bytes = min(count, 256) * 128  # Runtime reuses one stripe's pSUM storage.
            for address, length in ((partial, partial_bytes), (destination, count * 32)):
                if (address < parent["address"] + input_groups * rows * 32 and
                        parent["address"] < address + length):
                    raise ValueError("Resumed grouped pointwise workspace overlaps parent")
                tracker.write(6, np.arange(address, address + length), -1)
            if partial < base + len(group_rows) * rows * 32 and base < partial + partial_bytes:
                raise ValueError("Resumed grouped pointwise partial overlaps output groups")
            channels = min(32, output[3] - group * 32)
            check = dict(command=index, tensor_role="resumed_pointwise_conv", address=destination,
                         rows=count, channels=channels, tile_cols=output[2], row_stride=output[2] * 32,
                         reference_first_row=first_row, reference_first_channel=group * 32)
            output_indices = checkpoint_indices(check, local_bytes)
            row = np.arange(first_row, first_row + count)[:, None]
            channel = np.arange(group * 32, group * 32 + channels)[None, :]
            tracker.write(6, output_indices.ravel(), (2 * _TOKEN_STRIDE + row * output[3] + channel).ravel())
            group_rows[group] += count
            checks.append(check)
        elif kind == 12:
            src_region, src_index, source = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if (group_rows != [rows] * len(group_rows) or flags or
                    (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or
                    source != base or length != rows * len(group_rows) * 32):
                raise ValueError("Resumed grouped pointwise LUT must explicitly load and consume all groups")
            shape = dict(rows=rows, channels=output[3], tile_cols=output[2], row_stride=output[2] * 32,
                         group_stride=rows * 32, reference_first_row=0, reference_first_channel=0)
            source_indices = checkpoint_indices({**shape, "address": source}, local_bytes)
            if not np.array_equal(tracker.read(6, source_indices.ravel()),
                                  2 * _TOKEN_STRIDE + np.arange(rows * output[3])):
                raise ValueError("Stale resumed grouped pointwise LUT input")
            tracker.write(6, np.arange(destination, destination + length), -1)
            checks.append(dict(command=index, tensor_role="resumed_pointwise_activation", address=destination, **shape))
            validate_grouped_coverage(checks, shapes, RESUMED_POINTWISE_ROLES)
            return checks, index
        else:
            raise ValueError(f"Unsupported command {kind} before resumed pointwise completion")
    raise ValueError("Incomplete resumed grouped pointwise checkpoints")
