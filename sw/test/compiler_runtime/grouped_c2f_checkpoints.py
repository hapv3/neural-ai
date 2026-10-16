"""Complete multi-input/output-group C2f chains with external pSUM provenance."""

import struct

import numpy as np

from grouped_linebuffer_checkpoints import validate_grouped_coverage
from resumed_c2f_checkpoints import CONCAT_ROLE, PARENT_ROLE, chain_roles
from yolo_c2f_checkpoints import _MemoryLabels, checkpoint_indices


_TOKEN_STRIDE = 1 << 32


def native_values(matrix):
    """Serialize logical R,C labels in native group-major C32 memory order."""
    return matrix.reshape(matrix.shape[0], -1, 32).transpose(1, 0, 2).ravel()


def plan_grouped_c2f_checkpoints(commands, parent, shapes, layout, local_bytes, l2_bytes):
    """Require all input-group partials, independent outputs and full concat."""
    count = layout["blocks"]
    source = shapes[PARENT_ROLE]
    if (type(count) is not int or not 1 <= count <= 8 or len(source) != 4 or source[0] != 1 or
            any(type(v) is not int or v <= 0 for v in source) or source[3] not in (128, 256)):
        raise ValueError("Unsupported grouped C2f source layout")
    channels = source[3] // 2
    groups = channels // 32
    rows = source[1] * source[2]
    roles = chain_roles(count)
    if (any(shapes[r] != source[:3] + [channels] for r in roles[:-1]) or
            shapes[CONCAT_ROLE] != source[:3] + [channels * (count + 2)] or
            sorted((layout["branch_start"], layout["bypass_start"])) != [0, channels] or
            sorted(layout["concat_order"]) != sorted(["branch", "bypass"] + [f"residual_{b}" for b in range(count)]) or
            (parent["rows"], parent["channels"], parent.get("group_stride"),
             parent["tile_cols"], parent["row_stride"]) !=
            (rows, source[3], rows * 32, source[2], source[2] * 32)):
        raise ValueError("Unsupported grouped C2f tensor/slice layout")
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    bases = {role: (i + 1) * _TOKEN_STRIDE for i, role in enumerate((PARENT_ROLE,) + roles)}
    values = {r: bases[r] + np.arange(rows * channels).reshape(rows, channels) for r in roles[:-1]}
    parent_labels = bases[PARENT_ROLE] + np.arange(rows * source[3]).reshape(rows, source[3])
    tracker.write(6, checkpoint_indices(parent, local_bytes).ravel(), parent_labels.ravel())
    labels = {name: parent_labels[:, layout[f"{name}_start"]:layout[f"{name}_start"] + channels]
              for name in ("branch", "bypass")}
    labels.update({f"residual_{b}": values[f"resumed_c2f_{b}_residual"] for b in range(count)})
    conv_roles = [r for r in roles if r.endswith(("conv1", "conv2"))]
    conv_sources = {}
    for block in range(count):
        conv_sources[f"resumed_c2f_{block}_conv1"] = labels["branch" if block == 0 else f"residual_{block - 1}"]
        conv_sources[f"resumed_c2f_{block}_conv2"] = values[f"resumed_c2f_{block}_activation1"]
    suffix = commands[parent["command"]:]
    layers = list(dict.fromkeys(struct.unpack_from("<I", c, 8)[0]
                               for c in suffix if struct.unpack_from("<H", c)[0] in (9, 29)))[:2 * count]
    if len(layers) != len(conv_roles):
        raise ValueError("Grouped C2f is missing Conv layers")
    layer_roles = dict(zip(layers, conv_roles))
    pointwise = next((c for c in suffix if struct.unpack_from("<H", c)[0] == 10), None)
    if (pointwise is None or struct.unpack_from("<HH", pointwise, 24) != (6, 0) or
            struct.unpack_from("<I", pointwise, 52)[0] != groups * (count + 2) or
            struct.unpack_from("<I", pointwise, 64)[0] != rows * 32):
        raise ValueError("Grouped C2f concat has no matching pointwise consumer")
    concat = dict(address=struct.unpack_from("<I", pointwise, 28)[0], rows=rows,
                  channels=channels * (count + 2), tile_cols=source[2], row_stride=source[2] * 32,
                  group_stride=rows * 32, reference_first_row=0, reference_first_channel=0)
    concat_indices = checkpoint_indices(concat, local_bytes)
    concat_labels = np.concatenate([labels[r] for r in layout["concat_order"]], axis=1)
    covered = {r: np.zeros((rows, channels), np.bool_) for r in roles[:-1]}
    progress = {r: [0] * groups for r in conv_roles}
    output_bases = {}
    rq_bases = {}
    qparam = None
    qparam_layer = None
    active = None
    live_lut = struct.unpack_from("<HHI", commands[parent["command"] - 1], 32)
    checks = []

    def geometry(address, first=0, size=rows, column=0, width=channels):
        check = dict(address=address, rows=size, channels=width, tile_cols=source[2], row_stride=source[2] * 32,
                     reference_first_row=first, reference_first_channel=column)
        if width > 32:
            check["group_stride"] = rows * 32
        return check

    def complete(index, role, check):
        first, column = check["reference_first_row"], check["reference_first_channel"]
        size, width = check["rows"], check["channels"]
        region = covered[role][first:first + size, column:column + width]
        if (min(first, column) < 0 or min(size, width) <= 0 or first + size > rows or column + width > channels or
                region.any()):
            raise ValueError("Duplicate/out-of-range grouped C2f cells")
        region[:] = True
        tracker.write(6, checkpoint_indices(check, local_bytes).ravel(),
                      values[role][first:first + size, column:column + width].ravel())
        checks.append(dict(command=index, tensor_role=role, **check))

    for index, command in enumerate(suffix, parent["command"] + 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (2, 3, 4, 24, 25, 26):
            tracker.dma(command)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid grouped C2f DMA wait")
            tracker.wait(direction)
        elif kind == 1:
            tracker.wait(0)
            tracker.wait(1)
        elif kind == 5:
            qparam = struct.unpack_from("<3I", command, 16)
            qparam_layer = layer
        elif kind == 9:
            frame, height, width, input_channels, cols, sh, sw, ph, pw = struct.unpack_from("<I8H", command, 16)
            pitch, pixel, ow_step, oh_step = struct.unpack_from("<4I", command, 36)
            weight, _, psum, address, size, mode, output_pitch, output_cols, psum_pitch = struct.unpack_from(
                "<9I", command, 96)
            if (flags or layer not in layer_roles or layer != qparam_layer or qparam is None or
                    qparam[1] != 32 or qparam[2] >= groups or not size or size % source[2] or
                    (input_channels, cols, sh, sw, ph, pw) != (32, source[2], 1, 1, 0, 0) or
                    (pixel, ow_step, oh_step) != (32, 32, pitch) or height != size // source[2] + 2 or
                    width != source[2] + 2 or pitch < width * 32 or
                    (output_pitch, output_cols, psum_pitch) != (source[2] * 32, source[2], source[2] * 128) or
                    struct.unpack_from("<4H", command, 52) != (3, 3, 0, 0) or
                    struct.unpack_from("<H", command, 72)[0] != 32 or
                    struct.unpack_from("<2I", command, 80) != (9, size) or
                    any(v % 32 for v in (frame, weight, psum, address))):
                raise ValueError("Unsupported grouped C2f Conv geometry or missing current RQ")
            role, group = layer_roles[layer], qparam[2]
            rq_bases.setdefault(role, qparam[0] - group * 32)
            if qparam[0] != rq_bases[role] + group * 32:
                raise ValueError("Stale grouped C2f RQ group")
            first = progress[role][group]
            output_bases.setdefault(role, address - group * rows * 32 - first * 32)
            if first + size > rows or address != output_bases[role] + group * rows * 32 + first * 32:
                raise ValueError("Noncontiguous/out-of-range grouped C2f output")
            input_group = 0 if active is None else active[-1]
            expected_mode = 1 if input_group == 0 else 2 if input_group == groups - 1 else 3
            stripe = (role, group, first, size, psum, address, weight - input_group * 9216)
            if mode != expected_mode or (active is not None and active[:-1] != stripe):
                raise ValueError("Missing/reordered grouped C2f partial accumulation")
            yy, xx = np.arange(height) + first // source[2] - 1, np.arange(width) - 1
            valid = (yy[:, None] >= 0) & (yy[:, None] < source[1]) & (xx[None, :] >= 0) & (xx[None, :] < source[2])
            frame_indices = (frame + np.arange(height)[:, None, None] * pitch +
                             np.arange(width)[None, :, None] * 32 + np.arange(32)[None, None, :])
            staged = tracker.read(6, frame_indices.ravel()).reshape(height, width, 32)
            spatial = yy[:, None] * source[2] + xx[None, :]
            expected = conv_sources[role][spatial[valid]][:, input_group * 32:(input_group + 1) * 32]
            if not np.array_equal(staged[valid], expected):
                raise ValueError("Grouped C2f staged input pixels/channels are stale or misaligned")
            psum_indices = np.arange(psum, psum + size * 128)
            carry = -(2 + (conv_roles.index(role) * groups + group) * rows * 128 + first * 128 + np.arange(size * 128))
            if active is not None and not np.array_equal(tracker.read(6, psum_indices), carry):
                raise ValueError("Grouped C2f partial sums were overwritten")
            frame_end = frame + (height - 1) * pitch + width * 32
            ofm = output_bases[role]
            if (max(psum, frame) < min(psum + size * 128, frame_end) or
                    max(address, frame) < min(address + size * 32, frame_end) or
                    max(psum, ofm) < min(psum + size * 128, ofm + groups * rows * 32)):
                raise ValueError("Grouped C2f workspace overlaps live input/output")
            if input_group + 1 < groups:
                tracker.write(6, psum_indices, carry)
                active = (*stripe, input_group + 1)
            else:
                complete(index, role, geometry(address, first, size, group * 32, 32))
                progress[role][group] += size
                active = None
        elif kind == 12:
            sr, si, src = struct.unpack_from("<HHI", command, 16)
            dr, di, dst = struct.unpack_from("<HHI", command, 24)
            lut = struct.unpack_from("<HHI", command, 32)
            length = struct.unpack_from("<I", command, 40)[0]
            if ((sr, si, dr, di) != (6, 0, 6, 0) or flags & ~4 or length != rows * channels or
                    active is not None or (flags & 4 and lut != live_lut)):
                raise ValueError("Grouped C2f LUT requires a complete native tensor and valid configuration")
            data = tracker.read(6, np.arange(src, src + length))
            role = next((r for r in conv_roles if data[0] == bases[r]), None)
            if (role is None or not covered[role].all() or
                    not np.array_equal(data, native_values(values[role]))):
                raise ValueError("Stale/incomplete grouped C2f LUT input")
            complete(index, role.replace("conv", "activation"), geometry(dst))
            live_lut = lut
        elif kind in (13, 34):
            lr, li, lhs = struct.unpack_from("<HHI", command, 16)
            rr, ri, rhs = struct.unpack_from("<HHI", command, 24)
            dr, di, dst = struct.unpack_from("<HHI", command, 32)
            length = struct.unpack_from("<I", command, 40)[0]
            operation = struct.unpack_from("<I", command, 44 if kind == 13 else 92)[0]
            if ((lr, li, rr, ri, dr, di) != (6, 0, 6, 0, 6, 0) or flags or
                    length != rows * channels or operation != (2 if kind == 13 else 0) or active is not None):
                raise ValueError("Unsupported grouped C2f residual Add mode/layout")
            right = tracker.read(6, np.arange(rhs, rhs + length))
            block = next((b for b in range(count) if right[0] == bases[f"resumed_c2f_{b}_activation2"]), None)
            if block is None:
                raise ValueError("Grouped C2f residual has no completed activation")
            role = f"resumed_c2f_{block}_activation2"
            left = labels["branch" if block == 0 else f"residual_{block - 1}"]
            if (not covered[role].all() or not np.array_equal(right, native_values(values[role])) or
                    not np.array_equal(tracker.read(6, np.arange(lhs, lhs + length)), native_values(left))):
                raise ValueError("Grouped C2f residual inputs are stale or logically misaligned")
            complete(index, f"resumed_c2f_{block}_residual", geometry(dst))
            live_lut = None
        else:
            if kind == 10:
                tracker.read(6, concat_indices.ravel())
            raise ValueError(f"Unsupported command {kind} before complete grouped C2f concat")
        if (all(mask.all() for mask in covered.values()) and
                np.array_equal(tracker.memory[6][concat_indices], concat_labels)):
            tracker.read(6, concat_indices.ravel())
            checks.append(dict(command=index, tensor_role=CONCAT_ROLE, **concat))
            validate_grouped_coverage(checks, shapes, roles)
            return checks, index
    raise ValueError("Incomplete grouped C2f checkpoints")
