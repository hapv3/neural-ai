"""Bounded numerical mapping for a single-group C2f residual/concat block."""

import struct

import numpy as np


C2F_ROLES = ("branch_input", "block_conv1", "block_activation1", "block_conv2",
             "block_activation2", "block_residual", "block_concat")
C2F_OUTPUT_ROLES = ("block_output_conv", "block_output_activation")
_TOKEN_STRIDE = 1 << 32


def checkpoint_indices(check, memory_bytes):
    rows, channels, cols = (check[k] for k in ("rows", "channels", "tile_cols"))
    stride = check.get("element_stride", 32)
    group_stride = check.get("group_stride", 0)
    if (any(type(v) is not int or v <= 0 for v in (rows, channels, cols, stride)) or
            type(check["address"]) is not int or check["address"] < 0 or
            type(group_stride) is not int or group_stride < 0 or
            type(check["row_stride"]) is not int or
            (not channels <= stride <= 32 and not (channels > 32 and stride == 32 and group_stride > 0)) or
            check["row_stride"] < cols * stride):
        raise ValueError("Unsupported checkpoint storage layout")
    row = np.arange(rows, dtype=np.int64)[:, None]
    channel = np.arange(channels, dtype=np.int64)[None, :]
    indices = (check["address"] + row // cols * check["row_stride"] + row % cols * stride +
               channel // 32 * group_stride + channel % 32)
    spatial_span = (rows - 1) // cols * check["row_stride"] + (rows - 1) % cols * stride + stride
    if channels > 32 and group_stride < spatial_span:
        raise ValueError("Checkpoint channel groups overlap")
    if indices.max() >= memory_bytes:
        raise ValueError("Checkpoint storage exceeds memory")
    return indices


def extract_checkpoint(memory, check):
    return np.frombuffer(memory, dtype=np.int8)[checkpoint_indices(check, len(memory))].copy()


class _MemoryLabels:
    """Byte identities, not numerical values; DMA preserves labels exactly."""

    def __init__(self, local_bytes, l2_bytes):
        self.memory = {6: np.full(local_bytes, -1, dtype=np.int64),
                       5: np.full(l2_bytes, -1, dtype=np.int64)}
        self.pending = {0: [], 1: []}
        self.compute = None

    def guard(self, region, indices, writing=False):
        if region not in self.memory:
            return
        if indices.size == 0 or indices.min() < 0 or indices.max() >= self.memory[region].size:
            raise ValueError("C2f transfer exceeds memory")
        for queue in self.pending.values():
            for source, src, destination, dst, _ in queue:
                accesses = [(destination, dst)] + ([(source, src)] if writing else [])
                for other_region, other in accesses:
                    if region == other_region and np.intersect1d(indices, other, assume_unique=False).size:
                        raise ValueError("C2f access before DMA completion")
        if self.compute is not None and region == 6 and np.intersect1d(indices, self.compute).size:
            raise ValueError("C2f access before systolic completion")

    def read(self, region, indices):
        self.guard(region, indices)
        return self.memory[region][indices].copy() if region in self.memory else np.full(indices.size, -1)

    def write(self, region, indices, values):
        self.guard(region, indices, writing=True)
        if region in self.memory:
            self.memory[region][indices] = values

    def wait(self, direction):
        for _, _, region, indices, values in self.pending[direction]:
            self.memory[region][indices] = values
        self.pending[direction].clear()

    def dma(self, command):
        kind = struct.unpack_from("<H", command)[0]
        source, source_index, src = struct.unpack_from("<HHI", command, 16)
        destination, destination_index, dst = struct.unpack_from("<HHI", command, 24)
        length = struct.unpack_from("<I", command, 32)[0]
        if source_index or destination_index:
            raise ValueError("C2f mapping requires binding index zero")
        src_stride = dst_stride = length
        rows = planes = 1
        src_plane = dst_plane = 0
        direction = struct.unpack_from("<I", command, 36)[0]
        if kind in (3, 4, 25, 26):
            src_stride, dst_stride, rows, direction = struct.unpack_from("<4I", command, 36)
        if kind in (4, 26):
            src_plane, dst_plane, planes, direction = struct.unpack_from("<4I", command, 48)
        if not length or not rows or not planes or length * rows * planes > 2 * 1024 * 1024:
            raise ValueError("Invalid C2f DMA dimensions")
        byte = np.arange(length, dtype=np.int64)
        row = np.arange(rows, dtype=np.int64)[:, None]
        plane = np.arange(planes, dtype=np.int64)[:, None, None]
        src_indices = (src + plane * src_plane + row * src_stride + byte).ravel()
        dst_indices = (dst + plane * dst_plane + row * dst_stride + byte).ravel()
        values = self.read(source, src_indices)
        self.write(destination, dst_indices, -1)
        if kind in (24, 25, 26) and direction in (0, 1):
            if destination not in self.memory:
                raise ValueError("Unsupported C2f DMA destination")
            self.pending[direction].append((source, src_indices, destination, dst_indices, values))
        else:
            self.write(destination, dst_indices, values)
        return destination, dst_indices


def plan_c2f_checkpoints(commands, post_checks, shapes, layout, local_bytes, l2_bytes):
    """Map Conv/LUT/biased-Add and the first materialized multi-group concat.

    Source graph resolves the branch channel slices and concat order. ABI
    geometry resolves all physical addresses, DMA strides and completion IDs.
    Unknown layouts, incomplete rows and unsafe asynchronous accesses fail closed.
    """
    parent_shape = shapes["pointwise_activation"]
    if (len(parent_shape) != 4 or parent_shape[0] != 1 or parent_shape[3] != 32 or
            any(type(v) is not int or v <= 0 for v in parent_shape)):
        raise ValueError("C2f mapping requires a single-batch C32 parent")
    total_rows = parent_shape[1] * parent_shape[2]
    channels = shapes["branch_input"][3]
    if (channels != 16 or any(shapes[r] != parent_shape[:3] + [channels] for r in C2F_ROLES[:-1]) or
            shapes["block_concat"] != parent_shape[:3] + [3 * channels] or
            sorted(layout["concat_order"]) != ["branch", "bypass", "residual"] or
            sorted((layout["branch_start"], layout["bypass_start"])) != [0, channels]):
        raise ValueError("Unsupported C2f source graph layout")
    roles = ("pointwise_activation",) + C2F_ROLES
    bases = {role: (i + 1) * _TOKEN_STRIDE for i, role in enumerate(roles)}
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    seed = {c["command"]: c for c in post_checks if c["tensor_role"] == "pointwise_activation"}
    if not seed:
        raise ValueError("C2f mapping has no verified parent checkpoints")
    begin = max(seed)
    linebuffer_layers = list(dict.fromkeys(struct.unpack_from("<I", c, 8)[0]
        for c in commands[begin:] if struct.unpack_from("<H", c)[0] in (9, 29)))[:2]
    if len(linebuffer_layers) != 2:
        raise ValueError("C2f mapping requires two following linebuffer Conv layers")
    conv_roles = dict(zip(linebuffer_layers, ("block_conv1", "block_conv2")))
    next_pointwise = next((c for c in commands[begin:] if struct.unpack_from("<H", c)[0] == 10), None)
    if (next_pointwise is None or struct.unpack_from("<HH", next_pointwise, 24) != (6, 0) or
            struct.unpack_from("<I", next_pointwise, 52)[0] != 2 or
            struct.unpack_from("<I", next_pointwise, 64)[0] != total_rows * 32):
        raise ValueError("C2f concat has no supported two-group pointwise consumer")
    concat = dict(address=struct.unpack_from("<I", next_pointwise, 28)[0], rows=total_rows,
                  channels=3 * channels, tile_cols=parent_shape[2], row_stride=parent_shape[2] * 32,
                  group_stride=total_rows * 32)
    concat_indices = checkpoint_indices(concat, local_bytes)
    parent = bases["pointwise_activation"] + np.arange(total_rows * 32).reshape(total_rows, 32)
    source_labels = {name: parent[:, layout[f"{name}_start"]:layout[f"{name}_start"] + channels]
                     for name in ("branch", "bypass")}
    source_labels["residual"] = bases["block_residual"] + np.arange(total_rows * channels).reshape(total_rows, channels)
    concat_labels = np.concatenate([source_labels[role] for role in layout["concat_order"]], axis=1)
    checks = []
    covered = {role: set() for role in C2F_ROLES}
    conv_rows = {role: 0 for role in conv_roles.values()}
    pending_conv = None

    def capture(index, role, check, first_row, submit=None):
        logical = set(range(first_row, first_row + check["rows"]))
        if not logical or max(logical) >= total_rows or covered[role] & logical:
            raise ValueError("Duplicate/out-of-range C2f checkpoint rows")
        covered[role].update(logical)
        check = {**check, "command": index, "tensor_role": role, "reference_first_row": first_row}
        if submit is not None:
            check["submit_command"] = submit
        checks.append(check)

    def completed_conv(index, role, check, first_row, submit=None):
        physical = np.arange(check["address"], check["address"] + check["rows"] * 32)
        tracker.write(6, physical, -1)
        indices = checkpoint_indices(check, local_bytes)
        tracker.write(6, indices.ravel(), bases[role] + np.arange(first_row * channels,
                      (first_row + check["rows"]) * channels))
        capture(index, role, check, first_row, submit)

    for index, command in enumerate(commands, 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (2, 3, 4, 24, 25, 26):
            destination, indices = tracker.dma(command)
            if index > begin and not covered["branch_input"] and destination == 6:
                address = int(indices.min())
                expected = source_labels["branch"].ravel()
                if address + expected.size <= local_bytes and np.array_equal(
                        tracker.memory[6][address:address + expected.size], expected):
                    capture(index, "branch_input", dict(address=address, rows=total_rows, channels=channels,
                        element_stride=channels, tile_cols=parent_shape[2], row_stride=parent_shape[2] * channels), 0)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid C2f DMA wait direction")
            tracker.wait(direction)
        elif kind == 1:
            tracker.wait(0)
            tracker.wait(1)
        elif kind in (9, 29):
            address, rows, accum, pitch, cols = struct.unpack_from("<5I", command, 108)
            tracker.write(6, np.arange(address, address + rows * 32), -1)
            role = conv_roles.get(layer) if index > begin else None
            if role is not None:
                if accum or cols != parent_shape[2] or pitch != cols * 32 or flags:
                    raise ValueError("Unsupported C2f linebuffer output layout")
                check = dict(address=address, rows=rows, channels=channels, row_stride=pitch, tile_cols=cols)
                first_row = conv_rows[role]
                conv_rows[role] += rows
                if kind == 29:
                    pending_conv = (role, check, first_row, index)
                else:
                    completed_conv(index, role, check, first_row)
            if kind == 29:
                tracker.compute = np.arange(address, address + rows * 32)
        elif kind == 30:
            tracker.compute = None
            if pending_conv is not None:
                role, check, first_row, submit = pending_conv
                completed_conv(index, role, check, first_row, submit)
                pending_conv = None
        elif kind == 12:
            src_region, src_index, source = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0):
                raise ValueError("Unsupported C2f LUT region")
            labels = tracker.read(6, np.arange(source, source + length))
            tracker.write(6, np.arange(destination, destination + length), -1)
            active = np.flatnonzero(labels >= 0)
            if index > begin and active.size:
                role = next((r for r in conv_roles.values()
                             if labels[active[0]] // _TOKEN_STRIDE == bases[r] // _TOKEN_STRIDE), None)
                if role is None:
                    raise ValueError("C2f LUT source does not identify a completed Conv")
                first = int(labels[active[0]] - bases[role])
                rows = active.size // channels
                stride = length // rows
                if (first % channels or active.size % channels or stride not in (channels, 32) or
                        not np.array_equal(active, (np.arange(rows)[:, None] * stride + np.arange(channels)).ravel()) or
                        not np.array_equal(labels[active], bases[role] + np.arange(first, first + active.size))):
                    raise ValueError("Missing/stale/noncontiguous C2f LUT source rows")
                output_role = role.replace("conv", "activation")
                tracker.write(6, destination + active, bases[output_role] + labels[active] - bases[role])
                capture(index, output_role, dict(address=destination, rows=rows, channels=channels,
                    tile_cols=parent_shape[2], row_stride=parent_shape[2] * stride,
                    element_stride=stride), first // channels)
        elif kind == 10:
            region, _, address = struct.unpack_from("<HHI", command, 40)
            rows, _, groups = struct.unpack_from("<3I", command, 48)
            if index > begin:
                raise ValueError("C2f concat was not fully materialized before its consumer")
            tracker.write(region, np.arange(address, address + rows * groups * 32), -1)
        elif kind == 13:
            lhs_region, lhs_index, lhs = struct.unpack_from("<HHI", command, 16)
            rhs_region, rhs_index, rhs = struct.unpack_from("<HHI", command, 24)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 32)
            length, mode = struct.unpack_from("<2I", command, 40)
            if ((lhs_region, lhs_index, rhs_region, rhs_index, dst_region, dst_index) != (6, 0, 6, 0, 6, 0) or
                    mode != 2 or flags or not length or length % channels):
                raise ValueError("Unsupported C2f residual Add layout/mode")
            right = tracker.read(6, np.arange(rhs, rhs + length))
            first = int(right[0] - bases["block_activation2"])
            if (first < 0 or first % channels or first + length > total_rows * channels or
                    not np.array_equal(right, bases["block_activation2"] + np.arange(first, first + length)) or
                    not np.array_equal(tracker.read(6, np.arange(lhs, lhs + length)),
                                       source_labels["branch"].ravel()[first:first + length])):
                raise ValueError("C2f residual inputs are stale or logically misaligned")
            tracker.write(6, np.arange(destination, destination + length),
                          bases["block_residual"] + np.arange(first, first + length))
            capture(index, "block_residual", dict(address=destination, rows=length // channels, channels=channels,
                element_stride=channels, tile_cols=parent_shape[2],
                row_stride=parent_shape[2] * channels), first // channels)
        elif kind not in (0, 5):
            raise ValueError(f"Unsupported command {kind} in C2f mapping")
        if index in seed:
            check = seed[index]
            indices = checkpoint_indices(check, local_bytes)
            first = check["reference_first_row"] * parent_shape[3]
            tracker.write(6, indices.ravel(), bases["pointwise_activation"] + np.arange(first, first + indices.size))
        if (all(len(covered[role]) == total_rows for role in C2F_ROLES[:-1]) and
                np.array_equal(tracker.memory[6][concat_indices], concat_labels)):
            capture(index, "block_concat", concat, 0)
            return checks, index
    raise ValueError("Incomplete C2f checkpoints")


def plan_c2f_output_checkpoints(commands, c2f_checks, shapes, local_bytes, l2_bytes):
    """Follow a resident multi-input-group concat through pointwise/L2/SiLU."""
    concat, = [c for c in c2f_checks if c["tensor_role"] == "block_concat"]
    parent_shape = shapes["block_concat"]
    output_shape = shapes["block_output_conv"]
    if (any(len(s) != 4 or s[0] != 1 or any(type(v) is not int or v <= 0 for v in s)
            for s in (parent_shape, output_shape, shapes["block_output_activation"])) or
            parent_shape[:3] != output_shape[:3] or
            output_shape != shapes["block_output_activation"] or
            not 32 < parent_shape[3] <= 64 or not 0 < output_shape[3] <= 32 or
            (concat["rows"], concat["channels"]) != (parent_shape[1] * parent_shape[2], parent_shape[3])):
        raise ValueError("Unsupported C2f output tensor shapes")
    total_rows, channels = concat["rows"], output_shape[3]
    groups = (parent_shape[3] + 31) // 32
    group_stride = concat["group_stride"]
    if group_stride != total_rows * 32:
        raise ValueError("C2f output requires contiguous C32 input groups")
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    indices = checkpoint_indices(concat, local_bytes)
    tracker.write(6, indices.ravel(), _TOKEN_STRIDE + np.arange(indices.size))
    checks = []
    covered = {role: set() for role in C2F_OUTPUT_ROLES}
    layers = {}

    def capture(index, layer, role, address, rows, first_row):
        logical = set(range(first_row, first_row + rows))
        if (not logical or first_row < 0 or max(logical) >= total_rows or covered[role] & logical or
                (role in layers and layers[role] != layer)):
            raise ValueError("Duplicate/out-of-range C2f output checkpoint rows/layers")
        covered[role].update(logical)
        layers[role] = layer
        checks.append(dict(command=index, tensor_role=role, address=address, rows=rows, channels=channels,
                           tile_cols=output_shape[2], row_stride=output_shape[2] * 32,
                           reference_first_row=first_row))

    for index, command in enumerate(commands[concat["command"]:], concat["command"] + 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (2, 3, 4, 24, 25, 26):
            tracker.dma(command)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid C2f output DMA wait direction")
            tracker.wait(direction)
        elif kind == 1:
            tracker.wait(0)
            tracker.wait(1)
        elif kind == 10:
            src_region, src_index, source = struct.unpack_from("<HHI", command, 24)
            partial_region, partial_index, partial = struct.unpack_from("<HHI", command, 32)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 40)
            rows, input_groups, output_groups, _, input_stride, output_stride = struct.unpack_from("<6I", command, 48)
            if ((src_region, src_index, partial_region, partial_index, dst_region, dst_index) != (6, 0, 6, 0, 6, 0) or
                    flags or input_groups != groups or output_groups != 1 or not rows or
                    input_stride != group_stride or output_stride < rows * 32 or
                    source < concat["address"] or (source - concat["address"]) % 32):
                raise ValueError("Unsupported multi-group C2f output pointwise job")
            first_row = (source - concat["address"]) // 32
            if first_row + rows > total_rows:
                raise ValueError("C2f output pointwise exceeds input rows")
            source_indices = checkpoint_indices({**concat, "address": source, "rows": rows}, local_bytes)
            if not np.array_equal(tracker.read(6, source_indices.ravel()),
                    _TOKEN_STRIDE + np.arange(first_row * parent_shape[3], (first_row + rows) * parent_shape[3])):
                raise ValueError("Missing/stale multi-group C2f output input rows")
            for address, length in ((partial, rows * 32 * 4), (destination, rows * 32)):
                if address < concat["address"] + groups * group_stride and concat["address"] < address + length:
                    raise ValueError("C2f output/partial workspace overlaps resident concat")
                tracker.write(6, np.arange(address, address + length), -1)
            if max(partial, destination) < min(partial + rows * 128, destination + rows * 32):
                raise ValueError("C2f output overlaps partial workspace")
            out = dict(address=destination, rows=rows, channels=channels,
                       tile_cols=output_shape[2], row_stride=output_shape[2] * 32)
            tracker.write(6, checkpoint_indices(out, local_bytes).ravel(),
                          2 * _TOKEN_STRIDE + np.arange(first_row * channels, (first_row + rows) * channels))
            capture(index, layer, "block_output_conv", destination, rows, first_row)
        elif kind == 12:
            src_region, src_index, source = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or not length or length % 32:
                raise ValueError("Unsupported C2f output LUT layout")
            labels = tracker.read(6, np.arange(source, source + length))
            active = np.flatnonzero(labels >= 0)
            rows = length // 32
            first = int(labels[0] - 2 * _TOKEN_STRIDE)
            if (first < 0 or first % channels or first + rows * channels > total_rows * channels or
                    not np.array_equal(active, (np.arange(rows)[:, None] * 32 + np.arange(channels)).ravel()) or
                    not np.array_equal(labels[active], 2 * _TOKEN_STRIDE + np.arange(first, first + active.size))):
                raise ValueError("Missing/stale C2f output LUT input rows")
            tracker.write(6, np.arange(destination, destination + length), -1)
            tracker.write(6, destination + active, 3 * _TOKEN_STRIDE + np.arange(first, first + active.size))
            capture(index, layer, "block_output_activation", destination, rows, first // channels)
        elif kind != 5:
            raise ValueError(f"Unsupported command {kind} in C2f output mapping")
        if all(len(rows) == total_rows for rows in covered.values()):
            return checks, index
    raise ValueError("Incomplete C2f output checkpoints")
