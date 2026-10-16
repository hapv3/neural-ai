"""Source-edge and byte-provenance mapping for resident C32-group C2f chains."""

import struct

import numpy as np

from grouped_linebuffer_checkpoints import validate_grouped_coverage
from yolo_c2f_checkpoints import _MemoryLabels, checkpoint_indices


PARENT_ROLE = "resumed_pointwise_activation"
CONCAT_ROLE = "resumed_c2f_concat"
_TOKEN_STRIDE = 1 << 32


def chain_roles(count):
    return tuple(f"resumed_c2f_{block}_{stage}" for block in range(count)
                 for stage in ("conv1", "activation1", "conv2", "activation2", "residual")) + (CONCAT_ROLE,)


def resolve_c2f_chain(interpreter, parent_id):
    """Resolve channel slices, K3/SiLU residual chains and concat order."""
    ops = [op for op in interpreter._get_ops_details() if op["op_name"] != "DELEGATE"]

    def unique(name, predicate):
        matches = [op for op in ops if op["op_name"] == name and predicate(op)]
        if len(matches) != 1:
            raise ValueError(f"C2f source needs exactly one matching {name}")
        return matches[0]

    def silu(source):
        sigmoid = unique("LOGISTIC", lambda op: list(map(int, op["inputs"])) == [source])
        mul = unique("MUL", lambda op: set(map(int, op["inputs"])) ==
                     {source, int(sigmoid["outputs"][0])})
        return int(mul["outputs"][0])

    def conv(source):
        op = unique("CONV_2D", lambda op: int(op["inputs"][0]) == source)
        weights = interpreter.get_tensor(int(op["inputs"][1]))
        if weights.ndim != 4 or weights.shape[1:3] != (3, 3):
            raise ValueError("C2f chain requires K3 convolutions")
        return int(op["outputs"][0])

    slices = [op for op in ops if op["op_name"] == "STRIDED_SLICE" and int(op["inputs"][0]) == parent_id]
    if len(slices) != 2:
        raise ValueError("C2f chain requires two source channel slices")
    branch = unique("CONV_2D", lambda op: int(op["inputs"][0]) in
                    {int(s["outputs"][0]) for s in slices})
    current = int(branch["inputs"][0])
    branch_id = current
    channels = int(interpreter.get_tensor(branch_id).shape[-1])
    if channels not in (32, 64, 128):
        raise ValueError("C2f chain requires complete C32-group source slices")
    bypass_id, = {int(s["outputs"][0]) for s in slices} - {branch_id}
    layout = {}
    for name, tensor in (("branch", branch_id), ("bypass", bypass_id)):
        op = next(s for s in slices if int(s["outputs"][0]) == tensor)
        begin, end, stride = [interpreter.get_tensor(int(t)).reshape(-1).tolist() for t in op["inputs"][1:]]
        if (len(begin) != 4 or len(end) != 4 or begin[:3] != [0, 0, 0] or
                stride != [1, 1, 1, 1] or end[3] - begin[3] != channels):
            raise ValueError("C2f chain requires full-spatial C32-group slices")
        layout[f"{name}_start"] = int(begin[3])
    ids = {PARENT_ROLE: parent_id}
    concat_sources = {branch_id: "branch", bypass_id: "bypass"}
    for block in range(8):
        first = conv(current)
        activation1 = silu(first)
        second = conv(activation1)
        activation2 = silu(second)
        residual = unique("ADD", lambda op: set(map(int, op["inputs"])) == {current, activation2})
        current = int(residual["outputs"][0])
        ids.update(zip(chain_roles(block + 1)[block * 5:block * 5 + 5],
                       (first, activation1, second, activation2, current)))
        concat_sources[current] = f"residual_{block}"
        matches = [op for op in ops if op["op_name"] == "CONCATENATION" and
                   set(map(int, op["inputs"])) == set(concat_sources) and
                   len(op["inputs"]) == len(concat_sources)]
        if matches:
            if len(matches) != 1:
                raise ValueError("Ambiguous C2f chain concat")
            concat = matches[0]
            ids[CONCAT_ROLE] = int(concat["outputs"][0])
            layout.update(blocks=block + 1, concat_order=[concat_sources[int(t)] for t in concat["inputs"]])
            tensors = {role: interpreter.get_tensor(tensor) for role, tensor in ids.items()}
            parent_shape = tensors[PARENT_ROLE].shape
            if (len(parent_shape) != 4 or parent_shape[0] != 1 or parent_shape[3] != 2 * channels or
                    any(t.dtype != np.int8 for t in tensors.values()) or
                    any(tensors[r].shape != parent_shape[:3] + (channels,) for r in chain_roles(block + 1)[:-1]) or
                    tensors[CONCAT_ROLE].shape != parent_shape[:3] + (channels * len(concat_sources),)):
                raise ValueError("Unsupported C2f chain source shapes")
            return ids, {role: list(map(int, tensor.shape)) for role, tensor in tensors.items()}, layout
    raise ValueError("C2f chain has no supported complete concat")


def plan_resumed_c2f_checkpoints(commands, parent, shapes, layout, local_bytes, l2_bytes):
    """Track completed Conv/LUT/Add bytes through DMA to a multi-group concat.

    Only the retained parent is seeded. Constants have unknown semantic labels;
    Conv results acquire labels only at completion and are separately compared
    to independent numerical oracles. DMA cannot manufacture tensor identities.
    """
    if shapes[PARENT_ROLE][-1] != 64:
        # Multiple input groups require explicit external-pSUM accumulation;
        # keep the already verified single-group/async mapper unchanged.
        from grouped_c2f_checkpoints import plan_grouped_c2f_checkpoints
        return plan_grouped_c2f_checkpoints(commands, parent, shapes, layout, local_bytes, l2_bytes)
    count = layout["blocks"]
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("Unsupported C2f chain length")
    roles = chain_roles(count)
    source = shapes[PARENT_ROLE]
    if len(source) != 4:
        raise ValueError("Unsupported resident C2f chain layout")
    rows = source[1] * source[2]
    if (len(source) != 4 or source[0] != 1 or source[3] != 64 or
            any(type(v) is not int or v <= 0 for v in source) or
            any(shapes[r] != source[:3] + [32] for r in roles[:-1]) or
            shapes[CONCAT_ROLE] != source[:3] + [32 * (count + 2)] or
            sorted((layout["branch_start"], layout["bypass_start"])) != [0, 32] or
            sorted(layout["concat_order"]) != sorted(["branch", "bypass"] + [f"residual_{b}" for b in range(count)]) or
            (parent["rows"], parent["channels"], parent.get("group_stride")) != (rows, 64, rows * 32)):
        raise ValueError("Unsupported resident C2f chain layout")
    tracker = _MemoryLabels(local_bytes, l2_bytes)
    bases = {role: (i + 1) * _TOKEN_STRIDE for i, role in enumerate((PARENT_ROLE,) + roles)}
    parent_labels = bases[PARENT_ROLE] + np.arange(rows * 64).reshape(rows, 64)
    tracker.write(6, checkpoint_indices(parent, local_bytes).ravel(), parent_labels.ravel())
    labels = {name: parent_labels[:, layout[f"{name}_start"]:layout[f"{name}_start"] + 32]
              for name in ("branch", "bypass")}
    for block in range(count):
        labels[f"residual_{block}"] = (bases[f"resumed_c2f_{block}_residual"] +
                                         np.arange(rows * 32).reshape(rows, 32))
    suffix = commands[parent["command"]:]
    layers = list(dict.fromkeys(struct.unpack_from("<I", c, 8)[0]
                               for c in suffix if struct.unpack_from("<H", c)[0] in (9, 29)))[:2 * count]
    conv_roles = [r for r in roles if r.endswith(("conv1", "conv2"))]
    if len(layers) != len(conv_roles):
        raise ValueError("C2f chain is missing Conv layers")
    layer_roles = dict(zip(layers, conv_roles))
    pointwise = next((c for c in suffix if struct.unpack_from("<H", c)[0] == 10), None)
    if (pointwise is None or struct.unpack_from("<HH", pointwise, 24) != (6, 0) or
            struct.unpack_from("<I", pointwise, 52)[0] != count + 2 or
            struct.unpack_from("<I", pointwise, 64)[0] != rows * 32):
        raise ValueError("C2f chain concat needs a matching grouped pointwise consumer")
    concat = dict(address=struct.unpack_from("<I", pointwise, 28)[0], rows=rows, channels=32 * (count + 2),
                  tile_cols=source[2], row_stride=source[2] * 32, group_stride=rows * 32, reference_first_row=0)
    concat_indices = checkpoint_indices(concat, local_bytes)
    concat_labels = np.concatenate([labels[r] for r in layout["concat_order"]], axis=1)
    covered = {role: 0 for role in roles[:-1]}
    checks = []
    qparams = {}
    qparam_layer = None
    pending = None
    live_lut = struct.unpack_from("<HHI", commands[parent["command"] - 1], 32)

    def capture(index, role, address, length, first, submit=None):
        if first != covered[role] or not length or length % 32 or first + length // 32 > rows:
            raise ValueError("Duplicate/out-of-range C2f chain rows")
        covered[role] += length // 32
        checks.append(dict(command=index, tensor_role=role, address=address, rows=length // 32,
                           channels=32, tile_cols=source[2], row_stride=source[2] * 32,
                           reference_first_row=first, **({"submit_command": submit} if submit else {})))

    def complete(index, role, address, length, first, submit=None):
        tracker.write(6, np.arange(address, address + length), bases[role] + np.arange(first * 32, first * 32 + length))
        capture(index, role, address, length, first, submit)

    for index, command in enumerate(suffix, parent["command"] + 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (2, 3, 4, 24, 25, 26):
            tracker.dma(command)
        elif kind == 27:
            direction = struct.unpack_from("<I", command, 16)[0]
            if direction not in (0, 1):
                raise ValueError("Invalid C2f chain DMA wait")
            tracker.wait(direction)
        elif kind == 1:
            tracker.wait(0)
            tracker.wait(1)
        elif kind == 5:
            qparams[layer] = struct.unpack_from("<3I", command, 16)
            qparam_layer = layer
        elif kind in (9, 29):
            address, size, accum, pitch, cols = struct.unpack_from("<5I", command, 108)
            if (layer not in layer_roles or layer != qparam_layer or qparams[layer][1:] != (32, 0) or
                    flags or accum or pitch != source[2] * 32 or cols != source[2] or pending is not None):
                raise ValueError("C2f chain Conv needs fresh C32 RQ and supported output geometry")
            role = layer_roles[layer]
            tracker.write(6, np.arange(address, address + size * 32), -1)
            if kind == 29:
                pending = (role, address, size * 32, covered[role], index)
                tracker.compute = np.arange(address, address + size * 32)
            else:
                complete(index, role, address, size * 32, covered[role])
        elif kind == 30:
            if pending is None:
                raise ValueError("C2f chain systolic wait has no pending job")
            tracker.compute = None
            role, address, length, first, submit = pending
            complete(index, role, address, length, first, submit)
            pending = None
        elif kind == 12:
            src_region, src_index, src = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, dst = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            lut = struct.unpack_from("<HHI", command, 32)
            if ((src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or
                    flags & ~4 or not length or length % 32 or (flags & 4 and lut != live_lut)):
                raise ValueError("Unsupported C2f chain LUT layout or stale configuration")
            values = tracker.read(6, np.arange(src, src + length))
            role = next((r for r in conv_roles if values[0] // _TOKEN_STRIDE == bases[r] // _TOKEN_STRIDE), None)
            if role is None:
                raise ValueError("C2f chain LUT does not consume completed Conv bytes")
            first = int(values[0] - bases[role])
            if first < 0 or first % 32 or not np.array_equal(values, bases[role] + np.arange(first, first + length)):
                raise ValueError("Stale/noncontiguous C2f chain LUT input")
            live_lut = lut
            complete(index, role.replace("conv", "activation"), dst, length, first // 32)
        elif kind in (13, 34):
            lhs_region, lhs_index, lhs = struct.unpack_from("<HHI", command, 16)
            rhs_region, rhs_index, rhs = struct.unpack_from("<HHI", command, 24)
            dst_region, dst_index, dst = struct.unpack_from("<HHI", command, 32)
            length = struct.unpack_from("<I", command, 40)[0]
            operation = struct.unpack_from("<I", command, 44 if kind == 13 else 92)[0]
            if ((lhs_region, lhs_index, rhs_region, rhs_index, dst_region, dst_index) != (6, 0, 6, 0, 6, 0) or
                    flags or operation != (2 if kind == 13 else 0) or not length or length % 32):
                raise ValueError("Unsupported C2f chain residual Add mode/layout")
            right = tracker.read(6, np.arange(rhs, rhs + length))
            block = next((b for b in range(count) if right[0] // _TOKEN_STRIDE ==
                          bases[f"resumed_c2f_{b}_activation2"] // _TOKEN_STRIDE), None)
            if block is None:
                raise ValueError("C2f chain residual has no completed second activation")
            role = f"resumed_c2f_{block}_activation2"
            first = int(right[0] - bases[role])
            left = labels["branch" if block == 0 else f"residual_{block - 1}"].ravel()
            if (first < 0 or first % 32 or first + length > left.size or
                    not np.array_equal(right, bases[role] + np.arange(first, first + length)) or
                    not np.array_equal(tracker.read(6, np.arange(lhs, lhs + length)), left[first:first + length])):
                raise ValueError("C2f chain residual inputs are stale or logically misaligned")
            complete(index, f"resumed_c2f_{block}_residual", dst, length, first // 32)
            live_lut = None
        else:
            if kind == 10:
                tracker.read(6, concat_indices.ravel())
            raise ValueError(f"Unsupported command {kind} before complete C2f chain concat")
        if (all(v == rows for v in covered.values()) and
                np.array_equal(tracker.memory[6][concat_indices], concat_labels)):
            tracker.read(6, concat_indices.ravel())
            checks.append(dict(command=index, tensor_role=CONCAT_ROLE, **concat))
            validate_grouped_coverage(checks, shapes, roles)
            return checks, index
    raise ValueError("Incomplete resumed C2f chain checkpoints")
