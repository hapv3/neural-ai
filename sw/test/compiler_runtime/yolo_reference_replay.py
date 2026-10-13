"""Prepare pinned model full graphs or YOLO cascades against the Neural-AI hybrid reference."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
for directory in (AI_ROOT / "hw/rtl/cluster/tb", AI_ROOT / "hw/rtl/cluster/tb/tests"):
    sys.path.insert(0, str(directory))

import test_compiled_model as runtime
from npu_test_utils import NPU_CMD_TCDM_BASE
from test_yolo_stem_byte_exact import compare_bytes, extract_c32
from yolo_c2f_checkpoints import C2F_ROLES, extract_checkpoint, plan_c2f_checkpoints


def sha(data):
    return hashlib.sha256(data).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path.read_bytes())}


def checked_read(record):
    data = Path(record["path"]).read_bytes()
    if sha(data) != record["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {record['path']}")
    return data


def plan_lut_restore(model, first_command, end_command):
    """Recover one live constant LUT for this bounded initial-cascade replay.

    Memory snapshots omit AFU SRAM. Identify the original firmware LUT loader;
    host AXI cannot access AFU MMIO, so restoration needs a safe replay preamble.
    Reject other AFU modes because their residency effects are not modeled here.
    """
    commands, _ = runtime._logical_command_records(model)
    active_ref = None
    loaded_at = None
    for index, command in enumerate(commands[:end_command], 1):
        kind, _, flags = struct.unpack_from("<HHI", command)
        if kind in (13, 14, 28, 31, 32, 34):
            raise ValueError("LUT restore does not model intervening non-LUT AFU modes")
        if kind != 12:
            continue
        ref = struct.unpack_from("<HHI", command, 32)
        if index > first_command:
            if not flags & 4:
                return None  # This command loads the LUT itself.
            if active_ref is None or ref != active_ref:
                raise ValueError("LUT reuse has no matching known pre-snapshot load")
            if ref[:2] != (1, 0):
                raise ValueError("Only model-constant LUT restoration is supported")
            section_count, table = struct.unpack_from("<2I", model, 20)
            for section in range(section_count):
                kind, _, offset, size = struct.unpack_from("<4I", model, table + section * 32)
                if kind == 2:
                    if ref[2] + 256 > size:
                        raise ValueError("Resident LUT exceeds constants section")
                    data = model[offset + ref[2]:offset + ref[2] + 256]
                    return {"source_command": loaded_at, "reuse_command": index,
                            "constant_offset": ref[2], "data_hex": data.hex(), "sha256": sha(data)}
            raise ValueError("Missing constants section")
        if not flags & 4:
            active_ref, loaded_at = ref, index
        elif ref != active_ref:
            raise ValueError("Pre-snapshot LUT reuse reference differs from loaded LUT")
    return None


def plan_restore_preamble(model, first_command, restore, memory):
    """Allow replay of the immediately preceding, idempotent LUT command only.

    Original source bytes must still exist, source/destination must be disjoint,
    and the complete physical destination must already equal LUT(source).
    Firmware scratch is excluded from both tensor ranges. This is deliberately
    not a general reconstruction of arbitrary hardware snapshot state.
    """
    if restore is None:
        return None
    if restore["source_command"] != first_command:
        raise ValueError("LUT preamble must immediately precede the replay range")
    commands, _ = runtime._logical_command_records(model)
    command = commands[first_command - 1]
    kind, size, flags = struct.unpack_from("<HHI", command)
    if kind != 12 or size != 64 or flags:
        raise ValueError("LUT preamble requires an unflagged plain LUT command")
    src_region, src_index, src = struct.unpack_from("<HHI", command, 16)
    dst_region, dst_index, dst = struct.unpack_from("<HHI", command, 24)
    length = struct.unpack_from("<I", command, 40)[0]
    scratch = NPU_CMD_TCDM_BASE - 0x10100000
    if (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0):
        raise ValueError("LUT preamble requires local TCDM tensors")
    if (not length or any(v % 32 for v in (src, dst, length)) or
            max(src, dst) + length > min(len(memory), scratch) or
            max(src, dst) < min(src + length, dst + length)):
        raise ValueError("Unsafe LUT preamble ranges: overlap, alignment, or scratch")
    lut = bytes.fromhex(restore["data_hex"])
    expected = bytes(lut[value] for value in memory[src:src + length])
    if memory[dst:dst + length] != expected:
        raise ValueError("LUT preamble is not byte-idempotent on the saved snapshot")
    runtime._require_restart_safe_boundary(model, first_command - 1)
    return {"command": first_command, "source": src, "destination": dst,
            "bytes": length, "destination_sha256": sha(expected)}


def consumer_store_location(commands, submit, layer, address, length):
    """Find a staged output's L2 destination before its local bytes are reused.

    Support the initial cascade's contiguous DMA1D stores only. Async stores
    require an explicit local-to-external queue wait before the replay can end.
    """
    for index in range(submit, len(commands)):
        command = commands[index]
        kind, _, _, command_layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (9, 29):
            begin, rows = struct.unpack_from("<2I", command, 108)
            if begin < address + length and address < begin + rows * 32:
                raise ValueError("Consumer staging overwritten before its L2 store")
        if kind in (2, 3, 4, 24, 25, 26, 12):
            region, _, destination = struct.unpack_from("<HHI", command, 24)
            size = struct.unpack_from("<I", command, 40 if kind == 12 else 32)[0]
            if kind in (3, 4, 25, 26):
                stride, repetitions = struct.unpack_from("<2I", command, 40)
                size += stride * (repetitions - 1)
            if kind in (4, 26):
                stride, repetitions = struct.unpack_from("<2I", command, 52)
                size += stride * (repetitions - 1)
            if region == 6 and destination < address + length and address < destination + size:
                raise ValueError("Consumer staging overwritten before its L2 store")
        if kind not in (2, 24) or command_layer != layer:
            continue
        source_region, source_index, source = struct.unpack_from("<HHI", command, 16)
        region, binding, destination = struct.unpack_from("<HHI", command, 24)
        size, direction = struct.unpack_from("<2I", command, 32)
        if ((source_region, source_index, region, binding, direction) != (6, 0, 5, 0, 1) or
                not source <= address < address + length <= source + size):
            continue
        complete = index + 1
        if kind == 24:
            complete = next((wait + 1 for wait in range(index + 1, len(commands))
                             if struct.unpack_from("<H", commands[wait])[0] == 27 and
                             struct.unpack_from("<I", commands[wait], 16)[0] == 1), None)
            if complete is None:
                raise ValueError("Consumer L2 store has no completion wait")
        return {"command": index + 1, "complete_command": complete,
                "address": destination + address - source}
    raise ValueError("Consumer staging has no matching L2 store")


def plan_checkpoints(commands, first_command, conv_shape, consumer_shape, consumer_tiles=2):
    """Map a bounded C<=32 cascade, including rolling producer storage.

    Addresses/counts come from the ABI, not historical command numbers. Track
    completed producer C32 rows separately from an outstanding producer job;
    a LUT may read only completed rows, including across adjacent output tiles.
    Async consumers are checked at their explicit wait, not at submission.
    Partial/multi-group jobs remain deliberately unsupported. None requests all
    tiles through completion of the initial consumer tensor.
    Mapping assumes the first two distinct linebuffer layers follow the first two
    source Conv operators, with producer stripes issued in logical raster order.
    """
    if ((consumer_tiles is not None and consumer_tiles < 1) or
            any(len(s) != 4 or s[0] != 1 or not 0 < s[3] <= 32
                for s in (conv_shape, consumer_shape))):
        raise ValueError("Only single-batch, single-C32-group initial cascades are supported")
    producer_layer = None
    consumer_layer = None
    producer_rows = 0
    producer_map = {}
    pending_producer = None
    pending_consumer = None
    activation_layer = None
    consumer_base = None
    consumer_staged = False
    consumer_rows = 0
    count = 0
    checks = []

    def invalidate(address, length):
        for physical in list(producer_map):
            if address < physical + 32 and physical < address + length:
                del producer_map[physical]

    def complete_producer(tile):
        address, rows, first_row = tile
        invalidate(address, rows * 32)
        producer_map.update((address + row * 32, first_row + row) for row in range(rows))

    def overlaps_consumer(address, length):
        if pending_consumer is None:
            return False
        start = pending_consumer["address"]
        return address < start + pending_consumer["rows"] * 32 and start < address + length

    def finished():
        return (consumer_rows == consumer_shape[1] * consumer_shape[2]
                if consumer_tiles is None else count == consumer_tiles)

    def completed_end(index):
        return max([index] + [c["l2_store"]["complete_command"] for c in checks if "l2_store" in c])

    for index, command in enumerate(commands, 1):
        kind, _, _, layer, _ = struct.unpack_from("<HHIII", command)
        if kind == 32:
            raise ValueError("Async AFU requires a completion-aware mapping")
        if kind in (2, 3, 4, 24, 25, 26):
            src_region, _, source = struct.unpack_from("<HHI", command, 16)
            source_length = struct.unpack_from("<I", command, 32)[0]
            if kind in (3, 4, 25, 26):
                stride = struct.unpack_from("<I", command, 36)[0]
                repetitions = struct.unpack_from("<I", command, 44)[0]
                source_length += stride * (repetitions - 1)
            if kind in (4, 26):
                stride = struct.unpack_from("<I", command, 48)[0]
                repetitions = struct.unpack_from("<I", command, 56)[0]
                source_length += stride * (repetitions - 1)
            if src_region == 6 and overlaps_consumer(source, source_length):
                raise ValueError("DMA reads outstanding consumer output before its wait")
            region, ref_index, destination = struct.unpack_from("<HHI", command, 24)
            if region == 6:
                length = struct.unpack_from("<I", command, 32)[0]
                if kind in (3, 4, 25, 26):
                    stride, repetitions = struct.unpack_from("<2I", command, 40)
                    length += stride * (repetitions - 1)
                if kind in (4, 26):
                    stride, repetitions = struct.unpack_from("<2I", command, 52)
                    length += stride * (repetitions - 1)
                if ref_index or length <= 0:
                    raise ValueError("Invalid local DMA destination")
                if overlaps_consumer(destination, length):
                    raise ValueError("DMA touches outstanding consumer output before its wait")
                # Conservatively invalidate stride gaps as well. Missing maps
                # are safer than treating DMA-overwritten bytes as Conv output.
                invalidate(destination, length)
                if pending_producer is not None:
                    start, count_rows, _ = pending_producer
                    if destination < start + count_rows * 32 and start < destination + length:
                        raise ValueError("DMA touches outstanding producer output before its wait")
        elif kind == 30:
            if pending_producer is not None:
                complete_producer(pending_producer)
                pending_producer = None
            elif pending_consumer is not None:
                if index <= first_command:
                    raise ValueError("Snapshot is already past the first consumer tile")
                checks.append({**pending_consumer, "command": index})
                pending_consumer = None
                count += 1
                if finished():
                    return checks, completed_end(index)
            else:
                raise ValueError("Systolic wait has no mapped outstanding job")
        elif kind in (9, 29):
            if producer_layer is None:
                producer_layer = layer
            address, rows, accum, pitch, cols = struct.unpack_from("<5I", command, 108)
            if (rows <= 0 or accum or address % 32 or
                    address + rows * 32 > NPU_CMD_TCDM_BASE - 0x10100000):
                raise ValueError("Partial/empty linebuffer outputs cannot be checked as INT8")
            if pending_producer is not None:
                raise ValueError("Linebuffer job starts before the mapped producer wait")
            if pending_consumer is not None:
                raise ValueError("Linebuffer job starts before the mapped consumer wait")
            if layer == producer_layer:
                if cols != conv_shape[2] or pitch != cols * 32:
                    raise ValueError("Producer stripes are not spatial C32 rows")
                tile = (address, rows, producer_rows)
                producer_rows += rows
                if producer_rows > conv_shape[1] * conv_shape[2]:
                    raise ValueError("Producer rows exceed the source tensor")
                if kind == 29:
                    invalidate(address, rows * 32)
                    pending_producer = tile
                else:
                    complete_producer(tile)
            else:
                if consumer_layer is None:
                    consumer_layer = layer
                    consumer_staged = any(struct.unpack_from("<H", c)[0] in (2, 24) and
                                          struct.unpack_from("<I", c, 8)[0] == layer and
                                          struct.unpack_from("<H", c, 24)[0] == 5 for c in commands)
                if layer != consumer_layer or cols != consumer_shape[2] or pitch != cols * 32:
                    raise ValueError("Unexpected consumer layer/layout")
                store = consumer_store_location(commands, index, layer, address, rows * 32) if consumer_staged else None
                logical_address = store["address"] if store is not None else address
                if consumer_base is None:
                    consumer_base = logical_address
                if logical_address != consumer_base + consumer_rows * 32:
                    raise ValueError("Consumer tiles are not contiguous spatial C32 rows")
                if consumer_rows + rows > consumer_shape[1] * consumer_shape[2]:
                    raise ValueError("Consumer rows exceed the source tensor")
                if index <= first_command:
                    raise ValueError("Snapshot is already past the first consumer tile")
                invalidate(address, rows * 32)
                check = {"command": index, "tensor_role": "consumer_conv", "address": address,
                         "rows": rows, "channels": consumer_shape[3], "row_stride": pitch,
                         "tile_cols": cols, "reference_first_row": consumer_rows}
                if store is not None:
                    check["l2_store"] = store
                consumer_rows += rows
                if kind == 29:
                    pending_consumer = {**check, "submit_command": index}
                else:
                    checks.append(check)
                    count += 1
                    if finished():
                        return checks, completed_end(index)
        elif kind == 12 and producer_rows:
            if activation_layer is None:
                activation_layer = layer
            if layer != activation_layer and pending_consumer is None:
                raise ValueError("Requested consumer tiles cross another activation layer")
            src_region, src_index, src = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, dst = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if ((src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or
                    length <= 0 or any(value % 32 for value in (src, dst, length)) or
                    max(src, dst) + length > NPU_CMD_TCDM_BASE - 0x10100000):
                raise ValueError("Expected an initial C32 TCDM LUT activation")
            if any(overlaps_consumer(address, length) for address in (src, dst)):
                raise ValueError("LUT touches outstanding consumer output before its wait")
            if layer != activation_layer:
                # The scheduler may overlap the next layer's disjoint LUT tiles
                # with the last async consumer. These are not cascade checks.
                invalidate(dst, length)
                continue
            if pending_producer is not None:
                start, count_rows, _ = pending_producer
                if any(address < start + count_rows * 32 and start < address + length
                       for address in (src, dst)):
                    raise ValueError("LUT touches outstanding producer output before its wait")
            rows = length // 32
            mapped = [producer_map.get(src + row * 32) for row in range(rows)]
            first_row = mapped[0]
            if first_row is None or mapped != list(range(first_row, first_row + rows)):
                raise ValueError("LUT source does not map to completed contiguous producer rows")
            invalidate(dst, length)
            if index > first_command:
                checks.append({"command": index, "tensor_role": "producer_activation", "address": dst,
                               "rows": rows, "channels": conv_shape[3], "row_stride": conv_shape[2] * 32,
                               "tile_cols": conv_shape[2], "reference_first_row": first_row})
        elif kind not in (0, 1, 5, 27):
            raise ValueError(f"Unsupported command {kind} in initial-cascade mapping")
    raise ValueError("Not enough completed supported consumer tiles")


def cascade_l2_location(checks, commands, end):
    """Locate the complete consumer in L2 and reject post-store overwrites."""
    consumers = [c for c in checks if c["tensor_role"] == "consumer_conv"]
    staged = [c for c in consumers if "l2_store" in c]
    if not staged:
        return None
    if len(staged) != len(consumers):
        raise ValueError("Mixed resident/staged consumer checkpoints")
    first = staged[0]
    base = first["l2_store"]["address"] - first["reference_first_row"] * 32
    rows = sum(c["rows"] for c in staged)
    stores = {c["l2_store"]["command"] for c in staged}
    for check in staged:
        store = check["l2_store"]
        submit = check.get("submit_command", check["command"])
        encoded = commands[submit - 1]
        layer = struct.unpack_from("<I", encoded, 8)[0]
        if (struct.unpack_from("<H", encoded)[0] not in (9, 29) or
                struct.unpack_from("<2I", encoded, 108) != (check["address"], check["rows"]) or
                consumer_store_location(commands, submit, layer, check["address"], check["rows"] * 32) != store):
            raise ValueError("Stale L2 consumer store mapping")
        if (store["address"] != base + check["reference_first_row"] * 32 or
                store["complete_command"] > end):
            raise ValueError("Incomplete or noncontiguous L2 consumer stores")
    for index, command in enumerate(commands[:end], 1):
        kind = struct.unpack_from("<H", command)[0]
        if kind not in (2, 3, 4, 24, 25, 26) or index in stores:
            continue
        region, _, destination = struct.unpack_from("<HHI", command, 24)
        if region != 5:
            continue
        length = struct.unpack_from("<I", command, 32)[0]
        if kind in (3, 4, 25, 26):
            stride, repetitions = struct.unpack_from("<2I", command, 40)
            length += stride * (repetitions - 1)
        if kind in (4, 26):
            stride, repetitions = struct.unpack_from("<2I", command, 52)
            length += stride * (repetitions - 1)
        if destination < base + rows * 32 and base < destination + length:
            raise ValueError("Unmapped DMA overwrites the L2 consumer before final comparison")
    return {"address": base, "rows": rows, "channels": first["channels"],
            "row_stride": first["row_stride"], "tile_cols": first["tile_cols"]}


def completed_replay_boundary(model, end):
    """Extend a bounded replay only until all queued engines are drained."""
    commands, _ = runtime._logical_command_records(model)
    for boundary in range(end, len(commands) + 1):
        outstanding = runtime._restart_boundary_outstanding(commands, boundary)
        if not any(outstanding.values()):
            return boundary
    raise ValueError("No completed replay boundary after the requested checkpoints")


POST_CASCADE_ROLES = ("consumer_activation", "pointwise_conv", "pointwise_activation")


def plan_post_cascade_checkpoints(commands, cascade_checks, shapes):
    """Follow completed L2 loads, single-group pointwise Conv and its LUT.

    Logical rows come from the verified consumer's stores, never from reused
    local addresses or command IDs. Multi-group/partial jobs are rejected.
    This is a bounded extension, not a numerical mapper for the whole graph.
    """
    for role in POST_CASCADE_ROLES:
        shape = shapes[role]
        if (len(shape) != 4 or shape[0] != 1 or
                any(type(v) is not int or v <= 0 for v in shape) or shape[3] > 32):
            raise ValueError("Post-cascade mapping requires single-batch, single-C32-group tensors")
    if len({tuple(shapes[role][:3]) for role in POST_CASCADE_ROLES}) != 1:
        raise ValueError("Post-cascade pointwise tensors must have equal spatial extents")
    consumers = [c for c in cascade_checks if c["tensor_role"] == "consumer_conv"]
    if (any(c["channels"] != shapes["consumer_activation"][3] for c in consumers) or
            shapes["pointwise_conv"][3] != shapes["pointwise_activation"][3]):
        raise ValueError("Post-cascade activation channels differ from their Conv")
    l2_rows = {}
    for check in consumers:
        if "l2_store" not in check:
            raise ValueError("Post-cascade mapping requires staged consumer stores")
        store = check["l2_store"]
        for row in range(check["rows"]):
            address = store["address"] + row * 32
            if address in l2_rows:
                raise ValueError("Overlapping post-cascade L2 rows")
            l2_rows[address] = (check["reference_first_row"] + row, store["complete_command"])
    total_rows = shapes[POST_CASCADE_ROLES[0]][1] * shapes[POST_CASCADE_ROLES[0]][2]
    if sorted(row for row, _ in l2_rows.values()) != list(range(total_rows)):
        raise ValueError("Incomplete post-cascade L2 input mapping")
    local_rows = {}
    pending_loads = []
    checks = []
    layers = {}
    covered = {role: set() for role in POST_CASCADE_ROLES}
    limit = NPU_CMD_TCDM_BASE - 0x10100000

    def invalidate(address, length):
        if length <= 0 or address + length > limit:
            raise ValueError("Post-cascade local access exceeds tensor memory")
        if any(address < dst + size and dst < address + length for dst, size, _ in pending_loads):
            raise ValueError("Post-cascade access overlaps an outstanding DMA load")
        for physical in list(local_rows):
            if address < physical + 32 and physical < address + length:
                del local_rows[physical]

    def source_rows(address, length):
        if any(address < dst + size and dst < address + length for dst, size, _ in pending_loads):
            raise ValueError("Post-cascade input read before DMA completion")
        mapped = [local_rows.get(physical) for physical in range(address, address + length, 32)]
        if not any(value is not None for value in mapped):
            return None
        if address % 32 or length <= 0 or length % 32 or any(value is None for value in mapped):
            raise ValueError("Missing/stale post-cascade source rows")
        role, first_row = mapped[0]
        if mapped != [(role, first_row + row) for row in range(len(mapped))]:
            raise ValueError("Noncontiguous post-cascade source rows")
        return role, first_row

    def capture(index, layer, role, address, length, first_row):
        rows = length // 32
        if (address % 32 or length <= 0 or length % 32 or
                first_row + rows > total_rows or
                (role in layers and layers[role] != layer)):
            raise ValueError("Unsupported post-cascade output layout/layer")
        logical = set(range(first_row, first_row + rows))
        if covered[role] & logical:
            raise ValueError("Duplicate post-cascade logical output rows")
        invalidate(address, length)
        local_rows.update((address + row * 32, (role, first_row + row)) for row in range(rows))
        layers[role] = layer
        covered[role].update(logical)
        shape = shapes[role]
        checks.append({"command": index, "tensor_role": role, "address": address,
                       "rows": rows, "channels": shape[3], "row_stride": shape[2] * 32,
                       "tile_cols": shape[2], "reference_first_row": first_row})

    for index, command in enumerate(commands, 1):
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (2, 3, 4, 24, 25, 26):
            src_region, src_index, source = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 24)
            length, direction = struct.unpack_from("<2I", command, 32)
            span = length
            if kind in (3, 4, 25, 26):
                stride, repetitions = struct.unpack_from("<2I", command, 40)
                span += stride * (repetitions - 1)
            if kind in (4, 26):
                stride, repetitions = struct.unpack_from("<2I", command, 52)
                span += stride * (repetitions - 1)
            if dst_region != 6:
                continue
            invalidate(destination, span)
            mapped = []
            if src_region == 5:
                if (kind not in (2, 24) or src_index or dst_index or direction or
                        any(v % 32 for v in (source, destination, length))):
                    raise ValueError("Unsupported post-cascade L2 load layout")
                for row in range(length // 32):
                    logical = l2_rows.get(source + row * 32)
                    if logical is None or logical[1] > index:
                        raise ValueError("Post-cascade L2 input is missing or its store is incomplete")
                    mapped.append((destination + row * 32, ("consumer_conv", logical[0])))
            if kind in (24, 25, 26) and direction == 0:
                pending_loads.append((destination, span, mapped))
            else:
                local_rows.update(mapped)
        elif kind == 27 and struct.unpack_from("<I", command, 16)[0] == 0:
            for _, _, mapped in pending_loads:
                local_rows.update(mapped)
            pending_loads.clear()
        elif kind == 1:
            for _, _, mapped in pending_loads:
                local_rows.update(mapped)
            pending_loads.clear()
        elif kind in (9, 29):
            address, rows = struct.unpack_from("<2I", command, 108)
            invalidate(address, rows * 32)
        elif kind == 12:
            src_region, src_index, source = struct.unpack_from("<HHI", command, 16)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if (src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0):
                raise ValueError("Post-cascade LUT requires local input/output")
            mapped = source_rows(source, length)
            if mapped is None:
                invalidate(destination, length)
            else:
                role, first_row = mapped
                output_role = {"consumer_conv": "consumer_activation",
                               "pointwise_conv": "pointwise_activation"}.get(role)
                if output_role is None:
                    raise ValueError("Unexpected post-cascade LUT source role")
                capture(index, layer, output_role, destination, length, first_row)
        elif kind == 10:
            src_region, src_index, source = struct.unpack_from("<HHI", command, 24)
            dst_region, dst_index, destination = struct.unpack_from("<HHI", command, 40)
            rows, input_groups, output_groups = struct.unpack_from("<3I", command, 48)
            if ((src_region, src_index, dst_region, dst_index) != (6, 0, 6, 0) or flags or
                    (input_groups, output_groups) != (1, 1) or
                    struct.unpack_from("<HHI", command, 32) != (0, 0, 0)):
                raise ValueError("Unsupported post-cascade pointwise job")
            mapped = source_rows(source, rows * 32)
            if mapped is None or mapped[0] != "consumer_activation":
                raise ValueError("Pointwise input is not completed post-cascade activation")
            capture(index, layer, "pointwise_conv", destination, rows * 32, mapped[1])
        elif kind not in (0, 5, 27, 30):
            raise ValueError(f"Unsupported command {kind} in post-cascade mapping")
        if all(len(rows) == total_rows for rows in covered.values()):
            validate_cascade_coverage(checks, shapes, POST_CASCADE_ROLES)
            return checks, index
    raise ValueError("Incomplete post-cascade checkpoints")


def validate_cascade_coverage(checks, shapes, roles=("producer_activation", "consumer_conv")):
    """Prove every logical row of the requested tensors is compared once."""
    coverage = {}
    for role in roles:
        shape = shapes[role]
        if len(shape) != 4 or shape[0] != 1 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Invalid complete-cascade tensor shape")
        rows = 0
        for check in sorted((c for c in checks if c["tensor_role"] == role),
                            key=lambda c: c["reference_first_row"]):
            if check["reference_first_row"] != rows or check["rows"] <= 0 or check["channels"] != shape[3]:
                raise ValueError(f"Incomplete or overlapping cascade checkpoints: {role}")
            rows += check["rows"]
        if rows != shape[1] * shape[2]:
            raise ValueError(f"Incomplete cascade checkpoints: {role}")
        coverage[role] = {"rows": rows, "channels": shape[3], "bytes": rows * shape[3]}
    return coverage


def silu_output(ops, input_id):
    sigmoid = next(op for op in ops if op["op_name"] == "LOGISTIC" and input_id in op["inputs"])
    activation = next(op for op in ops if op["op_name"] == "MUL" and
                      set(map(int, op["inputs"])) == {input_id, int(sigmoid["outputs"][0])})
    return int(activation["outputs"][0])


def c2f_source_graph(interpreter, parent_id):
    """Resolve one residual C2f block through source edges and channel slices."""
    ops = [op for op in interpreter._get_ops_details() if op["op_name"] != "DELEGATE"]
    slices = {int(op["outputs"][0]): op for op in ops if op["op_name"] == "STRIDED_SLICE" and
              int(op["inputs"][0]) == parent_id}
    conv1 = next(op for op in ops if op["op_name"] == "CONV_2D" and int(op["inputs"][0]) in slices)
    branch_id = int(conv1["inputs"][0])
    first_id = int(conv1["outputs"][0])
    activation1_id = silu_output(ops, first_id)
    conv2 = next(op for op in ops if op["op_name"] == "CONV_2D" and int(op["inputs"][0]) == activation1_id)
    second_id = int(conv2["outputs"][0])
    activation2_id = silu_output(ops, second_id)
    residual = next(op for op in ops if op["op_name"] == "ADD" and
                    set(map(int, op["inputs"])) == {branch_id, activation2_id})
    residual_id = int(residual["outputs"][0])
    concat = next(op for op in ops if op["op_name"] == "CONCATENATION" and
                  {branch_id, residual_id}.issubset(set(map(int, op["inputs"]))))
    bypass_id, = set(map(int, concat["inputs"])) - {branch_id, residual_id}
    if bypass_id not in slices:
        raise ValueError("C2f bypass is not a slice of the verified pointwise activation")
    ids = dict(pointwise_activation=parent_id, branch_input=branch_id, block_conv1=first_id,
               block_activation1=activation1_id, block_conv2=second_id, block_activation2=activation2_id,
               block_residual=residual_id, block_concat=int(concat["outputs"][0]))
    tensors = {role: interpreter.get_tensor(tensor) for role, tensor in ids.items()}
    if any(tensor.dtype != np.int8 for tensor in tensors.values()):
        raise ValueError("C2f checkpoints require INT8 tensors")
    for conv in (conv1, conv2):
        weights = interpreter.get_tensor(int(conv["inputs"][1]))
        if weights.ndim != 4 or weights.shape[1:3] != (3, 3):
            raise ValueError("C2f source Conv is not a K3 layer")
    layout = {"concat_order": [{branch_id: "branch", bypass_id: "bypass", residual_id: "residual"}[int(t)]
                               for t in concat["inputs"]]}
    for role, tensor in (("branch", branch_id), ("bypass", bypass_id)):
        op = slices[tensor]
        begin, end, stride = [interpreter.get_tensor(int(t)).reshape(-1).tolist() for t in op["inputs"][1:]]
        if len(begin) != 4 or begin[:3] != [0, 0, 0] or stride != [1, 1, 1, 1] or end[3] <= begin[3]:
            raise ValueError("Unsupported C2f source channel slice")
        layout[f"{role}_start"] = int(begin[3])
    return ids, {role: list(map(int, tensor.shape)) for role, tensor in tensors.items()}, layout


def reset_buffers(model):
    bindings = runtime._model_binding_descriptors(model)
    sizes = {(direction, index): size for direction, index, size, _ in bindings}
    if len(bindings) != 3 or set(sizes) != {(1, 0), (2, 0), (3, 0)}:
        raise ValueError("Fresh replay requires one input, output and temporary binding, all index zero")
    # Input/output each have 1 MiB. Temporary starts at 0x80200000 and
    # may extend up to the model at 0x81000000 (14 MiB, not only 1 MiB).
    limits = {1: 0x100000, 2: 0x100000, 3: 0xE00000}
    if any(size <= 0 or size > limits[direction] for (direction, _), size in sizes.items()):
        raise ValueError("Fresh replay binding exceeds its non-overlapping L2 window")
    return bytes(runtime.TCDM_TOTAL_BYTES), bytes(sizes[3, 0]), bytes(sizes[2, 0]), sizes[1, 0]


def validate_output_scope(case, full_model):
    """A full-graph claim requires reset, every command, and a pinned output oracle."""
    full_graph = case.get("full_graph", False)
    if not isinstance(full_graph, bool):
        raise ValueError("full_graph must be boolean")
    timeout = case.get("timeout_cycles", 1000000)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout_cycles must be a positive integer")
    final_output = case.get("final_output")
    if not full_graph:
        if final_output is not None:
            raise ValueError("Bounded replay cannot claim full-graph output verification")
        return
    commands, _ = runtime._logical_command_records(full_model)
    if (case.get("start_mode") != "reset" or case["first_command"] != 0 or
            case["snapshot"] is not None or case["end_command"] != len(commands)):
        raise ValueError("Full-graph replay must start from reset and run every command")
    if not isinstance(final_output, dict) or "expected" not in final_output:
        raise ValueError("Full-graph replay requires a pinned final output oracle")
    shape = final_output.get("shape", [])
    if (final_output.get("dtype") != "int8" or not shape or
            any(type(dimension) is not int or dimension <= 0 for dimension in shape) or
            math.prod(shape) != case["output_bytes"] or
            len(checked_read(final_output["expected"])) != case["output_bytes"]):
        raise ValueError("Final output shape/dtype/size differs from the output binding")


def execution_model(case, full_model):
    """Keep the original encoded stream, including affine loops, for full runs."""
    if case.get("full_graph", False):
        return full_model
    first = case["first_command"] - (case["restore_preamble"] is not None)
    return runtime._extract_selected_yolo320_command_range(full_model, first, case["end_command"])


def invocation_address(model):
    """Place control data above the encoded model, on a separate 4 MiB window."""
    return max(0x81400000, (0x81000000 + len(model) + 0x3FFFFF) & ~0x3FFFFF)


def output_comparison(final_output, actual):
    """Compare all output bytes; no tolerance or command-completion shortcut."""
    expected = checked_read(final_output["expected"])
    return compare_bytes(np.frombuffer(actual, np.int8), np.frombuffer(expected, np.int8))


def save_checkpoint_result(folder, report, check, actual, fail_fast=False):
    """Persist byte evidence and the failure verdict before stopping simulation."""
    folder = Path(folder)
    expected = np.frombuffer(checked_read(check["expected"]), np.int8).reshape(actual.shape)
    stats = compare_bytes(actual, expected)
    (folder / f"command-{check['command']}-actual.bin").write_bytes(actual.tobytes())
    report["checkpoints"].append({**check, **stats})
    if fail_fast and stats["mismatches"]:
        report.update(status="FAIL", failed_command=check["command"],
                      error=f"Byte-exact mismatch at command {check['command']}, tensor {check['tensor']}")
    (folder / "byte_exact.json").write_text(json.dumps(report, indent=2) + "\n")
    return stats


def load_case(path):
    case = json.loads(Path(path).read_text())
    if case.get("version") != 1:
        raise ValueError("Unsupported replay case version")
    if type(case.get("fail_fast", False)) is not bool:
        raise ValueError("fail_fast must be boolean")
    if type(case.get("all_cascade_checkpoints", False)) is not bool:
        raise ValueError("all_cascade_checkpoints must be boolean")
    if type(case.get("post_cascade_checkpoints", False)) is not bool:
        raise ValueError("post_cascade_checkpoints must be boolean")
    if type(case.get("c2f_checkpoints", False)) is not bool:
        raise ValueError("c2f_checkpoints must be boolean")
    full_model = checked_read(case["model"])
    invocation = case.get("invocation_base", 0x81400000)
    if (type(invocation) is not int or invocation % 32 or invocation > 0xFFFFEFC0 or
            invocation < 0x81000000 + len(full_model)):
        raise ValueError("Invocation/control data overlaps the model or exceeds address space")
    checked_read(case["source"])
    checked_read(case["firmware"])
    input_data = checked_read(case["input"])
    validate_output_scope(case, full_model)
    if case.get("start_mode", "snapshot") == "reset":
        if case["snapshot"] is not None or case["first_command"] != 0:
            raise ValueError("Fresh replay cannot reuse a snapshot or skip initial commands")
        first = 0
        memory, temporary, output, input_bytes = reset_buffers(full_model)
        if (len(input_data), case["input_bytes"], case["temporary_bytes"], case["output_bytes"]) != (
                input_bytes, input_bytes, len(temporary), len(output)):
            raise ValueError("Fresh replay binding sizes differ from pinned model")
    elif case.get("start_mode", "snapshot") == "snapshot":
        snapshot = checked_read(case["snapshot"])
        first, memory, temporary, output = runtime._verify_yolo320_snapshot(
            full_model, snapshot, expected_boundary=case["first_command"],
            expected_tcdm=runtime.TCDM_TOTAL_BYTES,
            expected_temporary=case["temporary_bytes"], expected_output=case["output_bytes"])
    else:
        raise ValueError("Unsupported replay start mode")
    runtime._require_restart_safe_boundary(full_model, first)
    runtime._require_restart_safe_boundary(full_model, case["end_command"])
    if not first < case["end_command"]:
        raise ValueError("Empty replay range")
    # A full reset executes all configuration commands itself. No hardware
    # residency has to be reconstructed, including for models without any LUT.
    restore = None if case.get("full_graph", False) else plan_lut_restore(
        full_model, first, case["end_command"])
    if "afu_lut_restore" not in case or case["afu_lut_restore"] != restore:
        raise ValueError("Missing or stale AFU LUT restore plan; prepare a new replay case")
    if "restore_preamble" not in case or case["restore_preamble"] != plan_restore_preamble(
            full_model, first, case["afu_lut_restore"], memory):
        raise ValueError("Missing or stale firmware restore preamble; prepare a new replay case")
    commands, _ = runtime._logical_command_records(full_model)
    loaded_qparams = False
    for command in commands[first:case["end_command"]]:
        kind = struct.unpack_from("<H", command)[0]
        loaded_qparams |= kind == 5
        if kind in (9, 29) and not loaded_qparams:
            raise ValueError("First replay Conv would reuse missing requant registers")
    seen = set()
    for check in case["checkpoints"]:
        command = check["command"]
        if not first < command <= case["end_command"] or command in seen:
            raise ValueError("Checkpoint is duplicated or outside replay range")
        seen.add(command)
        expected = checked_read(check["expected"])
        if len(expected) != check["rows"] * check["channels"]:
            raise ValueError("Expected tensor size mismatch")
        extract_checkpoint(memory, check)
    if not seen and not case.get("full_graph", False):
        raise ValueError("Replay has no byte-exact checkpoints")
    if case.get("all_cascade_checkpoints", False):
        if case.get("start_mode") != "reset" or case.get("full_graph", False):
            raise ValueError("Complete-cascade checkpoint batches require bounded reset replay")
        coverage = validate_cascade_coverage(case["checkpoints"], case["cascade_shapes"])
        if coverage != case["cascade_coverage"]:
            raise ValueError("Stale complete-cascade coverage")
        saved_location = case.get("cascade_l2_output")
        comparison_command = (saved_location or {}).get("command", case["end_command"])
        if (type(comparison_command) is not int or not first < comparison_command <= case["end_command"] or
                (comparison_command != case["end_command"] and not case.get("c2f_checkpoints", False))):
            raise ValueError("Invalid live cascade L2 comparison boundary")
        location = cascade_l2_location(case["checkpoints"], commands, comparison_command)
        if location is not None:
            if saved_location is None or any(saved_location.get(k) != v for k, v in location.items()):
                raise ValueError("Missing or stale cascade L2 output comparison")
            expected = checked_read(saved_location["expected"])
            if len(expected) != location["rows"] * location["channels"]:
                raise ValueError("Cascade L2 expected tensor size mismatch")
            extract_c32(temporary, **location)
        elif saved_location is not None:
            raise ValueError("Unexpected cascade L2 output comparison")
    if case.get("post_cascade_checkpoints", False):
        if not case.get("all_cascade_checkpoints", False):
            raise ValueError("Post-cascade checkpoints require complete initial-cascade coverage")
        shapes = case["post_cascade_shapes"]
        planned, end = plan_post_cascade_checkpoints(commands, case["checkpoints"], shapes)
        saved = [c for c in case["checkpoints"] if c["tensor_role"] in POST_CASCADE_ROLES]
        if (len(saved) != len(planned) or any(
                any(actual.get(key) != value for key, value in expected.items())
                for actual, expected in zip(saved, planned))):
            raise ValueError("Missing or stale post-cascade checkpoint mapping")
        post_boundary = completed_replay_boundary(full_model, end)
        if post_boundary != case["end_command"] and not case.get("c2f_checkpoints", False):
            raise ValueError("Post-cascade replay must end at its first drained boundary")
        if validate_cascade_coverage(saved, shapes, POST_CASCADE_ROLES) != case["post_cascade_coverage"]:
            raise ValueError("Stale post-cascade coverage")
    if case.get("c2f_checkpoints", False):
        if not case.get("post_cascade_checkpoints", False):
            raise ValueError("C2f checkpoints require complete post-cascade coverage")
        if (case["cascade_l2_output"].get("command") != post_boundary or
                case["c2f_shapes"]["pointwise_activation"] != case["post_cascade_shapes"]["pointwise_activation"]):
            raise ValueError("Stale C2f parent/L2 comparison boundary")
        planned, end = plan_c2f_checkpoints(commands, case["checkpoints"], case["c2f_shapes"],
            case["c2f_layout"], len(memory), len(temporary))
        saved = [c for c in case["checkpoints"] if c["tensor_role"] in C2F_ROLES]
        if (len(saved) != len(planned) or any(
                any(actual.get(key) != value for key, value in expected.items())
                for actual, expected in zip(saved, planned))):
            raise ValueError("Missing or stale C2f checkpoint mapping")
        if completed_replay_boundary(full_model, end) != case["end_command"]:
            raise ValueError("C2f replay must end at its first drained boundary")
        if validate_cascade_coverage(saved, case["c2f_shapes"], C2F_ROLES) != case["c2f_coverage"]:
            raise ValueError("Stale C2f coverage")
    return case, full_model, memory, temporary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path, default=AI_ROOT.parent / "neural-compiler")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        help="Source TFLite (default: YOLO320); use --consumer-tiles 0 for other full graphs")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--snapshot", type=Path)
    start.add_argument("--from-reset", action="store_true")
    parser.add_argument("--consumer-tiles", type=int, default=2,
                        help="YOLO cascade checkpoints; 0 skips these only when verifying full output")
    parser.add_argument("--all-cascade-checkpoints", action="store_true",
                        help="Run every supported initial-cascade checkpoint in one bounded reset replay")
    parser.add_argument("--post-cascade-checkpoints", action="store_true",
                        help="Also check the next SiLU, single-group pointwise Conv and its SiLU")
    parser.add_argument("--c2f-checkpoints", action="store_true",
                        help="Also check the first C2f branch, Conv/SiLU pair, residual and concat")
    parser.add_argument("--full-graph", action="store_true",
                        help="Run the complete pinned NAI and compare the final output binding as well")
    parser.add_argument("--timeout-cycles", type=int,
                        help="Execution timeout (default: 1M bounded, 50M full graph)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop at the first intermediate byte mismatch, preserving failure evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.c2f_checkpoints:
        args.post_cascade_checkpoints = True
    if args.post_cascade_checkpoints:
        args.all_cascade_checkpoints = True
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    if args.full_graph and not args.from_reset:
        parser.error("--full-graph requires --from-reset; memory-only snapshots omit hardware registers")
    if args.all_cascade_checkpoints and (not args.from_reset or args.full_graph or not args.consumer_tiles):
        parser.error("--all-cascade-checkpoints requires bounded --from-reset with cascade mapping enabled")
    if args.timeout_cycles is not None and args.timeout_cycles <= 0:
        parser.error("--timeout-cycles must be positive")
    if args.consumer_tiles < 0 or (args.consumer_tiles == 0 and not args.full_graph):
        parser.error("--consumer-tiles 0 requires --full-graph; negative counts are invalid")
    pinned = json.loads(args.manifest.read_text())
    model_path = args.manifest.with_suffix(".nai")
    firmware_path = args.manifest.with_suffix(".elf")
    full_model = model_path.read_bytes()
    if sha(full_model) != pinned[".nai"] or sha(firmware_path.read_bytes()) != pinned[".elf"]:
        raise ValueError("Pinned model/ELF identity mismatch")
    source = args.source or args.compiler_root / "test/model/yolov8n_320_int8.tflite"
    if args.from_reset:
        first = 0
        memory, temporary, output, input_bytes = reset_buffers(full_model)
        saved = {"checkpoints": []}
    else:
        saved = json.loads((args.snapshot.parent / "byte_exact.json").read_text())
        if saved["source_sha256"] != sha(source.read_bytes()) or saved["model_sha256"] != sha(full_model):
            raise ValueError("Snapshot source/model provenance mismatch")
        first, memory, temporary, output = runtime._verify_yolo320_snapshot(
            full_model, args.snapshot.read_bytes(), expected_boundary=saved["prefix_commands"])
    spec = importlib.util.spec_from_file_location(
        "neural_ai_host_reference", args.compiler_root / "tools/neural_ai_tflite_reference/runtime.py")
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    interpreter = reference_module.make_interpreter(source, preserve=True)
    coverage = reference_module.coverage(interpreter)
    if coverage["mac_fallback_nodes"]:
        raise ValueError("Unsupported MAC fallback in the Neural-AI reference")
    inp = interpreter.get_input_details()[0]
    if args.from_reset:
        input_path = args.output / "input.bin"
        input_data = ((np.arange(input_bytes, dtype=np.uint32) * 37 + 11) & 255).astype(np.uint8).tobytes()
    else:
        input_path = args.snapshot.parent / "input.bin"
        input_data = input_path.read_bytes()
    interpreter.set_tensor(inp["index"], np.frombuffer(input_data, np.int8).reshape(inp["shape"]))
    interpreter.invoke()
    initial = []
    for check in saved["checkpoints"]:
        actual = extract_c32(memory, check["tcdm_address"], check["rows"], check["channels"],
                             check["row_stride"], check["tile_cols"])
        tensor = interpreter.get_tensor(check["tensor"]).reshape(-1, check["channels"])
        offset = check["reference_first_row"]
        stats = compare_bytes(actual, tensor[offset:offset + check["rows"]])
        if stats["mismatches"]:
            raise ValueError(f"Starting snapshot already differs at command {check['command']}: {stats}")
        initial.append({"command": check["command"], **stats})
    commands, _ = runtime._logical_command_records(full_model)
    checks, end = [], len(commands)
    if args.consumer_tiles:
        ops = interpreter._get_ops_details()
        convs = [op for op in ops if op["op_name"] == "CONV_2D"]
        producer_id, consumer_id = (int(op["outputs"][0]) for op in convs[:2])
        activation_id = silu_output(ops, producer_id)
        # Ensure the second source Conv really consumes this activation through PAD.
        consumer_input = int(convs[1]["inputs"][0])
        if consumer_input != activation_id:
            pad = next((op for op in ops if consumer_input in op["outputs"] and op["op_name"] == "PAD"), None)
            if pad is None or int(pad["inputs"][0]) != activation_id:
                raise ValueError("Second Conv is not the stem activation consumer")
        producer = interpreter.get_tensor(producer_id)
        consumer = interpreter.get_tensor(consumer_id)
        checks, end = plan_checkpoints(commands, first, producer.shape, consumer.shape,
                                       None if args.all_cascade_checkpoints else args.consumer_tiles)
        tensor_ids = {"producer_activation": activation_id, "consumer_conv": consumer_id}
        if args.post_cascade_checkpoints:
            consumer_activation_id = silu_output(ops, consumer_id)
            pointwise = next(op for op in convs if int(op["inputs"][0]) == consumer_activation_id)
            weights = interpreter.get_tensor(int(pointwise["inputs"][1]))
            if weights.ndim != 4 or weights.shape[1:3] != (1, 1):
                raise ValueError("Post-cascade source consumer is not a pointwise Conv")
            pointwise_id = int(pointwise["outputs"][0])
            tensor_ids.update(consumer_activation=consumer_activation_id, pointwise_conv=pointwise_id,
                              pointwise_activation=silu_output(ops, pointwise_id))
            post_shapes = {role: list(map(int, interpreter.get_tensor(tensor_ids[role]).shape))
                           for role in POST_CASCADE_ROLES}
            post_checks, end = plan_post_cascade_checkpoints(commands, checks, post_shapes)
            checks = sorted(checks + post_checks, key=lambda check: check["command"])
            if args.c2f_checkpoints:
                l2_comparison_command = completed_replay_boundary(full_model, end)
                c2f_ids, c2f_shapes, c2f_layout = c2f_source_graph(interpreter, tensor_ids["pointwise_activation"])
                tensor_ids.update(c2f_ids)
                c2f_checks, end = plan_c2f_checkpoints(commands, checks, c2f_shapes, c2f_layout,
                    len(memory), len(temporary))
                checks = sorted(checks + c2f_checks, key=lambda check: check["command"])
    if args.full_graph:
        end = len(commands)
    else:
        end = completed_replay_boundary(full_model, end)
    runtime._require_restart_safe_boundary(full_model, end)
    args.output.mkdir(parents=True)
    if args.from_reset:
        input_path.write_bytes(input_data)
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    for check in checks:
        tensor_id = tensor_ids[check["tensor_role"]]
        tensor = interpreter.get_tensor(tensor_id).reshape(-1, check["channels"])
        offset = check["reference_first_row"]
        filename = args.output / f"command-{check['command']}-expected.bin"
        filename.write_bytes(tensor[offset:offset + check["rows"]].tobytes())
        check.update({"tensor": tensor_id, "tensor_name": details[tensor_id]["name"], "expected": artifact(filename)})
    final_output = None
    if args.full_graph:
        outputs = interpreter.get_output_details()
        if len(outputs) != 1 or outputs[0]["dtype"] != np.int8:
            raise ValueError("Full-graph replay requires exactly one INT8 reference output")
        detail = outputs[0]
        expected = interpreter.get_tensor(detail["index"])
        if expected.nbytes != len(output):
            raise ValueError("Reference output size differs from the compiled output binding")
        filename = args.output / "output-expected.bin"
        filename.write_bytes(expected.tobytes())
        final_output = {"tensor": int(detail["index"]), "tensor_name": detail["name"],
                        "shape": list(map(int, expected.shape)), "dtype": str(expected.dtype),
                        "expected": artifact(filename)}
    case = {"version": 1, "reference": "Neural-AI hybrid Conv/Depthwise/FC + TFLite fallback",
            "coverage": coverage, "reference_build": json.loads((args.compiler_root /
                "tools/neural_ai_tflite_reference/build/manifest.json").read_text()),
            "pinned_manifest": str(args.manifest.resolve()), "model": artifact(model_path),
            "firmware": artifact(firmware_path), "source": artifact(source),
            "start_mode": "reset" if args.from_reset else "snapshot",
            "snapshot": None if args.from_reset else artifact(args.snapshot),
            "input": artifact(input_path), "first_command": first, "end_command": end,
            "invocation_base": invocation_address(full_model),
            "fail_fast": args.fail_fast,
            "full_graph": args.full_graph, "final_output": final_output,
            "timeout_cycles": args.timeout_cycles if args.timeout_cycles is not None else
                (50000000 if args.full_graph else 1000000),
            "afu_lut_restore": None if args.full_graph else plan_lut_restore(full_model, first, end),
            "input_bytes": len(input_data), "temporary_bytes": len(temporary), "output_bytes": len(output),
            "initial_snapshot_checks": initial, "checkpoints": checks}
    case["restore_preamble"] = plan_restore_preamble(full_model, first, case["afu_lut_restore"], memory)
    if args.all_cascade_checkpoints:
        case["all_cascade_checkpoints"] = True
        case["cascade_shapes"] = {"producer_activation": list(map(int, producer.shape)),
                                  "consumer_conv": list(map(int, consumer.shape))}
        case["cascade_coverage"] = validate_cascade_coverage(checks, case["cascade_shapes"])
        location = cascade_l2_location(checks, commands, l2_comparison_command if args.c2f_checkpoints else end)
        if location is not None:
            filename = args.output / "cascade-l2-expected.bin"
            filename.write_bytes(consumer.tobytes())
            case["cascade_l2_output"] = {**location, "tensor": consumer_id, "expected": artifact(filename)}
            if args.c2f_checkpoints:
                case["cascade_l2_output"]["command"] = l2_comparison_command
    if args.post_cascade_checkpoints:
        case.update(post_cascade_checkpoints=True, post_cascade_shapes=post_shapes,
                    post_cascade_coverage=validate_cascade_coverage(checks, post_shapes, POST_CASCADE_ROLES))
    if args.c2f_checkpoints:
        case.update(c2f_checkpoints=True, c2f_shapes=c2f_shapes, c2f_layout=c2f_layout,
                    c2f_coverage=validate_cascade_coverage(checks, c2f_shapes, C2F_ROLES))
    filename = args.output / "case.json"
    filename.write_text(json.dumps(case, indent=2) + "\n")
    load_case(filename)
    print(f"Verified {case['start_mode']} start at {first}; replay {first + 1}-{end}; "
          f"byte-exact checkpoints {[c['command'] for c in checks]}; {filename}")


if __name__ == "__main__":
    main()
