"""Prepare a pinned first-cascade replay against the Neural-AI hybrid reference."""

import argparse
import hashlib
import importlib.util
import json
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


def plan_checkpoints(commands, first_command, conv_shape, consumer_shape, consumer_tiles=2):
    """Bounded diagnostic: initial C<=32 producer and first consumer tiles only.

    Addresses/counts come from the ABI, not historical command numbers. Reject
    schedules with async/partial/multi-group producers rather than guessing maps.
    Mapping assumes the first two distinct linebuffer layers follow the first two
    source Conv operators, and their first stripes begin at logical row zero.
    """
    if consumer_tiles < 1 or any(len(s) != 4 or s[0] != 1 or not 0 < s[3] <= 32
                                 for s in (conv_shape, consumer_shape)):
        raise ValueError("Only single-batch, single-C32-group initial cascades are supported")
    producer_layer = None
    consumer_layer = None
    producer_base = None
    producer_rows = 0
    consumer_base = None
    consumer_rows = 0
    count = 0
    checks = []
    for index, command in enumerate(commands, 1):
        kind, _, _, layer, _ = struct.unpack_from("<HHIII", command)
        if kind in (29, 32):
            raise ValueError("Async linebuffer requires a completion-aware mapping")
        if kind == 9:
            if producer_layer is None:
                producer_layer = layer
            address, rows, accum, pitch, cols = struct.unpack_from("<5I", command, 108)
            if rows <= 0 or accum:
                raise ValueError("Partial/empty linebuffer outputs cannot be checked as INT8")
            if layer == producer_layer:
                if consumer_layer is not None:
                    raise ValueError("Requested consumer tiles cross a producer restart")
                if producer_base is None:
                    producer_base = address
                if (address != producer_base + producer_rows * 32 or
                        cols != conv_shape[2] or pitch != cols * 32):
                    raise ValueError("Producer stripes are not contiguous spatial C32 rows")
                producer_rows += rows
                if producer_rows > conv_shape[1] * conv_shape[2]:
                    raise ValueError("Producer rows exceed the source tensor")
            else:
                if consumer_layer is None:
                    consumer_layer = layer
                    consumer_base = address
                if layer != consumer_layer or cols != consumer_shape[2] or pitch != cols * 32:
                    raise ValueError("Unexpected consumer layer/layout")
                if address != consumer_base + consumer_rows * 32:
                    raise ValueError("Consumer tiles are not contiguous spatial C32 rows")
                if consumer_rows + rows > consumer_shape[1] * consumer_shape[2]:
                    raise ValueError("Consumer rows exceed the source tensor")
                if index <= first_command:
                    raise ValueError("Snapshot is already past the first consumer tile")
                checks.append({"command": index, "tensor_role": "consumer_conv", "address": address,
                               "rows": rows, "channels": consumer_shape[3], "row_stride": pitch,
                               "tile_cols": cols, "reference_first_row": consumer_rows})
                consumer_rows += rows
                count += 1
                if count == consumer_tiles:
                    return checks, index
        elif kind == 12 and producer_base is not None and consumer_layer is None:
            src_region, _, src = struct.unpack_from("<HHI", command, 16)
            dst_region, _, dst = struct.unpack_from("<HHI", command, 24)
            length = struct.unpack_from("<I", command, 40)[0]
            if src_region != 6 or dst_region != 6 or length <= 0 or length % 32:
                raise ValueError("Expected an initial C32 TCDM LUT activation")
            first_row, remainder = divmod(src - producer_base, 32)
            rows = length // 32
            if remainder or first_row < 0 or first_row + rows > producer_rows:
                raise ValueError("LUT source does not map to the initial producer rows")
            if index > first_command:
                checks.append({"command": index, "tensor_role": "producer_activation", "address": dst,
                               "rows": rows, "channels": conv_shape[3], "row_stride": conv_shape[2] * 32,
                               "tile_cols": conv_shape[2], "reference_first_row": first_row})
    raise ValueError("Not enough supported consumer tiles")


def reset_buffers(model):
    bindings = runtime._model_binding_descriptors(model)
    sizes = {(direction, index): size for direction, index, size, _ in bindings}
    if len(bindings) != 3 or set(sizes) != {(1, 0), (2, 0), (3, 0)}:
        raise ValueError("Fresh replay requires one input, output and temporary binding, all index zero")
    if any(size <= 0 or size > 0x100000 for size in sizes.values()):
        raise ValueError("Fresh replay binding exceeds its non-overlapping L2 window")
    return bytes(runtime.TCDM_TOTAL_BYTES), bytes(sizes[3, 0]), bytes(sizes[2, 0]), sizes[1, 0]


def load_case(path):
    case = json.loads(Path(path).read_text())
    if case.get("version") != 1:
        raise ValueError("Unsupported replay case version")
    full_model = checked_read(case["model"])
    checked_read(case["source"])
    checked_read(case["firmware"])
    input_data = checked_read(case["input"])
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
    if "afu_lut_restore" not in case or case["afu_lut_restore"] != plan_lut_restore(
            full_model, first, case["end_command"]):
        raise ValueError("Missing or stale AFU LUT restore plan; prepare a new replay case")
    if "restore_preamble" not in case or case["restore_preamble"] != plan_restore_preamble(
            full_model, first, case["afu_lut_restore"], memory):
        raise ValueError("Missing or stale firmware restore preamble; prepare a new replay case")
    commands, _ = runtime._logical_command_records(full_model)
    loaded_qparams = False
    for command in commands[first:case["end_command"]]:
        kind = struct.unpack_from("<H", command)[0]
        loaded_qparams |= kind == 5
        if kind == 9 and not loaded_qparams:
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
        extract_c32(memory, check["address"], check["rows"], check["channels"],
                    check["row_stride"], check["tile_cols"])
    if not seen:
        raise ValueError("Replay has no byte-exact checkpoints")
    return case, full_model, memory, temporary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path, default=AI_ROOT.parent / "neural-compiler")
    parser.add_argument("--manifest", type=Path, required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--snapshot", type=Path)
    start.add_argument("--from-reset", action="store_true")
    parser.add_argument("--consumer-tiles", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    pinned = json.loads(args.manifest.read_text())
    model_path = args.manifest.with_suffix(".nai")
    firmware_path = args.manifest.with_suffix(".elf")
    full_model = model_path.read_bytes()
    if sha(full_model) != pinned[".nai"] or sha(firmware_path.read_bytes()) != pinned[".elf"]:
        raise ValueError("Pinned model/ELF identity mismatch")
    source = args.compiler_root / "test/model/yolov8n_320_int8.tflite"
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
    ops = interpreter._get_ops_details()
    convs = [op for op in ops if op["op_name"] == "CONV_2D"]
    producer_id, consumer_id = (int(op["outputs"][0]) for op in convs[:2])
    sigmoid = next(op for op in ops if op["op_name"] == "LOGISTIC" and producer_id in op["inputs"])
    activation_op = next(op for op in ops if op["op_name"] == "MUL" and
                         set(map(int, op["inputs"])) == {producer_id, int(sigmoid["outputs"][0])})
    activation_id = int(activation_op["outputs"][0])
    # Ensure the second source Conv really consumes this activation through PAD.
    consumer_input = int(convs[1]["inputs"][0])
    if consumer_input != activation_id:
        pad = next((op for op in ops if consumer_input in op["outputs"] and op["op_name"] == "PAD"), None)
        if pad is None or int(pad["inputs"][0]) != activation_id:
            raise ValueError("Second Conv is not the stem activation consumer")
    producer = interpreter.get_tensor(producer_id)
    consumer = interpreter.get_tensor(consumer_id)
    commands, _ = runtime._logical_command_records(full_model)
    checks, end = plan_checkpoints(commands, first, producer.shape, consumer.shape, args.consumer_tiles)
    runtime._require_restart_safe_boundary(full_model, end)
    args.output.mkdir(parents=True)
    if args.from_reset:
        input_path.write_bytes(input_data)
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    for check in checks:
        tensor_id = consumer_id if check["tensor_role"] == "consumer_conv" else activation_id
        tensor = interpreter.get_tensor(tensor_id).reshape(-1, check["channels"])
        offset = check["reference_first_row"]
        filename = args.output / f"command-{check['command']}-expected.bin"
        filename.write_bytes(tensor[offset:offset + check["rows"]].tobytes())
        check.update({"tensor": tensor_id, "tensor_name": details[tensor_id]["name"], "expected": artifact(filename)})
    case = {"version": 1, "reference": "Neural-AI hybrid Conv/Depthwise/FC + TFLite fallback",
            "coverage": coverage, "reference_build": json.loads((args.compiler_root /
                "tools/neural_ai_tflite_reference/build/manifest.json").read_text()),
            "pinned_manifest": str(args.manifest.resolve()), "model": artifact(model_path),
            "firmware": artifact(firmware_path), "source": artifact(source),
            "start_mode": "reset" if args.from_reset else "snapshot",
            "snapshot": None if args.from_reset else artifact(args.snapshot),
            "input": artifact(input_path), "first_command": first, "end_command": end,
            "afu_lut_restore": plan_lut_restore(full_model, first, end),
            "input_bytes": len(input_data), "temporary_bytes": len(temporary), "output_bytes": len(output),
            "initial_snapshot_checks": initial, "checkpoints": checks}
    case["restore_preamble"] = plan_restore_preamble(full_model, first, case["afu_lut_restore"], memory)
    filename = args.output / "case.json"
    filename.write_text(json.dumps(case, indent=2) + "\n")
    load_case(filename)
    print(f"Verified {case['start_mode']} start at {first}; replay {first + 1}-{end}; "
          f"byte-exact checkpoints {[c['command'] for c in checks]}; {filename}")


if __name__ == "__main__":
    main()
