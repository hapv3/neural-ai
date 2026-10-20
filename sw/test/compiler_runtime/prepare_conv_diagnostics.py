"""Prepare complete Conv observations from source quantization and NAI ABI."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np

from grouped_linebuffer_checkpoints import validate_grouped_coverage
from yolo_reference_replay import artifact, checked_read, load_case, runtime


def plan_conv_observations(commands, qparams, sources, first, end):
    signatures = {}
    for source in sources:
        signatures.setdefault(tuple(source["bias_signature"]), []).append(source)
    layers = {struct.unpack_from("<I", c, 8)[0] for c in commands[first:end]
              if struct.unpack_from("<H", c)[0] in (9, 10, 29)}
    mapping = {}
    for command in commands[first:end]:
        kind, _, _, layer, _ = struct.unpack_from("<HHIII", command)
        if kind != 5 or layer not in layers or struct.unpack_from("<I", command, 24)[0] != 0:
            continue
        base, count = struct.unpack_from("<2I", command, 16)
        if count != 32 or (base + count) * 32 > len(qparams):
            raise ValueError("Conv observation RQ table is incomplete")
        bias = tuple(struct.unpack_from("<i", qparams, (base + channel) * 32)[0] for channel in range(32))
        candidates = signatures.get(bias, [])
        if len(candidates) != 1:
            raise ValueError(f"Conv layer {layer} has no unique source bias signature")
        if layer in mapping and mapping[layer] != candidates[0]:
            raise ValueError("Conv layer source changes within the replay")
        mapping[layer] = candidates[0]
    if set(mapping) != layers:
        raise ValueError(f"Unmapped replay Conv layers: {sorted(layers - set(mapping))}")
    checks, progress, groups = [], {}, {}
    for number in range(first + 1, end + 1):
        command = commands[number - 1]
        kind, _, flags, layer, _ = struct.unpack_from("<HHIII", command)
        if kind == 5:
            groups[layer] = struct.unpack_from("<I", command, 24)[0]
        if kind not in (9, 10, 29):
            continue
        source = mapping[layer]
        shape, tensor = source["shape"], source["tensor"]
        if flags or len(shape) != 4 or shape[0] != 1 or shape[3] <= 0:
            raise ValueError("Unsupported Conv observation source/layout")
        if kind in (9, 29):
            address, count, mode, pitch, cols = struct.unpack_from("<5I", command, 108)
            if list(struct.unpack_from("<2H", command, 52)) != source["kernel"]:
                raise ValueError("Conv source kernel differs from linebuffer ABI")
            if mode in (1, 3):
                continue
            if mode not in (0, 2) or layer not in groups:
                raise ValueError("Conv observation has no finalized RQ output")
            group = groups[layer]
        else:
            if any(struct.unpack_from("<HH", command, offset) != (6, 0) for offset in (24, 40)):
                raise ValueError("Pointwise observation requires local compute addresses")
            address, count, _, outputs, group = struct.unpack_from("<5I", command, 44)
            if outputs != 1 or source["kernel"] != [1, 1]:
                raise ValueError("Unsupported pointwise observation output/kernel")
            cols, pitch = shape[2], shape[2] * 32
        first_row = progress.get((tensor, group), 0)
        channels = min(32, shape[3] - group * 32)
        if not count or cols != shape[2] or not 0 < channels <= 32 or first_row + count > shape[1] * shape[2]:
            raise ValueError("Conv observation has incomplete/out-of-range output geometry")
        progress[tensor, group] = first_row + count
        completion = number
        if kind == 29:
            completion = next((index for index in range(number + 1, end + 1)
                               if struct.unpack_from("<H", commands[index - 1])[0] == 30), None)
            if completion is None:
                raise ValueError("Async Conv observation has no completion")
            wait = commands[completion - 1]
            if (struct.unpack_from("<HHI", wait) != (30, 32, 0) or wait[8:16] != command[8:16] or
                    any(struct.unpack_from("<H", c)[0] in (9, 10, 29, 30)
                        for c in commands[number:completion - 1])):
                raise ValueError("Async Conv observation has an ambiguous completion")
        check = dict(command=completion, tensor=tensor, tensor_role=f"conv_{tensor}", address=address,
                     rows=count, channels=channels, tile_cols=cols, row_stride=pitch,
                     reference_first_row=first_row, reference_first_channel=group * 32,
                     reference_shape=shape, command_sha256=hashlib.sha256(command).hexdigest())
        if kind == 29:
            check.update(submit_command=number,
                         completion_sha256=hashlib.sha256(commands[completion - 1]).hexdigest())
        checks.append(check)
    shapes = {f"conv_{s['tensor']}": s["shape"] for s in mapping.values()}
    validate_grouped_coverage(checks, shapes, tuple(shapes))
    return sorted(checks, key=lambda c: c["command"]), mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--compiler-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh observation directory")
    case, model, *_ = load_case(args.case)
    if case.get("remaining_graph") is not True:
        raise ValueError("Conv diagnostics require a verified complete continuation")
    root = args.compiler_root / "tools/neural_ai_tflite_reference"
    if json.loads((root / "build/manifest.json").read_text()) != case["reference_build"]:
        raise ValueError("Conv diagnostic reference build changed")
    spec = importlib.util.spec_from_file_location("independent_reference", root / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    interpreter = module.make_interpreter(case["source"]["path"], preserve=True)
    if module.coverage(interpreter)["mac_fallback_nodes"]:
        raise ValueError("Conv diagnostics cannot use a MAC fallback")
    detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(detail["index"], np.frombuffer(checked_read(case["input"]), np.int8).reshape(detail["shape"]))
    interpreter.invoke()
    if interpreter.get_tensor(case["final_output"]["tensor"]).tobytes() != checked_read(case["final_output"]["expected"]):
        raise ValueError("Conv diagnostic final reference differs from pinned oracle")
    details = {int(t["index"]): t for t in interpreter.get_tensor_details()}
    sources = []
    for op in interpreter._get_ops_details():
        if op["op_name"] != "CONV_2D":
            continue
        source, weights, bias = map(int, op["inputs"])
        tensor = int(op["outputs"][0])
        w = interpreter.get_tensor(weights).astype(np.int64)
        zp = details[source]["quantization"][1]
        adjusted = interpreter.get_tensor(bias).astype(np.int64) - zp * w.sum(axis=(1, 2, 3))
        signature = np.zeros(32, np.int64)
        signature[:min(32, adjusted.size)] = adjusted[:32]
        sources.append(dict(tensor=tensor, shape=list(map(int, details[tensor]["shape"])),
                            kernel=list(map(int, w.shape[1:3])), bias_signature=signature.tolist()))
    commands, _ = runtime._logical_command_records(model)
    sections, table = struct.unpack_from("<2I", model, 20)
    entry = next(table + index * 32 for index in range(sections)
                 if struct.unpack_from("<I", model, table + index * 32)[0] == 5)
    offset, size = struct.unpack_from("<2I", model, entry + 8)
    checks, mapping = plan_conv_observations(commands, model[offset:offset + size], sources,
                                            case["first_command"], case["end_command"])
    prefix = {check["command"]: check for check in case["checkpoints"]}
    selected = []
    args.output.mkdir(parents=True)
    for check in checks:
        shape = check["reference_shape"]
        value = interpreter.get_tensor(check["tensor"]).reshape(-1, shape[3])
        row, channel = check["reference_first_row"], check["reference_first_channel"]
        expected = value[row:row + check["rows"], channel:channel + check["channels"]].tobytes()
        if check["command"] in prefix:
            saved = prefix[check["command"]]
            if saved["tensor"] != check["tensor"] or expected != checked_read(saved["expected"]):
                raise ValueError("Diagnostic prefix differs from its validated checkpoint")
            continue
        path = args.output / f"command-{check['command']}-expected.bin"
        path.write_bytes(expected)
        check["expected"] = artifact(path)
        selected.append(check)
    manifest = {key: case[key] for key in ("model", "source", "input", "reference_build")}
    manifest.update(checkpoints=selected, source_layers={str(layer): s["tensor"] for layer, s in mapping.items()})
    path = args.output / "observations.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(selected)} observations / {sum(c['rows'] * c['channels'] for c in selected)} bytes; "
          f"{len(mapping)} source Conv layers; {path}")
    del interpreter


if __name__ == "__main__":
    main()
