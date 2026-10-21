"""Prepare one full reset run with bounded compute dumps and all source tensors."""

import argparse
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np

from compute_dumps import plan_compute_dumps, select_compute_dumps
from yolo_reference_replay import artifact, checked_read, load_case, runtime


def safe_layer_boundaries(model, commands):
    last = {}
    for number, command in enumerate(commands, 1):
        last[struct.unpack_from('<I', command, 8)[0]] = number
    boundaries = set()
    for final in last.values():
        # Save at the first globally drained boundary after a layer terminates.
        for boundary in range(final, len(commands)):
            try:
                runtime._require_restart_safe_boundary(model, boundary)
            except ValueError:
                continue
            boundaries.add(boundary)
            break
    return sorted(boundaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-compute-checkpoints', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh full-checkpoint output directory')
    case, model, *_ = load_case(args.case)
    if not case.get('full_graph') or case.get('start_mode') != 'reset':
        raise ValueError('Full compute checkpoints require an original full reset case')
    root = args.compiler_root / 'tools/neural_ai_tflite_reference'
    if json.loads((root / 'build/manifest.json').read_text()) != case['reference_build']:
        raise ValueError('Reference build changed')
    spec = importlib.util.spec_from_file_location('independent_reference', root / 'runtime.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    interpreter = reference.make_interpreter(case['source']['path'], preserve=True)
    if reference.coverage(interpreter)['mac_fallback_nodes']:
        raise ValueError('Source reference has unsupported MAC fallbacks')
    detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(detail['index'], np.frombuffer(checked_read(case['input']), np.int8).reshape(detail['shape']))
    interpreter.invoke()
    if interpreter.get_tensor(case['final_output']['tensor']).tobytes() != checked_read(case['final_output']['expected']):
        raise ValueError('Full-checkpoint preparation changed the pinned final oracle')
    details = {int(t['index']): t for t in interpreter.get_tensor_details()}
    arrays, source_ops = {}, []
    for op in interpreter._get_ops_details():
        if op['op_name'] == 'DELEGATE':
            continue
        outputs = []
        for index in map(int, op['outputs']):
            if index < 0:
                continue
            tensor = interpreter.get_tensor(index)
            arrays[f'tensor_{index}'] = tensor
            outputs.append(dict(tensor=index, name=details[index]['name'], shape=list(map(int, tensor.shape)),
                                dtype=str(tensor.dtype), bytes=tensor.nbytes,
                                quantization=list(details[index]['quantization'])))
        source_ops.append(dict(operator=int(op['index']), type=op['op_name'], outputs=outputs))
    commands, _ = runtime._logical_command_records(model)
    planned = plan_compute_dumps(commands, 0, len(commands))
    total_compute = len(planned)
    planned = select_compute_dumps(planned, args.max_compute_checkpoints)
    args.output.mkdir(parents=True)
    tensors = args.output / 'reference-tensors.npz'
    np.savez_compressed(tensors, **arrays)
    manifest = {key: case[key] for key in ('model', 'source', 'firmware', 'input', 'reference_build')}
    manifest.update(reference_tensors=artifact(tensors), source_operators=source_ops, compute_checkpoints=planned,
                    compute_checkpoint_limit=args.max_compute_checkpoints, total_compute_commands=total_compute,
                    selection='All layer-terminal compute commands, then uniform interior fill if budget remains',
                    scope='Reference source tensors and raw compute dumps; intermediate command-to-source comparisons are deferred')
    path = args.output / 'compute-reference.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    case.update(compute_dump_reference=artifact(path), snapshot_boundaries=safe_layer_boundaries(model, commands),
                fail_fast=False)
    case_path = args.output / 'case.json'
    case_path.write_text(json.dumps(case, indent=2) + '\n')
    load_case(case_path)
    print(f'Prepared full commands 1-{len(commands)}, {len(planned)}/{total_compute} compute checkpoints, '
          f'{len(source_ops)} reference source operators / {len(arrays)} tensors, '
          f'{len(case["snapshot_boundaries"])} full-memory intermediate snapshots: {case_path}')
    del interpreter


if __name__ == '__main__':
    main()
