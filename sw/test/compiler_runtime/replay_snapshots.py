"""Coherent intermediate memory snapshots with explicitly limited byte coverage."""

import hashlib
import json
from pathlib import Path


def parse_snapshot_boundaries(value):
    try:
        values = [int(part.strip()) for part in value.split(',')]
    except (ValueError, AttributeError) as error:
        raise ValueError('Snapshot boundaries must be comma-separated integers') from error
    if not values or any(v <= 0 for v in values) or values != sorted(set(values)):
        raise ValueError('Snapshot boundaries must be positive, unique and ascending')
    return values


def validate_snapshot_boundaries(case, model, runtime):
    values = case.get('snapshot_boundaries', [])
    if (not isinstance(values, list) or any(type(v) is not int for v in values) or
            values != sorted(set(values))):
        raise ValueError('Invalid intermediate snapshot boundary list')
    for boundary in values:
        if not case['first_command'] < boundary < case['end_command']:
            raise ValueError('Intermediate snapshot is outside execution range; END is saved automatically')
        runtime._require_restart_safe_boundary(model, boundary)
    return values


def observation_verdict(planned, report, boundary):
    planned_prefix = [c['command'] for c in planned if c['command'] <= boundary]
    expected = set(planned_prefix)
    if len(expected) != len(planned_prefix):
        raise ValueError('Intermediate snapshot plan duplicates byte comparisons')
    seen = [c for key in ('checkpoints', 'diagnostic_checkpoints') for c in report.get(key, [])
            if c['command'] <= boundary]
    if len(seen) != len(expected) or {c['command'] for c in seen} != expected:
        raise ValueError('Intermediate snapshot is missing or duplicates expected byte comparisons')
    if any(type(c.get('bytes')) is not int or type(c.get('mismatches')) is not int or
           not 0 <= c['mismatches'] <= c['bytes'] for c in seen):
        raise ValueError('Intermediate snapshot has an invalid comparison verdict')
    mismatches = sum(c['mismatches'] for c in seen)
    return dict(status='FAIL_LISTED_CHECKS' if mismatches else 'PASS_LISTED_CHECKS' if seen else 'UNVERIFIED',
                commands=sorted(expected), comparisons=len(seen), bytes=sum(c['bytes'] for c in seen),
                mismatches=mismatches,
                scope='Only listed observations, not all memory bytes or full inference output')


async def capture_with_paused_clock(clock, level, capture):
    clock.stop()
    try:
        return await capture()
    finally:
        # Restart at the held level, avoiding an extra rising edge.
        clock.start(start_high=bool(level))


def save_intermediate_snapshot(case_path, case, model, report, planned, boundary, state, runtime):
    case_path = Path(case_path).resolve()
    folder = case_path.parent
    path = folder / f'command-{boundary}.snapshot'
    sidecar = folder / f'command-{boundary}.snapshot.json'
    if path.exists() or sidecar.exists():
        raise ValueError('Refusing to overwrite intermediate snapshot evidence')
    if json.loads(case_path.read_text()) != case or hashlib.sha256(model).hexdigest() != case['model']['sha256']:
        raise ValueError('Intermediate snapshot case/model identity changed')
    validate_snapshot_boundaries({**case, 'snapshot_boundaries': [boundary]}, model, runtime)
    tcdm, temporary, output = map(bytes, state)
    if (len(tcdm), len(temporary), len(output)) != (
            runtime.TCDM_TOTAL_BYTES, case['temporary_bytes'], case['output_bytes']):
        raise ValueError('Intermediate snapshot memory sizes differ from the prepared case')
    verdict = observation_verdict(planned, report, boundary)
    encoded = runtime._encode_yolo320_snapshot(model, boundary, tcdm, temporary, output)
    with path.open('xb') as stream:
        stream.write(encoded)
    runtime._verify_yolo320_snapshot(model, path.read_bytes(), expected_boundary=boundary,
        expected_tcdm=tcdm, expected_temporary=temporary, expected_output=output)
    snapshot = dict(path=str(path), sha256=hashlib.sha256(encoded).hexdigest())
    metadata = dict(version=1, boundary=boundary, snapshot=snapshot,
                    case=dict(path=str(case_path), sha256=hashlib.sha256(case_path.read_bytes()).hexdigest()),
                    integrity='VERIFIED', capture='Clock paused; TCDM plus L2 temporary and output',
                    register_state='Not saved; resume needs self-configuring commands or validated configuration restore',
                    async_queues='Model-command queues drained according to command ABI; '
                                 'private command-reader jobs are outside this guarantee',
                    reader_state='Not saved; reset/re-entry discards and rebuilds private '
                                 'command prefetch in unallocated scratch', observations=verdict,
                    memory_bytes=dict(tcdm=len(tcdm), temporary=len(temporary), output=len(output)))
    metadata.update({key: case[key] for key in ('model', 'source', 'firmware', 'input', 'reference_build')})
    metadata['comparison_evidence'] = sorted(
        [c for key in ('checkpoints', 'diagnostic_checkpoints') for c in report.get(key, [])
         if c['command'] <= boundary], key=lambda c: c['command'])
    with sidecar.open('x') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    return dict(command=boundary, snapshot=snapshot,
                evidence=dict(path=str(sidecar), sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest()),
                integrity='VERIFIED', observations=verdict)
