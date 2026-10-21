"""Selected command-end TCDM evidence, explicitly not a numerical verdict."""

import hashlib
import json
from pathlib import Path
import struct
import zlib

COMPUTE_TYPES = frozenset(range(6, 21)) | {28, 29, 31, 32, 34}


def plan_compute_dumps(commands, first, end):
    planned = []
    for number in range(first + 1, end + 1):
        command = commands[number - 1]
        kind, size, flags, layer, tile = struct.unpack_from('<HHIII', command)
        if kind not in COMPUTE_TYPES:
            continue
        completion = number
        if kind in (29, 32):
            completion = None
            for later in range(number + 1, end + 1):
                following = commands[later - 1]
                following_type = struct.unpack_from('<H', following)[0]
                if following_type in (6, 7, 8, 9, 10, 11, 29, 31, 32):
                    raise ValueError('Async compute dump has ambiguous completion')
                if following_type == 30:
                    if following[8:16] != command[8:16]:
                        raise ValueError('Async compute dump wait has different layer/tile')
                    completion = later
                    break
            if completion is None:
                raise ValueError('Async compute dump has no matching wait')
        planned.append(dict(command=number, completion_command=completion, type=kind, size=size,
                            flags=flags, layer_id=layer, tile_id=tile,
                            command_sha256=hashlib.sha256(command).hexdigest()))
    return planned


def select_compute_dumps(planned, limit):
    """Keep every layer's terminal compute; fill spare slots across interiors."""
    if type(limit) is not int or limit <= 0:
        raise ValueError('Compute checkpoint limit must be a positive integer')
    last = {check['layer_id']: check for check in planned}
    if len(last) > limit:
        raise ValueError('Compute limit cannot cover every layer terminal; increase the limit')
    selected = {check['command']: check for check in last.values()}
    available = [check for check in planned if check['command'] not in selected]
    count = min(limit - len(selected), len(available))
    for index in range(count):
        # Uniform midpoint samples without duplicates; layer ends stay mandatory.
        check = available[((2 * index + 1) * len(available)) // (2 * count)]
        selected[check['command']] = check
    return sorted(selected.values(), key=lambda check: check['command'])


def validate_compute_dump_case(case, commands, checked_read):
    record = case.get('compute_dump_reference')
    if record is None:
        return []
    if case.get('full_graph') is not True or case.get('start_mode') != 'reset':
        raise ValueError('Compute checkpoint dumping requires a full reset run')
    manifest = json.loads(checked_read(record))
    for key in ('model', 'source', 'firmware', 'input', 'reference_build'):
        if manifest.get(key) != case.get(key):
            raise ValueError(f'Compute dump {key} provenance mismatch')
    checked_read(manifest['reference_tensors'])
    planned = plan_compute_dumps(commands, case['first_command'], case['end_command'])
    if 'compute_checkpoint_limit' in manifest:
        planned = select_compute_dumps(planned, manifest['compute_checkpoint_limit'])
    if not planned or manifest.get('compute_checkpoints') != planned:
        raise ValueError('Stale or empty compute dump plan')
    return planned


class ComputeDumpWriter:
    def __init__(self, folder, planned, local_bytes):
        self.folder = Path(folder) / 'compute-checkpoints'
        self.folder.mkdir()  # Refuse to overwrite even partial earlier evidence.
        self.path = self.folder / 'checkpoints.jsonl'
        self.local_bytes = local_bytes
        self.by_completion = {}
        for check in planned:
            self.by_completion.setdefault(check['completion_command'], []).append(check)
        self.planned = {check['command'] for check in planned}
        if len(self.planned) != len(planned):
            raise ValueError('Duplicate compute dump commands')
        self.seen = set()
        self.compressed_bytes = 0

    def wants(self, completion):
        return completion in self.by_completion

    def capture(self, completion, memory):
        checks = self.by_completion.get(completion, [])
        if not checks:
            return
        if len(memory) != self.local_bytes:
            raise ValueError('Compute dump TCDM size mismatch')
        memory = bytes(memory)
        compressed = zlib.compress(memory, level=1)
        digest = hashlib.sha256(memory).hexdigest()
        for check in checks:
            number = check['command']
            if number in self.seen:
                raise ValueError('Refusing to overwrite compute checkpoint')
            path = self.folder / f'command-{number}.tcdm.zlib'
            with path.open('xb') as stream:
                stream.write(compressed)
            # Check persisted compression/integrity, not numerical correctness.
            if zlib.decompress(path.read_bytes()) != memory:
                raise ValueError('Compute dump readback mismatch')
            entry = dict(check, path=str(path.resolve()), tcdm_bytes=len(memory), tcdm_sha256=digest,
                         compressed_bytes=len(compressed), compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                         integrity='VERIFIED', numerical_status='NOT_COMPARED',
                         scope='Instantaneous TCDM only; async queues/registers/L2 are not restart state')
            with self.path.open('a') as stream:
                stream.write(json.dumps(entry, separators=(',', ':')) + '\n')
                stream.flush()
            self.seen.add(number)
            self.compressed_bytes += len(compressed)

    def summary(self, complete=False):
        if complete and self.seen != self.planned:
            raise ValueError('Missing selected compute checkpoints')
        result = dict(planned=len(self.planned), captured=len(self.seen), compressed_bytes=self.compressed_bytes,
                      numerical_status='NOT_COMPARED',
                      scope='Raw per-compute evidence; only independently compared outputs can establish PASS')
        if self.path.exists():
            result['manifest'] = dict(path=str(self.path.resolve()), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        return result
