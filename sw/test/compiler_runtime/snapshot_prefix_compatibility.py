"""Prove memory-snapshot compatibility without rewriting its original identity."""

import hashlib
import struct

import test_compiled_model as runtime


def prove_snapshot_prefix(previous, current, boundary):
    def sections(model):
        if len(model) < 224:
            raise ValueError("Snapshot compatibility model is truncated")
        header = struct.unpack_from("<16I", model)
        if header[4] != len(model) or header[5:7] != (5, 64):
            raise ValueError("Snapshot compatibility requires the complete five-section ABI")
        result = {}
        for index in range(5):
            entry = struct.unpack_from("<8I", model, 64 + index * 32)
            kind, _, offset, size = entry[:4]
            if kind != index + 1 or offset < 224 or offset + size > len(model):
                raise ValueError("Snapshot compatibility section layout is invalid")
            result[kind] = (entry, model[offset:offset + size])
        return header, result

    old_header, old_sections = sections(previous)
    new_header, new_sections = sections(current)
    if any(old_header[i] != new_header[i] for i in range(16) if i not in (4, 8)):
        raise ValueError("Snapshot compatibility ABI, memory requirements or binding counts changed")
    for kind in range(1, 6):
        old_entry, old_data = old_sections[kind]
        new_entry, new_data = new_sections[kind]
        mutable = (2, 3, 5) if kind == 1 else (2,)
        if any(old_entry[i] != new_entry[i] for i in range(8) if i not in mutable):
            raise ValueError("Snapshot compatibility section metadata changed")
        if kind != 1 and old_data != new_data:
            raise ValueError("Snapshot compatibility constants, qparams or binding layout changed")
    old_commands, _ = runtime._logical_command_records(previous)
    new_commands, _ = runtime._logical_command_records(current)
    if type(boundary) is not int or not 0 < boundary < min(len(old_commands), len(new_commands)):
        raise ValueError("Snapshot compatibility boundary is invalid")
    if old_commands[:boundary] != new_commands[:boundary]:
        raise ValueError("Snapshot compatibility executed command prefix changed")
    runtime._require_restart_safe_boundary(previous, boundary)
    runtime._require_restart_safe_boundary(current, boundary)
    return dict(boundary=boundary,
                previous_model_sha256=hashlib.sha256(previous).hexdigest(),
                current_model_sha256=hashlib.sha256(current).hexdigest(),
                prefix_sha256=hashlib.sha256(b"".join(old_commands[:boundary])).hexdigest())
