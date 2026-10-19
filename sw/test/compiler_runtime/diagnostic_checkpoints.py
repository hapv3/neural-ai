"""Opt-in Conv observations; these do not replace validated gates/final output."""

import hashlib
import json
import math
import struct

from yolo_c2f_checkpoints import checkpoint_indices


def load_diagnostic_checkpoints(case, commands, local_bytes, checked_read):
    record = case.get("diagnostic_checkpoints")
    if record is None:
        return []
    if case.get("remaining_graph") is not True:
        raise ValueError("Diagnostic observations require a remaining-graph continuation")
    manifest = json.loads(checked_read(record))
    for key in ("model", "source", "input", "reference_build"):
        if manifest.get(key) != case.get(key):
            raise ValueError(f"Diagnostic {key} provenance mismatch")
    checks = manifest.get("checkpoints")
    if not isinstance(checks, list) or not checks:
        raise ValueError("Diagnostic observations require nonempty checkpoints")
    seen = {check["command"] for check in case["checkpoints"]}
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Invalid diagnostic checkpoint record")
        number = check.get("command")
        if (type(number) is not int or not case["first_command"] < number <= case["end_command"] or
                number in seen):
            raise ValueError("Diagnostic checkpoint is duplicated or outside execution range")
        seen.add(number)
        submit = check.get("submit_command", number)
        if type(submit) is not int or not case["first_command"] < submit <= number:
            raise ValueError("Invalid diagnostic Conv submission")
        command = commands[submit - 1]
        kind, _, flags = struct.unpack_from("<HHI", command)
        if check.get("command_sha256") != hashlib.sha256(command).hexdigest():
            raise ValueError("Diagnostic command bytes changed")
        if submit != number:
            wait = commands[number - 1]
            if (kind != 29 or struct.unpack_from("<HHI", wait) != (30, 32, 0) or
                    wait[8:16] != command[8:16] or
                    check.get("completion_sha256") != hashlib.sha256(wait).hexdigest() or
                    any(struct.unpack_from("<H", c)[0] in (9, 10, 29, 30) for c in commands[submit:number - 1])):
                raise ValueError("Diagnostic asynchronous Conv lacks its unambiguous matching wait")
        elif kind == 29:
            raise ValueError("Diagnostic asynchronous Conv requires completion, not submission")
        if kind not in (9, 10, 29) or flags or (kind in (9, 29) and
                struct.unpack_from("<I", command, 116)[0] not in (0, 2)):
            raise ValueError("Diagnostic Conv requires final requant output")
        address, rows = struct.unpack_from("<2I", command, 108 if kind in (9, 29) else 44)
        cols = struct.unpack_from("<I", command, 124)[0] if kind in (9, 29) else check.get("tile_cols")
        if type(cols) is not int or cols <= 0:
            raise ValueError("Invalid diagnostic output columns")
        if kind == 10 and struct.unpack_from("<HH", command, 40) != (6, 0):
            raise ValueError("Diagnostic pointwise output must be local TCDM")
        pitch = struct.unpack_from("<I", command, 120)[0] if kind in (9, 29) else cols * 32
        shape = check.get("reference_shape")
        if (not isinstance(shape, list) or len(shape) != 4 or shape[0] != 1 or
                any(type(v) is not int or v <= 0 for v in shape) or
                any(type(check.get(key)) is not int for key in
                    ("address", "rows", "channels", "tile_cols", "row_stride",
                     "reference_first_row", "reference_first_channel", "tensor")) or
                not 0 < check["channels"] <= 32 or
                (check["address"], check["rows"], check["tile_cols"], check["row_stride"]) !=
                (address, rows, cols, pitch) or cols != shape[2] or
                check["reference_first_row"] < 0 or check["reference_first_channel"] < 0 or
                check["reference_first_row"] + rows > math.prod(shape[1:3]) or
                check["reference_first_channel"] + check["channels"] > shape[3] or
                any(key in check for key in ("reference_spans", "group_stride", "element_stride"))):
            raise ValueError("Diagnostic output geometry differs from Conv ABI or reference shape")
        checkpoint_indices(check, local_bytes)
        if len(checked_read(check["expected"])) != rows * check["channels"]:
            raise ValueError("Diagnostic oracle size differs from output geometry")
    return checks
