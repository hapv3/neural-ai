"""Lossless host-side PMU reports; no simulator activity or counter deltas."""

import csv
import json
import os
from pathlib import Path
import re


def write_pmu_report(counters, report_name, counter_names, *, status, output_dir=None):
    """Save an aggregate firmware-window snapshot, not per-command/kernel timing.

    IRQ_REACHED is not a numerical PASS. Missing counters remain missing, peak
    values are copied verbatim, and overflow invalidates performance analysis.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", report_name):
        raise ValueError("Invalid PMU report name")
    if status not in ("IRQ_REACHED", "PASS", "FAIL", "TIMEOUT", "ERROR"):
        raise ValueError("Invalid PMU report status")
    names = list(counter_names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate PMU counter names")
    raw = dict(counters)
    for name, value in raw.items():
        if not isinstance(name, str) or type(value) is not int or value < 0:
            raise ValueError("PMU values must be nonnegative integers")
        if name in names and value >= 1 << 64:
            raise ValueError("PMU counter exceeds 64 bits")
    missing = [name for name in names if name not in raw]
    complete = not missing and raw.get("num_counters", len(names)) == len(names)
    metadata = {
        "report_name": report_name,
        "status": status,
        "measurement_scope": "standalone_operator_firmware_window",
        "full_counter_set": complete,
        "performance_valid": status == "PASS" and complete and
                             raw.get("num_counters") == len(names) and raw.get("overflow_status") == 0 and
                             raw.get("cycle", 0) > 0,
    }
    if set(metadata).intersection(raw):
        raise ValueError("PMU counter names collide with report metadata")
    if output_dir is None:
        output_dir = os.environ.get("SIM_BUILD")
        if not output_dir:
            output_dir = Path(os.environ.get("COCOTB_RESULTS_FILE", "results.xml")).parent
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{report_name}_pmu.json"
    csv_path = folder / f"{report_name}_pmu.csv"
    payload = {
        "schema_version": 1,
        **metadata,
        "expected_counter_count": len(names),
        "captured_counter_count": sum(name in raw for name in names),
        "missing_counters": missing,
        "scope_note": "Includes firmware setup/control and in-firmware validation. Not isolated kernel "
                      "latency, per-case PMU, or per-command ABI attribution.",
        "counters": raw,
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    fields = [*metadata, *names, *(name for name in raw if name not in names)]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**metadata, **raw})
    return json_path, csv_path
