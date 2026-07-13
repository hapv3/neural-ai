#!/usr/bin/env python3
"""Functional coverage for the systolic_controller case matrix.

The coverage database is implemented with cocotb-coverage CoverPoint and
CoverCross objects.  The script can report either the full planned matrix or a
newline-delimited list of cases that actually passed regression.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from cocotb_coverage.coverage import CoverCross, CoverPoint, coverage_db


REPO_ROOT = Path(__file__).resolve().parents[4]
TB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TB_DIR))

from systolic_controller_matrix import ALL_CASES, DirectCase, LinebufCase  # noqa: E402


ROOT = "systolic_ctrl"


DIRECT_M_BINS = {
    "m_001_016": lambda m: 1 <= m <= 16,
    "m_017_032": lambda m: 17 <= m <= 32,
    "m_033_064": lambda m: 33 <= m <= 64,
    "m_065_096": lambda m: 65 <= m <= 96,
    "m_097_128": lambda m: 97 <= m <= 128,
}


def direct_m_bin(dim_m: int | None) -> str | None:
    if dim_m is None:
        return None
    for name, pred in DIRECT_M_BINS.items():
        if pred(dim_m):
            return name
    return None


def padding_bin(case: LinebufCase) -> str:
    return "pad0" if case.pad_h == 0 and case.pad_w == 0 else "padded"


def aspect_bin(case: LinebufCase) -> str:
    if case.input_h == case.input_w:
        return "square"
    if case.input_w > case.input_h:
        return "rect_wide"
    return "rect_tall"


def channel_bin(case: LinebufCase) -> str:
    if case.mode == "bypass" and case.input_c == 33 and case.c_base == 1:
        return "c33_cross"
    if case.mode == "bypass" and case.input_c == 33 and case.c_base == 32:
        return "c33_tail"
    if case.mode == "bypass" and case.input_c == 65 and case.c_base == 33:
        return "c65_cross"
    if case.mode == "bypass" and case.input_c == 65 and case.c_base == 64:
        return "c65_tail"
    logical = case.logical_channels or case.input_c
    return f"c{logical}"


def channel_class(channel: str) -> str:
    if channel in {"c33_cross", "c65_cross"}:
        return "cross_beat"
    if channel in {"c33_tail", "c65_tail"}:
        return "tail"
    if channel == "c1":
        return "c1"
    if channel == "c3":
        return "rgb"
    if channel == "c32":
        return "c32"
    if channel[1:].isdigit() and int(channel[1:]) > 32:
        return "yolo_multi_c32"
    return "sub_c32"


def c_base_class(case: LinebufCase) -> str:
    if case.c_base == 0:
        return "zero"
    if case.c_base % 32 == 0:
        return "tail_aligned"
    return "cross_beat"


def area_bin(value: int | None) -> str | None:
    if value is None:
        return None
    if value <= 4:
        return "area_001_004"
    if value <= 9:
        return "area_005_009"
    if value <= 16:
        return "area_010_016"
    return "area_017_025"


def spatial_m_bin(value: int | None) -> str | None:
    if value is None:
        return None
    if value <= 8:
        return "spatial_001_008"
    if value <= 16:
        return "spatial_009_016"
    if value <= 32:
        return "spatial_017_032"
    return "spatial_033_plus"


def shape_size_bin(case: LinebufCase) -> str:
    max_dim = max(case.input_h, case.input_w)
    if max_dim <= 3:
        return "tiny"
    if max_dim <= 7:
        return "small"
    return "medium"


def flag(value: bool) -> str:
    return "on" if value else "off"


def yolo_channel_bin(case: LinebufCase) -> str | None:
    if case.mode != "kgen_c32fast":
        return None
    return f"yc{case.logical_channels or case.input_c}"


def kernel_cross_bin(kernel_h: int, kernel_w: int) -> str:
    return f"kw{kernel_w}_kh{kernel_h}"


def case_attrs(case) -> dict[str, object]:
    if isinstance(case, DirectCase):
        return {
            "case_id": case.case_id,
            "mode": "direct",
            "direct_m_exact": case.dim_m,
            "direct_m_bin": direct_m_bin(case.dim_m),
            "kernel": None,
            "kernel_cross": None,
            "kernel_h": None,
            "kernel_w": None,
            "kernel_area": None,
            "stride": None,
            "padding": None,
            "aspect": None,
            "shape_size": None,
            "spatial_m": None,
            "spatial_m_bin": None,
            "channel": None,
            "channel_class": None,
            "c_base_class": None,
            "coalesce": "off",
            "kgen": "off",
            "c32_fast": "off",
            "pool": "off",
            "yolo_channel": None,
        }

    assert isinstance(case, LinebufCase)
    channel = channel_bin(case)
    kernel_area = case.kernel_h * case.kernel_w
    return {
        "case_id": case.case_id,
        "mode": case.mode,
        "direct_m_exact": None,
        "direct_m_bin": None,
        "kernel": f"{case.kernel_h}x{case.kernel_w}",
        "kernel_cross": kernel_cross_bin(case.kernel_h, case.kernel_w),
        "kernel_h": case.kernel_h,
        "kernel_w": case.kernel_w,
        "kernel_area": area_bin(kernel_area),
        "stride": f"s{case.stride_h}x{case.stride_w}",
        "padding": padding_bin(case),
        "aspect": aspect_bin(case),
        "shape_size": shape_size_bin(case),
        "spatial_m": case.spatial_m,
        "spatial_m_bin": spatial_m_bin(case.spatial_m),
        "channel": channel,
        "channel_class": channel_class(channel),
        "c_base_class": c_base_class(case),
        "coalesce": flag(case.coalesce),
        "kgen": flag(case.kgen),
        "c32_fast": flag(case.c32_fast),
        "pool": flag(case.pool),
        "yolo_channel": yolo_channel_bin(case),
    }


def planned_attrs() -> list[dict[str, object]]:
    return [case_attrs(case) for case in ALL_CASES.values()]


def ordered_bins(attrs: list[dict[str, object]], key: str):
    values = {item[key] for item in attrs if item[key] is not None}
    return sorted(values, key=lambda value: (str(type(value)), value))


def collect_bins(attrs: list[dict[str, object]]):
    bins = {
        "mode": ["direct", "bypass", "coalesce", "kgen_c32fast", "pool"],
        "direct_m_exact": list(range(1, 129)),
        "direct_m_bin": list(DIRECT_M_BINS),
        "kernel": ordered_bins(attrs, "kernel"),
        "kernel_cross": ordered_bins(attrs, "kernel_cross"),
        "kernel_h": ordered_bins(attrs, "kernel_h"),
        "kernel_w": ordered_bins(attrs, "kernel_w"),
        "kernel_area": ordered_bins(attrs, "kernel_area"),
        "stride": ordered_bins(attrs, "stride"),
        "padding": ordered_bins(attrs, "padding"),
        "aspect": ordered_bins(attrs, "aspect"),
        "shape_size": ordered_bins(attrs, "shape_size"),
        "spatial_m_bin": ordered_bins(attrs, "spatial_m_bin"),
        "channel": ordered_bins(attrs, "channel"),
        "channel_class": ordered_bins(attrs, "channel_class"),
        "c_base_class": ordered_bins(attrs, "c_base_class"),
        "coalesce": ["off", "on"],
        "kgen": ["off", "on"],
        "c32_fast": ["off", "on"],
        "pool": ["off", "on"],
        "yolo_channel": ordered_bins(attrs, "yolo_channel"),
    }
    return bins


COVERPOINT_KEYS = [
    "mode",
    "direct_m_exact",
    "direct_m_bin",
    "kernel",
    "kernel_cross",
    "kernel_h",
    "kernel_w",
    "kernel_area",
    "stride",
    "padding",
    "aspect",
    "shape_size",
    "spatial_m_bin",
    "channel",
    "channel_class",
    "c_base_class",
    "coalesce",
    "kgen",
    "c32_fast",
    "pool",
    "yolo_channel",
]


CROSSES = {
    "direct_m_bin_x_exact": ["mode", "direct_m_bin", "direct_m_exact"],
    "mode_x_kernel": ["mode", "kernel"],
    "mode_x_kernel_cross": ["mode", "kernel_cross"],
    "mode_x_kernel_x_stride": ["mode", "kernel", "stride"],
    "mode_x_kernel_cross_x_stride": ["mode", "kernel_cross", "stride"],
    "mode_x_kernel_x_channel": ["mode", "kernel", "channel"],
    "mode_x_kernel_cross_x_channel": ["mode", "kernel_cross", "channel"],
    "kernel_x_stride_x_padding_x_aspect": ["kernel", "stride", "padding", "aspect"],
    "kernel_cross_x_stride_x_padding_x_aspect": ["kernel_cross", "stride", "padding", "aspect"],
    "kernel_x_shape_x_spatial": ["kernel", "shape_size", "spatial_m_bin"],
    "kernel_cross_x_shape_x_spatial": ["kernel_cross", "shape_size", "spatial_m_bin"],
    "channel_x_class_x_cbase": ["channel", "channel_class", "c_base_class"],
    "mode_x_flags": ["mode", "coalesce", "kgen", "c32_fast", "pool"],
    "mode_x_c32fast_x_channel": ["mode", "c32_fast", "channel"],
    "yolo_channel_x_kernel_x_stride": ["yolo_channel", "kernel", "stride"],
    "yolo_channel_x_kernel_cross_x_stride": ["yolo_channel", "kernel_cross", "stride"],
    "kernel_h_x_kernel_w_x_area": ["kernel_h", "kernel_w", "kernel_area"],
    "kernel_cross_x_area": ["kernel_cross", "kernel_area"],
}


def valid_cross_bins(attrs: list[dict[str, object]], keys: list[str]):
    bins = set()
    for item in attrs:
        values = tuple(item[key] for key in keys)
        if all(value is not None for value in values):
            bins.add(values)
    return bins


def ignored_cross_bins(all_bins: dict[str, list], keys: list[str], valid_bins: set[tuple]):
    cartesian = itertools.product(*(all_bins[key] for key in keys))
    return [values for values in cartesian if values not in valid_bins]


def register_coverage_model():
    coverage_db.clear()
    attrs = planned_attrs()
    bins = collect_bins(attrs)

    coverpoints = {
        key: CoverPoint(f"{ROOT}.{key}", xf=lambda item, key=key: item[key], bins=bins[key])
        for key in COVERPOINT_KEYS
    }
    crosses = {}
    for cross_name, keys in CROSSES.items():
        item_names = [f"{ROOT}.{key}" for key in keys]
        valid_bins = valid_cross_bins(attrs, keys)
        ignore_bins = ignored_cross_bins(bins, keys, valid_bins)
        crosses[cross_name] = CoverCross(
            f"{ROOT}.{cross_name}",
            items=item_names,
            ign_bins=ignore_bins,
        )

    sample = lambda item: None
    for cross_name, keys in CROSSES.items():
        sample = crosses[cross_name](sample)

    for key in reversed(COVERPOINT_KEYS):
        sample = coverpoints[key](sample)

    return sample


def load_case_ids(path: Path | None):
    if path is None:
        return list(ALL_CASES)
    case_ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            case_ids.append(line)
    return case_ids


def sample_cases(case_ids: list[str]):
    sample = register_coverage_model()
    unknown = []
    for case_id in case_ids:
        case = ALL_CASES.get(case_id)
        if case is None:
            unknown.append(case_id)
            continue
        sample(case_attrs(case))
    return unknown


def db_item_summary(name: str):
    item = coverage_db[name]
    detailed = getattr(item, "detailed_coverage", {})
    covered = [str(bin_name) for bin_name, hits in detailed.items() if hits]
    missing = [str(bin_name) for bin_name, hits in detailed.items() if not hits]
    return {
        "covered_count": len(covered),
        "total_count": len(detailed),
        "percent": round(float(item.cover_percentage), 2),
        "covered": covered,
        "missing": missing,
    }


def make_json_report(case_ids: list[str], unknown: list[str]):
    groups = {}
    for key in COVERPOINT_KEYS:
        groups[key] = db_item_summary(f"{ROOT}.{key}")
    for cross_name in CROSSES:
        groups[cross_name] = db_item_summary(f"{ROOT}.{cross_name}")
    return {
        "coverage_engine": "cocotb-coverage",
        "case_count": len(case_ids),
        "known_case_count": len(case_ids) - len(unknown),
        "unknown_cases": unknown,
        "root_coverage_percent": round(float(coverage_db[ROOT].cover_percentage), 2),
        "groups": groups,
    }


def write_markdown(report: dict, path: Path):
    lines = [
        "# Systolic Controller Cocotb Coverage",
        "",
        f"Engine: `{report['coverage_engine']}`",
        f"Cases sampled: `{report['known_case_count']}`",
        f"Root coverage: `{report['root_coverage_percent']:.2f}%`",
        "",
        "| Cover item | Covered | Total | Percent |",
        "|---|---:|---:|---:|",
    ]
    for name, group in report["groups"].items():
        lines.append(
            f"| `{name}` | {group['covered_count']} | {group['total_count']} | {group['percent']:.2f}% |"
        )
    lines.extend(["", "## Missing Bins", ""])
    any_missing = False
    for name, group in report["groups"].items():
        missing = group["missing"]
        if missing:
            any_missing = True
            preview = ", ".join(missing[:80])
            suffix = "" if len(missing) <= 80 else f", ... ({len(missing) - 80} more)"
            lines.append(f"- `{name}`: {preview}{suffix}")
    if not any_missing:
        lines.append("- None")
    if report["unknown_cases"]:
        lines.extend(["", "## Unknown Cases", ""])
        for case_id in report["unknown_cases"]:
            lines.append(f"- `{case_id}`")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-list", type=Path, help="Optional newline-delimited list of passed case IDs")
    parser.add_argument("--out-json", type=Path, default=TB_DIR / "coverage" / "systolic_controller_coverage.json")
    parser.add_argument("--out-md", type=Path, default=TB_DIR / "coverage" / "systolic_controller_coverage.md")
    parser.add_argument("--out-yaml", type=Path, default=TB_DIR / "coverage" / "systolic_controller_coverage.yml")
    parser.add_argument("--out-xml", type=Path, default=TB_DIR / "coverage" / "systolic_controller_coverage.xml")
    args = parser.parse_args()

    case_ids = load_case_ids(args.case_list)
    unknown = sample_cases(case_ids)
    report = make_json_report(case_ids, unknown)

    for path in [args.out_json, args.out_md, args.out_yaml, args.out_xml]:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    write_markdown(report, args.out_md)
    coverage_db.export_to_yaml(filename=str(args.out_yaml))
    coverage_db.export_to_xml(filename=str(args.out_xml))

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_yaml}")
    print(f"wrote {args.out_xml}")


if __name__ == "__main__":
    main()
