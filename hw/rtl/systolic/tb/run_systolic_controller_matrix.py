#!/usr/bin/env python3
"""Run systolic_controller matrix cases as independent cocotb simulations."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TB_DIR = REPO_ROOT / "hw" / "rtl" / "systolic" / "tb"
SYSTOLIC_DIR = REPO_ROOT / "hw" / "rtl" / "systolic"
SHARED_SIM_BUILD = TB_DIR / "sim" / "test_systolic_controller_matrix_case"
SHARED_VTOP = SHARED_SIM_BUILD / "Vtop"
MATRIX_RESULTS_DIR = TB_DIR / "sim" / "matrix_results"
COVERAGE_TOOL = TB_DIR / "systolic_controller_cocotb_coverage.py"
sys.path.insert(0, str(TB_DIR))

from systolic_controller_matrix import ALL_CASES, default_smoke_case_ids  # noqa: E402


def base_env(case_id: str):
    env = os.environ.copy()
    env.setdefault("CCACHE_DIR", "/tmp/ccache")
    env.setdefault("CCACHE_TEMPDIR", "/tmp/ccache-tmp")
    env["SYSTOLIC_CTRL_MATRIX_CASE"] = case_id
    env["PYTHONPATH"] = str(TB_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["COCOTB_TEST_MODULES"] = "test_systolic_controller_matrix_case"
    env["COCOTB_TOPLEVEL"] = "tb_systolic_controller"
    env["TOPLEVEL_LANG"] = "verilog"
    env.setdefault("PYGPI_PYTHON_BIN", sys.executable)
    env["COCOTB_RESULTS_FILE"] = str(MATRIX_RESULTS_DIR / f"{case_id}.xml")
    return env


def shared_run_cmd(jobs: int):
    return [
        str(SHARED_VTOP),
        "-Wno-fatal",
        "--timing",
        "-j",
        str(jobs),
        f"+incdir+{REPO_ROOT / 'hw' / 'common_cells' / 'include'}",
        "release",
    ]


def isolated_build_cmd(case_id: str, jobs: int):
    sim_build = REPO_ROOT / "hw" / "rtl" / "systolic" / "tb" / "sim" / f"matrix_{case_id}"
    return [
        "make",
        "-C",
        "hw/rtl/systolic",
        "MODULE=test_systolic_controller_matrix_case",
        f"SIM_BUILD={sim_build}",
        f"VERILATOR_JOBS={jobs}",
    ]


def ensure_shared_build(jobs: int, log_dir: Path, rebuild: bool):
    MATRIX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SHARED_VTOP.exists() and not rebuild:
        return
    env = base_env("direct_m_001")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "shared_build.log"
    cmd = [
        "make",
        "-C",
        "hw/rtl/systolic",
        "MODULE=test_systolic_controller_matrix_case",
        f"VERILATOR_JOBS={jobs}",
    ]
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_one(case_id: str, jobs: int, timeout_s: int, log_dir: Path, isolated_build: bool):
    MATRIX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = base_env(case_id)
    cmd = isolated_build_cmd(case_id, jobs) if isolated_build else shared_run_cmd(jobs)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case_id}.log"
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=SYSTOLIC_DIR, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = 124
            timed_out = True
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
    return case_id, returncode, log_path, timed_out


def write_coverage(case_list: Path, prefix: Path):
    cmd = [
        sys.executable,
        str(COVERAGE_TOOL),
        "--case-list",
        str(case_list),
        "--out-json",
        str(prefix.with_suffix(".json")),
        "--out-md",
        str(prefix.with_suffix(".md")),
        "--out-yaml",
        str(prefix.with_suffix(".yml")),
        "--out-xml",
        str(prefix.with_suffix(".xml")),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run the full matrix. Default runs a smoke subset.")
    parser.add_argument("--case", action="append", help="Run one case ID. Can be repeated.")
    parser.add_argument("--randomize", action="store_true", help="Shuffle the selected case order before running.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed used with --randomize.")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--verilator-jobs", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--quiet", action="store_true", help="Only print failures and periodic progress.")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--log-dir", type=Path, default=TB_DIR / "coverage" / "matrix_logs")
    parser.add_argument("--passed-out", type=Path, default=TB_DIR / "coverage" / "systolic_controller_passed_cases.txt")
    parser.add_argument("--coverage-prefix", type=Path, default=TB_DIR / "coverage" / "systolic_controller_regression_coverage")
    parser.add_argument("--no-coverage", action="store_true", help="Do not generate cocotb-coverage reports from passed cases.")
    parser.add_argument("--isolated-build", action="store_true", help="Compile a separate Verilator build per case.")
    parser.add_argument("--rebuild-shared", action="store_true", help="Run make before shared-executable regression.")
    args = parser.parse_args()

    if args.case:
        case_ids = args.case
    elif args.all:
        case_ids = sorted(ALL_CASES)
    else:
        case_ids = default_smoke_case_ids()

    unknown = [case_id for case_id in case_ids if case_id not in ALL_CASES]
    if unknown:
        raise SystemExit(f"unknown case IDs: {', '.join(unknown)}")
    if args.randomize:
        rng = random.Random(args.seed)
        rng.shuffle(case_ids)
        print(f"randomized {len(case_ids)} cases with seed={args.seed}")

    passed = []
    failed = []
    completed = 0
    if not args.isolated_build:
        ensure_shared_build(args.verilator_jobs, args.log_dir, args.rebuild_shared)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(
                run_one,
                case_id,
                args.verilator_jobs,
                args.timeout_s,
                args.log_dir,
                args.isolated_build,
            )
            for case_id in case_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            case_id, returncode, log_path, timed_out = future.result()
            completed += 1
            if returncode == 0:
                if not args.quiet:
                    print(f"PASS {case_id} ({log_path})")
                passed.append(case_id)
            else:
                status = "TIMEOUT" if timed_out else "FAIL"
                print(f"{status} {case_id} rc={returncode} ({log_path})")
                failed.append(case_id)
            if args.quiet and args.progress_every and (
                completed % args.progress_every == 0 or completed == len(case_ids)
            ):
                print(f"progress: {completed}/{len(case_ids)} passed={len(passed)} failed={len(failed)}")

    args.passed_out.parent.mkdir(parents=True, exist_ok=True)
    args.passed_out.write_text("\n".join(sorted(passed)) + ("\n" if passed else ""))
    print(f"passed: {len(passed)} failed: {len(failed)}")
    print(f"wrote {args.passed_out}")
    if passed and not args.no_coverage:
        write_coverage(args.passed_out, args.coverage_prefix)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
