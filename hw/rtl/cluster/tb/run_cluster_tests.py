#!/usr/bin/env python3
"""Parallel cocotb runner for the NPU cluster testbench.

The cluster Makefile can run multiple cocotb modules in one simulator process,
but those modules execute serially.  This runner builds one shared Verilator
binary per RTL generic set, then launches that binary multiple times in
parallel with different cocotb modules, result XMLs, and logs.

Conv-perf case sweeps are handled specially: one `conv_perf.bin` firmware image
is used for every case.  The cocotb host writes the requested case selector into
L2 before releasing fetch, so a case sweep no longer needs firmware rebuilds.
By default the sweep uses the normal shared binary and runs each case as a
separate simulator process with its own result XML.  The firmware binary is not
rebuilt between cases.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CLUSTER_DIR = REPO_ROOT / "hw" / "rtl" / "cluster"
TEST_DIR = CLUSTER_DIR / "tb" / "tests"

HELPER_MODULES = {
    "npu_test_utils",
    "systolic_independent_common",
}

DEFAULT_OFM_ENV = {
    "test_systolic_ofm_fifo_small_depth": {"SYSTOLIC_OFM_FIFO_DEPTH": "2"},
    "test_systolic_ofm_fifo_stall_sweep": {
        "SYSTOLIC_OFM_FIFO_DEPTH": "8",
        "SYSTOLIC_OTCDM_STALL_PERIOD": "5",
        "SYSTOLIC_OTCDM_STALL_HOLD": "3",
    },
}

GENERIC_ENV_KEYS = (
    "SYSTOLIC_OFM_FIFO_DEPTH",
    "SYSTOLIC_OTCDM_STALL_PERIOD",
    "SYSTOLIC_OTCDM_STALL_HOLD",
)

FIRMWARE_TARGETS = {
    "sw/test/boot",
    "sw/test/afu",
    "sw/test/spatz_ops",
    "sw/test/conv_perf",
    "sw/test/conv3x3_multi_linebuf",
    "sw/test/depthwise_conv",
    "sw/test/pointwise_conv",
    "sw/test/micro_yolo",
    "sw/test/micro_mobilenet",
    "sw/test/systolic_requant",
    "sw/test/independent_memory",
    "sw/test/matmul",
    "sw/test/pmu",
    "sw/test/spatz_vector",
    "sw/test/independent_systolic",
}


@dataclass(frozen=True)
class Job:
    name: str
    module: str
    env: dict[str, str] = field(default_factory=dict)
    sim_name: str | None = None
    result_name: str | None = None

    def make_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("CCACHE_DISABLE", "1")
        env.setdefault("COCOTB_LOG_LEVEL", "WARNING")
        env.setdefault("COCOTB_REDUCED_LOG_FMT", "1")
        env.setdefault("PYGPI_PYTHON_BIN", sys.executable)
        env["PYTHONPATH"] = ":".join(
            [
                str(CLUSTER_DIR / "tb"),
                str(TEST_DIR),
                env.get("PYTHONPATH", ""),
            ]
        )
        env.update(self.env)
        return env

    def generic_env(self) -> dict[str, str]:
        env = self.make_env()
        return {key: env[key] for key in GENERIC_ENV_KEYS if key in env}

    def command(self) -> list[str]:
        sim_name = self.sim_name or self.name
        result_name = self.result_name or f"{sim_name}/results.xml"
        return [
            "make",
            "-C",
            str(CLUSTER_DIR),
            "sim",
            f"CLUSTER_SIM_NAME={sim_name}",
            f"COCOTB_RESULTS_FILE=tb/sim/{result_name}",
            f"COCOTB_TEST_MODULES={self.module}",
        ]

    def result_path(self, sim_name: str) -> Path:
        result_name = self.result_name or f"{self.name}.xml"
        return CLUSTER_DIR / "tb" / "sim" / sim_name / result_name

    def log_path(self, sim_name: str) -> Path:
        return CLUSTER_DIR / "tb" / "sim" / sim_name / f"{self.name}.log"


def discover_modules() -> list[str]:
    modules = []
    for path in sorted(TEST_DIR.glob("test_*.py")):
        if path.stem not in HELPER_MODULES:
            modules.append(path.stem)
    return modules


def parse_list(value: str, all_values: list[str]) -> list[str]:
    if value == "all":
        return all_values
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(all_values))
    if unknown:
        raise SystemExit(f"Unknown test module(s): {', '.join(unknown)}")
    return requested


def parse_case_list(value: str) -> list[int]:
    if value == "all":
        return list(range(24))
    cases: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = item.split("-", 1)
            cases.extend(range(int(first), int(last) + 1))
        else:
            cases.append(int(item))
    return cases


def build_firmware() -> None:
    for target in sorted(FIRMWARE_TARGETS):
        print(f"[fw] make -C {target}", flush=True)
        subprocess.run(["make", "-C", str(REPO_ROOT / target)], check=True)


def run_make_job(job: Job) -> int:
    print(f"[run] {job.name}: {' '.join(job.command())}", flush=True)
    completed = subprocess.run(job.command(), cwd=REPO_ROOT, env=job.make_env())
    print(f"[done] {job.name}: rc={completed.returncode}", flush=True)
    return completed.returncode


def tail_file(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def sim_name_for_generic(generic_env: dict[str, str]) -> str:
    if not generic_env:
        return "cluster_shared"
    suffix = "_".join(f"{key.lower()}_{value}" for key, value in sorted(generic_env.items()))
    return "cluster_shared_" + sanitize_name(suffix)


def build_shared_binary(sim_name: str, generic_env: dict[str, str]) -> int:
    vtop = CLUSTER_DIR / "tb" / "sim" / sim_name / "Vtop"
    env = os.environ.copy()
    env.setdefault("CCACHE_DISABLE", "1")
    env.setdefault("COCOTB_LOG_LEVEL", "WARNING")
    env.setdefault("COCOTB_REDUCED_LOG_FMT", "1")
    env.setdefault("PYGPI_PYTHON_BIN", sys.executable)
    env.update(generic_env)

    cmd = [
        "make",
        "-C",
        str(CLUSTER_DIR),
        "sim",
        f"CLUSTER_SIM_NAME={sim_name}",
        f"COCOTB_RESULTS_FILE=tb/sim/{sim_name}/prewarm.xml",
        "COCOTB_TEST_MODULES=test_snitch_boot",
    ]
    print(f"[build] {sim_name}: ensure {vtop}", flush=True)
    rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
    if rc == 0 and not vtop.exists():
        print(f"[build] {sim_name}: {vtop} was not produced", file=sys.stderr, flush=True)
        return 1
    return rc


def run_shared_job(job: Job, sim_name: str) -> int:
    sim_dir = CLUSTER_DIR / "tb" / "sim" / sim_name
    vtop = sim_dir / "Vtop"
    result_path = job.result_path(sim_name)
    log_path = job.log_path(sim_name)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    env = job.make_env()
    env["COCOTB_TEST_MODULES"] = job.module
    env.setdefault("COCOTB_TESTCASE", "")
    env.setdefault("COCOTB_TEST_FILTER", "")
    env["COCOTB_TOPLEVEL"] = "tb_npu_cluster"
    env["TOPLEVEL_LANG"] = "verilog"
    env["COCOTB_RESULTS_FILE"] = str(result_path)

    cmd = [str(vtop), "release"]
    print(f"[run] {job.name}: {vtop.name} -> {result_path.relative_to(CLUSTER_DIR)}", flush=True)
    with log_path.open("w") as log_file:
        completed = subprocess.run(cmd, cwd=CLUSTER_DIR, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if completed.returncode == 0 and result_path.exists():
        completed = subprocess.run(
            [sys.executable, "-m", "cocotb_tools.check_results", str(result_path)],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    if completed.returncode != 0:
        print(f"[fail] {job.name}: rc={completed.returncode}, log={log_path}", file=sys.stderr, flush=True)
        tail = tail_file(log_path)
        if tail:
            print(tail, file=sys.stderr, flush=True)
    else:
        print(f"[done] {job.name}: rc=0", flush=True)
    return completed.returncode


def run_parallel_make(jobs: list[Job], jobs_count: int) -> int:
    if not jobs:
        return 0
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs_count) as executor:
        future_to_job = {executor.submit(run_make_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                rc = future.result()
            except Exception as exc:  # pragma: no cover - runner fault path
                print(f"[error] {job.name}: {exc}", file=sys.stderr, flush=True)
                failures.append(job.name)
                continue
            if rc != 0:
                failures.append(job.name)
    if failures:
        print(f"Failed jobs: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def run_parallel_shared(jobs: list[Job], jobs_count: int) -> int:
    if not jobs:
        return 0

    grouped: dict[tuple[tuple[str, str], ...], list[Job]] = {}
    generic_by_key: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for job in jobs:
        generic_env = job.generic_env()
        key = tuple(sorted(generic_env.items()))
        grouped.setdefault(key, []).append(job)
        generic_by_key[key] = generic_env

    failures: list[str] = []
    for key, group_jobs in grouped.items():
        generic_env = generic_by_key[key]
        sim_name = sim_name_for_generic(generic_env)
        if build_shared_binary(sim_name, generic_env) != 0:
            failures.extend(job.name for job in group_jobs)
            continue

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs_count) as executor:
            future_to_job = {
                executor.submit(run_shared_job, job, sim_name): job
                for job in group_jobs
            }
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    rc = future.result()
                except Exception as exc:  # pragma: no cover - runner fault path
                    print(f"[error] {job.name}: {exc}", file=sys.stderr, flush=True)
                    failures.append(job.name)
                    continue
                if rc != 0:
                    failures.append(job.name)

    if failures:
        print(f"Failed jobs: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def make_regular_jobs(modules: list[str]) -> list[Job]:
    jobs = []
    for module in modules:
        if module == "test_conv_perf":
            continue
        jobs.append(Job(name=module, module=module, env=DEFAULT_OFM_ENV.get(module, {})))
    return jobs


def make_conv_perf_case_jobs(cases: list[int]) -> list[Job]:
    jobs = []
    for case in cases:
        jobs.append(
            Job(
                name=f"test_conv_perf_case_{case}",
                module="test_conv_perf",
                result_name=f"test_conv_perf_case_{case}.xml",
                env={"CONV_PERF_CASE": str(case), "CONV_PERF_GROUP": "0"},
            )
        )
    return jobs


def main() -> int:
    all_modules = discover_modules()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", default="all", help="Comma-separated test modules, or 'all'.")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--build-fw", action="store_true", help="Build all firmware images before simulation.")
    parser.add_argument(
        "--isolated-builds",
        action="store_true",
        help="Use the older one-sim-build-per-job Makefile flow instead of a shared Vtop binary.",
    )
    parser.add_argument(
        "--conv-perf-cases",
        default="",
        help="Comma/range list, e.g. '0,1,20-23', or 'all'. Runs one conv_perf case per simulation.",
    )
    args = parser.parse_args()

    modules = parse_list(args.tests, all_modules)

    if args.build_fw:
        build_firmware()

    jobs = make_regular_jobs(modules)
    if "test_conv_perf" in modules or args.conv_perf_cases:
        cases = parse_case_list(args.conv_perf_cases or "all")
        jobs.extend(make_conv_perf_case_jobs(cases))

    if args.isolated_builds:
        return run_parallel_make(jobs, args.jobs)
    return run_parallel_shared(jobs, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
