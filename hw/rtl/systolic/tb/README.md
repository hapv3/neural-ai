# Systolic Controller Test Matrix And Coverage

This directory contains the `tb_systolic_controller` cocotb tests, the matrix
runner, and the functional coverage model.  Generated coverage files are written
under `coverage/`, which is intentionally ignored by git.

## Tools

- `run_systolic_controller_matrix.py`
  Runs matrix cases as independent cocotb simulations.  By default it runs a
  smoke subset; use `--all` for the full matrix.
- `systolic_controller_cocotb_coverage.py`
  Builds the functional coverage database using `cocotb-coverage` `CoverPoint`
  and `CoverCross`, then exports JSON, Markdown, YAML, and XML reports.

## Current Coverage Model

The matrix covers:

- Direct GEMM `M=1..128`.
- Linebuffer modes: bypass, coalesce, KGEN + C32 fast, and pool.
- Full kernel Cartesian product: `kernel_h=1..5` and `kernel_w=1..5`.
- `kernel_cross = kernel_w x kernel_h`, represented as bins like
  `kw5_kh1`, crossed with mode, stride, padding, aspect, channel, and YOLO
  channel bins.
- Input shape classes: tiny, small, medium, square, wide, tall, padded,
  unpadded, stride-1, and stride-2.
- Channel classes: C1, RGB/C3, sub-C32, C32, cross-beat, tail, and YOLO-style
  multi-C32 logical channels.

## Running

Run these commands from `hw/rtl/systolic/tb`.

Smoke run:

```bash
env PYTHONUNBUFFERED=1 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  ./run_systolic_controller_matrix.py \
  --randomize --seed 2525 \
  --parallel 8 --verilator-jobs 8 --timeout-s 240 \
  --quiet --progress-every 10 \
  --coverage-prefix coverage/systolic_controller_runner_coverage
```

Full coverage run:

```bash
env PYTHONUNBUFFERED=1 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  ./run_systolic_controller_matrix.py \
  --all --randomize --seed 20260715 \
  --parallel 8 --verilator-jobs 8 --timeout-s 240 \
  --quiet --progress-every 100 \
  --coverage-prefix coverage/systolic_controller_runner_coverage
```

Run only coverage export from a passed-case list:

```bash
./systolic_controller_cocotb_coverage.py \
  --case-list coverage/systolic_controller_passed_cases.txt \
  --out-json coverage/systolic_controller_runner_coverage.json \
  --out-md coverage/systolic_controller_runner_coverage.md \
  --out-yaml coverage/systolic_controller_runner_coverage.yml \
  --out-xml coverage/systolic_controller_runner_coverage.xml
```

## Last Full Run

The latest full randomized run used:

- Seed: `20260715`
- Cases: `1969/1969` passed
- Failures: `0`
- Root coverage: `100.00%`
- `kernel_h`: `1..5`
- `kernel_w`: `1..5`
- `kernel_cross`: `25/25` bins covered

The generated outputs from that run were:

- `coverage/systolic_controller_passed_cases.txt`
- `coverage/systolic_controller_runner_coverage.md`
- `coverage/systolic_controller_runner_coverage.json`
- `coverage/systolic_controller_runner_coverage.yml`
- `coverage/systolic_controller_runner_coverage.xml`

These files are not committed.  Re-run the commands above to regenerate them.
