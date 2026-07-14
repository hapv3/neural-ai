# Micro-YOLO Raw-Head Firmware Test

This directory contains the active Micro-YOLO raw-head firmware used by
`test_micro_yolo_e2e.py`.

Current checkpoint: **Phase 3j**

The firmware no longer represents older intermediate checkpoints such as
Conv-only, SiLU-only, MaxPool-only, or materialized Concat. Those phases are
documented in `docs/micro_yolov8_integration_test_plan.md`, but this firmware
entrypoint is the current end-to-end raw-head graph.

## Graph

Logical model:

```text
96x96x3 HWC input
  -> Conv_Stem 3x3/s2/p1 C3->C32
  -> SiLU = x * sigmoid(x)
  -> C2f_Conv 3x3/s1/p1 C32->C32
  -> residual Add with SiLU output
  -> Conv_Down 3x3/s2/p1 C32->C32
  -> MaxPool 5x5/s1/p2
  -> Upsample nearest 2x
  -> logical Concat with preserved SiLU branch
  -> Head_Conv 3x3/s1/p1 C64->C32
  -> 48x48x32 INT8 raw-head output in L2
```

Implementation detail: the Concat tensor is **not materialized**. The graph
uses:

```text
Conv([upsample, skip], W) = Conv(upsample, W[0:32]) + Conv(skip, W[32:64])
```

Layer 16 runs the first C32 chunk into INT32 psum, then runs the second C32
chunk with accumulate + systolic requant and writes the INT8 output tile to L2.

## Memory Contract

L2 payloads are provided by cocotb:

| Symbol | Address | Payload |
|---|---:|---|
| `L2_INPUT` | `0x80000000` | input image |
| `L2_WEIGHT0` | `0x80008000` | stem weights |
| `L2_SIG_LUT` | `0x80009000` | sigmoid LUT |
| `L2_WEIGHT1` | `0x8000A000` | C2f weights |
| `L2_WEIGHT2` | `0x8000D000` | downsample weights |
| `L2_WEIGHT3` | `0x80010000` | head weights, two C32 chunks |
| `L2_OUTPUT` | `0x80020000` | final raw-head output |
| `L2_SKIP` | `0x80040000` | temporary skip checkpoint |

TCDM scratch is statically aliased in `main.c`. The important aliases are:

| Buffer | Lifetime sequence |
|---|---|
| `act_a` | `T_STEM -> T_OUT -> T_POOL -> T_SKIP_RELOAD` |
| `act_c` | `T_SILU -> T_DOWN -> T_UPSAMPLE` |
| `psum_or_sig` | `T_SIG -> INT32 psum scratch` |
| `head_tile` | dedicated output tile, must not alias input branches |

Do not insert or reorder layers without updating the lifetime map in `main.c`.
The graph can compile while still corrupting data if an alias is reused too
early.

## Build

```sh
make -C sw/test/micro_yolo
```

The firmware no longer compiles the linebuffer/GEMM job descriptors into
`.data`. The host side generates a small runtime descriptor manifest plus binary
descriptor blobs with `tools/npu_linebuf_precompute.py`, writes the manifest to
L2 at `0x80052000`, and writes each blob at the L2 address listed by that
manifest. Firmware copies the manifest, then copies each blob into scratch/TCDM
before graph setup and only dispatches the received pointer/count pairs.

For the cocotb flow, `test_micro_yolo_e2e.py` calls the same Python planner and
loads the descriptor blobs into L2 before releasing firmware fetch. For an
external host flow, use:

```sh
tools/npu_linebuf_precompute.py micro-yolo-blob --output-dir /tmp/micro_yolo_desc
```

The generated `micro_yolo_linebuf_manifest.bin` is the runtime ABI consumed by
firmware. `micro_yolo_linebuf_manifest.txt` is a readable sidecar that lists
each blob name, L2 address, count, byte size, and filename.

Optional compile-time tile knobs:

```sh
make -C sw/test/micro_yolo \
  MICRO_YOLO_C2F_TILE_OH=16 MICRO_YOLO_C2F_TILE_OW=16 \
  MICRO_YOLO_DOWN_TILE_OH=16 MICRO_YOLO_DOWN_TILE_OW=16 \
  MICRO_YOLO_HEAD_TILE_OH=16 MICRO_YOLO_HEAD_TILE_OW=16
```

The default `16x16` head tile is chosen to fit the current 512 KiB TCDM budget.

## Run E2E RTL Test

```sh
CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster COCOTB_TEST_MODULES=test_micro_yolo_e2e
```

The cocotb test:

- generates deterministic input/weights/golden in Python;
- loads the firmware ELF sections and L2 payloads;
- runs the Snitch graph firmware;
- prints per-layer PMU counters;
- compares all `48*48*32` output bytes with zero tolerance.

## Current Reference Result

Latest passing raw-head run:

```text
total cycles: 386044
head conv cycles: 149256
head sys_compute: 41472
```

Head compute is exactly 2x the C32 Conv compute because the logical C64 input is
split into two C32 chunks. Total head latency is higher than 2x because the
second chunk also reads psum, accumulates, requants, and writes INT8 tiles.
The current firmware preloads the next linebuffer tile into RTL shadow
registers while the current tile is running, so tile-to-tile MMIO setup is
partially hidden behind systolic execution. The Micro-YOLO graph now also uses
host-generated linebuffer/GEMM job descriptors, so Snitch no longer rebuilds
per-tile linebuffer config in the critical layer path.
