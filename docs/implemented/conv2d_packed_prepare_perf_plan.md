# Conv2D Packed Prepare Performance Plan

## Decision

RTL `conv2d_feeder` is dropped from the Conv2D performance roadmap.

The active performance path is:

```text
software scheduler
  -> iDMA 2D/3D packed tile prepare for contiguous regions
  -> Spatz RVV packed tile prepare for RGB, padding, border, and irregular regions
  -> systolic GEMM32 / accumulated GEMM32 / final requant
```

`conv2d_feeder.sv` and `conv2d_feeder_cache.sv` remain only as legacy
debug/reference regressions until removal is explicitly scheduled. No new
performance features should be added to that RTL path.

## Rationale

The measured P0/P1 data shows that scalar software prepare was the first
bottleneck, not systolic compute:

| Case | Old scalar prepare | New prepare | GEMM | Backend |
|------|--------------------|-------------|------|---------|
| Conv1x1 `M=20`, `IC=33` | `108592` cycles | `10658` cycles | `666` cycles | iDMA 2D, 2 K tiles |
| Conv3x3 `M=25`, `IC=3`, pad1 | `95822` cycles | `42108` cycles | `202` cycles | Spatz RVV, 1 K tile |

The new path proves the right backend selection:

- Conv1x1 contiguous K tiles use iDMA 2D pack.
- K-tail lanes are zero-filled before copying valid bytes.
- RGB/padding uses Spatz RVV segment copies with explicit zero injection.
- Scalar prepare is no longer used by the performance test.

The remaining bottleneck is now packed-tile preparation policy and overhead,
not the byte-serial RTL feeder. Continuing to optimize the existing RTL feeder
would split effort across two Conv2D lowering paths without proving model-level
benefit.

## Active Scope

### Conv1x1 / Contiguous K Tile

- Keep input in L2 when possible.
- Use iDMA 2D pack:
  - `length = valid_k_lanes`
  - `src_stride = IC`
  - `dst_stride = 32`
  - `reps = M`
- Pre-clear the destination tile so K-tail lanes are zero.
- Feed the packed `M x 32` tile to systolic GEMM32.

### Larger-IC Conv3x3

- Use iDMA only when the current K tile maps to one contiguous
  spatial/channel segment.
- Use iDMA 3D for repeated output rows/columns when source and destination
  strides are regular.
- Fall back to Spatz RVV when a tile crosses spatial segments or touches a
  border.

### RGB / IC < 32 / Padding / Border

- Pre-clear the packed destination tile.
- Use Spatz RVV segment copies for in-bound contiguous channel runs.
- Leave invalid padding/tail lanes as zero.

### Unsupported

- Dilation greater than `1` remains unsupported.
- Depthwise/grouped convolution is a separate operator path.
- Full hardware line-buffer/direct-conv engine is not part of this phase.

## Verification Gates

The performance roadmap should be gated by `sw/test/conv_perf` and
`test_conv_perf`, not by RTL feeder tests.

Required coverage before micro-model use:

- Conv1x1 `M=1024`, `IC=32`, `OC=32`.
- Conv1x1 `M=1024`, `IC=33`, `OC=32`, proving K-tail handling.
- Conv1x1 `M=1024`, `IC=64`, `OC=32`, proving multiple full K tiles.
- Conv1x1 `OC=64`, proving output-channel tiling.
- Conv3x3 `M=1024`, `IC=3`, pad1, proving RGB/padding path.
- Conv3x3 `M=1024`, `IC=32`, `OC=32`, proving larger-IC segment path.
- Conv3x3 `IC=32`, `OC=64`, proving K/OC/M tile scheduling together.

Pass criteria:

- Output tensors match Python golden exactly.
- Firmware status is `NPU_CONV2D_PACKED_OK`.
- Backend tile counters match expected path selection.
- `prepare_scalar_tiles == 0`.
- Cycle stats are recorded for prepare, GEMM, total, and final K tile.

## Legacy RTL Policy

`test_conv_feeder_rtl` remains useful for:

- checking the old address generator against Python im2col golden,
- preserving direct feeder-to-systolic stream coverage,
- debugging historical RTL behavior if a regression appears.

It is not a performance gate for YOLO/CNN/ViT work. Any future Conv2D
performance work should improve the packed prepare scheduler, iDMA usage,
Spatz RVV kernels, or systolic scheduling/requant path instead.
