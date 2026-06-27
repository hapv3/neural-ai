# Conv2D Packed Prepare Performance Plan

## Decision

The old direct Conv2D RTL lowering path is dropped from the Conv2D performance roadmap.

The active performance path is:

```text
software scheduler
  -> iDMA 2D/3D packed tile prepare for contiguous regions
  -> iDMA 3D multi-spatial packed tile prepare for regular L2 Conv2D segments
  -> Spatz RVV packed tile prepare for L1/TCDM fallback and irregular regions
  -> systolic GEMM32 / accumulated GEMM32 / final requant
```

The old direct Conv2D RTL modules and matching firmware/debug tests have since
been removed from the active tree. No new performance features should be added to
that removed path.

Update: the original RGB/padding Spatz-only path has been superseded for
L2-resident regular Conv2D tiles by the generic multi-spatial iDMA path in
`sw/lib/conv2d_packed.c`. Spatz remains the fallback backend.

## Rationale

The measured P0/P1 data shows that scalar software prepare was the first
bottleneck, not systolic compute:

| Case | Old scalar prepare | New prepare | GEMM | Backend |
|------|--------------------|-------------|------|---------|
| Conv1x1 `M=20`, `IC=33` | `108592` cycles | `10658` cycles | `666` cycles | iDMA 2D, 2 K tiles |
| Conv3x3 `M=25`, `IC=3`, pad1 | `95822` cycles | `42108` cycles | `202` cycles | Spatz RVV, 1 K tile |
| Conv3x3 `M=25`, `IC=3`, pad1 | `42108` cycles | `11630` cycles | `200` cycles | iDMA 3D multi-spatial, 1 K tile / 9 queued transfers |

The new path proves the right backend selection:

- Conv1x1 contiguous K tiles use iDMA 2D pack.
- K-tail lanes are zero-filled before copying valid bytes.
- L2-resident regular RGB/padding segments use iDMA 3D multi-spatial pack.
- L1/TCDM or irregular cases use Spatz RVV segment copies with explicit zero
  injection.
- Scalar prepare is no longer used by the performance test.

The remaining bottleneck is now packed-tile preparation policy and overhead,
not the byte-serial RTL lowering path. Continuing to optimize that old RTL path
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
- Use iDMA 3D multi-spatial segment copies for regular L2-resident tiles.
- Use Spatz RVV segment copies for L1/TCDM or irregular fallback runs.
- Leave invalid padding/tail lanes as zero.

### Unsupported

- Dilation greater than `1` remains unsupported.
- Depthwise/grouped convolution is a separate operator path.
- Full hardware line-buffer/direct-conv engine is not part of this phase.

## Verification Gates

The performance roadmap should be gated by `sw/test/conv_perf` and
`test_conv_perf`, not by removed RTL lowering tests.

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
- `prepare_idma_transfers` records actual iDMA commands, which may be larger
  than `prepare_idma_tiles` for multi-spatial Conv2D K tiles.
- The iDMA frontend has a configurable job FIFO, so regular multi-spatial
  Conv2D prepare can queue segment transfers and wait only for the final ID.
- `prepare_scalar_tiles == 0`.
- Cycle stats are recorded for prepare, GEMM, total, and final K tile.

## Removed RTL Policy

The previous direct Conv2D-to-systolic stream path is not a performance gate
for YOLO/CNN/ViT work and is no longer present in active RTL/software. Any
future Conv2D performance work should improve the packed prepare scheduler,
iDMA usage, Spatz RVV kernels, or systolic scheduling/requant path instead.
