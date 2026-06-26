# Gemmini-Style Conv2D Feeder Plan

## Goal

Add a Gemmini-style Conv2D execution path around the existing `32x32` systolic
array without redesigning the MAC array. Conv2D is lowered to tiled GEMM:

- `M = OH * OW` rows, tiled by `Mtile`.
- `N = OC` columns, tiled by `32`.
- `K = IC * KH * KW`, tiled by `32`.

The first implementation target is correctness and reusable control flow. Full
performance optimization, line-buffer reuse, and direct-conv dataflow are later
work.

## Current Baseline

The current cluster has no dedicated IFM SRAM. Input activation, weights,
partial sums, and outputs live in Shared Data TCDM. The systolic controller
reads weight and IFM rows from TCDM, stages them through small FIFOs, feeds the
array, and drains OFM rows back to TCDM.

The systolic array already exposes a `psum_data_i` boundary, but the controller
currently drives it as zero. Therefore native GEMM32 only supports one
`K=32` block per invocation unless partial sums are accumulated outside the
array.

## Architecture

```text
L2
 │
 DMA
 │
Shared Data TCDM
 │
 ├─ input tensor
 ├─ weight tiles packed as K32 x OC32
 ├─ INT32 psum tiles
 └─ INT8/INT32 output tiles
        │
        ▼
conv2d loop controller / feeder
        │
        ├─ generate tile-local IFM rows: Mtile x 32
        ├─ load weight tile: 32 x 32
        ├─ load or reuse INT32 psum tile for K>32
        ▼
systolic_controller
        ▼
32x32 systolic array
        ▼
OFM INT32 rows
        ▼
psum writeback or requant + activation + store
```

## K-Tile Accumulation

For `K <= 32`, one systolic invocation is sufficient. For `K > 32`, firmware or
a future graph controller issues multiple GEMM32 blocks:

```text
for oc_tile in 0..OC step 32:
  for m_tile in 0..OH*OW step Mtile:
    for k_tile in 0..IC*KH*KW step 32:
      build_or_feed A[Mtile, 32]
      load W[32, 32]
      if k_tile == 0:
        psum = A * W
      else:
        psum = psum + A * W
      if k_tile is last:
        requant/activation/store final output
      else:
        store INT32 psum tile
```

Phase 0 implements psum accumulation in the systolic output drain path. This is
intentionally conservative: the array computes one `A*W` block, the controller
reads the previous psum row from TCDM, adds it to the OFM row, then either
writes the accumulated INT32 row back or feeds the accumulated row into the
requant pipeline for the final K-block.

## Conv2D Feeder Contract

The feeder maps each logical GEMM row and K lane back to Conv2D coordinates:

```text
m      -> oh, ow
k_lane -> kh, kw, ic
ih = oh * stride_h + kh * dilation_h - pad_h
iw = ow * stride_w + kw * dilation_w - pad_w
```

If `(ih, iw)` is outside the input image, the feeder injects zero. Otherwise it
loads the corresponding activation byte from Shared TCDM. The feeder output is
a `32` byte row matching the current systolic IFM input format.

## Operator Coverage

The verification plan must cover the feeder family, not only `3x3`:

- Pointwise Conv `1x1`.
- Standard Conv `3x3`, `5x5`, `7x7`.
- Asymmetric Conv `1x3`, `3x1`, `1x5`, `5x1`.
- Stride `1/2`, padding `0/1`, dilation when enabled.
- Channel boundary cases: `IC/OC = 1, 3, 31, 32, 33, 64`.
- `K` boundary cases: `<32`, `=32`, `>32`, non-multiple of `32`.
- Depthwise/grouped Conv as separate functional paths; they are not the
  performance target for dense systolic GEMM.

## Implementation Phases

### P0: GEMM K-Block Foundation

- Add systolic MMIO fields for psum pointer and accumulation enable.
- Add controller support for `OFM + previous_psum -> OFM`.
- Add HAL helper for tiled `K>32` accumulation.
- Add exact data-output tests for `K=64` INT32 accumulation and fused final
  requant.

### P1: Software Tile Feeder

- Generate tile-local IFM rows in firmware into TCDM.
- Verify Conv `1x1`, then Conv `3x3` with padding.
- Keep the systolic RTL unchanged except P0 accumulation support.

Current implementation status:

- `sw/lib/conv2d_feeder_sw.*` implements NHWC software materialization for
  `M x 32` IFM rows and `OC=32` execution.
- `sw/test/conv_feeder` verifies Conv `1x1` with `IC=33` so the path exercises
  two K-blocks and INT32 psum accumulation.
- The same suite verifies Conv `3x3`, stride `1`, pad `1`, `IC=3`, including
  padding zero injection.
- The cocotb test checks both dumped im2col K tiles and full INT32 output
  tensors against Python golden.
- All temporary/output buffers must stay inside the Shared Data TCDM address
  window; the P1 OFM buffer is `0x1014_0000`.

### P2: RTL Conv2D Feeder

- Move address generation into an RTL feeder.
- Support padding/stride/dilation zero injection.
- Keep debug mode that can materialize the feeder tile in TCDM.

### P3: Fused Final Store

- Intermediate K-blocks store INT32 psum.
- Final K-block accumulates and enables activation-specific postprocess.
- Requant final K-block is already supported by P0; richer activation fusion is
  future work.

### P4: Performance

- Add double-buffering for IFM/weight tiles.
- Prefetch next K-block.
- Revisit output ready/valid so accumulation tiles can use larger `Mtile`.
- Add PMU counters for feeder bandwidth, TCDM stalls, and systolic utilization.

## Initial Limits

- Phase 0 supports INT32 accumulation and fused requant on the final accumulated
  K-block.
- Accumulation mode uses small `Mtile` until the systolic output path has true
  output backpressure.
- No full im2col tensor is created in L2.
