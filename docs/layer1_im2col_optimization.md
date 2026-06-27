# Layer-1 Im2Col Performance Optimization

## Problem

The first Conv2D layer in YOLO/CNN models often has small input-channel count,
for example RGB input with `IC=3`, `KH=3`, `KW=3`, stride `2`, pad `1`.
The effective `K` dimension is only `27`, so the systolic compute cost is low,
but packed tile preparation can dominate if firmware materializes im2col with
scalar or many tiny vector copies.

For a `640x640x3 -> 320x320x48` stem layer:

- Output rows: `320 * 320 = 102400`.
- `K = 3 * 3 * 3 = 27`, padded to the systolic `K_TILE=32`.
- A full materialized packed tile is `102400 * 32 = 3.2 MiB`, which is larger
  than the local TCDM budget, so the real scheduler must process smaller
  `M` tiles and repeat prepare/GEMM across the full layer.

The previous packed prepare path only used iDMA when one K tile mapped to a
single spatial index. Layer-1 RGB `3x3` crosses nine spatial positions, so it
fell back to Spatz RVV segment copies. Correctness was good, but the prepare
stage was command/control dominated and much slower than GEMM for small tiles.

## Implemented Optimization

`sw/lib/conv2d_packed.c` now has a generic multi-spatial iDMA path between the
existing single-spatial iDMA path and the Spatz fallback:

1. Zero-fill the `M x 32` packed tile first, preserving tail and padding lanes.
2. Split the K tile into contiguous channel runs for each spatial kernel point.
3. For each valid spatial run, issue one `idma_L2ToL1_3d()`:
   - `size_1 = valid channel run`
   - `size_2 = valid output width`
   - `size_3 = valid output height`
   - source strides walk NHWC input in L2
   - destination strides place each run into packed `M x 32` TCDM rows
4. Use Spatz RVV segmented copies only when the source is already in L1/TCDM or
   the shape is not representable by the regular multi-spatial iDMA segments.

This is not a hardcoded `3x3 RGB` path. It also covers regular larger-IC
multi-spatial cases, for example `3x3 IC32`, where each kernel point maps to a
full contiguous 32-byte channel run.

## Measured RTL Results

Measured with `test_conv_perf` after the implementation:

| Case | Rows | K tiles | Backend | iDMA transfers | Prepare cycles | GEMM cycles | Total cycles |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Conv1x1 IC33 OC32 | 20 | 2 | iDMA 2D | 2 | 10834 | 662 | 11692 |
| Conv3x3 IC3 pad1 OC32 | 25 | 1 | iDMA 3D multi-spatial | 9 | 11630 | 200 | 11944 |
| Conv3x3 IC32 pad0 OC32 | 4 | 9 | iDMA 3D multi-spatial | 9 | 13910 | 1844 | 16482 |
| Conv3x3 IC32 pad1 OC32 | 16 | 9 | iDMA 3D multi-spatial | 9 | 43190 | 2544 | 46462 |
| Conv1x1 IC64 requant | 16 | 2 | iDMA 2D | 2 | 8764 | 546 | 9508 |

The earlier small Conv3x3 RGB Spatz path was about `42108` prepare cycles for
the same small fixture, so the multi-spatial iDMA path gives about `3.6x`
prepare reduction on the RTL regression case (`42108 -> 11630`). This is still
not bandwidth-limited because `M=25` is tiny and each transfer has MMIO/wait
overhead. Larger `M` tiles should amortize command overhead better, but the
full-layer result must be measured with realistic TCDM tiling.

The iDMA controller now has a configurable frontend job FIFO
(`IDMA_JOB_FIFO_DEPTH=16` by default), and the multi-spatial Conv2D prepare path
queues all segment transfers before waiting for the final transfer ID. This
removes the per-segment blocking wait, but the measured gain is modest on the
small RGB fixture (`12062 -> 11630` prepare cycles) because command issue/MMIO
overhead still dominates.

The firmware stats now expose both `prepare_idma_tiles` and
`prepare_idma_transfers`. For Layer-1 RGB, this distinction matters: one K tile
uses nine iDMA 3D transfers, one per kernel spatial point.

## Engineering Conclusion

- The performance path should be software scheduler + iDMA/Spatz packed prepare
  rather than the dropped byte-serial RTL Conv2D feeder.
- iDMA is the preferred backend for L2-resident, regular Conv2D segments.
- Spatz remains necessary for L1/TCDM sources, irregular small-channel fallback,
  and future tensor transforms that are not expressible as iDMA 2D/3D copies.
- The next performance work is not another feeder; it is measuring larger
  `M` tiles, adding transfer counters, and choosing tile sizes that amortize
  iDMA command overhead without exceeding TCDM capacity.
