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

## Spatz Strided/Indexed Gather Opportunity

Spatz supports strided and indexed vector memory operations in RTL, and the
upstream Spatz tests include `vlse*` and `vloxei/vluxei*` cases. This is useful
for the Spatz fallback path, but it does not replace the current iDMA path for
L2-resident layer-1 input because Spatz VLSU only reaches local L1/TCDM in this
cluster. If the source tensor remains in L2, iDMA is still the right first
backend.

The real bottleneck in the current Spatz fallback is not the tiny vector copy;
it is the Snitch scalar loop in `prepare_spatz_segmented()`:

```text
rows * kernel_spatial scalar iterations
  -> compute oh/ow/kh/kw/ih/iw
  -> bounds check
  -> call spatz_vec_copy_i8() for a very short channel run
```

For `Conv3x3 IC=3 M=1024`, that is `1024 * 9 = 9216` scalar loop iterations.
Using Spatz gather/scatter can collapse this into vector chunks:

- For each `(kh, kw, ic)`, load many output rows with `vluxei8.v` or row-local
  `vlse8.v`.
- Store into packed `M x 32` rows with `vsse8.v` stride `32`, or `vsuxei8.v`
  when the valid output rows are sparse because of padding.
- Zero-fill the packed tile first; invalid padding/tail rows remain zero.
- Reuse a shape-local offset table instead of recomputing address arithmetic
  inside the row loop.

The expected gain is significant, but not the theoretical `~230x` instruction
reduction. That estimate omits several real costs:

- `IC=3` requires three gather/store streams per spatial point.
- Raster order has a row-wrap discontinuity, so a single `vlse8.v` cannot cover
  the full `H*W` tile unless the tile is split per output row; full-tile gather
  needs indexed offsets.
- Indexed/strided VLSU operations are implemented as single-element memory
  operations in Spatz, so throughput is closer to one element per cycle plus
  TCDM arbitration, not a wide burst.
- Index vectors must be loaded or generated; a `1024`-row tile with 32-bit
  source offsets costs `4 KiB` of index storage per reusable row-pattern table.
- Masked indexed loads/stores are not part of the local verified test suite yet,
  so padding should initially be handled by splitting valid rectangles or using
  indexed stores only for valid row lists.

Recommended next implementation:

1. Add dedicated local RVV tests for `vlse8/vsse8`, `vluxei8/vsuxei8`, and
   `vloxei8/vsoxei8` with exact TCDM output checks.
2. Add `spatz_im2col_gather_i8()` as an assembly helper for L1/TCDM-resident
   input tiles only.
3. Start with the unmasked interior path: per `(kh, kw, ic)`, process valid
   output rectangles using `vlse8.v` load stride `IC` and `vsse8.v` store stride
   `32`.
4. Add indexed-offset mode for full raster tiles and irregular border regions
   after the basic strided path passes.
5. Keep iDMA multi-spatial as the default for L2-resident regular layer-1 tiles;
   use Spatz gather only when the scheduler has already staged the source tile
   into TCDM or when iDMA cannot express the shape efficiently.

Current local RVV gate status: `strided_indexed` passes the RTL regression with
exact firmware and cocotb checks for `vlse8/vsse8`, LMUL `m8` stride-32 store,
`vluxei8/vsuxei8`, and `vloxei8/vsoxei8`. The first production Spatz gather
helper still starts with the lower-risk rectangle path using `vlse8.v` plus
`vsse8.v`; indexed load/store is now unblocked for follow-up irregular
sparse-row gather/scatter.

The first rectangle-strided fallback is implemented for L1/TCDM-resident Conv2D
input tiles when the current software tile covers the full output rectangle
(`rows == output_h * output_w`). It zero-fills the packed tile, splits padding
into valid output rectangles, and copies each channel stream with a 2D Spatz
helper using `vlse8.v` load stride `stride_w * IC` plus `vsse8.v` store stride
`32`. The helper uses LMUL `m8`, which is now covered by the local RTL RVV
regression, fuses the channel loop into assembly, and moves the row loop into
assembly so C only calls once per valid command rectangle instead of once per
output row/channel.

The runtime now builds a per-layer prepare command list before the measured
K-tile loop. Each command captures `k_block`, source/destination base,
channel count, valid output rectangle, and source/destination strides. The hot
prepare loop consumes this command list directly, so it no longer recalculates
`output_valid_range`, `kh/kw`, lane placement, or channel grouping for each
K tile. Packed-tile zero-fill is also done by Spatz vector word stores
(`vse32.v`) when the im2col buffer and byte count are 32-bit aligned, falling
back to byte stores only for unaligned generic use.

On the focused RTL fixture `CONV_PERF_CASE=6` (`Conv5x5 pad2 IC3`, L1 source,
`rows=16`, `k_tiles=3`), the path is correctness-clean and reduces prepare
cycles from `75560` to `11454` versus the older segmented Spatz fallback. This
is a useful `~6.5x` improvement and also removes most C-side shape/control
overhead from the measured prepare loop. The next step remains indexed-offset
gather for larger row tiles and sparse border lists if larger Layer-1 fixtures
still show strided VLSU memory latency as the dominant cost.

## Engineering Conclusion

- The performance path should be software scheduler + iDMA/Spatz packed prepare
  rather than the dropped byte-serial RTL Conv2D feeder.
- iDMA is the preferred backend for L2-resident, regular Conv2D segments.
- Spatz remains necessary for L1/TCDM sources, irregular small-channel fallback,
  and future tensor transforms that are not expressible as iDMA 2D/3D copies.
  The next Spatz optimization should use strided/indexed memory instructions
  rather than many short `spatz_vec_copy_i8()` calls.
- The next performance work is not another feeder; it is measuring larger
  `M` tiles, adding transfer counters, and choosing tile sizes that amortize
  iDMA command overhead without exceeding TCDM capacity.
