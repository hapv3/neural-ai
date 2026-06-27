# Packed Systolic Conv2D Plan

## Goal

Add a packed Conv2D execution path around the existing `32x32` systolic array
without redesigning the MAC array. Conv2D is lowered to tiled GEMM:

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
conv2d packed scheduler
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
      prepare packed A[Mtile, 32]
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

## Packed Tile Contract

The packed prepare path maps each logical GEMM row and K lane back to Conv2D
coordinates:

```text
m      -> oh, ow
k_lane -> kh, kw, ic
ih = oh * stride_h + kh * dilation_h - pad_h
iw = ow * stride_w + kw * dilation_w - pad_w
```

If `(ih, iw)` is outside the input image, the prepare path injects zero by
leaving the pre-cleared destination lane untouched. Otherwise it copies the
corresponding activation byte into a packed `32` byte row matching the current
systolic IFM input format.

## Operator Coverage

The verification plan must cover the packed Conv2D operator family, not only
`3x3`:

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

### Removed Legacy Conv2D Lowering Paths

The previous software materializer test and RTL debug path have been dropped
from active RTL/software. Conv2D
lowering now uses the packed prepare path in `sw/lib/conv2d_packed.*`, with
iDMA for contiguous K tiles and Spatz RVV copies for segmented/padded tiles.

Active Conv2D verification lives in `sw/test/conv_perf` and
`test_conv_perf`. Do not add new roadmap features or acceptance criteria to the
removed path. If prepare performance is insufficient, add a new
wide-segment extractor architecture instead of reviving byte-serial lowering.

### P3: Packed Conv2D Functional Completeness

Goal: turn the software+iDMA+Spatz packed prepare path into a reusable Conv2D
operator backend for model layers. There is no active direct-Conv2D RTL dependency for this phase.

Features to complete:

- **Output-channel tiling (`OC > 32`)**:
  - Add firmware scheduler loops for `oc_tile = 0..OC step 32`.
  - Define weight layout for `K32 x OC32` tiles across multiple output channel
    groups.
  - Store each OC tile to the correct output tensor offset.
- **K tiling edge cases**:
  - Keep `K_TILE=32`, but formally support `K < 32`, `K = 32`,
    `K > 32`, and non-multiple K tails through zero-injected lanes.
  - Ensure only final K-block can enable requant/final store; intermediate
    blocks must store INT32 psum.
- **Final block postprocess**:
  - Use existing systolic requant path for final accumulated block.
  - Support final output as either INT32 debug output or INT8 requantized output.
  - Add optional ReLU via requant clamp configuration (`clamp_min = 0`) for
    integer-only activation fusion.
- **Shape coverage**:
  - Support standard Conv `1x1`, `3x3`, `5x5`, `7x7`.
  - Support asymmetric Conv `1x3`, `3x1`, `1x5`, `5x1`.
  - Support stride `1` and `2`.
  - Support symmetric and asymmetric padding.
  - Support dilation `1` only. Dilation greater than `1` is out of scope for
    the current YOLO/CNN/Vision Transformer target path.
- **Backend policy**:
  - Conv1x1/contiguous K tiles use iDMA 2D pack.
  - Larger-IC Conv3x3 uses iDMA only for regular one-segment tiles.
  - RGB, padding, border, and tile-crossing cases use Spatz RVV pack.
  - Scalar prepare must remain disabled for performance tests.
- **Unsupported in P3**:
  - Depthwise/grouped conv are tracked as separate paths; dense systolic Conv2D
    should not claim them.
  - No performance overlap, no line buffer, no weight reuse cache.
  - Dilation greater than `1` remains out of scope.

P3 implementation status:

- **Pointwise coverage**:
  - Conv1x1 `IC=1`, `IC=3`, `IC=31`, `IC=32`, `IC=33`, `IC=64`, `OC=32`
    exact INT32 compare is covered by `test_conv_perf`.
  - Conv1x1 `IC=33`, `OC=64` is covered as two OC tiles and two K tiles.
- **Kernel coverage**:
  - Conv3x3 pad0 and pad1 with `IC=32`, `OC=32` are covered.
  - Conv5x5 pad2 with `IC=3`, `OC=32` is covered.
  - Conv7x7 pad3 with `IC=1`, `OC=32` is covered.
  - Conv1x3, Conv3x1, Conv1x5, Conv5x1 are covered with exact INT32 compare.
- **Stride/padding coverage**:
  - Conv3x3 stride2 pad1 is covered.
  - Asymmetric kernel coverage is implemented; true asymmetric top/bottom or
    left/right padding remains a future register/API extension because the
    current packed scheduler config exposes symmetric `pad_h/pad_w`.
- **Tail and zero-injection coverage**:
  - `K < 32`: Conv3x3 `IC=1` (`K=9`) and Conv1x1 `IC=3` are covered.
  - `K = 32`: Conv1x1 `IC=32` is covered.
  - `K > 32` non-multiple: Conv1x1 `IC=33` and Conv3x3 `IC=5` (`K=45`)
    are covered.
  - Padding/tail zero lanes are checked through exact final output compare.
- **Final store coverage**:
  - INT32 output mode exact compare is covered.
  - Final-block requant INT32-to-INT8 exact compare is covered for Conv1x1
    `IC=64`.
  - ReLU-through-clamp uses the same requant clamp mechanism; a dedicated
    clamp-min-zero fixture remains optional.
- **Regression gates**:
  - `test_conv_perf` covers packed prepare shape/backend coverage.
  - `CONV_PERF_GROUP=1` is the dedicated OC tiling/pointwise regression.
  - `CONV_PERF_GROUP=2` is the dedicated kernel/stride/tail-K regression.
  - `CONV_PERF_GROUP=3` is the dedicated final-block requant regression.
  - Existing `test_independent_systolic`, `test_systolic_requant`, and
    `test_dma_tcm` must continue passing.

P3 known limits:

- Dilation greater than `1` remains unsupported by policy.
- True asymmetric top/bottom or left/right padding needs an API/register
  extension; current coverage is symmetric padding plus asymmetric kernels.
- Depthwise/grouped convolution remains a separate path, not dense systolic
  Conv2D.

### P4: Packed Prepare Performance

Goal: improve throughput without changing the correctness contract established
in P3.

Features to complete:

- **Tile scheduling**:
  - Choose `Mtile` policy for small/large feature maps.
  - Keep output FIFO high-water protection until true OFM backpressure exists.
  - Decide when to split large `M` into multiple controller invocations.
- **DMA/compute overlap**:
  - Double-buffer input, weight, and output/psum tiles where TCDM capacity
    allows it.
  - Prefetch next K tile or next OC tile while current tile computes.
  - Keep a non-overlapped debug mode for deterministic bring-up.
- **Weight/data reuse**:
  - Reuse weight tile across multiple `Mtile` chunks when TCDM placement allows.
  - Avoid full L2 im2col tensors; packed `M x 32` TCDM tiles are the compute
    interface to systolic.
- **Performance counters**:
  - Revisit OFM ready/valid into systolic controller so FIFO depth can be
    reduced safely.
  - Add counters for packed prepare cycles, iDMA wait cycles, Spatz pack cycles,
    systolic active cycles, OFM FIFO stalls, TCDM grant stalls, and DMA wait cycles.
- **Unsupported in P4**:
  - Re-architecting the MAC array or adding a full line-buffer direct-conv
    engine is out of scope unless packed prepare measurements cannot meet target.

P4 required tests:

- **Performance invariant tests**:
  - Run the same P3 correctness suite and compare output bit-exactly while
    checking backend tile counters.
  - Randomized bounded Conv1x1 and Conv3x3 fixtures with `M` crossing tile
    boundaries.
- **Large-shape smoke tests**:
  - Conv1x1 with `M=1024`, `IC=32`, `OC=32`.
  - Conv3x3 pad1 with `M=1024`, `IC=3`, `OC=32`.
  - Conv3x3 with `IC=32`, `OC=64`, `M` split into multiple tiles.
- **Stress tests**:
  - TCDM bank-conflict fixture for input/weight/output addresses.
  - Back-to-back Conv layers sharing output/input buffers.
  - DMA overlap test with intentionally delayed TCDM grants if testbench
    support exists.
- **Performance reporting tests**:
  - Log cycles per output element and systolic utilization for Conv1x1,
    Conv3x3, and Conv3x3 with `IC=32`.
  - Record packed prepare breakdown: iDMA wait, Spatz pack, systolic active,
    OFM drain wait.
  - Compare against current `test_conv_perf` baseline before accepting P4 changes.

## Initial Limits

- Phase 0 supports INT32 accumulation and fused requant on the final accumulated
  K-block.
- Accumulation mode uses small `Mtile` until the systolic output path has true
  output backpressure.
- No full im2col tensor is created in L2.
