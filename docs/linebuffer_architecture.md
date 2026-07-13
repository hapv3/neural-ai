# NPU Linebuffer Architecture

This document records the performance review of the legacy
`conv_channel_linebuf_packer`, why it is no longer used as the performance
path, and the current `conv_linebuf_stream_packer` architecture that feeds the
systolic array for supported Conv2D linebuffer shapes.

Current decisions:

- `conv_channel_linebuf_packer` is kept only as a legacy/reference RTL block
  for correctness coverage. It is not the default performance path.
- The active linebuffer is `conv_linebuf_stream_packer`, redesigned around a
  window cache plus segment/row prefetch so it can stream IFM vectors directly
  into the systolic array.
- The steady-state target for supported shapes is one 256-bit IFM vector per
  cycle.
- Native stride support is focused on `stride=1` and `stride=2`. Larger strides
  must be decomposed by the compiler/scheduler or routed through the packed
  prepare fallback.

## 0. Current Implementation Status

The implementation has moved the Micro-YOLO performance path to
`conv_linebuf_stream_packer`. Micro-YOLO no longer depends on
`conv_channel_linebuf_packer`.

Important points:

- **Host/Python owns the Conv2D linebuffer schedule.**
  `tools/npu_linebuf_precompute.py` generates
  `sw/test/micro_yolo/micro_yolo_linebuf_precompute.h`, which contains full
  `npu_conv2d_linebuf_job_desc_t` job arrays for each Conv tile.
- **Snitch firmware does not build linebuffer tiles in the Micro-YOLO hot
  path.** `npu_layer_t` stores only descriptor pointer/count fields. The graph
  executor calls `npu_conv2d_packed_run_linebuf_job_descs()`, and that runner
  preloads shadow registers for job N+1 while job N is running.
- **The generic C planner remains as a fallback** for graphs/layers that do not
  provide host-planned job descriptors. This fallback still uses
  `make_linebuf_output_tile_cfg()` and `linebuf_job_from_tile_cfg()`.
- **The C32-aligned fast path is the main path for YOLO layers after the
  stem.** The descriptor must present `input_c=32`,
  `pixel_stride_bytes=32`, `row_stride_bytes=input_w*32`,
  `input_c_base=0`, `lane_base=0`, `block_valid_bytes=32`, and a 32-byte
  aligned base/channel offset. In this path the RTL can use `obi_rdata_i` or a
  window word directly and avoid `merge_beats` on the main data path.
- **`merge_beats` is still required** for the RGB stem, sub-C32/tail chunks,
  `lane_base != 0`, raw/NHWC or non-aligned fallback, and bypass reads crossing
  a 32-byte beat boundary.
- **The Micro-YOLO test firmware uses initialized D-TCM descriptors.** Cocotb
  uses an ELF section-aware loader: `.text` is written into I-TCM through AXI,
  while the `.data` descriptor table is initialized in D-TCM through testbench
  hierarchy/backdoor. This is the current verification mechanism; it does not
  mean the normal host AXI path exposes D-TCM.

## 1. Measured Legacy RTL Contract

This section describes the old `conv_channel_linebuf_packer` contract. It is
useful for semantic comparison, but it is not the current performance path.

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| `K_MAX` | `5` | Native kernel height/width support is `1..5`. `7x7/9x9` kernels must be decomposed by compiler/scheduler before using the linebuffer. |
| `MAX_INPUT_W` | `640` | SRAM line depth is based on real input width, excluding padding. Tiles/stripes with `input_w > 640` must be split first. |
| `DATA_WIDTH` | `256-bit` | One SRAM word stores one C-block, `32 x INT8`, matching the systolic IFM row width. |
| `C_BLOCK` | `32 channels` | Each run processes one `c_base`/C-block. Changing `c_base` flushes the linebuffer on `start_i`. |

`conv_channel_linebuf_packer` supports padding through zero injection and does
not materialize padding in SRAM. Therefore `MAX_INPUT_W=640` is the real input
width; `pad_h/pad_w` only affect bounds checks and output shape. This contract
is useful for preserving semantics, but it is not sufficient for the
performance target.

## 2. SRAM Sizing

For the legacy configuration:

- One line: `640 words x 32B = 20 KiB`.
- Ring buffer: `5 lines x 20 KiB = 100 KiB`.
- Each SRAM word is a 32-byte vector at one `x` coordinate for one `c_base`.

The legacy design does not tag by `c_base/cblk`; the row tag is only `ih`.
This matches the rule that each run uses one `c_base` and flushes on start. If
multiple C-blocks must be resident in the same ring buffer, the tag must be
extended to `{ih, cblk}`.

## 3. Legacy Dataflow

This section describes the old channel-linebuffer dataflow. The current stream
linebuffer dataflow is described in Section 8.

Legacy flow:

1. Host/scheduler configures shape, stride, padding, `c_base`, and byte
   strides. `input_base` is the origin of the padded output row: firmware sets
   it to `input_addr - pad_h * row_stride_bytes`. Horizontal padding is still
   handled through `pad_w` and bounds checks, not materialized in SRAM.
2. The packer ensures all input rows required by the current window are present
   in the ring SRAM.
3. For each output spatial/kernel position, the packer reads `C_BLOCK=32` bytes
   from row SRAM, or emits zero when the coordinate is outside input bounds.
4. `1x1` without padding uses a bypass path: it reads OBI/TCDM directly and
   skips the ring SRAM.
5. The output row is fed to the systolic IFM stream through ready/valid.

For `KH*KW*IC <= 32`, RTL supports `coalesce` mode: each output spatial
position emits one IFM row containing the full window in lane order
`{kh, kw, ic}`, matching the K-major systolic weight layout.

For `K > 32`, RTL has KGEN support for micro-tiles:

1. Host/Python computes the layer/tile config, `{kh,kw,ic}` seed, and
   `k_tile_count`.
2. Snitch writes `REG_LB_K_SEED`, `REG_LB_K_TILES`, and enables
   `REG_LB_CTRL_KGEN`.
3. Snitch starts systolic once and waits once.
4. `systolic_controller` loops over K tiles internally and advances the 32 lane
   descriptors per tile in `{kh,kw,ic}` order.
5. The first tile writes INT32 OFM; later tiles read psum and accumulate into
   the same OFM.

The historical KGEN v0 gate used a very small
`M <= SYSTOLIC_GEMM32_ACCUM_TILE_M` firmware/test limit. The current software
limit has been raised to `NPU_CONV2D_LINEBUF_KGEN_MAX_M = 1024`; Micro-YOLO
uses `M=256` jobs for `16x16` tiles. Shapes outside the supported
linebuffer/KGEN envelope still use the generic planner or packed-prepare
fallback.

Unaligned vectors are handled by `merge_beats`: if a 32-byte vector crosses a
256-bit beat boundary, the packer issues two reads and merges them.

## 4. Legacy Compiler/Scheduler Rules

- Native linebuffer path: `kernel_h <= 5`, `kernel_w <= 5`, `input_w <= 640`.
- Coalesced fast path: only when `KH*KW*IC <= 32` and the output spatial tile
  does not exceed the current GEMM linebuffer launch limit.
- KGEN fast path: only when `KH*KW*IC > 32`,
  `M <= NPU_CONV2D_LINEBUF_KGEN_MAX_M` (software default `1024`), and the
  channel shape does not cut across a C-block. For YOLO C32-blocked tensors,
  Host/Python should emit descriptors that each see exactly one C32 block:
  `input_c=32`, `input_c_base=0`, `lane_base=0`, `block_valid_bytes=32`.
- `7x7`, `9x9`, or tile widths larger than `640` must either be decomposed into
  smaller sub-kernels/tiles or fall back to packed prepare through iDMA/Spatz.
- `1x1` pad0 should use the bypass path. `1x1` with padding still uses the
  ring/zero-injection fallback for correctness.
- The old Conv2D scheduler once prioritized this linebuffer before packed
  prepare. After the performance review below, that rule was replaced: only the
  newer `conv_linebuf_stream_packer` should be prioritized as the performance
  path.
- Halo retention/cascade across stripes is not implemented; `start_i` currently
  clears ring valid state.

### 4.1 Internal C32-Blocked Layout Contract

Firmware/Python host must distinguish the raw input layout from the NPU
internal layout:

- Boundary input from camera/model loader may be default NHWC/RGB. The first
  layer may use a tail/raw path or a dedicated DMA/repack path because `IC=3`
  is small.
- After the first layer, systolic OFM should be treated as an internal
  **C32-blocked/block-major** tensor, not compact full-C NHWC:
  `block[cblk][spatial][32C]`.
- When a later layer reads one C32 block, the host must configure the command as
  an independent tensor view:
  - `input_addr = base_of_cblk`
  - `input_c = 32` for a full block, or tail width for the final block
  - `input_c_base = 0`
  - `input_c_stride = 32`
  - `row_stride_bytes = input_w * 32`
  - `pixel_stride_bytes = 32`
- For `IC > 32`, the Python planner splits work into multiple C32 views and
  accumulates into the same OFM/psum. Snitch firmware only runs the commands
  generated by the planner; it does not need to gate the layout again.
- If the host passes `input_c = full_IC` and `input_c_base = 32/64/...` over
  C32-blocked storage, the linebuffer will apply the wrong offset. The correct
  convention is to move `input_addr` to the block base and keep
  `input_c_base = 0`.

`conv_linebuf_stream_packer` has fast handling for C32-aligned views:
`pixel_stride=32`, `row_stride=input_w*32`, `input_c_base=0`, `lane_base=0`,
and a 32-byte aligned origin base. Raw NHWC/cross-beat handling remains the
correctness fallback for boundary and tail cases.

## 5. Legacy Verification

Direct regression coverage for this contract lives in
`hw/rtl/systolic/tb/test_conv_channel_linebuf_packer.py`:

- Positive tests: `1x1`, `3x3`, `5x5`, coalesced `3x3/C3`, stride `1/2`,
  padding, tail channel, unaligned/cross-beat, and width boundary `640`.
- Sweep: all kernels `1x1..5x5`.
- Negative tests: reject `7x7`, `9x9`, and `input_w=641`.

## 6. Performance Review

The goal of this review was to determine whether the old linebuffer could
replace packed im2col prepare for Conv2D. The most important case is deep
Conv2D with `IC > 32`, because it dominates the YOLO/CNN workload after the
first layer.

### 6.1 Baseline Before Linebuffer

Earlier measurements showed that scalar software packed prepare could not be
the performance path:

- `Conv1x1 IC=33`: `prepare=108592`, `gemm=662`, `total=109382`.
- `Conv3x3 IC=3`: `prepare=95822`, `gemm=200`, `total=96096`.

After optimizing `Conv1x1 IC=33` with iDMA contiguous/2D packing, the
pointwise path was effectively solved:

- `Conv1x1 IC=33`: `gemm=666`, `idma=2`, `spatz=0`, `scalar=0`.

Conclusion from this stage: the remaining core issue is spatial Conv2D
`KH x KW > 1`, especially `3x3/5x5` with large `IC`. The linebuffer was added
to avoid materializing im2col in TCDM.

### 6.2 Main Measurement Case: Conv3x3 IC120

Command:

```bash
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  CONV_PERF_CASE=20 \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf
```

Test result:

```text
TESTS=1 PASS=1 FAIL=0
```

PMU summary:

```text
cycles=24170
systolic: compute=544 (2.25%) ifm_req=2038 ofm_req=4288 ofm_stall=0
tcdm: req=7588 gnt=7562 stall=26 read=4216 write=3372
```

Python monitor:

```text
conv_perf state monitor events:
cycles=24448
compute=544
weight_load=1088
ofm_valid=544
linebuf_row_valid=544
linebuf_row_ready=544
linebuf_obi_req=950
linebuf_obi_stall=26
ofm_fifo_empty=20736
ofm_fifo_full=0
```

Systolic state cycles:

```text
total=24448 active=18700
COMPUTE      15202 total=62.18% active=81.29%
IDLE          5748 total=23.51%
WAIT_DRAIN    2374 total=9.71%  active=12.70%
LOAD_WEIGHTS  1122 total=4.59%  active=6.00%
DONE             2 total=0.01%
```

OFM drain state cycles:

```text
DRAIN_IDLE         21280
DRAIN_ACCUM_READ    2640
DRAIN_ACCUM_WRITE    528
DRAIN_ACCUM_REQUANT    0
```

Linebuffer state cycles:

```text
total=24448 active=15202
CH_IDLE           9246
CH_COAL_PREP      4896 active=32.21%
CH_COAL_READ_REQ  3400 active=22.37%
CH_COAL_READ_WAIT 3400 active=22.37%
CH_FILL_REQ0       570
CH_ENSURE          544
CH_FILL_WAIT0      544
CH_FILL_WRITE      544
CH_COAL_EMIT       544
CH_FILL_REQ1       380
CH_FILL_WAIT1      380
```

Firmware stats:

```text
linebuffer split conv3x3 IC120 stats:
rows=16 k_tiles=34 prepare=0 gemm=19418 total=19490
last_prepare=0 last_gemm=4162 idma=0 idma_tx=0 spatz=0 scalar=0
```

Meaning of the case:

- `M=16` output spatial rows in the micro-tile.
- `k_tiles=34`; each K tile creates one IFM vector for each output row.
- The expected IFM vector count is `16 * 34 = 544`.
- `systolic.compute=544` exactly matches the number of real vectors entering
  the MAC array.
- `linebuf_row_valid=544` and `linebuf_row_ready=544`, so no output data is
  lost and the linebuffer output is not backpressured.

### 6.3 Bottleneck Analysis

`conv_channel_linebuf_packer` was not blocked by TCDM grants:

- `linebuf_obi_stall=26` out of `950` OBI requests.
- `tcdm stall=26` out of `7588` requests.
- OFM FIFO was not full: `ofm_fifo_full=0`.
- Linebuffer output was always accepted: `row_valid=row_ready=544`.

The bottleneck was the FSM sequence used to create each IFM vector:

- True MAC activity was only `544` cycles.
- The controller stayed in `COMPUTE` for `15202` cycles.
- Bubbles inside compute state were `15202 - 544 = 14658` cycles.
- Effective throughput inside compute state was `544 / 15202 = 0.0358`
  vector/cycle, or about `27.9` cycles/vector.

The three largest linebuffer states were:

- `CH_COAL_PREP=4896`
- `CH_COAL_READ_REQ=3400`
- `CH_COAL_READ_WAIT=3400`
- Total: `11696 / 15202 = 76.9%` of active linebuffer cycles.

Architectural causes:

- Coalesce/KGEN mixed-tap generation scanned `kh/kw/ic` sequentially.
- Each output vector required multiple prepare, request, wait, and merge steps
  before `CH_COAL_EMIT`.
- Row SRAM was used as a direct tap read source for each output vector, not as
  backing storage for a ready window cache.
- Therefore correctness was good, but the design could not reach
  `1 vector/cycle`.

### 6.4 Effect of IC Tail Split

Before better IC tail splitting, the IC120 case could fall into a slow path:

```text
gemm=85008 cycles=89316
```

After splitting IC into more appropriate commands:

```text
gemm=19418 cycles=24170
```

Command splitting fixed the overly slow fallback/tail behavior, but did not fix
the root linebuffer bottleneck: only `544` vectors were emitted during about
`15202` controller compute cycles. This was a necessary first optimization for
correctness/performance, but it did not change the decision to redesign the
linebuffer.

## 7. Redesign Decision

`conv_channel_linebuf_packer` was removed from the performance roadmap because
there was no small upgrade path to `1 vector/cycle`:

- Adding FIFO only hides downstream backpressure; it does not reduce
  `CH_COAL_PREP` or `CH_COAL_READ_*`.
- Increasing SRAM depth does not help because the stall does not come from
  capacity.
- Scheduler optimization cannot remove the sequential scan inside RTL.
- A one-read row SRAM used as the direct tap source is insufficient for
  coalesce/KGEN mixed taps when the design must emit steadily every cycle.

The new linebuffer must use a different dataflow:

```text
Shared TCDM / stripe in TCDM
        |
        v
row SRAM banks / segment prefetch
        |
        v
window cache: K_MAX x K_MAX x 32B
        |
        v
lane descriptor generator
        |
        v
32-lane byte mux / zero inject
        |
        v
skid FIFO
        |
        v
systolic IFM stream
```

Block roles:

- Row SRAM banks hold input rows or prefetched segments.
- Window cache holds the window currently being emitted; the design should not
  read taps directly from SRAM for every lane of every output.
- Lane descriptor generator maps each lane to `{kh,kw,ic}`, including KGEN seed
  and K-tile increments.
- Lane mux selects bytes from the window cache or zero padding.
- Skid FIFO decouples the large mux timing from systolic ready/valid.

## 8. Implemented `conv_linebuf_stream_packer`

Required features:

- Non-KGEN/non-coalesce: emit vectors according to descriptor
  `{oh, ow, kh, kw, c_base}`.
- Coalesce non-KGEN: when `KH*KW*valid_channels <= 32`, emit one vector
  containing the whole window in lane order `{kh, kw, ic}`.
- KGEN mixed tap: Host/Python only provides seed `{kh,kw,ic}` and
  `k_tile_count`; RTL generates 32 lane descriptors per K tile.
- Padding: zero injection, no materialized padding in SRAM.
- `1x1` pad0: dedicated bypass path, not through window cache.
- Native stride: `1`, `2`.

Current implementation details:

- The banked row/window path uses `ROW_SLOTS = K_MAX + STRIDE_MAX` and
  `BANKS = ROW_SLOTS * STRIDE_MAX`. Each bank is a 256-bit dual-port SRAM word
  array.
- The fill path has a beat metadata FIFO to track OBI responses. The response
  engine writes into bank SRAM and handles single-beat and cross-beat reads.
- C32-aligned mode (`cfg_c32_fast_i`) is only valid when the descriptor
  guarantees a full 32-byte block and a 32-byte aligned channel offset. In that
  case fill/bypass can use the raw 256-bit beat without byte merge.
- KGEN fast path (`c32_kgen_fast`) emits lanes `0..31` directly from the C32
  window word corresponding to the `{kh,kw}` lane descriptor. This is the
  steady-state path for C32 YOLO conv.
- Non-C32/coalesce/tail paths still use byte muxing and `merge_beats` to keep
  correct semantics when an address is not aligned or a vector crosses two OBI
  beats.
- Lightweight background prefetch/fill exists, but it is not DMA 2D/3D
  coalesced fill. Request granularity optimization by row/segment remains a
  separate work item if `ifm_req` must be reduced.

### 8.1 Current Pipeline

`conv_linebuf_stream_packer` is currently a sequential main FSM with several
side pipelines: beat FIFO/response writeback, banked SRAM read/write, output
stage, and a lightweight background fill FSM. It is not yet a fully decoupled
producer/consumer design.

```text
                  cfg/start/next_tile/prefetch
                             |
                             v
+----------------------+  main control FSM  +----------------------+
| CH_IDLE              |------------------->| CH_ENSURE            |
| setup spatial walk   |                    | check needed rows    |
+----------------------+                    +----------+-----------+
                                                       |
                                                       | row miss
                                                       v
                                            +----------+-----------+
                                            | CH_FILL_REQ0/REQ1    |
                                            | issue OBI beat reads |
                                            +----------+-----------+
                                                       |
                                                       v
                                            +----------+-----------+
                                            | beat metadata FIFO   |
                                            | addr_lsb, bytes,     |
                                            | row slot, x, cross   |
                                            +----------+-----------+
                                                       |
                                  OBI rvalid           v
+----------------------+                    +----------+-----------+
| OBI/TCDM read port   |------------------->| response writeback   |
| 256-bit beat         |                    | merge/full-beat mux  |
+----------------------+                    +----------+-----------+
                                                       |
                                                       v
                                            +----------+-----------+
                                            | banked row SRAM      |
                                            | ROW_SLOTS=7          |
                                            | BANKS=14             |
                                            | 256b dual-port banks |
                                            +----------+-----------+
                                                       |
                                                       | row ready
                                                       v
+----------------------+                    +----------+-----------+
| BG fill FSM          |<------------------>| CH_WINDOW_REQ/WAIT   |
| BG_SCAN/REQ/DRAIN    |                    | read window columns  |
| prefetch next row    |                    +----------+-----------+
| during window/emit   |                               |
+----------------------+                               v
                                            +----------+-----------+
                                            | window_q 5x5x256b    |
                                            | pad zero injection   |
                                            +----------+-----------+
                                                       |
                                                       v
                                            +----------+-----------+
                                            | CH_STREAM_PRIME      |
                                            | latch tap coords     |
                                            +----------+-----------+
                                                       |
                                                       v
                                            +----------+-----------+
                                            | CH_STREAM_EMIT       |
                                            | build_emit_row mux   |
                                            | C32/KGEN/coalesce    |
                                            +----------+-----------+
                                                       |
                                                       v
                                            +----------+-----------+
                                            | row_valid/row_ready  |
                                            | to systolic IFM      |
                                            +----------------------+
```

Existing stages:

- **Control/setup**: `CH_IDLE` accepts `start_i`, resets the spatial walk, and
  selects bypass or window path.
- **Row ensure/fill**: `CH_ENSURE` checks rows needed by the current window. If
  a row misses, `CH_FILL_REQ0/REQ1` issues OBI reads one 256-bit beat at a
  time; accesses crossing a beat boundary need two requests.
- **OBI request tracking**: every request pushes metadata into `beat_fifo_q`
  (`addr_lsb`, `valid_bytes`, row slot, `x`, and cross/single flag). The FIFO
  depth is `4`, so this is a shallow pipeline, not a large burst queue.
- **Response writeback**: on `obi_rvalid_i`, the response engine consumes the
  matching metadata and writes bank SRAM. C32 aligned/full-beat accesses write
  `obi_rdata_i` directly. Tail or non-aligned accesses use `merge_beats`.
  Cross-beat accesses keep beat0 in `resp_beat0_q`.
- **Banked row SRAM**: `ROW_SLOTS = K_MAX + STRIDE_MAX = 7` and
  `BANKS = ROW_SLOTS * STRIDE_MAX = 14`. Bank index is based on
  `{row_slot, x % STRIDE_MAX}` to support window-column reads and response
  writes.
- **Window load/slide**: `CH_WINDOW_REQ/WAIT` reads SRAM one `kw` at a time to
  populate `window_q`. When sliding along the same output row, `slide_window`
  reuses old columns and reads only the new column(s) required by `stride_w`.
- **Emit stage**: `CH_STREAM_PRIME` latches tap coordinates, and
  `CH_STREAM_EMIT` calls `build_emit_row()` and drives `row_valid_o`. The
  formatter supports `c32_kgen_fast`, `coalesce+kgen`, `coalesce`, and
  non-coalesce modes.
- **Background fill**: `BG_IDLE/BG_SCAN/BG_REQ0/BG_REQ1/BG_DRAIN` runs only
  while the main FSM is in `CH_WINDOW_REQ`, `CH_WINDOW_WAIT`,
  `CH_STREAM_PRIME`, or `CH_STREAM_EMIT`. It uses OBI only when the main path
  is not issuing a request, and fetches rows for the next output row.

Already overlapped or pipelined:

- OBI request and response are decoupled by the beat metadata FIFO.
- Response writeback can write bank SRAM independently of the main FSM state.
- SRAM has separate read/window and write/response paths, so window reads can
  overlap response writes when resources do not conflict.
- Background fill can run during window load and stream emit.
- Emit has a coordinate stage `stg1` and an output register
  `row_data_q/row_valid_out_q`.

Not fully pipelined yet:

- `CH_ENSURE`, `CH_FILL_REQ*`, and `CH_FILL_DRAIN` still belong to the main
  FSM. If a row misses on the critical path, stream emit cannot proceed.
- Initial window load is still serialized by `kw` through `CH_WINDOW_REQ/WAIT`.
- There is only one OBI read port; main fill/bypass has priority over
  background fill.
- Beat FIFO depth is small and request granularity is still pixel/C32-word
  based, not row/segment coalesced DMA.
- Background fill only prefetches the next row. There is no context ping-pong
  or fully independent fill scheduler yet.

Performance contract:

- With a window-cache hit and systolic ready, the emit path should produce one
  256-bit vector per cycle.
- Refill/prefetch should not be serialized on the critical emit path in steady
  state.
- Counter/PMU support should measure `window_refill`, `emit_valid`,
  `emit_ready`, `emit_stall`, `k_tile_transition`, and `padding_zero`
  separately.

Storage estimate:

- `K_MAX=5`, `C_BLOCK=32B` gives a window cache of
  `5*5*32B = 800B = 6.4Kb`.
- This is small for ASIC. The main risk is mux/timing from `25 x 32B` into
  32 lanes, so the mux should be pipelined or split by lane group if timing
  requires it.

Stride policy:

- `stride=1`: with row banks, reading one column per bank per cycle is enough
  to update a `5x5` window when sliding to the next output.
- `stride=2`: this must not be implemented by scanning stride-1 outputs and
  discarding half of them. It needs a wider prefetch segment or x-banking so
  the next columns are already available before emit.
- `stride>=3`: not a native performance target in this phase. The compiler must
  decompose/rewrite the operation or use the packed prepare backup.

## 9. Verification for the Stream Linebuffer

Required unit tests before cluster integration:

- `1x1` bypass: pad0, stride1, tail width, unaligned address.
- Non-KGEN single tap: `3x3`, `5x5`, pad0/pad1, stride1/2.
- Coalesce: `3x3 IC=3`, `5x5 IC=1`, tail lanes zero.
- KGEN mixed tap: `3x3 IC=32`, `3x3 IC=96`, `3x3 IC=120`, `5x5 IC=32`.
- Padding: top/bottom/left/right/asymmetric.
- Read alignment: same-beat and cross-beat 256-bit reads.
- Backpressure: random `ifm_ready_i` stalls while output data remains exact.
- Performance assertion: after warm-up, accepted vectors must be one per cycle
  for supported steady-state regions.

Cluster tests:

- Run existing `CONV_PERF_CASE=17/18/20`.
- Add a large-shape first-layer YOLO case: `3x3/s2/p1/IC=3`.
- Collect PMU after every run and fail the performance test if linebuffer
  active cycles/vector exceeds the configured threshold.

## 10. Pipelining / Decoupled FSM Roadmap

This section describes what has been partially implemented and what remains on
the roadmap. `conv_linebuf_stream_packer` is no longer a purely sequential FSM:
it already has dual-port banks, a beat-FIFO response engine, background fill
state, and a pipelined emit stage. However, fill/window/emit still have
sequential points in row ensure and request granularity, so the design has not
yet reached ideal DMA/segment streaming behavior for all shapes.

### 10.1 Problem with the Single Main FSM

The current main flow still lets one FSM own both producer and consumer work:

- When the main FSM is in `CH_FILL_*`, linebuffer stream emission is stopped.
- This serializes fill and emit for critical row misses, which creates cycle
  bubbles on large or unaligned cases.
- Background fill helps only when the stream path is already in window/emit
  states and the OBI read port is idle.

### 10.2 Decoupled Producer/Consumer Direction

The banked SRAM structure can support a cleaner two-FSM design:

1. **Fill FSM (producer, SRAM write side)**
   - Owns row scanning, OBI request issue, response accounting, and row-ready
     scoreboarding.
   - Writes data into row SRAM and marks `row_slot_valid_q` /
     `row_slot_ih_q` when a row is complete.
   - Runs whenever there is a future row miss to cover, not only when the main
     FSM reaches `CH_ENSURE`.
2. **Stream FSM (consumer, SRAM read side)**
   - Owns `CH_WINDOW_REQ`, `CH_WINDOW_WAIT`, `CH_STREAM_PRIME`,
     `CH_STREAM_EMIT`, and `CH_STREAM_DONE` behavior.
   - Reads SRAM to slide the window and emits vectors into systolic.
   - Synchronizes through a row-ready scoreboard. If the needed row is valid,
     it continues streaming; if not, it stalls only on that missing row.

The key design point is that the stream FSM should not enter fill states. It
should only observe row readiness and either emit or wait.

### 10.3 C-Block Double Buffering / Ping-Pong Caching

To hide fill latency when switching between K tiles or C-blocks, the row SRAM
space can be split into two contexts:

- While the stream FSM reads context 0, the fill FSM can prefetch context 1.
- At tile/context boundary, the contexts swap without a long empty transition.
- The tradeoff is more row state: valid bits, row tags, pending counters, and
  possibly duplicated bank address context per ping-pong side.

This is not implemented in the current RTL. The current implementation only has
lightweight background fill for the next row, not a full context ping-pong
cache.
