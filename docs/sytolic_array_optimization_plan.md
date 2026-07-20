# Performance Gap Analysis: 68,968 → 8,704 cycles

## 0. Current Implementation Status

This document preserves the historical case-20/case-22 optimization analysis.
It is no longer a complete description of the current Micro-YOLO execution
path.

Current implemented flow:

- The active Conv2D performance path is `conv_linebuf_stream_packer` inside
  `systolic_controller`, not the legacy `conv_channel_linebuf_packer`.
- Micro-YOLO no longer asks Snitch firmware to build each linebuffer tile in
  the hot path. `hw/rtl/cluster/tb/npu_linebuf_precompute.py` generates a runtime L2
  descriptor manifest plus blobs containing `npu_conv2d_linebuf_job_desc_t`
  arrays; firmware DMA-copies those blobs into scratch/TCDM and
  `npu_conv2d_packed_run_linebuf_job_descs()` dispatches them.
- The descriptor runner preloads the next linebuffer/GEMM shadow-register image
  while the current job runs. This removed most per-job C planner arithmetic
  from Snitch.
- C32-blocked layers use the RTL `C32_FAST` path when the host descriptor
  guarantees `input_c=32`, `input_c_base=0`, `lane_base=0`,
  `block_valid_bytes=32`, and 32-byte aligned base/stride/channel offsets.
  This bypasses byte-level `merge_beats` on the main C32 data path.
- `merge_beats` is still required for RGB stem, sub-C32/tail chunks,
  `lane_base != 0`, raw/NHWC fallback, and any non-32B-aligned access.
- The large "Stripe-Stationary resident cache" idea is not active in RTL. The
  current hardware keeps a row-ring/window state sized around kernel/stride and
  relies on host spatial job descriptors plus C32 alignment to reduce schedule
  overhead.

Latest Micro-YOLO E2E snapshot after runtime L2 descriptor manifest/blobs:

| Counter | Value |
|---|---:|
| PMU cycles | 388,146 |
| Previous field-precompute bridge snapshot | 403,128 |
| Delta | -14,982 cycles (-3.7%) |
| Total systolic useful compute | 69,696 cycles |

The remaining gap is dominated by IFM request volume/window setup and graph
operator overhead, not by Snitch recomputing linebuffer register fields. See
`linebuffer_architecture.md`, `conv2d_packed_systolic_plan.md`, and
`micro_yolov8_integration_test_plan.md` for the current architecture.

## 1. Cycle Budget Decomposition

Historical case-20 snapshot before the later C32/group/host-descriptor work:

Target: 8,704 cycles (systolic compute). Measured: 68,968 cycles. Gap: 60,264 cycles (87.4% overhead).

### 1.1 Controller FSM Time (from PMU)

| State | Cycles | % of Total | Pipelined? |
|---|---|---|---|
| `WAIT_DRAIN` | 31,101 | 44.91% | Partially — weight preload + linebuf prefetch overlap drain, but compute does not |
| `IDLE` | 23,297 | 33.64% | No — firmware setup + DMA between controller invocations |
| `COMPUTE` | 14,236 | 20.56% | Yes — includes 8,704 systolic cycles + 5,532 cycles array flush + stalls |
| `LOAD_WEIGHTS` | 600 | 0.87% | Partially — only first tile; subsequent tiles use preload in `WAIT_DRAIN` |
| `DONE` | 12 | 0.02% | N/A |

### 1.2 Drain Engine Time (from PMU)

| State | Cycles | % of Total |
|---|---|---|
| `DRAIN_IDLE` | 60,184 | 86.91% |
| `DRAIN_ACCUM_WRITE` | 8,647 | 12.49% |
| `DRAIN_ACCUM_READ` | 415 | 0.60% |

The drain engine is idle 86.91% of the time. It only performs useful work for 9,062 cycles (12.49% + 0.60%).

### 1.3 Linebuffer Time (from PMU)

| State | Cycles | % of Total |
|---|---|---|
| `CH_IDLE` | 24,485 | 35.36% |
| `CH_FILL_REQ0` | 11,648 | 16.82% |
| `CH_STREAM_DONE` | 9,617 | 13.89% |
| `CH_STREAM_EMIT` | 8,704 | 12.57% |
| `CH_FILL_REQ1` | 8,632 | 12.47% |
| `CH_WINDOW_REQ/WAIT` | 3,264 | 4.72% |
| `CH_FILL_DRAIN` | 1,456 | 2.10% |
| `CH_ENSURE` | 896 | 1.29% |
| `CH_STREAM_PRIME` | 544 | 0.79% |

Linebuffer issues 20,280 OBI requests with 0 stalls. It spends 20,280 cycles in `CH_FILL_REQ0 + CH_FILL_REQ1` fetching IFM data.

---

## 2. Root Cause Identification (5 Time Sinks)

### Sink 1: Firmware/DMA Overhead — 23,297 cycles (33.6%)

**Evidence:** `IDLE = 23,297`, `CH_IDLE = 24,485`.

**Cause:** Case 20 firmware splits the 16×16 output into 6 `oh_base` chunks × 2 IC passes (96 + 24). Each chunk requires:
- Snitch MMIO register writes to configure the controller (34,411 instructions total)
- DMA transfers for IFM/weights/OFM between L2 and TCDM (36 DMA transactions, 3,806 busy cycles)
- Controller startup latency per invocation

**This overhead is entirely sequential and cannot be overlapped with compute.**

### Sink 2: WAIT_DRAIN Blocking Compute — 31,101 cycles (44.9%)

**Evidence:** `WAIT_DRAIN = 31,101`. During this state, `compute_en = 0`.

**Cause:** The main FSM at [systolic_controller.sv:1089](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1089) requires `drain_cnt_q == 0 && ofm_fifo_empty` before transitioning to the next tile. This means compute cannot begin until all 256 output rows from the previous tile have been fully written to TCDM or PSum buffer.

**What is pipelined:** Weight preload and linebuffer prefetch execute during `WAIT_DRAIN` ([systolic_controller.sv:1020-1070](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1020-L1070)). This saves ~600 cycles per tile that would otherwise be spent in `LOAD_WEIGHTS`.

**What is NOT pipelined:** Compute of tile(N+1) cannot overlap with drain of tile(N).

### Sink 3: Array Flush Latency — 5,532 cycles (8.0%)

**Evidence:** `COMPUTE = 14,236` but only 8,704 of those are actual systolic MAC operations (`perf_compute_en`). The remaining 5,532 cycles are `ARRAY_FLUSH_CYCLES = 2 * ARRAY_DIM = 64` cycles per tile × ~86 tiles needing flush, plus stalls from `ofm_fifo_full = 25`.

**Cause:** After the last IFM vector enters the systolic array, results take `ARRAY_DIM = 32` cycles to propagate through the pipeline. The flush counter at [systolic_controller.sv:1009](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1009) holds the FSM in `COMPUTE` (but not computing) for 64 cycles.

### Sink 4: Linebuffer Refill per K-Tile — 20,280 OBI requests

**Evidence:** `CH_FILL_REQ0 = 11,648`, `CH_FILL_REQ1 = 8,632`. Total fill = 20,280 cycles. `linebuf_obi_req = 20,280`.

**Cause:** With `input_h ≤ 5` (due to `oh_base` tiling), `row_cache_full_mode = 1` and `row_cache_reuse = 1`. The linebuffer fills its SRAM banks once, then replays from cache for subsequent K tiles sharing the same `c_base`. However, when `c_base` changes (IC=96 → IC=24), the cache must be invalidated and refilled. With 204 K tiles across 12 invocations, the linebuffer executes approximately 12 full refills.

**Cost per refill:** ~1,690 OBI requests × 1 cycle each = ~1,690 cycles.

### Sink 5: PSum Buffer → TCDM Write Traffic — 8,647 cycles

**Evidence:** `DRAIN_ACCUM_WRITE = 8,647`, `ofm_req = 3,264`.

**Cause:** `psum_buf_overlap_active = 1` when `psum_buf_active && linebuf_kgen_multi && !cfg_requant_en_i`. The PSum buffer (64 KB, dual-bank) stores partial sums on-chip across K tiles. Only the final tile writes to TCDM via 4× OBI ports. Each write drains one 1024-bit OFM row in 1 cycle (4× 256-bit ports). For 256 M rows on the final tile, this takes 256 cycles. The remaining 8,391 cycles are intermediate PSum buffer writes (combinational, 1 cycle per row) across non-final tiles.

---

## 3. What IS Already Pipelined

| Feature | Status | Location |
|---|---|---|
| Weight preload during `WAIT_DRAIN` | ✓ Active | [systolic_controller.sv:1020-1058](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1020-L1058) |
| Linebuffer prefetch during `WAIT_DRAIN` | ✓ Active | [systolic_controller.sv:1060-1070](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1060-L1070) |
| PSum buffer (on-chip accumulation) | ✓ Active | [systolic_controller.sv:285-291](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L285-L291) |
| OFM FIFO decoupling compute from drain | ✓ Active (depth=128) | [systolic_controller.sv:443-459](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L443-L459) |
| Linebuffer sliding window (1 vector/cycle emit) | ✓ Active | `CH_STREAM_EMIT = 8,704` = target |
| Beat FIFO for pipelined OBI fill | ✓ Active (depth=4) | [conv_linebuf_stream_packer.sv:97-112](/neural-ai/hw/rtl/systolic/conv_linebuf_stream_packer.sv#L97-L112) |

## 4. What Is NOT Pipelined (Ranked by Cycle Impact)

### Rank 1: Firmware Loop Overhead (23,297 cycles, 33.6%)

**Problem:** 6 `oh_base` × 2 IC passes = 12 controller invocations. Each invocation incurs:
- MMIO register configuration (~100 cycles)
- DMA setup and completion wait (~317 cycles average per DMA)
- Controller `IDLE → LOAD_WEIGHTS` transition

**Solution:** Eliminate `oh_base` loop — run full `output_h=16` in a single invocation. This requires the linebuffer to operate without `row_cache_full_mode` (since `input_h=16 > K_MAX=5`).

**Prerequisite:** Linebuffer must support efficient refill when `row_cache_reuse=0`. Current refill costs ~1,690 cycles per K tile. With 27 tiles and no cache reuse, that becomes 27 × 1,690 = 45,630 additional cycles — a net regression.

**Historical proposed fix:** Implement stripe-stationary linebuffer (Section
6.4 of [npu_refactor_and_review.md](/neural-ai/docs/npu_refactor_and_review.md#L277-L332)).
That idea used a resident cache holding 4 rows × 16 cols × 4 cblocks × 32B =
8 KB per context, with ping-pong 2 contexts = 16 KB total.

**Current status:** This large resident-cache path is not active. The current
implementation instead uses host-generated spatial descriptors, C32 fast path,
row-ring/window caching, and shadow-register preload. The next useful work is
to improve the row/window pipeline and C32 segment request efficiency without
reintroducing a full activation-tile cache.

**Estimated savings:** 23,297 cycles (100% of Sink 1).

### Rank 2: Compute/Drain Serialization (22,454 cycles, 32.6%)

**Problem:** `WAIT_DRAIN` = 31,101 cycles. Of this, weight preload uses ~600 cycles and linebuf prefetch uses ~8,047 cycles. The remaining ~22,454 cycles are pure stall waiting for `drain_cnt_q == 0 && ofm_fifo_empty`.

**Solution:** Allow compute of tile(N+1) to begin before drain of tile(N) completes. The `psum_buf_overlap_active` path at [systolic_controller.sv:1073-1088](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1073-L1088) already supports this transition — it checks `psum_buf_overlap_active && linebuf_has_next_k_tile && !psum_buf_needs_external && weight_preload_done_q && !linebuf_prefetch_busy && (array_flush_cnt_q == '0)`.

**Current blocker:** The transition at line 1075 requires `!psum_buf_needs_external`, which is false when `cfg_sys_accum_en_i=1 && k_tile_idx_q==0`. For subsequent tiles where `psum_buf_needs_external=0`, the overlap path should fire. The fact that `WAIT_DRAIN` still shows 31,101 cycles suggests the conditions at line 1073-1075 are not being met — likely because `linebuf_prefetch_busy` or `array_flush_cnt_q != 0` holds the transition.

**Estimated savings:** 15,000–22,000 cycles (depends on whether drain throughput can keep up with compute).

### Rank 3: Array Flush Overhead (5,532 cycles, 8.0%)

**Problem:** `ARRAY_FLUSH_CYCLES = 64` per tile. With ~86 internal tiles (204 K-tiles distributed across 12 invocations, each having ~17 tiles), flush overhead = 86 × 64 = 5,504 cycles.

**Solution:** If compute/drain overlap is achieved (Rank 2), flush cycles are hidden under drain. No RTL change needed — the flush counter already runs in `WAIT_DRAIN` at [systolic_controller.sv:1016-1018](/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1016-L1018).

**Estimated savings:** 5,532 cycles (fully hidden by Rank 2 fix).

### Rank 4: Linebuffer TCDM Fill Latency (~20,280 cycles, overlappable)

**Problem:** Linebuffer fetches IFM data at 1 OBI request/cycle. With `row_cache_reuse=1`, this cost is amortized across K tiles sharing the same `c_base`. But each `c_base` change triggers a full refill.

**Current overlap:** Linebuffer prefetch executes during `WAIT_DRAIN`, hiding ~8,000 of the 20,280 cycles under drain time.

**Historical stripe-stationary estimate:** Each stripe's IFM data (8 KB) would
require ~256 OBI beats at 256 bits/beat = 256 cycles to fill. With ping-pong
buffering, fill of stripe(N+1) could overlap under compute of stripe(N).

**Current status:** The active RTL does not implement that resident-cache mode.
The relevant target is now to make the row-ring/window path issue fewer and
more regular C32 reads, then overlap window setup with stream emit.

**Historical estimated savings:** Fully hidden under compute if the
stripe-stationary resident-cache path had been implemented. For the active
row-ring/window path, use the priorities in Section 6.5 instead.

---

## 5. Historical Achievable Target

| Scenario | Estimated Cycles | Utilization |
|---|---|---|
| Historical case-20 snapshot | 68,968 | 12.6% |
| After eliminating firmware loop (stripe-stationary) | ~45,000 | 19.3% |
| + Compute/drain overlap | ~18,000–25,000 | 35–48% |
| + Stripe-stationary with ping-pong fill | ~17,408 | 50.0% |
| Theoretical minimum (compute only) | 8,704 | 100% |

The 17,408-cycle target (2× compute) is the practical lower bound for this workload because weight load bandwidth equals compute bandwidth: each systolic cycle consumes one 256-bit IFM vector AND requires one weight row pre-loaded. With a single `obi_i` port shared between weight load and IFM fetch, the minimum is `max(compute, weight_load) + startup` ≈ 2× compute.

For this historical case, reaching below 2× compute would require either:
- A second OBI read port for weights (separate from IFM/linebuffer)
- Or pre-loading all weights into on-chip SRAM before compute begins (feasible for small K but not for 34 tiles × 32 rows × 32B = 34 KB per stripe)

---

## 6. Follow-up Pipeline Opportunities After C32-Blocked Group Mode

Snapshot: `CONV_PERF_CASE=22`, `16x16x120`, C32-blocked input memory, channel split `96 + 24`.

Measured after enabling C32-blocked group addressing:

| Counter | Cycles / Count |
|---|---:|
| PMU cycles | 25,854 |
| Monitor cycles | 26,132 |
| Systolic useful compute | 8,704 |
| `COMPUTE` state | 13,645 |
| `WAIT_DRAIN` | 3,292 |
| `LOAD_WEIGHTS` | 68 |
| `linebuf_obi_req` | 8,750 |
| `linebuf_obi_stall` | 46 |
| `CH_ENSURE` | 2,176 |
| `CH_WINDOW_WAIT` | 1,632 |
| `CH_WINDOW_REQ` | 544 |
| `CH_STREAM_PRIME` | 544 |
| `CH_STREAM_DONE` | 1,913 |
| `CH_STREAM_EMIT` | 8,771 |
| `DRAIN_ACCUM_WRITE` | 8,693 |
| `DRAIN_ACCUM_READ` | 284 |

Compared with the previous C32-blocked `32 + 32 + 32 + 24` split, the `96 + 24` group mode reduced PMU cycles from 28,340 to 25,854. The improvement mainly comes from fewer controller invocations and less OFM/psum traffic, not from lower IFM fetch count: `linebuf_obi_req` stayed nearly flat (`8,820 -> 8,750`).

### 6.1 Highest Priority: Pipeline Linebuffer Window Setup

Remaining linebuffer overhead is still large:

- `CH_ENSURE = 2,176`
- `CH_WINDOW_WAIT = 1,632`
- `CH_WINDOW_REQ = 544`
- `CH_STREAM_PRIME = 544`
- `CH_STREAM_DONE = 1,913`

These states account for roughly 6.8k cycles of non-emit linebuffer activity.

Recommended RTL work:

- Merge or pipeline `CH_WINDOW_REQ`, `CH_WINDOW_WAIT`, and `CH_STREAM_PRIME`. With 1-cycle SRAM read latency, the linebuffer should issue the next window or slide-column bank read while the current row vector is being emitted.
- Convert `CH_ENSURE` into a row-ready scoreboard path. If the current output window rows are already resident, stream immediately and let missing future rows fill in the background.
- Reduce `CH_STREAM_DONE` by arming the next K tile earlier. When the controller asserts `next_tile_i`, the linebuffer should already have preserved enough row/window state to enter `CH_WINDOW_REQ` or `CH_STREAM_PRIME`, not repeat a long reset/setup path.

Estimated gain: 3k-5k cycles for case 22 if implemented correctly.

### 6.2 Controller WAIT_DRAIN and K-Tile Transition

Weight load is mostly hidden now: `LOAD_WEIGHTS = 68`, so weight preload is not the main problem. Remaining `WAIT_DRAIN = 3,292` cycles comes from drain/transition conditions.

Current behavior:

- Within one invocation, `psum_buf_overlap_active` can start the next K tile before final drain completes, but it still requires `weight_preload_done_q`, `!linebuf_prefetch_busy`, and `array_flush_cnt_q == 0`.
- Across the `IC=96 -> IC=24` boundary, there is effectively no hardware overlap because firmware calls `npu_conv2d_packed_run_oc32()` twice. The first invocation must finish and drain before the tail invocation starts.

Recommended work:

- Allow next K-tile compute to start once the linebuffer has the first valid window, not only after full prefetch completes.
- Keep the psum/window state across channel groups by adding a group-level command descriptor. A single logical command should contain:
  - group 0: C32-blocked base block 0, `input_c=96`, `accumulate=0`
  - group 1: C32-blocked base block 3, `input_c=24`, `accumulate=1`
  - final drain only after the last group
- Continue using the current `psum_buf_overlap_active` mechanism for intra-group K tiles.

Estimated gain: 1k-3k cycles from better K-tile transition, plus roughly 2k cycles if the `96 -> 24` firmware boundary is removed.

### 6.3 OFM/PSum Drain Path

Current drain state distribution:

- `DRAIN_ACCUM_WRITE = 8,693`
- `DRAIN_ACCUM_READ = 284`

The psum double buffer avoids most external psum reads. The remaining cost is mostly one row per cycle psum-buffer update or final output write. This is useful work, but it can still backpressure transitions and FIFO progress.

Recommended work:

- Add a small skid buffer between OFM FIFO and psum-buffer write so transient psum-buffer port conflicts do not stall OFM FIFO pop.
- Batch or pipeline psum-buffer write arbitration so intermediate tiles do not force controller-visible bubbles.
- Prioritize final output write once `final_tile=1`; avoid read-prefetch arbitration stealing cycles from final drain.

Estimated gain: several hundred cycles to about 1k cycles.

### 6.4 Requant Path

Case 22 does not enable requant, so requant is not part of the current bottleneck.

The current `requant_pipeline.sv` already has 3-stage ready/valid flow. For requant-enabled layers, the likely bottleneck is not the math pipeline but the output write path:

- non-requant int32 output uses 4 OBI ports for 128 bytes per row
- requant int8 output uses one 256-bit port for 32 bytes per row

Because requant output is 4x smaller, one port is usually adequate. Optimize this only if a requant-enabled performance case shows `requant_out_valid` backpressure or high output-port stall.

### 6.5 Updated Priority Order

1. Pipeline linebuffer window setup: `WINDOW_REQ/WAIT/PRIME` plus rolling stream reads.
2. Start next K tile when the first next-tile window is ready, instead of waiting for full linebuffer prefetch completion.
3. Add a group-level scheduler so `96 + 24` channel groups execute as one logical command with a single final drain.
4. Add light drain skid/batch buffering around psum-buffer writes and final output writes.
5. Defer requant optimization until a requant-enabled layer proves it is limiting performance.

Expected next target for case 22 after items 1-3: about 18k-21k monitor cycles. Getting close to the 8,704-cycle compute floor requires a deeper linebuffer/window pipeline that emits near one vector per cycle with minimal setup and drain bubbles.
