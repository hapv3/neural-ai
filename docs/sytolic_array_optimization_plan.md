# Performance Gap Analysis: 68,968 → 8,704 cycles

## 1. Cycle Budget Decomposition

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

**Cause:** The main FSM at [systolic_controller.sv:1089](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1089) requires `drain_cnt_q == 0 && ofm_fifo_empty` before transitioning to the next tile. This means compute cannot begin until all 256 output rows from the previous tile have been fully written to TCDM or PSum buffer.

**What is pipelined:** Weight preload and linebuffer prefetch execute during `WAIT_DRAIN` ([systolic_controller.sv:1020-1070](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1020-L1070)). This saves ~600 cycles per tile that would otherwise be spent in `LOAD_WEIGHTS`.

**What is NOT pipelined:** Compute of tile(N+1) cannot overlap with drain of tile(N).

### Sink 3: Array Flush Latency — 5,532 cycles (8.0%)

**Evidence:** `COMPUTE = 14,236` but only 8,704 of those are actual systolic MAC operations (`perf_compute_en`). The remaining 5,532 cycles are `ARRAY_FLUSH_CYCLES = 2 * ARRAY_DIM = 64` cycles per tile × ~86 tiles needing flush, plus stalls from `ofm_fifo_full = 25`.

**Cause:** After the last IFM vector enters the systolic array, results take `ARRAY_DIM = 32` cycles to propagate through the pipeline. The flush counter at [systolic_controller.sv:1009](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1009) holds the FSM in `COMPUTE` (but not computing) for 64 cycles.

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
| Weight preload during `WAIT_DRAIN` | ✓ Active | [systolic_controller.sv:1020-1058](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1020-L1058) |
| Linebuffer prefetch during `WAIT_DRAIN` | ✓ Active | [systolic_controller.sv:1060-1070](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1060-L1070) |
| PSum buffer (on-chip accumulation) | ✓ Active | [systolic_controller.sv:285-291](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L285-L291) |
| OFM FIFO decoupling compute from drain | ✓ Active (depth=128) | [systolic_controller.sv:443-459](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L443-L459) |
| Linebuffer sliding window (1 vector/cycle emit) | ✓ Active | `CH_STREAM_EMIT = 8,704` = target |
| Beat FIFO for pipelined OBI fill | ✓ Active (depth=4) | [conv_linebuf_stream_packer.sv:97-112](file:///home/dev01/neural-ai/hw/rtl/systolic/conv_linebuf_stream_packer.sv#L97-L112) |

## 4. What Is NOT Pipelined (Ranked by Cycle Impact)

### Rank 1: Firmware Loop Overhead (23,297 cycles, 33.6%)

**Problem:** 6 `oh_base` × 2 IC passes = 12 controller invocations. Each invocation incurs:
- MMIO register configuration (~100 cycles)
- DMA setup and completion wait (~317 cycles average per DMA)
- Controller `IDLE → LOAD_WEIGHTS` transition

**Solution:** Eliminate `oh_base` loop — run full `output_h=16` in a single invocation. This requires the linebuffer to operate without `row_cache_full_mode` (since `input_h=16 > K_MAX=5`).

**Prerequisite:** Linebuffer must support efficient refill when `row_cache_reuse=0`. Current refill costs ~1,690 cycles per K tile. With 27 tiles and no cache reuse, that becomes 27 × 1,690 = 45,630 additional cycles — a net regression.

**True fix:** Implement stripe-stationary linebuffer (Section 6.4 of [npu_refactor_and_review.md](file:///home/dev01/neural-ai/docs/npu_refactor_and_review.md#L277-L332)). Resident cache holds 4 rows × 16 cols × 4 cblocks × 32B = 8 KB per context. Ping-pong 2 contexts = 16 KB total. This eliminates both the firmware loop overhead and per-tile refill.

**Estimated savings:** 23,297 cycles (100% of Sink 1).

### Rank 2: Compute/Drain Serialization (22,454 cycles, 32.6%)

**Problem:** `WAIT_DRAIN` = 31,101 cycles. Of this, weight preload uses ~600 cycles and linebuf prefetch uses ~8,047 cycles. The remaining ~22,454 cycles are pure stall waiting for `drain_cnt_q == 0 && ofm_fifo_empty`.

**Solution:** Allow compute of tile(N+1) to begin before drain of tile(N) completes. The `psum_buf_overlap_active` path at [systolic_controller.sv:1073-1088](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1073-L1088) already supports this transition — it checks `psum_buf_overlap_active && linebuf_has_next_k_tile && !psum_buf_needs_external && weight_preload_done_q && !linebuf_prefetch_busy && (array_flush_cnt_q == '0)`.

**Current blocker:** The transition at line 1075 requires `!psum_buf_needs_external`, which is false when `cfg_sys_accum_en_i=1 && k_tile_idx_q==0`. For subsequent tiles where `psum_buf_needs_external=0`, the overlap path should fire. The fact that `WAIT_DRAIN` still shows 31,101 cycles suggests the conditions at line 1073-1075 are not being met — likely because `linebuf_prefetch_busy` or `array_flush_cnt_q != 0` holds the transition.

**Estimated savings:** 15,000–22,000 cycles (depends on whether drain throughput can keep up with compute).

### Rank 3: Array Flush Overhead (5,532 cycles, 8.0%)

**Problem:** `ARRAY_FLUSH_CYCLES = 64` per tile. With ~86 internal tiles (204 K-tiles distributed across 12 invocations, each having ~17 tiles), flush overhead = 86 × 64 = 5,504 cycles.

**Solution:** If compute/drain overlap is achieved (Rank 2), flush cycles are hidden under drain. No RTL change needed — the flush counter already runs in `WAIT_DRAIN` at [systolic_controller.sv:1016-1018](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L1016-L1018).

**Estimated savings:** 5,532 cycles (fully hidden by Rank 2 fix).

### Rank 4: Linebuffer TCDM Fill Latency (~20,280 cycles, overlappable)

**Problem:** Linebuffer fetches IFM data at 1 OBI request/cycle. With `row_cache_reuse=1`, this cost is amortized across K tiles sharing the same `c_base`. But each `c_base` change triggers a full refill.

**Current overlap:** Linebuffer prefetch executes during `WAIT_DRAIN`, hiding ~8,000 of the 20,280 cycles under drain time.

**If stripe-stationary is implemented (Rank 1 fix):** Each stripe's IFM data (8 KB) requires ~256 OBI beats at 256 bits/beat = 256 cycles to fill. With ping-pong buffering, fill of stripe(N+1) overlaps entirely under compute of stripe(N) (8,704 / 8 stripes = 1,088 cycles per stripe, which is 4× the 256-cycle fill time).

**Estimated savings:** Fully hidden under compute after stripe-stationary implementation.

---

## 5. Achievable Target

| Scenario | Estimated Cycles | Utilization |
|---|---|---|
| Current | 68,968 | 12.6% |
| After eliminating firmware loop (stripe-stationary) | ~45,000 | 19.3% |
| + Compute/drain overlap | ~18,000–25,000 | 35–48% |
| + Stripe-stationary with ping-pong fill | ~17,408 | 50.0% |
| Theoretical minimum (compute only) | 8,704 | 100% |

The 17,408-cycle target (2× compute) is the practical lower bound for this workload because weight load bandwidth equals compute bandwidth: each systolic cycle consumes one 256-bit IFM vector AND requires one weight row pre-loaded. With a single `obi_i` port shared between weight load and IFM fetch, the minimum is `max(compute, weight_load) + startup` ≈ 2× compute.

To reach below 2× compute would require either:
- A second OBI read port for weights (separate from IFM/linebuffer)
- Or pre-loading all weights into on-chip SRAM before compute begins (feasible for small K but not for 34 tiles × 32 rows × 32B = 34 KB per stripe)
