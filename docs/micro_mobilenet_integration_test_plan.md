# Micro-MobileNet Integration Test Plan

## Objective

Prove that the NPU can run the operator pattern that defines MobileNet-style
networks, especially depthwise separable convolution:

```text
standard Conv -> Depthwise Conv -> Pointwise Conv -> activation/residual/pool
```

Input: **96x96 RGB INT8**

First end-to-end target: run a compact MobileNet-like graph with every
MobileNet operator represented at least once, and with each convolution type
represented at least twice:

- standard spatial Conv2D: at least 2 layers;
- DepthwiseConv2D: at least 2 layers;
- Pointwise Conv2D `1x1`: at least 2 layers.

Current status: **planning with first pointwise and first depthwise fast paths
implemented**.
Existing Micro-YOLO infrastructure provides the stem Conv3x3 linebuffer path,
AFU/vector activation ops, AFU Add, iDMA movement, and the graph/test harness.
`NPU_OP_CONV2D1X1_C32_REQUANT` now covers the direct C32 `1x1` fast path for
`C32->C32`. `NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT` now covers a linebuffer-fed
lane-wise depthwise `3x3/s1/p1` fast path for any channel count that is a
multiple of 32. `NPU_OP_DEPTHWISE3X3S2P1_C32_REQUANT` extends the same native
path to stride-2 downsample blocks. New work is still required for multi-C32
pointwise tiling, native GlobalAvgPool, and planner-side tail splitting when
`C % 32 != 0`.

The target implementation should stay on native accelerator paths. Scalar CPU
loops are allowed only for golden-model generation, setup/validation code, or
very small tail handling that cannot yet be expressed by the native datapaths.
They are not acceptable as the planned implementation for convolution,
activation, residual, pooling, or layout movement.

---

## 1. Topology

The topology intentionally includes every MobileNet operator class while keeping
all activation tensors C32-blocked. Logical shapes below use NxCxHxW notation;
the physical implementation should keep the existing ROW32/C32 packed layout.

| # | Layer | Op | Config | Output (logical NxCxHxW) | Hardware Unit |
|---|---|---|---|---|---|
| 0 | Input | - | RGB INT8 | 1x3x96x96 | L2 |
| 1 | Stem_Conv | **Standard Conv2D** | K=3x3, S=2, Pad=1, C3->C32 | 1x32x48x48 | existing systolic linebuffer + requant |
| 2 | Stem_ReLU6 | **ReLU6 / clamp** | clamp INT8 to quantized ReLU6 range | 1x32x48x48 | prefer requant clamp fusion; otherwise AFU clamp |
| 3 | DW0 | **DepthwiseConv2D** | K=3x3, S=1, Pad=1, C32 groups | 1x32x48x48 | RTL depthwise linebuffer |
| 4 | DW0_ReLU6 | **ReLU6 / clamp** | element-wise | 1x32x48x48 | prefer depthwise requant clamp fusion; otherwise AFU clamp |
| 5 | PW0 | **Pointwise Conv2D** | K=1x1, S=1, C32->C32 | 1x32x48x48 | systolic direct/packed GEMM + requant |
| 6 | Residual0_Add | **Add** | L2 + L5 | 1x32x48x48 | AFU Add |
| 7 | DW1_Down | **DepthwiseConv2D** | K=3x3, S=2, Pad=1, C32 groups | 1x32x24x24 | RTL depthwise linebuffer stride-2 dispatch |
| 8 | PW1_Expand | **Pointwise Conv2D** | K=1x1, S=1, C32->C64 | 1x64x24x24 | systolic direct/packed GEMM, 2 OC tiles |
| 9 | PW1_ReLU6 | **ReLU6 / clamp** | element-wise | 1x64x24x24 | prefer requant clamp fusion; otherwise AFU clamp |
| 10 | PW2_Expand | **Pointwise Conv2D** | K=1x1, S=1, C64->C128 | 1x128x24x24 | systolic direct/packed GEMM, 4 OC tiles |
| 11 | DW2 | **DepthwiseConv2D** | K=3x3, S=1, Pad=1, C128 groups | 1x128x24x24 | RTL depthwise linebuffer, one invocation over 4 C32 groups |
| 12 | PW3_Project | **Pointwise Conv2D** | K=1x1, S=1, C128->C64 | 1x64x24x24 | systolic direct/packed GEMM |
| 13 | Residual1_Add | **Add** | L8 + L12 | 1x64x24x24 | AFU Add |
| 14 | Validate_Conv | **Standard Conv2D** | K=3x3, S=1, Pad=1, C64->C64 | 1x64x24x24 | existing systolic linebuffer/KGEN + requant |
| 15 | GlobalAvgPool | **AveragePool2D** | pool 24x24 -> 1x1 per channel | 1x64x1x1 | native AFU C32 reduce-sum/avg mode required |
| 16 | Classifier_PW | **Pointwise Conv2D** | K=1x1, C64->C32 | 1x32x1x1 | systolic direct GEMM |
| 17 | Output_DMA | **DMA_OUT** | write final INT8 vector | 1x32x1x1 | iDMA |

Coverage counts:

| Operator class | Layers | Count |
|---|---|---:|
| Standard Conv2D | `Stem_Conv`, `Validate_Conv` | 2 |
| DepthwiseConv2D | `DW0`, `DW1_Down`, `DW2` | 3 |
| Pointwise Conv2D | `PW0`, `PW1_Expand`, `PW2_Expand`, `PW3_Project`, `Classifier_PW` | 5 |
| ReLU6 / clamp | `Stem_ReLU6`, `DW0_ReLU6`, `PW1_ReLU6` | 3 |
| Add / residual | `Residual0_Add`, `Residual1_Add` | 2 |
| GlobalAvgPool | `GlobalAvgPool` | 1 |
| DMA_IN / DMA_OUT | input/weights load, `Output_DMA` | >= 1 |

BatchNorm is treated as a compile-time folded transform into Conv weights,
bias, and requant parameters. It does not require a runtime operator in the
first Micro-MobileNet graph.

---

## 2. Native Datapath Assessment

### 2.1 Reuse Existing RTL/Accelerator Paths

| Op | Reuse path | Required planner/SW work | Performance note |
|---|---|---|---|
| Standard Conv3x3 C3->C32 | Existing systolic linebuffer + KGEN + requant | Reuse Micro-YOLO linebuffer job descriptors for stem stride-2/pad-1 RGB input | Native path; no im2col scalar prepare. |
| Standard Conv3x3 C64->C64 | Existing systolic linebuffer + psum accumulation + requant | Emit OC32 x IC32 chunks and accumulate final chunk before requant | Reuses current dense Conv path; scheduler overhead should be tracked per tile. |
| Pointwise Conv1x1 C32->C32 | `NPU_OP_CONV2D1X1_C32_REQUANT` using direct `systolic_gemm32_requant()` | Already covered by `sw/test/pointwise_conv` | No linebuffer/window states and no im2col scratch. |
| Pointwise Conv1x1 multi-C32 | Same direct systolic GEMM32 path, tiled over IC/OC C32 groups | Add graph/planner wrapper for `C32->C64`, `C64->C128`, `C128->C64`, and `C64->C32` | Should be compute-dense; avoid CPU pack/copy by keeping tensors C32-blocked. |
| Depthwise Conv3x3 S1/S2 P1, `C % 32 == 0` | RTL depthwise linebuffer path with row-ring reuse and one-start multi-group loop | Already supports C32/C64/C96-style group iteration and S2 downsample dispatch; add C128 coverage | Current 48x48 S1 PMU scales near-linearly: C32 `32524`, C64 `61476`, C96 `90428`; S2 C32 is `14122`. |
| ReLU6 / clamp after Conv/Depthwise | Existing requant clamp fields | Fold activation min/max into producer requant whenever possible | Best path is zero extra pass over activation tensor. |
| Standalone element-wise Add | AFU Add | Reuse Micro-YOLO residual Add path on C32-blocked tensors | Native vector datapath; no scalar loop. |
| DMA / layout movement | iDMA 1D/2D/3D helpers | Keep host fixtures and graph tensors in C32-blocked layout | Avoid runtime C-layout conversion. |

### 2.2 Native RTL/AFU Updates Still Required

| Gap | Needed update | Why it matters |
|---|---|---|
| Depthwise non-3x3 variants | Generalize depthwise HAL/graph op from fixed `3x3/p1` to descriptor-configured `kernel_h/kernel_w/pad_h/pad_w` if future MobileNet variants need other kernels. | The current micro graph needs 3x3 S1/S2 P1 and is covered by native dispatch. |
| Depthwise non-multiple-of-32 tails | Host planner should split `main_C = floor(C/32)*32` plus a tail job only when `C % 32 != 0`; RTL tail support can be added later if tails become common. | Most middle MobileNet channels are multiples of 32 in this micro graph. Tail scalar code should not be in the normal path. |
| Depthwise throughput beyond 1 tap/cycle | Optional RTL mode issuing 3 C32 taps/cycle or a small unrolled tap pipeline. | Current path is correct and memory-efficient, but depthwise active cycles are still `output_pixels * groups * 9`. This is the largest remaining depthwise speedup knob. |
| GlobalAvgPool native reduce | Add an AFU C32 reduce-sum/avg mode: accumulate C32 lanes across spatial rows into wider accumulators, apply reciprocal/shift, clamp to INT8. | Classification heads need pool-to-1x1. A scalar implementation would be slow and should not be part of the target plan. |
| Standalone ReLU6 if not fused | Add or expose AFU clamp-only mode for C32-blocked tensors. | Requant fusion handles Conv/Depthwise producers. A native clamp pass is still useful after Add or other non-requant producers. |
| Pointwise multi-C32 one-start mode | Optional systolic controller mode to loop IC/OC C32 tiles internally from a host descriptor. | Current graph can tile in firmware. Native one-start tile looping would reduce register/config overhead for `C64/C128` pointwise-heavy blocks. |
| Layer fusion | Optional RTL/SW stripe pipeline for `Depthwise -> Pointwise` and `Conv -> ReLU6`. | Avoids spilling full intermediate tensors to L2/TCDM when local capacity allows. This is a later E2E bandwidth optimization, not a correctness blocker. |

---

## 3. Convolution Type Requirements

### 3.1 Standard Conv2D

Requirement: at least two standard Conv2D instances.

Planned instances:

1. `Stem_Conv`: `3x3/s2/p1`, `C3->C32`, validates RGB/small-IC linebuffer.
2. `Validate_Conv`: `3x3/s1/p1`, `C64->C64`, validates multi-C32 input and
   multi-OC tiling with the standard dense convolution path.

Implementation expectation:

- reuse the existing systolic linebuffer/KGEN path;
- split `C64->C64` as two input C32 chunks and two output C32 tiles;
- final chunk performs psum accumulate + requant.
- do not materialize im2col with scalar code.

### 3.2 Pointwise Conv2D `1x1`

Requirement: at least two pointwise conv instances.

Planned instances:

1. `PW0`: `C32->C32`, base MobileNetV1 depthwise-separable block.
2. `PW1_Expand`: `C32->C64`, validates OC tiling.
3. `PW2_Expand`: `C64->C128`, validates multi-input and multi-output C32 tiles.
4. `PW3_Project`: `C128->C64`, validates projection after expansion.
5. `Classifier_PW`: `C64->C32` on a `1x1` spatial tensor.

Implementation expectation:

- bypass linebuffer/window;
- use direct C32-blocked tensor reads;
- map to GEMM with `M = output_h * output_w`, `K = input_c`,
  `N = output_c`;
- use systolic requant for final INT8 output.
- for multi-C32 pointwise, tile over IC/OC C32 groups in the graph planner
  first; consider a controller one-start loop later if config overhead becomes
  visible in PMU.

### 3.3 DepthwiseConv2D

Requirement: at least two depthwise conv instances.

Planned instances:

1. `DW0`: `3x3/s1/p1`, `C32`, validates same-shape residual block.
2. `DW1_Down`: `3x3/s2/p1`, `C32`, validates stride-2 spatial reduction.
3. `DW2`: `3x3/s1/p1`, `C128`, validates four C32 channel groups.

Depthwise must not be lowered as dense `IC x OC` GEMM for the optimized path.
The dense systolic formulation wastes roughly `C` work because the logical
weight matrix is diagonal per channel. The target optimized datapath is:

```text
for each output pixel and C32 group:
  acc[0..31] = bias[0..31]
  for kh, kw:
    acc[lane] += ifm_tap[lane] * weight[kh][kw][lane]
  requant acc[0..31] -> output C32 vector
```

First optimized target: **1 tap C32 vector/cycle**.

Performance lower-bound estimates for 3x3:

| Layer | Output pixels | C32 groups | Active tap cycles |
|---|---:|---:|---:|
| `DW0` 48x48x32 | 2,304 | 1 | 20,736 |
| `DW1_Down` 24x24x32 | 576 | 1 | 5,184 |
| `DW2` 24x24x128 | 576 | 4 | 20,736 |

If depthwise dominates the E2E PMU, a later 3-tap/cycle path can reduce these
active tap cycles by about 3x at the cost of more multipliers and tap routing.

---

## 4. Implementation Phases

### Phase 1: Golden and Fixtures

Objective: define the graph and deterministic Python golden before RTL/SW
changes.

Tasks:

| Step | Task |
|---|---|
| 1a | Create Python golden for the topology above using integer INT8 arithmetic and exact requant rules |
| 1b | Generate deterministic input and weights in cocotb |
| 1c | Define C32-blocked activation layout and weight layout for standard, depthwise, and pointwise conv |
| 1d | Add operator coverage assertions: every MobileNet operator appears, each conv type appears at least twice |

Acceptance: Python golden can run standalone and emits all intermediate tensor
shapes listed in Section 1.

### Phase 2: Pointwise Conv1x1

Objective: add graph-level pointwise conv support using the existing systolic
GEMM path.

Current status: **partially implemented**. The first graph op,
`NPU_OP_CONV2D1X1_C32_REQUANT`, directly maps a C32-blocked
`1x32x48x48 -> 1x32x48x48` pointwise layer to systolic GEMM + requant:

```text
weight[32x32], input[Mx32] -> systolic_gemm32_requant() -> output[Mx32]
```

It deliberately bypasses linebuffer and does not allocate an im2col/scratch
prepare buffer. Unit coverage lives in `sw/test/pointwise_conv` and
`test_pointwise_conv.py`.

Tasks:

| Step | Task |
|---|---|
| 2a | Add graph op `NPU_OP_CONV2D1X1_C32_REQUANT` or a generic Conv descriptor op |
| 2b | Cover `C32->C32`, `C32->C64`, `C64->C128`, `C128->C64` |
| 2c | Use direct/packed Conv1x1 path without linebuffer fill/window states |
| 2d | Add unit tests before E2E graph integration |

Acceptance: all pointwise instances match Python golden byte-exactly. Current
accepted subset: `C32->C32`, `48x48`, C32-blocked input/output.

### Phase 3: Native Depthwise Path

Objective: run depthwise convolution through the RTL linebuffer-fed lane-wise
MAC path, not through dense GEMM or non-native lowering.

Current implementation note: the production-facing path directly uses the
optimized RTL mode for `3x3/s1/p1`,
C32-blocked input/output, and channel counts that are exact multiples of 32.
The weight layout is `group, kh, kw, lane`. The systolic controller loops over
all C32 groups inside one accelerator start, so `C=64/96/128/...` does not need
to be split into multiple graph/firmware invocations. If `C % 32 != 0`, the
host planner should emit one main multiple-of-32 job and a separate tail job.

Tasks:

| Step | Task |
|---|---|
| 3a | Add `systolic_depthwise3x3s1p1_c32_requant()` API |
| 3b | Define depthwise weight layout as `kh, kw, lane` per C32 group |
| 3c | Add graph op `NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT` for the C32 subset |
| 3d | Add standalone C32, C64, and C96 tests; add C128/odd-tail coverage later |
| 3e | Add native stride-2 depthwise dispatch for `DW1_Down` |

Acceptance: standalone C32/C64/C96 `48x48` depthwise matches golden
byte-exactly. Current measured standalone PMU after row-ring reuse is:

| Shape | Cycles | IFM req | OFM req |
|---|---:|---:|---:|
| `48x48x32` | 32,524 | 2,313 | 2,304 |
| `48x48x64` | 61,476 | 4,626 | 4,608 |
| `48x48x96` | 90,428 | 6,939 | 6,912 |
| `48x48x32, stride=2` | 14,122 | 2,313 | 576 |

This removes the previous per-tap reread pattern (`ifm_req=6825` for C32)
without increasing SRAM. The IFM requests now track raw C32 input vectors plus
small boundary overhead.

### Phase 4: Depthwise Performance Extensions

Objective: extend the existing depthwise RTL path to any additional MobileNet
depthwise cases while preserving row-ring reuse and avoiding extra SRAM.

Status: implemented for `3x3/s1/p1` and `3x3/s2/p1`, `C % 32 == 0`.

Tasks:

| Step | Task |
|---|---|
| 4a | Keep current row-ring/window fetch for C32 tap vectors |
| 4b | Add generic depthwise descriptors for non-3x3/p1 variants if needed |
| 4c | Keep 1 tap C32/cycle as the baseline 3x3 kernel path |
| 4d | Add PMU counters for depthwise active, fill, tap wait, and store cycles |
| 4e | Evaluate 3 taps/cycle RTL unroll if PMU shows depthwise dominates E2E |

Acceptance: depthwise optimized path matches golden and keeps IFM requests close
to raw input-vector traffic. Implemented optimization keeps only K resident rows in
the existing row-ring cache, slides rows across output positions, and lets the
existing background-fill FSM fetch the next row while the stream path emits
current window taps. It does not add SRAM; it adds only control state for
depthwise row-ring enablement and C32 group iteration.

### Phase 5: Micro-MobileNet E2E

Objective: run the complete topology through `Output_DMA`.

Tasks:

| Step | Task |
|---|---|
| 5a | Add `sw/test/micro_mobilenet` firmware entrypoint |
| 5b | Add `test_micro_mobilenet_e2e.py` cocotb test |
| 5c | Load host-generated Conv/Depthwise descriptors from L2, matching the Micro-YOLO runtime flow |
| 5d | Use native AFU/RTL paths for activation, Add, and GlobalAvgPool; no scalar operator implementation in the E2E hot path |
| 5e | Compare final `1x32x1x1` output byte-exactly |
| 5f | Print PMU per layer and aggregate by operator type |

Acceptance: final output and selected intermediate checkpoints match Python
golden with zero mismatch.

### Phase 6: Native Optimization

Optimization order:

1. Fuse ReLU6/clamp into Conv/Depthwise requant where possible.
2. Keep pointwise direct path free of linebuffer states.
3. Extend native depthwise dispatch beyond 3x3/p1 only if future layers need
   other kernel/pad variants.
4. Add native AFU GlobalAvgPool C32 reduce-sum/avg.
5. Add or expose AFU clamp-only pass for unfused ReLU6 after Add.
6. Add graph/planner pointwise multi-C32 tiling without layout conversion.
7. Consider 3 taps/cycle depthwise if PMU proves depthwise remains dominant.
8. Stripe-fuse `Depthwise -> Pointwise` to avoid spilling full intermediate
   tensors to L2 when TCDM budget allows.

---

## 5. Verification Strategy

1. **Golden Model**
   - Deterministic Python loops for every layer.
   - Exact INT8 clamp/requant model.
   - BatchNorm folded into weight/bias/qparams before runtime.

2. **Operator Unit Tests**
   - Pointwise: C32, C64, C128 input/output combinations.
   - Depthwise: C32, C64, C128 with stride 1 and stride 2.
   - ReLU6: quantized clamp boundaries.
   - GlobalAvgPool: deterministic C32 reduce/average output through native AFU.
   - Confirm no E2E operator falls back to scalar CPU loops except explicit
     test-only validation or tail cases.

3. **E2E RTL Simulation**
   - Cocotb loads firmware and L2 fixtures.
   - Firmware runs graph descriptors.
   - Cocotb checks final output and optional intermediate checkpoints.
   - PMU report groups cycles by standard conv, depthwise, pointwise, activation,
     add, pool, and DMA.

4. **Coverage Rules**
   - Each operator class in Section 1 must execute at least once.
   - Standard Conv2D, DepthwiseConv2D, and Pointwise Conv2D must each execute at
     least twice.
   - At least one depthwise layer must use stride 2.
   - At least one pointwise layer must use `OC > 32`.
   - At least one pointwise layer must use `IC > 32`.
   - At least one residual add must use a non-trivial producer chain.

---

## 6. Expected Bottlenecks

Pointwise Conv should be compute-dense and map well to the existing systolic
array. The main risks are scheduler overhead, OC tiling, and psum accumulation
for `IC > 32`.

Depthwise Conv remains the largest native RTL bottleneck. Dense GEMM lowering is
not an acceptable optimized path because it wastes most MACs. The implemented
target design is a lane-wise C32 depthwise MAC fed by the existing
linebuffer/window machinery. The next meaningful RTL upgrade is tap parallelism
or a tighter depthwise stream pipeline, not more SRAM.

GlobalAvgPool is not expected to dominate this micro graph, but it must still
use a native AFU vector-reduction path because MobileNet classification heads
depend on it and scalar spatial reductions do not scale.
