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

Current status: **planning with pointwise, depthwise, and GlobalAvgPool native
building blocks implemented**.
Existing Micro-YOLO infrastructure provides the stem Conv3x3 linebuffer path,
AFU/vector activation ops, AFU Add, iDMA movement, and the graph/test harness.
`NPU_OP_CONV2D1X1_C32_REQUANT` now covers the direct C32 `1x1` fast path for
`C32->C32`. `NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT` now covers a linebuffer-fed
lane-wise depthwise `3x3/s1/p1` fast path for any C32-blocked channel count,
including non-multiple-of-32 tails. `NPU_OP_DEPTHWISE3X3S2P1_C32_REQUANT`
extends the same native path to stride-2 downsample blocks. Multi-C32
pointwise tiling is implemented in the graph/HAL wrapper. New work is still
required for Micro-MobileNet E2E graph assembly.
`NPU_OP_GLOBAL_AVGPOOL_C32_REDUCE` covers native C32-blocked spatial averaging
through AFU mode 7. `NPU_OP_CLAMP_I8` covers standalone quantized ReLU6/clamp
through the AFU E8 LUT datapath when the clamp cannot be fused into producer
requant.

The target implementation should stay on native accelerator paths. Scalar CPU
loops are allowed only for golden-model generation, setup/validation code, or
very small tail handling that cannot yet be expressed by the native datapaths.
They are not acceptable as the planned implementation for convolution,
activation, residual, pooling, or layout movement.

Planning rule: **Phases 1-5 are for functional coverage and byte-exact
integration only. All performance optimization work is tracked in Phase 6.**
Earlier phases may use already-existing native accelerator paths, but they
should not add new performance-only RTL/SW changes.

---

## 1. Topology

The topology intentionally includes every MobileNet operator class while keeping
all activation tensors C32-blocked. Logical shapes below use NxCxHxW notation;
the physical implementation should keep the existing ROW32/C32 packed layout.

| # | Layer | Op | Config | Output (logical NxCxHxW) | Hardware Unit |
|---|---|---|---|---|---|
| 0 | Input | - | RGB INT8 | 1x3x96x96 | L2 |
| 1 | Stem_Conv | **Standard Conv2D** | K=3x3, S=2, Pad=1, C3->C32 | 1x32x48x48 | existing systolic linebuffer + requant |
| 2 | Stem_ReLU6 | **ReLU6 / clamp** | clamp INT8 to quantized ReLU6 range | 1x32x48x48 | existing requant clamp field or AFU clamp |
| 3 | DW0 | **DepthwiseConv2D** | K=3x3, S=1, Pad=1, C32 groups | 1x32x48x48 | RTL depthwise linebuffer |
| 4 | DW0_ReLU6 | **ReLU6 / clamp** | element-wise | 1x32x48x48 | existing requant clamp field or AFU clamp |
| 5 | PW0 | **Pointwise Conv2D** | K=1x1, S=1, C32->C32 | 1x32x48x48 | systolic direct/packed GEMM + requant |
| 6 | Residual0_Add | **Add** | L2 + L5 | 1x32x48x48 | AFU Add |
| 7 | DW1_Down | **DepthwiseConv2D** | K=3x3, S=2, Pad=1, C32 groups | 1x32x24x24 | RTL depthwise linebuffer stride-2 dispatch |
| 8 | PW1_Expand | **Pointwise Conv2D** | K=1x1, S=1, C32->C64 | 1x64x24x24 | systolic direct/packed GEMM, 2 OC tiles |
| 9 | PW1_ReLU6 | **ReLU6 / clamp** | element-wise | 1x64x24x24 | existing requant clamp field or AFU clamp |
| 10 | PW2_Expand | **Pointwise Conv2D** | K=1x1, S=1, C64->C128 | 1x128x24x24 | systolic direct/packed GEMM, 4 OC tiles |
| 11 | DW2 | **DepthwiseConv2D** | K=3x3, S=1, Pad=1, C128 groups | 1x128x24x24 | RTL depthwise linebuffer, one invocation over 4 C32 groups |
| 12 | PW3_Project | **Pointwise Conv2D** | K=1x1, S=1, C128->C64 | 1x64x24x24 | systolic direct/packed GEMM |
| 13 | Residual1_Add | **Add** | L8 + L12 | 1x64x24x24 | AFU Add |
| 14 | Validate_Conv | **Standard Conv2D** | K=3x3, S=1, Pad=1, C64->C64 | 1x64x24x24 | existing systolic linebuffer/KGEN + requant |
| 15 | GlobalAvgPool | **AveragePool2D** | pool 24x24 -> 1x1 per channel | 1x64x1x1 | native AFU C32 reduce-sum/avg mode |
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

| Op | Reuse path | Required planner/SW work | Notes |
|---|---|---|---|
| Standard Conv3x3 C3->C32 | Existing systolic linebuffer + KGEN + requant | Reuse Micro-YOLO linebuffer job descriptors for stem stride-2/pad-1 RGB input | Native path; no im2col scalar prepare. |
| Standard Conv3x3 C64->C64 | Existing systolic linebuffer + psum accumulation + requant | Emit OC32 x IC32 chunks and accumulate final chunk before requant | Reuses current dense Conv path. Scheduler tuning belongs to Phase 6. |
| Pointwise Conv1x1 C32->C32 | `NPU_OP_CONV2D1X1_C32_REQUANT` using direct `systolic_gemm32_requant()` | Already covered by `sw/test/pointwise_conv` | No linebuffer/window states and no im2col scratch. |
| Pointwise Conv1x1 multi-C32 | Same direct systolic GEMM32 path, tiled over IC/OC C32 groups through `systolic_pointwise1x1_c32_multi_requant()` | Implemented graph/HAL wrapper for C32-multiple tensors; current regression covers `C32->C32` and `C64->C128` | Compute-dense path with no im2col or linebuffer states. Inputs/outputs stay C32-blocked; weights are packed `OCG -> ICG -> 32x32`. |
| Depthwise Conv3x3 S1/S2 P1 | RTL depthwise linebuffer path with row-ring reuse, one-start multi-group loop, and final-group lane masking | Supports C32/C64/C96-style group iteration, odd tails such as C33/C65, and S2 downsample dispatch; add C128 coverage | Current 48x48 S1 PMU scales near-linearly for full groups; tail groups run as one masked C32 group with padding lanes forced to zero. |
| ReLU6 / clamp after Conv/Depthwise | Existing requant clamp fields or AFU clamp | Use whichever existing native path is already supported by the producer graph op | New fusion work belongs to Phase 6. |
| Standalone element-wise Add | AFU Add | Reuse Micro-YOLO residual Add path on C32-blocked tensors | Native vector datapath; no scalar loop. |
| DMA / layout movement | iDMA 1D/2D/3D helpers | Keep host fixtures and graph tensors in C32-blocked layout | Avoid runtime C-layout conversion. |

### 2.2 Native Support Status

This table separates functional support from performance-only work. Anything
marked Phase 6 must not block Phases 1-5.

| Item | Status / owner | Phase |
|---|---|---|
| Depthwise non-3x3 variants | Not required by the first micro graph. Keep fixed `3x3/p1` dispatch unless a future MobileNet variant needs descriptor-configured `kernel_h/kernel_w/pad_h/pad_w`. | Phase 6, only if needed |
| Depthwise non-multiple-of-32 tails | Implemented in the native depthwise controller path. The final C32 group uses an effective lane count of `C % 32`; invalid lanes are not fetched/MACed and are masked to zero after requant. | Phase 3 functional coverage |
| Depthwise throughput beyond 1 tap/cycle | Optional RTL mode issuing multiple C32 taps/cycle or a small unrolled tap pipeline. | Phase 6 optimization |
| GlobalAvgPool native reduce | Implemented as AFU mode 7 plus graph op `NPU_OP_GLOBAL_AVGPOOL_C32_REDUCE`. It accumulates C32 lanes across C32-blocked spatial rows into signed 32-bit accumulators, divides by `H*W`, clamps to INT8, and writes one C32 vector per group. | Phase 4 functional wiring |
| Standalone ReLU6 if not fused | Implemented as graph op `NPU_OP_CLAMP_I8` and wrapper `npu_clamp_i8()`, using the existing AFU E8 LUT datapath with a generated 256-entry clamp table. | Phase 4 functional wiring |
| Pointwise RTL single-start mode | Optional systolic controller mode to loop IC/OC C32 tiles internally from a host descriptor. Current graph/HAL path already dispatches one pointwise graph op and uses a reusable psum scratch buffer, but it still issues multiple systolic starts internally. | Phase 6 optimization |
| Layer fusion | Optional RTL/SW stripe pipeline for `Depthwise -> Pointwise` and `Conv -> ReLU6`. | Phase 6 optimization |

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
  first. Controller one-start looping is a Phase 6 optimization.

### 3.3 DepthwiseConv2D

Requirement: at least two depthwise conv instances.

Planned instances:

1. `DW0`: `3x3/s1/p1`, `C32`, validates same-shape residual block.
2. `DW1_Down`: `3x3/s2/p1`, `C32`, validates stride-2 spatial reduction.
3. `DW2`: `3x3/s1/p1`, `C128`, validates four C32 channel groups.

Depthwise must not be lowered as dense `IC x OC` GEMM for the native path.
The dense systolic formulation wastes roughly `C` work because the logical
weight matrix is diagonal per channel. The functional native datapath is:

```text
for each output pixel and C32 group:
  acc[0..31] = bias[0..31]
  for kh, kw:
    acc[lane] += ifm_tap[lane] * weight[kh][kw][lane]
  requant acc[0..31] -> output C32 vector
```

Current native datapath target: **1 tap C32 vector/cycle**. Any tap-parallel
upgrade belongs to Phase 6.

Performance lower-bound estimates for 3x3:

| Layer | Output pixels | C32 groups | Active tap cycles |
|---|---:|---:|---:|
| `DW0` 48x48x32 | 2,304 | 1 | 20,736 |
| `DW1_Down` 24x24x32 | 576 | 1 | 5,184 |
| `DW2` 24x24x128 | 576 | 4 | 20,736 |

Tap-parallel depthwise upgrades are intentionally deferred to Phase 6.

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

Current status: **implemented for C32-multiple channel counts**. The graph op,
`NPU_OP_CONV2D1X1_C32_REQUANT`, maps C32-blocked pointwise layers to the native
systolic GEMM path:

```text
IC_groups=1:
  weight[32x32], input[Mx32] -> systolic_gemm32_requant() -> output[Mx32]

IC_groups>1:
  first ICG  -> systolic_gemm32() into int32 psum scratch
  middle ICG -> systolic_gemm32_accumulate() into psum scratch
  final ICG  -> systolic_gemm32_accumulate_requant() into i8 output
```

It deliberately bypasses linebuffer and does not allocate an im2col/scratch
prepare buffer. The only scratch allocation is one reusable `rows x 32 x int32`
psum tile for multi-IC-group accumulation. Weight layout is OCG-major,
ICG-major, with each tile stored as the existing native GEMM32 `32x32`
K-major/N-minor matrix.

Unit coverage lives in `sw/test/pointwise_conv` and `test_pointwise_conv.py`.

Tasks:

| Step | Task |
|---|---|
| 2a | Add graph op `NPU_OP_CONV2D1X1_C32_REQUANT` or a generic Conv descriptor op |
| 2b | Cover `C32->C32`, `C32->C64`, `C64->C128`, `C128->C64` |
| 2c | Use direct/packed Conv1x1 path without linebuffer fill/window states |
| 2d | Add unit tests before E2E graph integration |

Acceptance: all pointwise instances match Python golden byte-exactly. Current
accepted subset:

- `C32->C32`, `48x48`, C32-blocked input/output.
- `C64->C128`, `16x16`, C32-blocked input/output, multi-IC and multi-OC groups.

### Phase 3: Native Depthwise Functional Path

Objective: run depthwise convolution through the RTL linebuffer-fed lane-wise
MAC path, not through dense GEMM or non-native lowering.

Current implementation note: the production-facing path directly uses the
native RTL mode for `3x3/s1/p1`,
C32-blocked input/output, including final groups where `C % 32 != 0`. The
weight layout is `group, kh, kw, lane`, with padding lanes set to zero by host
fixtures/descriptors. The systolic controller loops over all C32 groups inside
one accelerator start, so `C=33/64/65/96/128/...` does not need to be split
into multiple graph/firmware invocations.

Tasks:

| Step | Task |
|---|---|
| 3a | Add `systolic_depthwise3x3s1p1_c32_requant()` API |
| 3b | Define depthwise weight layout as `kh, kw, lane` per C32 group |
| 3c | Add graph op `NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT` for the C32 subset |
| 3d | Add standalone C32, C64, C96, C128, C33-tail, and C65-tail tests |
| 3e | Add native stride-2 depthwise dispatch for `DW1_Down` |
| 3f | Add native final-group tail lane masking for non-multiple-of-32 channel counts |

Acceptance: standalone C32/C64/C96 and odd-tail `48x48` depthwise matches golden
byte-exactly. Current measured standalone PMU after row-ring reuse is:

| Shape | Cycles | IFM req | OFM req |
|---|---:|---:|---:|
| `48x48x32` | 32,524 | 2,313 | 2,304 |
| `48x48x64` | 61,476 | 4,626 | 4,608 |
| `48x48x96` | 90,428 | 6,939 | 6,912 |
| `48x48x32, stride=2` | 14,122 | 2,313 | 576 |

These PMU numbers are recorded as the current functional baseline. Follow-up
performance tuning belongs to Phase 6.

### Phase 4: Remaining Native Operator Wiring

Objective: wire the remaining non-pointwise/non-depthwise operators needed by
the first Micro-MobileNet graph, using already-available native accelerator
paths and avoiding scalar operator implementations.

Status: standard Conv2D, Add, GlobalAvgPool, and standalone clamp already have
native building blocks in the repository. Phase 4 is about graph integration
and focused unit coverage, not new performance work.

Tasks:

| Step | Task |
|---|---|
| 4a | Wire `Stem_Conv` and `Validate_Conv` to the existing standard Conv2D linebuffer/KGEN path |
| 4b | Wire AFU Add for `Residual0_Add` and `Residual1_Add` |
| 4c | Wire `NPU_OP_GLOBAL_AVGPOOL_C32_REDUCE` for the classification head |
| 4d | Wire `NPU_OP_CLAMP_I8` only where ReLU6/clamp cannot already be represented by existing requant clamp fields |
| 4e | Add operator-level unit tests for each wired native path before E2E graph assembly |

Acceptance: every non-convolution MobileNet operator needed by Section 1 has a
native graph dispatch and byte-exact unit coverage. No Phase 4 task may add
tap-parallelism, layer fusion, new buffering, or other performance-only RTL.

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

Objective: improve PMU results after the byte-exact E2E graph is stable. Phase
6 is the only phase that should add performance-only RTL/SW changes.

Entry criteria:

- Phase 5 E2E passes byte-exactly.
- PMU is printed per layer and grouped by operator type.
- No hot-path scalar operator remains in the E2E graph.

Optimization backlog:

| Priority | Item | Scope | Expected benefit / decision rule |
|---|---|---|---|
| 6a | PMU attribution baseline | Add or refine per-layer counters for standard Conv, depthwise, pointwise, activation, Add, GlobalAvgPool, DMA, and config overhead. | Required before changing RTL. Use this to choose the next optimization instead of guessing. |
| 6b | ReLU6/clamp fusion | Fold clamp min/max into Conv/Depthwise/Pointwise requant when the producer supports it; keep AFU clamp only for non-fused producers. | Removes full extra activation tensor passes. |
| 6c | Pointwise single-start IC/OC loop | Optional systolic controller mode to consume a pointwise descriptor and loop over IC/OC C32 groups internally. | Reduces register/config/start overhead for pointwise-heavy `C64/C128` layers. Prioritize only if PMU shows config or psum traffic is visible. |
| 6d | Depthwise stream tightening | Reduce depthwise stream overhead around fill/window/store states without changing the functional 1 tap/cycle contract. | Use if PMU shows non-tap overhead is significant. |
| 6e | Depthwise tap parallelism | Add 2- or 3-tap/cycle issue for C32 tap vectors, or a small unrolled tap pipeline. | Largest depthwise speedup knob; pursue only if depthwise active cycles dominate E2E after 6d. |
| 6f | Standard Conv linebuffer scheduling | Revisit row/window scheduler, coalesced reads, and drain overlap for the `Validate_Conv` dense C64 path. | Use if standard Conv linebuffer states dominate PMU. |
| 6g | Layer fusion | Stripe-fuse `Depthwise -> Pointwise` and optionally `Conv -> ReLU6` when TCDM budget allows. | Avoids spilling full intermediate tensors to L2/TCDM. Higher complexity; do after operator PMU is stable. |
| 6h | Descriptor preloading / shadow config | Preload next layer descriptors while the current layer drains. | Reduces inter-layer bubbles if PMU shows config/start latency. |
| 6i | DMA/layout tuning | Use C32-blocked host fixtures and aligned DMA transfers throughout; add 2D/3D DMA only where it avoids CPU copies. | Prevents layout conversion from masking accelerator gains. |
| 6j | Future depthwise kernel variants | Generalize depthwise descriptors beyond `3x3/p1` only if a target MobileNet variant actually requires it. | Avoids RTL churn for kernels not used by the graph. |

Acceptance: every optimization has before/after PMU numbers and preserves the
Phase 5 byte-exact E2E result.

---

## 5. Verification Strategy

1. **Golden Model**
   - Deterministic Python loops for every layer.
   - Exact INT8 clamp/requant model.
   - BatchNorm folded into weight/bias/qparams before runtime.

2. **Operator Unit Tests**
   - Pointwise: C32, C64, C128 input/output combinations.
   - Depthwise: C32, C33, C64, C65, C128 with stride 1 and stride 2.
   - ReLU6: quantized clamp boundaries.
   - GlobalAvgPool: deterministic C32 reduce/average output through native AFU.
   - Confirm no E2E operator falls back to scalar CPU loops except explicit
     test-only validation.

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

## 6. Phase 6 Performance Notes

These notes are inputs to Phase 6 only. They should not block functional
operator coverage or Phase 5 E2E correctness.

Pointwise Conv should be compute-dense and map well to the existing systolic
array. The main risks are scheduler overhead, OC tiling, and psum accumulation
for `IC > 32`.

Depthwise Conv remains the largest native RTL bottleneck. Dense GEMM lowering is
not an acceptable target path because it wastes most MACs. The implemented
native design is a lane-wise C32 depthwise MAC fed by the existing
linebuffer/window machinery. Phase 6 should use PMU to choose between tap
parallelism and a tighter depthwise stream pipeline.

GlobalAvgPool is not expected to dominate this micro graph, but it must still
use a native AFU vector-reduction path because MobileNet classification heads
depend on it and scalar spatial reductions do not scale.
