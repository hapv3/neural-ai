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

Current status: **planning with first pointwise fast path implemented**.
Existing Micro-YOLO infrastructure provides the stem Conv3x3 linebuffer path,
AFU/Spatz activation ops, AFU Add, iDMA movement, and the graph/test harness.
`NPU_OP_CONV2D1X1_C32_REQUANT` now covers the direct C32 `1x1` fast path for
`C32->C32`. New work is still required for multi-C32 pointwise tiling and
optimized depthwise conv.

---

## 1. Topology

The topology intentionally includes every MobileNet operator class while keeping
all activation tensors C32-blocked. Logical shapes below use NxCxHxW notation;
the physical implementation should keep the existing ROW32/C32 packed layout.

| # | Layer | Op | Config | Output (logical NxCxHxW) | Hardware Unit |
|---|---|---|---|---|---|
| 0 | Input | - | RGB INT8 | 1x3x96x96 | L2 |
| 1 | Stem_Conv | **Standard Conv2D** | K=3x3, S=2, Pad=1, C3->C32 | 1x32x48x48 | existing systolic linebuffer + requant |
| 2 | Stem_ReLU6 | **ReLU6 / clamp** | clamp INT8 to quantized ReLU6 range | 1x32x48x48 | AFU/Spatz clamp path |
| 3 | DW0 | **DepthwiseConv2D** | K=3x3, S=1, Pad=1, C32 groups | 1x32x48x48 | new depthwise path |
| 4 | DW0_ReLU6 | **ReLU6 / clamp** | element-wise | 1x32x48x48 | AFU/Spatz clamp path |
| 5 | PW0 | **Pointwise Conv2D** | K=1x1, S=1, C32->C32 | 1x32x48x48 | systolic direct/packed GEMM + requant |
| 6 | Residual0_Add | **Add** | L2 + L5 | 1x32x48x48 | AFU Add |
| 7 | DW1_Down | **DepthwiseConv2D** | K=3x3, S=2, Pad=1, C32 groups | 1x32x24x24 | new depthwise path |
| 8 | PW1_Expand | **Pointwise Conv2D** | K=1x1, S=1, C32->C64 | 1x64x24x24 | systolic direct/packed GEMM, 2 OC tiles |
| 9 | PW1_ReLU6 | **ReLU6 / clamp** | element-wise | 1x64x24x24 | AFU/Spatz clamp path |
| 10 | PW2_Expand | **Pointwise Conv2D** | K=1x1, S=1, C64->C128 | 1x128x24x24 | systolic direct/packed GEMM, 4 OC tiles |
| 11 | DW2 | **DepthwiseConv2D** | K=3x3, S=1, Pad=1, C128 groups | 1x128x24x24 | new depthwise path over 4 C32 groups |
| 12 | PW3_Project | **Pointwise Conv2D** | K=1x1, S=1, C128->C64 | 1x64x24x24 | systolic direct/packed GEMM |
| 13 | Residual1_Add | **Add** | L8 + L12 | 1x64x24x24 | AFU Add |
| 14 | Validate_Conv | **Standard Conv2D** | K=3x3, S=1, Pad=1, C64->C64 | 1x64x24x24 | existing systolic linebuffer/KGEN + requant |
| 15 | GlobalAvgPool | **AveragePool2D** | pool 24x24 -> 1x1 per channel | 1x64x1x1 | Spatz/scalar first, optional AFU reduce later |
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

## 2. Operator Feasibility Assessment

### 2.1 Ready or Mostly Ready

| Op | Existing path | Notes |
|---|---|---|
| Standard Conv3x3 C3/C32 | `npu_conv2d_packed_run_oc32_linebuf_requant()` | Already exercised by Micro-YOLO stem/C32 convs. Need OC64 tiling in graph usage for `Validate_Conv`. |
| Pointwise Conv1x1 C32->C32 | `NPU_OP_CONV2D1X1_C32_REQUANT` | Implemented as direct `systolic_gemm32_requant()` over C32-blocked input; no linebuffer or im2col/scratch prepare. |
| Pointwise Conv1x1 multi-C32 | future OC/IC tiled wrapper around the direct C32 fast path | Needed for `C32->C64`, `C64->C128`, and projection layers. |
| ReLU / clamp | `spatz_vec_relu_i8()` or simple clamp kernel | ReLU6 should use quantized clamp min/max, not floating-point runtime math. |
| Add | `spatz_add_i8()` / AFU Add | Already used by Micro-YOLO residual. |
| DMA | iDMA helpers | Existing L2/TCDM movement path. |

### 2.2 New or Incomplete

| Op | Difficulty | Recommended first path | Optimized path |
|---|---|---|---|
| DepthwiseConv2D C32 | High | Spatz/scalar correctness path | linebuffer-fed depthwise MAC/requant block |
| GlobalAvgPool | Medium | Spatz/scalar reduction over C32 groups | vector reduction or AFU reduction mode if it becomes hot |
| ReLU6 exact quantization | Low | clamp op using precomputed quantized `[0, 6]` range | fuse into requant when producer is Conv/depthwise |

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
- use direct C32 stream or packed prepare iDMA 2D;
- map to GEMM with `M = output_h * output_w`, `K = input_c`,
  `N = output_c`;
- use systolic requant for final INT8 output.

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

### Phase 3: Depthwise Correctness Path

Objective: unblock graph integration with a simple depthwise implementation.

Tasks:

| Step | Task |
|---|---|
| 3a | Add `npu_depthwise3x3_c32_requant()` API |
| 3b | Define depthwise weight layout as `C32_group, kh, kw, lane` |
| 3c | Implement scalar/Spatz correctness path for `s1/p1` and `s2/p1` |
| 3d | Add standalone tests for C32, C64, C128 |

Acceptance: `DW0`, `DW1_Down`, and `DW2` match golden byte-exactly, even if
performance is not final.

### Phase 4: Optimized Depthwise Linebuffer Path

Objective: replace the correctness path with a linebuffer-fed depthwise MAC.

Tasks:

| Step | Task |
|---|---|
| 4a | Reuse current row-ring/window fetch for C32 tap vectors |
| 4b | Add depthwise MAC/requant block or a dedicated RTL mode |
| 4c | Support 1 tap C32/cycle for 3x3 kernels |
| 4d | Add PMU counters for depthwise active, fill, tap wait, and store cycles |
| 4e | Keep scalar/Spatz fallback for debug and unsupported shapes |

Acceptance: depthwise optimized path matches golden and reduces PMU cycles vs
the correctness path.

### Phase 5: Micro-MobileNet E2E

Objective: run the complete topology through `Output_DMA`.

Tasks:

| Step | Task |
|---|---|
| 5a | Add `sw/test/micro_mobilenet` firmware entrypoint |
| 5b | Add `test_micro_mobilenet_e2e.py` cocotb test |
| 5c | Load host-generated Conv/Depthwise descriptors from L2, matching the Micro-YOLO runtime flow |
| 5d | Compare final `1x32x1x1` output byte-exactly |
| 5e | Print PMU per layer and aggregate by operator type |

Acceptance: final output and selected intermediate checkpoints match Python
golden with zero mismatch.

### Phase 6: Optimization

Optimization order:

1. Fuse ReLU6/clamp into Conv/Depthwise requant where possible.
2. Keep pointwise direct path free of linebuffer states.
3. Move depthwise from scalar/Spatz correctness path to 1 tap/cycle RTL path.
4. Reuse linebuffer/window state across adjacent depthwise spatial tiles.
5. Consider 3 taps/cycle depthwise if PMU proves depthwise remains dominant.
6. Stripe-fuse `Depthwise -> Pointwise` to avoid spilling full intermediate
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
   - GlobalAvgPool: deterministic reduced output.

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

Depthwise Conv is the main new bottleneck. Dense GEMM lowering is not an
acceptable optimized path because it wastes most MACs. The target design is a
lane-wise C32 depthwise MAC fed by the existing linebuffer/window machinery.

GlobalAvgPool is not expected to dominate this micro graph, but it should be
included because MobileNet classification heads depend on it.
