# Micro-YOLOv8 Integration Test Plan

## Objective

Prove that the NPU (Systolic Array + Spatz + iDMA) can continuously run a neural network graph with the characteristic structure of YOLOv8, not just isolated operators.

Input: **96x96 RGB INT8**

First end-to-end target: run the backbone/neck/head Conv path and write the
**raw head tensor as INT8**. DFL/Softmax/decode remain in the graph roadmap but
are not required for the first raw-head E2E pass.

---

## 1. Topology

| # | Layer | Op | Config | Output (logical NxCxHxW) | Hardware Unit |
|---|---|---|---|---|---|
| 0 | Input | - | - | 1x3x96x96 | L2 |
| 1 | Conv_Stem | **Conv2D** | K=3x3, S=2, Pad=1, OC=32 | 1x32x48x48 | Spatz/iDMA pack (IC=3) + Systolic |
| 2 | SiLU_Sig | **Logistic** | element-wise sigmoid | 1x32x48x48 | AFU LUT or Spatz LUT fallback |
| 3 | SiLU_Mul | **Mul** | L1 * L2 | 1x32x48x48 | Spatz (vmul + requant) |
| 4 | C2f_Route | **Identity/Slice** | route L3 as residual branch | 1x32x48x48 | Firmware metadata |
| 5 | C2f_Conv | **Conv2D** | K=3x3, S=1, Pad=1, IC=32, OC=32 | 1x32x48x48 | iDMA/Spatz pack + Systolic |
| 6 | C2f_Add | **Add** | L4 + L5, same shape | 1x32x48x48 | Spatz (vadd + clamp/requant) |
| 7 | Conv_Down | **Conv2D** | K=3x3, S=2, Pad=1, IC=32, OC=32 | 1x32x24x24 | iDMA/Spatz pack + Systolic |
| 8 | SPPF_Pool | **MaxPool2D** | K=5x5, S=1, Pad=2 | 1x32x24x24 | Spatz |
| 9 | Upsample | **Upsample** | Nearest 2x | 1x32x48x48 | Spatz / iDMA 2D |
| 10 | Concat | **Concat** | L9 + L3 along C-axis | 1x64x48x48 | iDMA/Spatz compact copy |
| 11 | Head_Conv | **Conv2D** | K=1x1, S=1, IC=64, OC=32 | 1x32x48x48 | Fast Path (IC>=32) + Systolic |
| 12 | Raw_Head_Out | **DMA_OUT** | write raw head INT8 to L2 | 1x32x48x48 | iDMA |
| 13 | Future_Head_Split | **StridedSlice** | split Box[0:16] and Class[16:32] | 1x16x48x48 x2 | Future postprocess |
| 14 | Future_Transpose | **Transpose** | [N,C,H,W] -> [N,H,W,C] | 1x48x48x16 | Future postprocess |
| 15 | Future_Reshape | **Reshape** | flatten -> [1, 2304, 16] | 1x2304x16 | Future postprocess |
| 16 | Future_DFL_Soft | **Softmax** | on Box branch (dim=-1) | 1x2304x16 | Future Spatz/AFU |
| 17 | Future_Cls_Sig | **Logistic** | on Class branch | 1x2304x16 | Future AFU/Spatz |

Internal implementation should keep tensors in the existing HWC/NHWC packed
layout used by `conv2d_packed`; the NxCxHxW notation above is only the logical
model shape.

First E2E unique operators: Conv2D, Logistic, Mul, Identity/Slice, Add,
MaxPool2D, Upsample, Concat, Requant, DMA_OUT. Transpose, Reshape, DFL Softmax,
and class Sigmoid are retained as future postprocess operators after raw-head
INT8 passes.

---

## 2. Operator Feasibility Assessment

### 2.1 ALREADY AVAILABLE (Ready)

| Op | API | File | Notes |
|---|---|---|---|
| Conv2D | `npu_conv2d_packed_run_oc32_requant()` | `sw/lib/conv2d_packed.c` | Supports the first E2E topology because every Conv output tile is OC=32. |
| Copy | `spatz_vec_copy_i8()` | `sw/lib/spatz_ops.h` | Used for Concat/Slice |
| ReLU | `spatz_vec_relu_i8()` | `sw/lib/spatz_ops.h` | Backup if SiLU is dropped |
| Requant | `spatz_requant_i32_to_i8()` | `sw/lib/spatz_ops.h` | INT32->INT8 with multiplier+shift |
| iDMA 1D/2D/3D | `idma_L2ToL1_2d()`, `idma_L2ToL1_3d()` | `sw/lib/idma_mm_utils.h` | Used for Transpose, Concat |

### 2.2 NEEDS NEW IMPLEMENTATION (8 operators)

| Op | Difficulty | Description | RVV instruction |
|---|---|---|---|
| **Add** | Easy | Add 2 saturated INT8 vectors | `vadd.vv` + `vmax`/`vmin` clamp |
| **Mul** | Easy | Multiply 2 INT8 vectors, needs requant | `vmul.vv` -> INT16 -> shift -> clamp |
| **Logistic** | Medium | Sigmoid using 256-byte LUT on TCDM | Prefer AFU LUT; Spatz indexed-load fallback is acceptable |
| **MaxPool2D** | Medium | 5x5 sliding window finding max | `vmax.vv` looped multiple times |
| **Upsample** | Medium | Nearest 2x: duplicate pixels | `vrgather` or iDMA 2D copy |
| **Softmax / DFL** | **Hard** | findmax -> sub -> exp(LUT) -> sum -> div | Future postprocess; not required for first raw-head E2E |
| **Concat** | Easy | Copy 2 memory regions into 1 contiguous | `spatz_vec_copy_i8` or iDMA |
| **Transpose** | Medium | NCHW->NHWC: iDMA 2D stride copy | `idma_L2ToL1_2d()` already available |

### 2.3 ZERO-COST (only metadata changes, no code needed)

| Op | Method |
|---|---|
| **Identity/Slice** | Zero-cost only when the downstream operator can consume the same packed layout/stride. Channel compaction is a real copy. |
| **Reshape** | Firmware changes the `shape[]` field in the tensor struct. No data copy. |

---

## 3. Key Architectural Issues

### 3.1 OC Policy: First E2E Uses OC=32 Everywhere

The `npu_conv2d_packed_run_oc32()` function currently only handles a **fixed OC = 32**.
The first Micro-YOLOv8 integration topology therefore sets every Conv2D output
channel count to exactly `32`. This avoids `OC<32` masked-store work and
`OC>32` multi-tile work while still exercising realistic `IC`, `K`, stride,
padding, concat, and activation patterns.

Later, if OC=64 or larger is needed, an outer OC loop will be added:

```
for (oc_tile = 0; oc_tile < OC; oc_tile += 32) {
    // load weight slice [oc_tile : oc_tile+32]
    // run npu_conv2d_packed_run_oc32() with this weight slice
    // write result to output[..., oc_tile : oc_tile+32]
}
```

**In this test:** all Conv2D layers produce exactly `OC=32`, so multi-OC tiling
and partial-OC masking are both deferred.

### 3.2 Add Shape Policy

Layer 6 (`C2f_Add`) must add two tensors with identical shape. The first E2E
topology therefore uses:

- L4 residual route: `1x32x48x48`
- L5 C2f Conv output: `1x32x48x48`

No zero-padding or broadcast semantics are required for the first raw-head E2E
test.

### 3.3 Sigmoid/Softmax on INT8 - Quantization concern

Sigmoid and Softmax are non-linear functions operating on the real number domain [0, 1].
When the input is quantized INT8 (scale + zero_point), we need:

1. **Sigmoid LUT:** Pre-calculate 256 `sigmoid(dequant(i))` values then re-quantize
   back to INT8. The whole LUT is only 256 bytes.
2. **Softmax / DFL:** More complex because it needs `exp()` and `sum()`. Must be done on
   higher precision (INT16 or INT32) then requantized back to INT8 at the last step.
   **This remains in the roadmap but is implemented after raw-head INT8 E2E passes.**

### 3.4 Memory budget for 96x96

| Tensor | Size | Bytes |
|---|---|---|
| Input 96x96x3 | | 27,648 |
| L1 Conv_Stem 48x48x32 | | 73,728 |
| L7 Conv_Down 24x24x32 | | 18,432 |
| L9 Upsample 48x48x32 | | 73,728 |
| L10 Concat 48x48x64 | | 147,456 |
| L11 Head_Conv 48x48x32 | | 73,728 |
| **Total if naively concurrent** | | ~415 KB |
| **Shared Data TCDM capacity** | | 512 KB logical; O-TCDM accumulator window is smaller |

The naive tensor total is close to the full Shared Data TCDM and does not
include packed `M x 32` tiles, weights, or INT32 psum scratch. Therefore the
test must use the **L2-Centric + Tiling** strategy:
- Store all intermediate tensors in L2.
- Only pull small tiles into L1 to compute, then push results back to L2.

---

## 4. Implementation Phase Breakdown

The implementation is staged so each milestone has one dominant failure mode.
Operator unit tests prove the kernels, then graph integration starts from a
minimal Conv-only graph and adds data movement, layout, and scheduler lifetime
one step at a time.

### Phase 1: Operator Library Unit Tests

Objective: implement each missing non-Conv operator independently before it is
allowed into the full graph scheduler.

| Step | Task | File |
|---|---|---|
| 1a | `spatz_add_i8()` saturated add | `sw/lib/spatz_ops.c`, `sw/lib/spatz_ops.h` |
| 1b | `spatz_mul_i8()` INT8 multiply with requant/clamp | `sw/lib/spatz_ops.c`, `sw/lib/spatz_ops.h` |
| 1c | `npu_logistic_i8()` using AFU LUT, with optional Spatz fallback | `sw/lib/hal_afu.h`, `sw/lib/spatz_ops.c` |
| 1d | `spatz_maxpool2d_i8()` for K=5, S=1, Pad=2 | `sw/lib/spatz_ops.c`, `sw/lib/spatz_ops.h` |
| 1e | `spatz_upsample_nearest_i8()` 2x nearest | `sw/lib/spatz_ops.c`, `sw/lib/spatz_ops.h` |
| 1f | `spatz_concat_c32_i8()` or equivalent C32-blocked concat helper | `sw/lib/spatz_ops.c`, `sw/lib/spatz_ops.h` |
| 1g | `spatz_softmax_i8()` / DFL postprocess | Future phase after raw-head E2E |
| 1h | Unit tests for all implemented operators | `sw/test/spatz_ops/main.c`, `test_spatz_operator_library.py` |

Acceptance: each operator has a deterministic Python golden and byte-exact RTL
cluster test coverage. Softmax/DFL remains disabled until after raw-head INT8
passes.

### Phase 2: Tensor Layout and Graph Runtime Contract

Objective: define layout and lifetime rules before introducing the full
96x96 topology. This prevents graph bugs from being hidden inside ad hoc copies.

Required decisions:

- Activation tensors with `C <= 32`: store as one C32 block with zero-padded tail.
- Activation tensors with `C > 32`: store as C32-blocked blocks, not compact NCHW.
- Concat along C: produce C32-blocked output blocks directly.
- Conv consumers: use the C32-blocked descriptor path for `IC=64` and larger.
- L2 is the owner of full intermediate tensors; TCDM only holds working tiles.

| Step | Task | File |
|---|---|---|
| 2a | Define `tensor_t` with addr, H/W/C, layout, scale, zero point | `sw/lib/npu_tensor.h` |
| 2b | Define C32-blocked tensor helpers and address math | `sw/lib/npu_tensor.h` or `sw/lib/npu_graph.h` |
| 2c | Add graph scratch allocator for TCDM tile buffers | `sw/test/micro_yolo/main.c` or `sw/lib/npu_graph.c` |
| 2d | Document qparam propagation for Requant/SiLU/Add/Mul | `docs/micro_yolov8_integration_test_plan.md` |

Acceptance: the minimal graph in Phase 3a can run using the same tensor
descriptors and layout rules that the full 96x96 graph will use.

### Phase 3: Incremental Graph Integration

Objective: grow the graph one operator at a time, with a byte-exact checkpoint
after each addition.

| Step | Graph Added | Expected Output |
|---|---|---|
| 3a | Minimal graph: `32x32x3 -> Conv3x3 -> Requant/Clamp -> Conv1x1 -> Requant/Clamp` | `32x32x32` INT8 |
| 3b | Conv_Stem only for 96x96 input | L1 `48x48x32` INT8 |
| 3c | Conv_Stem + SiLU | L3 `48x48x32` INT8 |
| 3d | Add C2f_Conv | L5 `48x48x32` INT8 |
| 3e | Add residual Add | L6 `48x48x32` INT8 |
| 3f | Add Conv_Down | L7 `24x24x32` INT8 |
| 3g | Add SPPF MaxPool | L8 `24x24x32` INT8 |
| 3h | Add Upsample | L9 `48x48x32` INT8 |
| 3i | Add Concat to C64 C32-blocked layout | L10 `48x48x64` INT8 |
| 3j | Add Head_Conv IC64 OC32 | Raw head `48x48x32` INT8 |

Acceptance: every step writes its checkpoint tensor to L2 and compares against
the Python golden before the next operator is enabled. Step 3a also creates the
`sw/test/micro_yolo/` firmware, Makefile, linker/start files, and makes the
existing `test_micro_yolo_e2e.py` style test pass.

### Phase 4: Full 96x96 Raw-Head E2E

Objective: run the complete raw-head topology through `Raw_Head_Out`.
DFL/Softmax/decode table entries may exist in metadata but must remain disabled.

| Step | Task |
|---|---|
| 4a | Generate deterministic Python golden for the full raw-head topology |
| 4b | Export input and weights as fixed `.bin` fixtures or generate them in cocotb |
| 4c | Run firmware scheduler over L2-centric tiled tensors |
| 4d | Compare raw head INT8 output byte-exactly |
| 4e | Collect PMU counters per layer for performance triage |

Acceptance: raw head `1x32x48x48` INT8 matches golden with 0 byte mismatch.

### Phase 5: Future Postprocess

Objective: extend beyond raw-head output after the Conv/activation path is stable.

| Step | Task |
|---|---|
| 5a | Head split into Box/Class branches |
| 5b | Transpose/reshape to flattened head layout |
| 5c | DFL Softmax implementation and unit test |
| 5d | Class sigmoid |
| 5e | Decode/NMS roadmap |

---

## 5. Verification strategy

1. **Golden Model (Python/PyTorch):**
   - Write a Python script using PyTorch defining the Micro-YOLOv8 network.
   - Assign random weights and quantize everything to INT8.
   - Export: `input_image.bin`, `weights_layer_N.bin`, `golden_raw_head_int8.bin`.
2. **RTL Simulation:**
   - Create a new cocotb testbench: `sw/test/micro_yolo/`
   - Load the `.bin` files into the simulated L2 memory.
   - Compile Firmware Scheduler and run Cocotb/Verilator simulation.
3. **Matching:**
   - Python testbench reads `raw_head_int8` from L2 and compares byte-by-byte.
   - Allowed error: 0 (since everything is INT8 deterministic).
