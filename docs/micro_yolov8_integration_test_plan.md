# Micro-YOLOv8 Integration Test Plan

## Objective

Prove that the NPU (Systolic Array + Spatz + iDMA) can continuously run a neural network graph with the characteristic structure of YOLOv8, not just isolated operators.

Input: **96x96 RGB INT8**

---

## 1. Topology

| # | Layer | Op | Config | Output (NxCxHxW) | Hardware Unit |
|---|---|---|---|---|---|
| 0 | Input | - | - | 1x3x96x96 | L2 |
| 1 | Conv_Stem | **Conv2D** | K=3x3, S=2, Pad=1, OC=16 | 1x16x48x48 | Spatz Im2Col (IC=3) + Systolic |
| 2 | SiLU_Sig | **Logistic** | element-wise sigmoid | 1x16x48x48 | Spatz (LUT-256) |
| 3 | SiLU_Mul | **Mul** | L1 * L2 | 1x16x48x48 | Spatz (vmul.vv) |
| 4 | C2f_Split | **StridedSlice** | channel split: [0:8], [8:16] | 2x (1x8x48x48) | Firmware pointer |
| 5 | C2f_Conv | **Conv2D** | K=3x3, S=1, Pad=1, IC=8, OC=16 | 1x16x48x48 | Spatz Im2Col (IC=8) + Systolic |
| 6 | C2f_Add | **Add** | L4_branch0 + L5 (broadcast/pad) | 1x16x48x48 | Spatz (vadd.vv) |
| 7 | Conv_Down | **Conv2D** | K=3x3, S=2, Pad=1, OC=32 | 1x32x24x24 | Systolic (IC=16 < 32 => Spatz prep) |
| 8 | SPPF_Pool | **MaxPool2D** | K=5x5, S=1, Pad=2 | 1x32x24x24 | Spatz |
| 9 | Upsample | **Upsample** | Nearest 2x | 1x32x48x48 | Spatz / iDMA 2D |
| 10 | Concat | **Concat** | L9 + L3 along C-axis | 1x48x48x48 | iDMA copy |
| 11 | Head_Conv | **Conv2D** | K=1x1, S=1, OC=32 | 1x32x48x48 | Fast Path (IC=48>=32) |
| 12 | Head_Split | **StridedSlice** | split Box[0:16] and Class[16:32] | 1x16x48x48 x2 | Firmware pointer |
| 13 | Transpose | **Transpose** | [N,C,H,W] -> [N,H,W,C] | 1x48x48x16 | iDMA 2D stride |
| 14 | Reshape | **Reshape** | flatten -> [1, 2304, 16] | 1x2304x16 | Zero-cost (metadata) |
| 15 | DFL_Soft | **Softmax** | on Box branch (dim=-1) | 1x2304x16 | Spatz |
| 16 | Cls_Sig | **Logistic** | on Class branch | 1x2304x16 | Spatz (LUT-256) |

**12 unique operators:** Conv2D, Logistic, Mul, StridedSlice, Add,
MaxPool2D, Upsample, Concat, Transpose, Reshape, Softmax + Requant (fused).

---

## 2. Operator Feasibility Assessment

### 2.1 ALREADY AVAILABLE (Ready)

| Op | API | File | Notes |
|---|---|---|---|
| Conv2D | `npu_conv2d_packed_run_oc32_requant()` | `sw/lib/conv2d_packed.c` | Fully supports 3x3, 1x1. Fixed OC=32. |
| Copy | `spatz_vec_copy_i8()` | `sw/lib/spatz_ops.h` | Used for Concat/Slice |
| ReLU | `spatz_vec_relu_i8()` | `sw/lib/spatz_ops.h` | Backup if SiLU is dropped |
| Requant | `spatz_requant_i32_to_i8()` | `sw/lib/spatz_ops.h` | INT32->INT8 with multiplier+shift |
| iDMA 1D/2D/3D | `idma_L2ToL1_2d()`, `idma_L2ToL1_3d()` | `sw/lib/idma_mm_utils.h` | Used for Transpose, Concat |

### 2.2 NEEDS NEW IMPLEMENTATION (8 operators)

| Op | Difficulty | Description | RVV instruction |
|---|---|---|---|
| **Add** | Easy | Add 2 saturated INT8 vectors | `vadd.vv` + `vmax`/`vmin` clamp |
| **Mul** | Easy | Multiply 2 INT8 vectors, needs requant | `vmul.vv` -> INT16 -> shift -> clamp |
| **Logistic** | Medium | Sigmoid using 256-byte LUT on TCDM | `vluxei8.v` (indexed load) |
| **MaxPool2D** | Medium | 5x5 sliding window finding max | `vmax.vv` looped multiple times |
| **Upsample** | Medium | Nearest 2x: duplicate pixels | `vrgather` or iDMA 2D copy |
| **Softmax** | **Hard** | findmax -> sub -> exp(LUT) -> sum -> div | Needs INT16/INT32 scratch buffer |
| **Concat** | Easy | Copy 2 memory regions into 1 contiguous | `spatz_vec_copy_i8` or iDMA |
| **Transpose** | Medium | NCHW->NHWC: iDMA 2D stride copy | `idma_L2ToL1_2d()` already available |

### 2.3 ZERO-COST (only metadata changes, no code needed)

| Op | Method |
|---|---|
| **StridedSlice** | Firmware only needs to recalculate `base_addr` and `channel_count` of the tensor. No data copy. |
| **Reshape** | Firmware changes the `shape[]` field in the tensor struct. No data copy. |

---

## 3. Key Architectural Issues

### 3.1 OC > 32: Multi-OC Tiling

The `npu_conv2d_packed_run_oc32()` function currently only handles a **fixed OC = 32**.
In Micro-YOLOv8, the `Conv_Down` layer has OC=32 (just enough), but if OC=64
or larger (like full YOLO), an outer OC loop needs to be added:

```
for (oc_tile = 0; oc_tile < OC; oc_tile += 32) {
    // load weight slice [oc_tile : oc_tile+32]
    // run npu_conv2d_packed_run_oc32() with this weight slice
    // write result to output[..., oc_tile : oc_tile+32]
}
```

**In this test:** All OC <= 32, so multi-OC tiling is NOT needed.
Recorded to implement later.

### 3.2 Add with mismatched shapes (C2f residual)

Layer 6 (`C2f_Add`) adds 2 tensors with different channel counts:
- L4_branch0: 1x8x48x48 (first half of L3 after Split)
- L5: 1x16x48x48 (output of C2f_Conv)

This **CANNOT** be added directly. Two options:
- **(A)** Zero-pad the 8-channel branch to 16 channels before adding. Simple but
  wastes memory.
- **(B)** Change topology: make C2f_Conv output 8 channels (OC=8) to match the
  branch. Then the Add is 8+8 => no issue.

**Recommendation:** Choose (B) to simplify the test. Change L5 to OC=8.

### 3.3 Sigmoid/Softmax on INT8 - Quantization concern

Sigmoid and Softmax are non-linear functions operating on the real number domain [0, 1].
When the input is quantized INT8 (scale + zero_point), we need:

1. **Sigmoid LUT:** Pre-calculate 256 `sigmoid(dequant(i))` values then re-quantize
   back to INT8. The whole LUT is only 256 bytes.
2. **Softmax:** More complex because it needs `exp()` and `sum()`. Must be done on
   higher precision (INT16 or INT32) then requantized back to INT8 at the last step.
   **Should be the last operator implemented.**

### 3.4 Memory budget for 96x96

| Tensor | Size | Bytes |
|---|---|---|
| Input 96x96x3 | | 27,648 |
| L1 Conv_Stem 48x48x16 | | 36,864 |
| L7 Conv_Down 24x24x32 | | 18,432 |
| L10 Concat 48x48x48 | | 110,592 |
| L11 Head_Conv 48x48x32 | | 73,728 |
| **Total (concurrent)** | | ~267 KB |
| **TCDM capacity** | | 128-256 KB |

Tensors L10 (Concat) and L11 (Head_Conv) **CANNOT** reside simultaneously in
TCDM. Must use the **L2-Centric + Tiling** strategy:
- Store all intermediate tensors in L2.
- Only pull small tiles into L1 to compute, then push results back to L2.

---

## 4. Implementation Phase Breakdown

### Phase 1: Operator Library (priority, tested individually)

Objective: Write and unit-test each new operator in `spatz_ops`.

| Step | Task | File |
|---|---|---|
| 1a | `spatz_add_i8()` | `sw/lib/spatz_ops.c` |
| 1b | `spatz_mul_i8()` | `sw/lib/spatz_ops.c` |
| 1c | `spatz_logistic_i8()` + LUT | `sw/lib/spatz_ops.c` |
| 1d | `spatz_maxpool2d_i8()` | `sw/lib/spatz_ops.c` |
| 1e | `spatz_upsample_nearest_i8()` | `sw/lib/spatz_ops.c` |
| 1f | `spatz_softmax_i8()` | `sw/lib/spatz_ops.c` |
| 1g | Unit test for all | `sw/test/spatz_ops/main.c` (expand) |

**Verify:** Expand the existing `test_spatz_operator_library` test. Every operator
is tested with Python golden data, compared byte-by-byte.

### Phase 2: Graph Scheduler Firmware

Objective: Write firmware to run the 16 steps of the graph sequentially.

| Step | Task | File |
|---|---|---|
| 2a | Define `tensor_t` struct (addr, shape, scale, zp) | `sw/lib/npu_tensor.h` |
| 2b | Write scheduler `micro_yolo_run()` calling operators sequentially | `sw/test/micro_yolo/main.c` |
| 2c | iDMA tiling logic for tensors exceeding TCDM | `sw/test/micro_yolo/main.c` |

### Phase 3: Golden Model + End-to-End Verify

| Step | Task |
|---|---|
| 3a | Python script: define network, quantize INT8, export `.bin` |
| 3b | Cocotb testbench: load `.bin` into L2, run firmware, read results |
| 3c | Compare output vs golden. Allowed error: **0 bytes** |

---

## 5. Verification strategy

1. **Golden Model (Python/PyTorch):**
   - Write a Python script using PyTorch defining the Micro-YOLOv8 network.
   - Assign random weights and quantize everything to INT8.
   - Export: `input_image.bin`, `weights_layer_N.bin`, `golden_output.bin`.
2. **RTL Simulation:**
   - Create a new cocotb testbench: `sw/test/micro_yolo/`
   - Load the `.bin` files into the simulated L2 memory.
   - Compile Firmware Scheduler and run Cocotb/Verilator simulation.
3. **Matching:**
   - Python testbench reads `final_output` from L2 and compares byte-by-byte.
   - Allowed error: 0 (since everything is INT8 deterministic).
