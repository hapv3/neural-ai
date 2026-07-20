# Micro-YOLOv8 Integration Test Plan

## Objective

Prove that the NPU (Systolic Array + Spatz + iDMA) can continuously run a neural network graph with the characteristic structure of YOLOv8, not just isolated operators.

Input: **96x96 RGB INT8**

First end-to-end target: run the backbone/neck/head Conv path, write the
**raw head tensor as INT8**, then run the first postprocess kernels for DFL and
class sigmoid.

Current status: the raw-head E2E path, fused DFL, and class sigmoid are
implemented and passing in `test_micro_yolo_e2e.py`.

---

## 1. Topology

| # | Layer | Op | Config | Output (logical NxCxHxW) | Hardware Unit |
|---|---|---|---|---|---|
| 0 | Input | - | - | 1x3x96x96 | L2 |
| 1 | Conv_Stem | **Conv2D** | K=3x3, S=2, Pad=1, OC=32 | 1x32x48x48 | systolic linebuffer + requant |
| 2 | SiLU_Sig | **Logistic** | element-wise sigmoid | 1x32x48x48 | AFU LUT or Spatz LUT fallback |
| 3 | SiLU_Mul | **Mul** | L1 * L2 | 1x32x48x48 | AFU Mul_Q7 |
| 4 | C2f_Route | **Identity/Slice** | route L3 as residual branch | 1x32x48x48 | Firmware metadata |
| 5 | C2f_Conv | **Conv2D** | K=3x3, S=1, Pad=1, IC=32, OC=32 | 1x32x48x48 | systolic linebuffer/KGEN + requant |
| 6 | C2f_Add | **Add** | L4 + L5, same shape | 1x32x48x48 | AFU Add |
| 7 | Conv_Down | **Conv2D** | K=3x3, S=2, Pad=1, IC=32, OC=32 | 1x32x24x24 | systolic linebuffer/KGEN + requant |
| 8 | SPPF_Pool | **MaxPool2D** | K=5x5, S=1, Pad=2 | 1x32x24x24 | Systolic linebuffer pool mode |
| 9 | Upsample | **Upsample** | Nearest 2x | 1x32x48x48 | Spatz C32-specialized |
| 10 | Concat | **Bypassed Concat** | logical L9 + L3 along C-axis | logical 1x64x48x48 | no materialized copy |
| 11 | Head_Conv | **Conv2D** | K=3x3, S=1, Pad=1, IC=64, OC=32 | 1x32x48x48 | two C32 linebuffer chunks + psum accumulate/requant |
| 12 | Raw_Head_Out | **DMA_OUT** | write raw head INT8 to L2 | 1x32x48x48 | iDMA row/tile writeback |
| 13 | Head_Split | **ROW32 view** | Box[0:16] and Class[16:32] | 1x16x48x48 x2 | no materialized split |
| 14 | DFL_Soft | **DFL softmax** | on Box branch, `reg_max=4` | 1x4x48x48 Q8.8 | fused AFU mode |
| 15 | Cls_Sig | **Logistic** | on Class branch | 1x16x48x48 INT8 | AFU class sigmoid |
| 16 | Future_Transpose | **Transpose** | optional postprocess layout change | 1x48x48x16 | Future postprocess |
| 17 | Future_Decode_NMS | **Decode/NMS** | boxes + class scores | detection output | Future firmware/host decision |

Internal implementation should keep tensors in the existing HWC/NHWC packed
layout used by `conv2d_packed`; the NxCxHxW notation above is only the logical
model shape.

First E2E unique operators: Conv2D, Logistic, Mul, Identity/Slice, Add,
MaxPool2D, Upsample, logical Concat, Requant, DMA_OUT, fused DFL, and class
sigmoid. The current raw-head path intentionally **does not materialize** the
Concat tensor. Instead, the Head_Conv consumes the two C32 producers as separate
input chunks and accumulates their partial sums. DFL and class sigmoid also
avoid materialized branch split: they consume the low/high halves of each ROW32
raw-head location directly.

---

## 2. Operator Feasibility Assessment

### 2.1 ALREADY AVAILABLE (Ready)

| Op | API | File | Notes |
|---|---|---|---|
| Conv2D | `npu_conv2d_packed_run_oc32_requant()` | `sw/lib/conv2d_packed.c` | Supports the first E2E topology because every Conv output tile is OC=32. |
| Copy | `spatz_vec_copy_i8()` | `sw/lib/spatz_ops.h` | Used for fallback copies and Slice-style movement |
| ReLU | `spatz_vec_relu_i8()` | `sw/lib/spatz_ops.h` | Backup if SiLU is dropped |
| Requant | systolic requant drain, `spatz_requant_i32_to_i8()` fallback | `sw/lib/hal_systolic.c`, `sw/lib/spatz_ops.h` | Conv outputs use systolic requant; Spatz fallback must avoid scratch overlap. |
| iDMA 1D/2D/3D | `idma_L2ToL1_2d()`, `idma_L2ToL1_3d()` | `sw/lib/idma_mm_utils.h` | Used for L2 checkpoints, future Transpose, and fallback materialized Concat. |

### 2.2 IMPLEMENTED AND MISSING NON-CONV OPS

| Op | Difficulty | Description | RVV instruction |
|---|---|---|---|
| **Add** | Easy | Add 2 saturated INT8 vectors | Implemented through AFU Add |
| **Mul** | Easy | Q7 INT8 multiply for SiLU | Implemented through AFU Mul_Q7 |
| **Logistic** | Medium | Sigmoid using 256-byte LUT on TCDM | Implemented through AFU LUT |
| **MaxPool2D** | Medium | 5x5 sliding window finding max | Implemented through systolic linebuffer pool mode |
| **Upsample** | Medium | Nearest 2x: duplicate pixels | Implemented as C32-specialized Spatz load-once/store-four kernel |
| **Softmax / DFL** | **Hard** | findmax -> sub -> exp(LUT) -> sum -> div | Implemented for YOLO `reg_max=4` as fused AFU ROW32 low16 mode |
| **Concat** | Easy | Logical C-axis join | Current raw-head path bypasses materialization by splitting the following Conv over two C32 sources. |
| **Transpose** | Medium | NCHW->NHWC: iDMA 2D stride copy | Future postprocess; iDMA 2D support already exists |
| **Class sigmoid** | Medium | Sigmoid LUT on class logits | Implemented as fused AFU ROW32 high16 compact-output mode |

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
include weights or INT32 psum scratch. Therefore the test uses a hybrid
**TCDM hot tensors + L2 checkpoint/lifetime** strategy:

- Full materialized Concat is skipped. This saves `48*48*64 = 147,456` bytes.
- The L3 skip tensor is written to L2 before it is overwritten in TCDM, then
  reloaded before Head_Conv.
- Head_Conv uses full INT32 psum scratch for the first C32 chunk and final
  accumulate+requant for the second C32 chunk.
- The final head INT8 output is produced tile-by-tile in TCDM and copied to L2.

Current TCDM allocation after 3j is approximately:

| Allocation | Bytes |
|---|---:|
| Input 96x96x3 | 27,648 |
| Stem weight | 1,024 |
| C2f weight | 9,216 |
| Down weight | 9,216 |
| Head weight, two C32 chunks | 18,432 |
| Sigmoid LUT | 256 |
| Activation buffer A / skip reload | 73,728 |
| INT32 psum / sigmoid scratch | 294,912 |
| Activation buffer C / upsample | 73,728 |
| Head output tile 16x16x32 | 8,192 |
| **Total** | **516,352** |

This fits the 512 KiB logical TCDM only narrowly because `512 KiB = 524,288`
bytes. The remaining head tile size must therefore stay conservative unless
the scratch allocator becomes lifetime-aware.

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
| 1g | `npu_dfl_softmax4_row32_i8_q8()` and class sigmoid postprocess | `sw/lib/spatz_ops.c`, `sw/lib/spatz_ops.h` |
| 1h | Unit tests for all implemented operators | `sw/test/spatz_ops/main.c`, `test_spatz_operator_library.py` |

Acceptance: each operator has a deterministic Python golden and byte-exact RTL
cluster test coverage. Fused DFL and class sigmoid are now enabled after the
raw-head INT8 layer in the Micro-YOLO E2E graph.

### Phase 2: Tensor Layout and Graph Runtime Contract

Objective: define layout and lifetime rules before introducing the full
96x96 topology. This prevents graph bugs from being hidden inside ad hoc copies.

Required decisions:

- Activation tensors with `C <= 32`: store as one C32 block with zero-padded tail.
- Activation tensors with `C > 32`: store as C32-blocked blocks, not compact NCHW.
- RGB model input is an exception to the internal C32 policy. The first
  `Conv_Stem` keeps the current raw `HWC` C3 input contract and uses the
  dedicated RGB linebuffer descriptor path (`input_c=3`, `input_c_stride=3`).
  Python/host must not pre-pack the RGB input into C32 for this path unless a
  future stem-specific performance tradeoff explicitly changes the contract.
- Middle and later Conv layers should use C32-aligned fast-path descriptors.
  Their producer OFM tensors should already be written in `ROW32` or
  `C32_BLOCKED` padded layout, and weights should be compiler/host-packed with
  padded lanes set to zero. Firmware should dispatch these descriptors without
  runtime channel packing or padding.
- Concat along C: prefer bypassing materialization when the consumer is Conv.
  Materialized fallback must produce C32-blocked output blocks directly.
- Conv consumers: use the C32-chunk descriptor path for `IC=64` and larger.
- L2 is the owner of full intermediate tensors; TCDM only holds working tiles.
- Python host / command generator must own the C32-aligned linebuffer schedule:
  when it emits a Conv2D tile that uses the RTL `C32_FAST` linebuffer path, it
  must guarantee `input_base`, `pixel_stride_bytes`, `row_stride_bytes`, and
  `channel_addr_offset` are 32-byte aligned. It must also emit the complete
  linebuffer/GEMM job descriptor, including the fast-path fields:
  `block_valid_bytes`, `channel_addr_offset`, and `coalesce_k_bytes =
  kernel_h * kernel_w * block_valid_bytes`. Firmware should treat these fields
  as part of the descriptor, not re-plan the schedule on Snitch.
- The Python host helper in `hw/rtl/cluster/tb/npu_linebuf_precompute.py` now emits full
  Micro-YOLO linebuffer/GEMM job arrays as runtime L2 descriptor blobs.
  Firmware DMA-copies those blobs into scratch/TCDM, stores pointers to the
  copied arrays in `npu_layer_t`, and calls the descriptor runner in
  `sw/lib/conv2d_packed.c`. The C planner remains as a generic compatibility
  fallback for layers that do not provide host-planned job descriptors. Python
  must only set `C32_FAST` for one full C32 block (`block_valid_bytes == 32`,
  `lane_base == 0`, `c_base == 0`, 32-byte aligned base/offset). Tail chunks
  and non-C32 layouts must leave `C32_FAST` disabled and use the generic
  linebuffer path.

| Step | Task | File |
|---|---|---|
| 2a | Define `tensor_t` with addr, H/W/C, layout, scale, zero point | `sw/lib/npu_tensor.h` |
| 2b | Define C32-blocked tensor helpers and address math | `sw/lib/npu_tensor.h` or `sw/lib/npu_graph.h` |
| 2c | Add graph scratch allocator for TCDM tile buffers | `sw/test/micro_yolo/main.c` or `sw/lib/npu_graph.c` |
| 2d | Document qparam propagation for Requant/SiLU/Add/Mul | `docs/micro_yolov8_integration_test_plan.md` |
| 2e | Emit C32-aligned Conv2D linebuffer/GEMM job descriptors from Python host | Python command generator / graph export path |

Acceptance: the minimal graph in Phase 3a can run using the same tensor
descriptors and layout rules that the full 96x96 graph will use.

Current implementation status:

- `sw/lib/npu_tensor.h` defines the shared tensor descriptor, INT8/INT32 dtype
  enum, HWC/ROW32/C32-blocked layout enum, C32 byte-size helpers, and address
  helpers.
- `sw/lib/npu_graph.c` owns the small graph executor, tensor/layout validation,
  and a linear TCDM scratch allocator used by the Micro-YOLO firmware.
- Requant qparams are carried through graph-layer attributes. The current
  micro-yolo firmware uses uniform qparams per layer; per-channel qparams remain
  compatible with the systolic requant configuration API.
- `hw/rtl/cluster/tb/npu_linebuf_precompute.py` owns the current host-side C32 linebuffer
  job descriptor planning for the Micro-YOLO graph. The active flow is runtime
  descriptor delivery: Python/host emits a small descriptor manifest plus binary
  descriptor blobs, writes the manifest to L2 at `0x80052000`, writes each blob
  at the L2 address listed by the manifest, and firmware DMA-copies those blobs
  into scratch/TCDM before graph setup. Firmware dispatches only the received
  pointer/count pairs and does not compile the full job arrays into `.data`.
- The generated descriptor ABI is `npu_conv2d_linebuf_job_desc_t`: full
  `systolic_linebuf_cfg_t`, full `systolic_gemm32_req_t`, `rows`, and
  `k_tiles`. Head chunk1 uses `npu_conv2d_l2_copy_job_desc_t`, which wraps the
  linebuffer job plus compact tile-output and final L2 copy metadata.
- `test_micro_yolo_e2e` loads `micro_yolo.elf` section-by-section and also
  writes the Python-generated descriptor blobs into L2 before releasing fetch.
  `.text` is still written through AXI I-TCM; firmware then pulls runtime
  descriptor blobs from L2 into scratch.

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
| 3i | Add Concat to C64 C32-blocked layout | Bypassed; no materialized tensor |
| 3j | Add Head_Conv IC64 OC32 | Raw head `48x48x32` INT8 |

Acceptance: every step writes its checkpoint tensor to L2 and compares against
the Python golden before the next operator is enabled. Step 3a also creates the
`sw/test/micro_yolo/` firmware, Makefile, linker/start files, and makes the
existing `test_micro_yolo_e2e.py` style test pass.

Current implementation status:

- The active firmware entrypoint is Phase 3j raw-head E2E. Older Phase 3
  checkpoints are documented as integration history and should not be treated
  as separate current firmware modes.
- Phase 3h is committed as `6d61c4b Add micro yolo phase 3h upsample`.
- Phase 3j is committed as `abc827b Add micro yolo phase 3j head conv bypass concat`.
- `sw/test/micro_yolo/` now runs the raw-head path through `Head_Conv` and
  writes the `48x48x32` INT8 raw head tensor to L2.
- `test_micro_yolo_e2e.py` generates deterministic input/weights/golden in
  Python and compares the L2 raw-head output byte-exactly.
- Latest cluster RTL result: PASS.

Phase 3 checkpoint matrix:

| Step | Status | Runtime path | Checkpoint / output | Notes |
|---|---|---|---|---|
| 3a | Historical bootstrap | Minimal graph firmware skeleton | `32x32x32` INT8 | Brought up `sw/test/micro_yolo`, linker/start files, graph descriptors, and byte-exact cocotb harness. Not the active entrypoint now. |
| 3b | Folded into 3j | `Conv_Stem` 3x3/s2/p1 C3->C32 through systolic linebuffer + requant | L1 `48x48x32` INT8 | Uses direct linebuffer path, not scalar im2col. Input is HWC RGB; output is C32-blocked. |
| 3c | Folded into 3j | AFU Logistic LUT + AFU `Mul_Q7` for SiLU | L3 `48x48x32` INT8 | L3 is the skip branch later consumed by logical Concat. Current firmware saves it to L2 because the present TCDM lifetime map cannot keep it resident through downsample/pool/upsample. |
| 3d | Folded into 3j | `C2f_Conv` 3x3/s1/p1 C32->C32 through systolic linebuffer/KGEN + requant | L5 `48x48x32` INT8 | Confirms the C32 steady-state conv path used by later YOLO blocks. |
| 3e | Folded into 3j | AFU residual Add | L6 `48x48x32` INT8 | Adds C2f output with the preserved SiLU activation before the downsample branch. |
| 3f | Folded into 3j | `Conv_Down` 3x3/s2/p1 C32->C32 through systolic linebuffer/KGEN + requant | L7 `24x24x32` INT8 | Exercises stride-2 spatial reduction with the same C32 conv scheduler. |
| 3g | Folded into 3j | MaxPool 5x5/s1/p2 C32 through systolic linebuffer pool mode | L8 `24x24x32` INT8 | Replaced slower Spatz/scalar pooling paths. Reference PMU after this phase was about 18.5k cycles for MaxPool. |
| 3h | Committed | Upsample nearest 2x C32 through Spatz specialized kernel | L9 `48x48x32` INT8 | Uses one vector load and four vector stores per C32 element group. Phase 3h E2E reference was about 237.7k cycles total. |
| 3i | Deliberately bypassed | No materialized Concat tensor | No standalone tensor | Logical Concat is only represented in the consuming Head_Conv. This avoids allocating/copying `48x48x64` = 147,456 bytes. |
| 3j | Active | Fused logical Concat + `Head_Conv` C64->C32 | Raw head `48x48x32` INT8 in L2 | Runs two C32 conv chunks. Chunk 0 writes INT32 psum; chunk 1 accumulates, requants, and writes the final INT8 tile. |

Active Phase 3j graph:

```
DMA_IN input
DMA_IN weight0, weight1, weight2, weight3, sigmoid LUT
Conv_Stem 3x3/s2/p1 C3 -> C32, systolic linebuffer + requant
AFU Logistic LUT
AFU Mul_Q7 for SiLU
DMA_OUT L3 skip to L2
C2f_Conv 3x3/s1/p1 C32 -> C32, systolic linebuffer + requant
Residual Add via AFU
Conv_Down 3x3/s2/p1 C32 -> C32, systolic linebuffer + requant
MaxPool 5x5/s1/p2 C32, systolic linebuffer pool mode
Upsample nearest 2x C32, Spatz specialized kernel
DMA_IN L3 skip reload from L2
Head_Conv 3x3/s1/p1 logical C64 -> C32:
  chunk 0: upsample C32 -> INT32 psum
  chunk 1: skip C32 -> accumulate with psum -> systolic requant -> INT8 tile
DMA tile rows to L2 raw-head output
```

The materialized Concat step is deliberately bypassed. The mathematical
equivalence used by the scheduler is:

```
Conv([A, B], W) = Conv(A, W[0:32]) + Conv(B, W[32:64])
```

where `A` is the upsampled SPPF branch and `B` is the skip branch from L3.
Requantization is only applied after the second chunk has accumulated into the
INT32 psum. This preserves the logical Concat+Conv result while avoiding the
147,456-byte `48x48x64` concat copy.

Graph/runtime entries used by the active Phase 3 path:

| Item | Used by | Purpose |
|---|---|---|
| `npu_tensor_t` / `npu_layer_t` | 3a+ | Shared graph tensor/layer ABI for firmware-side scheduling. |
| `npu_layer_t.src2` | 3j | Second input tensor for fused logical Concat consumers. |
| `CONV2D_C32_LINEBUF_REQUANT` | 3d | C32 conv + final requant path for C2f. |
| `CONV2D_C32_DOWNSAMPLE_LINEBUF_REQUANT` | 3f | C32 stride-2 conv + final requant path for downsample. |
| `MAXPOOL2D_I8` | 3g | SPPF-style 5x5 pool through the linebuffer pool datapath. |
| `UPSAMPLE_NEAREST_I8` | 3h | nearest-neighbor 2x graph op. |
| `CONV2D_DUAL_SOURCE_C32_LINEBUF_REQUANT_L2` | 3j | fused logical Concat + Head_Conv. |
| `spatz_upsample_nearest2x_c32_i8()` | 3h | C32 fast path: one vector load, four vector stores. |
| `systolic_gemm32_linebuf_ktiles_accumulate_requant_strided()` | 3j | final chunk accumulates external psum and requants in the systolic controller. |
| `npu_conv2d_packed_run_oc32_linebuf_tile_accumulate_requant()` | 3j | per-spatial-tile final Head_Conv chunk helper. |

The head path uses the existing `systolic_controller.sv` psum-buffer/requant
datapath. No new RTL state was required for 3j: software now exposes the mode
by setting `accum_en`, `RQ_CTRL`, `psum_addr`, output stride, and psum stride
for the final C32 chunk.

### Phase 3j PMU Snapshot

Latest `test_micro_yolo_e2e.py` result after 3j:

| Layer | Op | Cycles | Key notes |
|---:|---|---:|---|
| 0 | DMA_IN input | 1,696 | input HWC |
| 1 | DMA_IN weight0 | 360 | stem weights |
| 2 | DMA_IN weight1 | 784 | C2f weights |
| 3 | DMA_IN weight2 | 784 | down weights |
| 4 | DMA_IN weight3 | 1,240 | two C32 head chunks |
| 5 | DMA_IN sigmoid LUT | 360 | 256-byte LUT |
| 6 | Conv_Stem | 21,134 | `sys_compute=2,304`, `ifm_req=14,805` |
| 7 | Logistic LUT | 26,248 | AFU |
| 8 | Mul_Q7 | 9,378 | AFU, `tcdm_stall=2,304` |
| 9 | DMA_OUT skip | 4,100 | preserve L3 for later logical concat |
| 10 | C2f_Conv | 49,802 | `sys_compute=20,736`, `ifm_req=26,928` |
| 11 | Residual Add | 9,388 | AFU, `tcdm_stall=2,304` |
| 12 | Conv_Down | 41,014 | `sys_compute=5,184`, `ifm_req=32,463` |
| 13 | MaxPool | 18,474 | linebuffer pool mode |
| 14 | Upsample | 35,462 | Spatz C32 fast path, `tcdm_stall=2,880` |
| 15 | DMA_IN skip reload | 4,062 | reload L3 branch |
| 16 | Head_Conv C32x2 | 149,256 | `sys_compute=41,472`, `ifm_req=54,277`, `ofm_req=21,042` |
| **Total** |  | **388,146** | PASS, runtime L2 descriptor manifest |

Head_Conv compute scales as expected:

- C2f C32 compute: `20,736` cycles.
- Head logical C64 compute: `41,472` cycles, exactly `2x`.
- Head total cycles are about `3.1x` C2f total because the second C32 chunk also
  reads external psum, accumulates, requants, and writes INT8 tiles to L2.
- Firmware now uses host-generated linebuffer/GEMM job descriptors delivered
  through a runtime L2 manifest/blob flow plus RTL shadow registers for
  linebuffer tile scheduling: tile N+1 config is staged while tile N is running,
  so the DONE-to-START gap only needs a start pulse plus the residual wait.
- This reduced total Micro-YOLO cycles from the previous field-precompute bridge
  snapshot (`403,128`) to `388,146`. The systolic compute count stayed
  `69,696`, confirming the gain comes from eliminating Snitch-side tile config
  preparation rather than changing MAC work. The runtime manifest/blob handoff
  adds one small pre-graph DMA setup compared with the earlier static
  descriptor-pointer verification flow.

Current head scheduler geometry:

- Spatial output: `48x48`.
- Tile: `16x16`.
- Spatial tiles: `3 * 3 = 9`.
- Logical C64 is split into two C32 chunks.
- Scheduler/linebuffer starts: `9 * 2 = 18`.
- Each C32 chunk has `3*3*32/32 = 9` KGEN tiles.
- Total KGEN phases in Head_Conv: `18 * 9 = 162`.

Next optimization target: reduce the 3j head overhead above pure compute by
reducing per-tile setup, reusing linebuffer/window state across adjacent tiles,
and avoiding L2 skip save/reload when a lifetime-aware TCDM allocator can keep
the skip branch resident.

### Phase 3k: Full 96x96 Raw-Head E2E

Objective: run the complete raw-head topology through `Raw_Head_Out`.
DFL/Softmax/decode table entries may exist in metadata but must remain disabled.

| Step | Task |
|---|---|
| 3k-a | Generate deterministic Python golden for the full raw-head topology |
| 3k-b | Export input and weights as fixed `.bin` fixtures or generate them in cocotb |
| 3k-c | Run firmware scheduler over L2-centric tiled tensors |
| 3k-d | Compare raw head INT8 output byte-exactly |
| 3k-e | Collect PMU counters per layer for performance triage |

Acceptance: raw head `1x32x48x48` INT8 matches golden with 0 byte mismatch.

Current status: **implemented and passing** through the Phase 3j raw-head
checkpoint. The test remains in `test_micro_yolo_e2e.py` rather than a separate
Phase 4 test file, but it now covers the full raw-head tensor. This closes the
old Phase 4 scope, so postprocess preparation now starts at the new Phase 4.

Current fixture policy:

- Cocotb generates deterministic input and all weights in Python.
- Firmware DMA-loads input, weights, and the sigmoid LUT from L2 into TCDM.
- Firmware writes final raw-head INT8 output to `L2_OUTPUT = 0x80020000`.
- Cocotb reads `48*48*32 = 73,728` output bytes and compares against the Python
  golden.

Current L2 map:

| Symbol | Address | Payload |
|---|---:|---|
| `L2_INPUT` | `0x80000000` | `96x96x3` INT8 input |
| `L2_WEIGHT0` | `0x80008000` | stem weight |
| `L2_SIG_LUT` | `0x80009000` | sigmoid LUT |
| `L2_WEIGHT1` | `0x8000A000` | C2f weight |
| `L2_WEIGHT2` | `0x8000D000` | down weight |
| `L2_WEIGHT3` | `0x80010000` | head weight, two C32 chunks |
| `L2_OUTPUT` | `0x80020000` | raw head output |
| `L2_SKIP` | `0x80040000` | temporary skip checkpoint |
| `L2_LINEBUF_DESC_BASE` | `0x80052000` | runtime linebuffer job manifest |
| `L2_DFL_EXP_LUT` | `0x80054000` | DFL exp LUT |
| `L2_DFL_RECIP_LUT` | `0x80054400` | DFL reciprocal LUT |
| `L2_DFL_OUTPUT` | `0x80060000` | DFL Q8.8 output |
| `L2_CLASS_OUTPUT` | `0x80065000` | class sigmoid INT8 output |

### Phase 4: Postprocess Preparation

Objective: extend beyond raw-head output after the Conv/activation path is stable.

| Step | Task |
|---|---|
| 4a | Head split into Box/Class branches |
| 4b | Transpose/reshape to flattened head layout |
| 4c | DFL Softmax implementation and unit test |
| 4d | Class sigmoid |
| 4e | Decode/NMS roadmap |

#### Phase 4a: Head split into Box/Class branches

Current status: **implemented and passing**.

The current micro-YOLO raw head remains a compact `48x48x32` INT8 ROW32 tensor
written to `L2_OUTPUT`. Phase 4a defines the first postprocess ABI on top of
that raw tensor:

| Branch | Channel range | Shape | Bytes |
|---|---:|---|---:|
| Box / DFL logits | `0..15` | `48x48x16` INT8 | `36,864` |
| Class logits | `16..31` | `48x48x16` INT8 | `36,864` |

The raw-head tensor stays in ROW32 physical layout:

```text
raw_head[location][0..15]  -> Box / DFL logits
raw_head[location][16..31] -> Class logits
```

The full raw head is still compared byte-exactly. The postprocess operators now
consume the two branch views directly from the ROW32 tensor, avoiding an extra
full-head materialized split/copy.

#### Phase 4c: DFL Softmax implementation and unit test

Current status: **implemented and passing**.

The current micro-YOLO Box branch uses `reg_max=4`, so each location has
`4 sides * 4 bins = 16` logits. DFL converts those logits into four Q8.8
distances per location:

```text
box_logits[location][side][bin] -> dfl_distance_q8[location][side]
```

The optimized implementation is a fused AFU mode:

- `NPU_AFU_MODE_DFL4_ROW32_Q8` reads one 32-byte ROW32 location at a time and
  consumes the low 16 bytes.
- AFU bank0 is interpreted as the DFL exp LUT and bank1 as the reciprocal LUT;
  bank swapping is disabled while the fused DFL mode is active.
- Firmware writes `48*48*4` Q8.8 distances to `L2_DFL_OUTPUT`.
- The cocotb golden uses the same LUT contents and checks the output
  byte-exactly.

Coverage:

- AFU block-level test covers the fused DFL datapath.
- `afu_ops_dfl_fused.bin` covers the C wrapper and cluster path.
- `test_micro_yolo_e2e.py` runs the graph layer and checks `L2_DFL_OUTPUT`.

#### Phase 4d: Class sigmoid

Current status: **implemented and passing**.

Class sigmoid consumes the high 16 bytes of each ROW32 raw-head location:

```text
raw_head[location][16..31] -> class_sigmoid[location][0..15]
```

The implementation uses the new AFU mode
`NPU_AFU_MODE_CLASS_SIGMOID_ROW32_HIGH16`:

- input length is expressed as raw ROW32 input bytes (`locations * 32`);
- AFU reads one 32-byte location, applies the active sigmoid LUT to bytes
  `16..31`, and writes a compact 16-byte class vector;
- the output is `48*48*16` INT8 bytes at `L2_CLASS_OUTPUT`;
- normal AFU ping-pong LUT behavior is preserved for non-DFL modes.

Coverage:

- `hw/rtl/afu/tb/tb_afu.sv` checks a 17-location unaligned-size case at block
  level.
- `afu_ops_class_sigmoid.bin` checks the C wrapper and cluster dispatch path.
- `test_micro_yolo_e2e.py` compares the materialized class sigmoid output
  byte-exactly against the Python golden.

#### Phase 4e: Decode/NMS roadmap

Current status: **not started**.

The next postprocess step should consume `L2_DFL_OUTPUT` and
`L2_CLASS_OUTPUT`, then decide whether decode/NMS should stay on firmware,
Spatz/AFU vector kernels, or a host-side postprocess path.

---

## 5. Verification strategy

1. **Golden Model:**
   - Current test uses deterministic Python loops in
     `hw/rtl/cluster/tb/tests/test_micro_yolo_e2e.py`.
   - The golden computes every Conv directly from the window loops; it does not
     materialize im2col.
   - Head_Conv golden applies the same bypass-Concat identity:
     `Conv(up, W0) + Conv(skip, W1)`, followed by clamp/requant.
2. **Firmware:**
   - `sw/test/micro_yolo/main.c` builds the graph descriptor, allocates linear
     TCDM scratch, and loads host-generated linebuffer job descriptors from L2
     runtime manifest/blob payloads.
   - `sw/lib/npu_graph.c` validates tensor contracts and dispatches graph ops.
   - `sw/lib/conv2d_packed.c` owns linebuffer/KGEN tile execution.
3. **RTL Simulation:**
   - Cocotb loads firmware and L2 fixtures.
   - PMU snapshots are taken per graph layer using the firmware trace page.
   - Output is checked byte-for-byte from L2.
4. **Matching:**
   - Allowed error: 0.
   - Any mismatch reports the byte index and signed INT8 expected/got values.

Recommended commands:

```sh
make -C sw/test/micro_yolo
CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster COCOTB_TEST_MODULES=test_micro_yolo_e2e
```

Relevant passing commits:

| Commit | Scope |
|---|---|
| `6d61c4b` | Phase 3h C32-specialized Upsample |
| `abc827b` | Phase 3j Head_Conv bypass Concat |
