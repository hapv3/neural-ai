# NPU Operator Support Matrix

This document is the current implementation reference for supported graph
operators, the hardware/software block that executes each operator, design
limits, and the source files that show how to use and test the path.

## Summary

| Operator / graph op | Execution block | Primary layout | Status | Main limits | Usage / tests |
|---|---|---|---|---|---|
| `NPU_OP_DMA_IN`, `NPU_OP_DMA_OUT` | iDMA blocking copy wrapper | Any tensor byte range | Supported | 1D blocking copy in graph path | `sw/lib/npu_graph.c`, `sw/lib/idma_mm_utils.h`, `sw/test/independent_memory` |
| `NPU_OP_SYSTOLIC_GEMM32` | Systolic array | `ROW32` i8 input, `ROW32` i32 output | Supported | K/N fixed to 32 lanes, M from `layer.dim_m` | `sw/lib/hal_systolic.c`, `sw/test/independent_systolic` |
| `NPU_OP_SYSTOLIC_GEMM32_REQUANT` | Systolic + RTL requant | `ROW32` i8 output | Supported | Uniform graph requant parameters across 32 lanes unless HAL is used directly | `sw/lib/npu_graph.c`, `sw/test/systolic_requant` |
| `NPU_OP_CONV2D1X1_C32_REQUANT` | Systolic direct GEMM32 loop | `ROW32` or `C32_BLOCKED` i8 | Supported | IC/OC must be multiples of 32; multi-IC uses one i32 psum scratch tile | `sw/test/pointwise_conv`, `hw/rtl/cluster/tb/tests/test_pointwise_conv.py` |
| `NPU_OP_CONV2D3X3S2P1_C3_LINEBUF_REQUANT` | Systolic linebuffer + KGEN + requant | input `HWC`, output `ROW32` | Supported for RGB stem | IC fixed to 3, OC fixed to 32, K=3x3, S=2, P=1 | `sw/test/micro_yolo/main.c`, `hw/rtl/cluster/tb/tests/test_micro_yolo_e2e.py` |
| `NPU_OP_CONV2D3X3S1P1_C32_LINEBUF` | Systolic linebuffer + KGEN | `ROW32` i8 to `ROW32` i32 | Supported | IC=32, OC=32, K=3x3, S=1, P=1, no final requant | `sw/lib/npu_graph.c`, `sw/test/conv_perf` |
| `NPU_OP_CONV2D3X3S1P1_C32_LINEBUF_REQUANT` | Systolic linebuffer + KGEN + requant | `ROW32` i8 | Supported | IC=32, OC=32, K=3x3, S=1, P=1; psum tensor required | `sw/test/micro_yolo`, `sw/test/conv_perf` |
| `NPU_OP_CONV2D3X3S2P1_C32_LINEBUF_REQUANT` | Systolic linebuffer + KGEN + requant | `ROW32` i8 | Supported | IC=32, OC=32, K=3x3, S=2, P=1; psum tensor required | `sw/test/micro_yolo`, `sw/test/conv_perf` |
| `NPU_OP_CONV2D3X3S1P1_C32X2_LINEBUF_REQUANT_L2` | Two C32 linebuffer convs + psum accumulate + DMA tile copy | `ROW32` i8 | Supported for logical concat consumer | Exactly two C32 sources, output OC=32, writes tiles to L2 | `sw/test/micro_yolo/main.c`, `hw/rtl/cluster/tb/tests/test_micro_yolo_e2e.py` |
| `NPU_OP_CONV2D3X3S1P1_C32_MULTI_LINEBUF_REQUANT` | Multi-C32 linebuffer conv + psum accumulation + requant | `C32_BLOCKED` or single-group `ROW32` i8 | Supported | IC/OC must be multiples of 32, K=3x3, S=1, P=1; one i32 psum scratch group reused per OC group | `sw/test/micro_mobilenet`, `hw/rtl/cluster/tb/tests/test_micro_mobilenet_e2e.py` |
| `NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT` | Systolic controller depthwise lane MAC + linebuffer | `C32_BLOCKED` or `ROW32` i8 | Supported | K=3x3, S=1, P=1; channel tails are masked | `sw/test/depthwise_conv`, `hw/rtl/cluster/tb/tests/test_depthwise_conv.py` |
| `NPU_OP_DEPTHWISE3X3S2P1_C32_REQUANT` | Systolic controller depthwise lane MAC + linebuffer | `C32_BLOCKED` or `ROW32` i8 | Supported | K=3x3, S=2, P=1; channel tails are masked | `sw/test/depthwise_conv`, `hw/rtl/cluster/tb/tests/test_depthwise_conv.py` |
| `NPU_OP_SPATZ_REQUANT` | Spatz vector helper | `ROW32` i32 to `ROW32` i8 | Supported / legacy graph path | Uses software wrapper and scratch; systolic fused requant is preferred after conv | `sw/lib/spatz_ops.c`, `sw/test/spatz_ops` |
| `NPU_OP_LOGISTIC_LUT_I8` | AFU E8 LUT | Any i8 byte tensor | Supported | Requires 256-entry LUT tensor; wrapper reloads LUT before start | `sw/lib/spatz_ops.c`, `sw/test/afu_ops`, `sw/test/spatz_ops` |
| `NPU_OP_CLAMP_I8` | AFU E8 LUT | Any i8 layout, same shape in/out | Supported | Out-of-place only; min/max in i8 range | `sw/test/afu_ops`, `hw/rtl/cluster/tb/tests/test_spatz_operator_library.py` |
| `NPU_OP_MUL_I8` | AFU MUL_Q7 or Spatz fallback | Any i8 byte tensor | Supported | AFU fast path only for Q7 config: multiplier=1, shift=7, clamp=-128..127 | `sw/lib/spatz_ops.c`, `sw/test/afu_ops` |
| `NPU_OP_ADD_I8` | AFU ADD_I8 or Spatz fallback | Same-shape i8 tensors | Supported | AFU fast path only for full i8 clamp; graph requires same H/W/C | `sw/lib/spatz_ops.c`, `sw/test/afu_ops` |
| `NPU_OP_MAXPOOL2D5X5S1P2_I8` | Systolic linebuffer pool for C32, Spatz fallback otherwise | i8 | Supported | Fast path requires C=32, K=5x5, S=1, P=2, out-of-place | `sw/test/spatz_ops`, `sw/test/micro_yolo` |
| `NPU_OP_UPSAMPLE_NEAREST2X_I8` | Spatz C32 specialized or generic wrapper | i8 | Supported | Scale fixed to 2x in graph op; out-of-place | `sw/lib/spatz_ops.c`, `sw/test/spatz_ops` |
| `NPU_OP_DFL_SOFTMAX4_I8_Q8` | AFU fused DFL mode | input `ROW32` i8, output u16 Q8 | Supported | Expects 32 input lanes per location; uses channels 0..15 as 4 sides x 4 bins | `sw/test/afu_ops`, `sw/test/micro_yolo` |
| `NPU_OP_CLASS_SIGMOID_ROW32_HIGH16_I8` | AFU class sigmoid mode | input `ROW32` i8, output packed i8 | Supported | Expects 32 input lanes; applies LUT to high 16 lanes | `sw/test/afu_ops`, `sw/test/micro_yolo` |
| `NPU_OP_GLOBAL_AVGPOOL_C32_REDUCE` | AFU global avgpool mode | `C32_BLOCKED` i8 | Supported | Output must be 1x1 with same channel count; channels may have tail lanes | `sw/test/afu_ops`, `hw/rtl/cluster/tb/tests/test_spatz_operator_library.py` |
| `NPU_OP_IM2COL3X3S1P1_C3_PAD32`, `NPU_OP_IM2COL3X3S2P1_C3_PAD32` | CPU firmware helper | HWC RGB to ROW32 | Legacy / avoid for new optimized paths | Scalar prepare path, kept for old fixtures | `sw/lib/npu_graph.c`, `sw/test/conv_perf` |

## Common Graph ABI

Graph operators are described by `npu_layer_t` in `sw/lib/npu_graph.h`.
Tensor shape, dtype, layout, byte size, and TCDM/L2 address are described by
`npu_tensor_t` in `sw/lib/npu_tensor.h`.

Supported layouts:

| Layout | Meaning | Byte size helper | Typical users |
|---|---|---|---|
| `NPU_LAYOUT_HWC` | Pixel-major H/W/C tensor | `npu_tensor_hwc_bytes()` | RGB input/stem only |
| `NPU_LAYOUT_ROW32` | Row-major M rows x 32 lanes | `npu_tensor_row32_bytes()` or `npu_tensor_i32_row32_bytes()` | GEMM, linebuffer conv, raw YOLO head |
| `NPU_LAYOUT_C32_BLOCKED` | Channel group-major C32 blocks: `[c_group][pixel][lane]` | `npu_tensor_c32_bytes()` | Pointwise, depthwise, MobileNet-style tensors |

The graph runner validates tensor metadata before dispatching an operator. The
dispatch table lives in `sw/lib/npu_graph.c::npu_graph_run()`.

## Systolic Operators

### GEMM32

Graph ops:

- `NPU_OP_SYSTOLIC_GEMM32`
- `NPU_OP_SYSTOLIC_GEMM32_REQUANT`

Execution block:

- HAL: `sw/lib/hal_systolic.c`
- RTL controller: `hw/rtl/systolic/systolic_controller.sv`
- Register file and shadow config: `hw/rtl/systolic/systolic_ctrl_regs.sv`
- Array datapath: `hw/rtl/systolic/npu_systolic_array.sv`
- Requant datapath: `hw/rtl/systolic/requant_pipeline.sv`

Contract:

- Input activation is `M x 32` i8 in `ROW32`.
- Weight is a native `32 x 32` i8 tile in `ROW32`/K-major order.
- Non-requant output is `M x 32` i32 in `ROW32`.
- Requant output is `M x 32` i8 in `ROW32`.
- `layer.dim_m` is required and selects M.

Limits:

- K and N are fixed to 32 lanes per hardware start.
- Larger K is scheduled by software/HAL through base + accumulate starts.
- Per-channel requant is available in the HAL, but graph convenience ops
  currently use uniform `layer.multiplier`, `layer.shift`, `min_val`, and
  `max_val` across all 32 lanes.

Reference tests:

- `sw/test/independent_systolic`
- `sw/test/systolic_requant`
- `hw/rtl/systolic/tb/test_systolic_controller.py`

### Pointwise Conv1x1 C32 Requant

Graph op:

- `NPU_OP_CONV2D1X1_C32_REQUANT`

Execution block:

- Graph wrapper: `sw/lib/npu_graph.c::run_conv2d1x1_c32_requant()`
- HAL wrapper: `sw/lib/hal_systolic.c::systolic_pointwise1x1_c32_multi_requant()`
- RTL: same systolic controller/GEMM/requant path as GEMM32.

Contract:

- Source and destination are i8 `ROW32` or `C32_BLOCKED`.
- IC and OC must both be nonzero multiples of 32.
- Weight tensor is i8 `ROW32`, packed as:

```text
weight[oc_group][ic_group][k_lane][n_lane]
```

- For `IC_groups == 1`, the HAL directly runs GEMM32 + requant per OC group.
- For `IC_groups > 1`, the HAL uses one i32 psum scratch tile:

```text
first ICG  -> systolic_gemm32() into psum
middle ICG -> systolic_gemm32_accumulate() into psum
final ICG  -> systolic_gemm32_accumulate_requant() into i8 output
```

Limits:

- This is one graph operator, but not yet a literal single RTL `START` that
  loops over all IC/OC groups inside the controller.
- Multi-IC requires `layer.aux2` to point to a `ROW32` i32 psum tensor of at
  least `H * W * 32 * 4` bytes.
- Channel tails are not supported by this pointwise graph op; use C32-multiple
  tensors for the fast path.

How to use:

- Firmware example: `sw/test/pointwise_conv/main.c`
- Build variants: `sw/test/pointwise_conv/Makefile`
- Golden/test: `hw/rtl/cluster/tb/tests/test_pointwise_conv.py`

### Standard Conv3x3 Linebuffer

Graph ops:

- `NPU_OP_CONV2D3X3S2P1_C3_LINEBUF_REQUANT`
- `NPU_OP_CONV2D3X3S1P1_C32_LINEBUF`
- `NPU_OP_CONV2D3X3S1P1_C32_LINEBUF_REQUANT`
- `NPU_OP_CONV2D3X3S2P1_C32_LINEBUF_REQUANT`
- `NPU_OP_CONV2D3X3S1P1_C32X2_LINEBUF_REQUANT_L2`
- `NPU_OP_CONV2D3X3S1P1_C32_MULTI_LINEBUF_REQUANT`

Execution block:

- Graph wrappers: `sw/lib/npu_graph.c`
- Conv scheduler/HAL: `sw/lib/conv2d_packed.c`, `sw/lib/conv2d_packed.h`
- Descriptor generator support: `tools/npu_linebuf_precompute.py`
- RTL controller: `hw/rtl/systolic/systolic_controller.sv`
- Linebuffer: `hw/rtl/systolic/conv_linebuf_stream_packer.sv`
- Requant: `hw/rtl/systolic/requant_pipeline.sv`

Contract:

- RGB stem path consumes HWC input, 3 input channels, 32 output channels,
  K=3x3, stride=2, pad=1.
- C32 paths consume `ROW32` i8 input, 32 input channels, 32 output channels,
  K=3x3, pad=1, stride 1 or 2 depending on op.
- The multi-C32 path consumes `C32_BLOCKED` i8 tensors, loops OC32 x IC32
  chunks in graph runtime, writes the first IC chunk to i32 psum, accumulates
  intermediate chunks without requant, and enables requant only for the final
  IC chunk.
- Requant variants require `aux2` psum tensor in `ROW32` i32 layout.
- Descriptor-based runs can use `layer.linebuf_jobs` and
  `layer.linebuf_job_count` instead of graph-side fallback tiling.

Limits:

- Current graph ops are specialized 3x3/pad1 paths, not a generic Conv2D
  descriptor op.
- Legacy C32 graph ops have one C32 output group per graph op. The multi-C32
  graph op handles wider IC/OC channel counts internally, but still requires
  channel counts to be exact multiples of 32.
- Generic non-square kernels are covered in lower-level `conv_perf` regression,
  but not exposed as stable high-level graph ops.
- For best performance, the host/Python scheduler should emit C32-aligned
  linebuffer job descriptors. Firmware should dispatch descriptors, not
  recompute the schedule.

How to use:

- Micro-YOLO graph: `sw/test/micro_yolo/main.c`
- Micro-MobileNet graph: `sw/test/micro_mobilenet/main.c`
- Host descriptor generation: `tools/npu_linebuf_precompute.py`
- Linebuffer architecture details: `docs/linebuffer_architecture.md`
- Cluster regression: `hw/rtl/cluster/tb/tests/test_micro_yolo_e2e.py`
- MobileNet E2E regression: `hw/rtl/cluster/tb/tests/test_micro_mobilenet_e2e.py`
- Performance regression: `sw/test/conv_perf`, `hw/rtl/cluster/tb/tests/test_conv_perf.py`

### Depthwise Conv3x3 C32 Requant

Graph ops:

- `NPU_OP_DEPTHWISE3X3S1P1_C32_REQUANT`
- `NPU_OP_DEPTHWISE3X3S2P1_C32_REQUANT`

Execution block:

- Graph wrapper: `sw/lib/npu_graph.c::run_depthwise3x3_c32_requant()`
- HAL: `sw/lib/hal_systolic.c::systolic_depthwise3x3_c32_requant_channels()`
- RTL controller depthwise datapath: `hw/rtl/systolic/systolic_controller.sv`
- Linebuffer: `hw/rtl/systolic/conv_linebuf_stream_packer.sv`

Contract:

- Input/output are i8 `ROW32` or `C32_BLOCKED`.
- Channel count may be any positive value; groups are `ceil(C / 32)`.
- Weight is i8 `ROW32`, packed by C32 group, 3x3 taps, 32 lanes.
- Invalid final-group lanes are masked to zero after requant.

Limits:

- Kernel is fixed to 3x3 and pad is fixed to 1.
- Stride is fixed to 1 or 2 through the two graph op variants.
- Current datapath is one C32 tap vector per cycle; no multi-tap unroll yet.
- Requant config uses the same uniform graph fields for all lanes unless the
  HAL is extended for per-channel graph metadata.

How to use:

- Firmware examples and build variants: `sw/test/depthwise_conv`
- Golden/tests: `hw/rtl/cluster/tb/tests/test_depthwise_conv.py`
- MobileNet plan and expected use: `docs/micro_mobilenet_integration_test_plan.md`

## AFU Operators

AFU modes and register definitions are in `sw/lib/npu_memory_map.h`. The C HAL
is in `sw/lib/hal_afu.h`; operator wrappers are in `sw/lib/spatz_ops.c`.
The RTL is split across:

- `hw/rtl/afu/afu.sv`
- `hw/rtl/afu/afu_frontend.sv`
- `hw/rtl/afu/afu_core.sv`
- `hw/rtl/afu/afu_backend.sv`

### LUT E8 Operators: Logistic and Clamp

Graph ops:

- `NPU_OP_LOGISTIC_LUT_I8`
- `NPU_OP_CLAMP_I8`

Contract:

- Source/destination are i8 byte tensors.
- Logistic requires `aux` to contain a 256-entry u8 LUT.
- Clamp generates and loads a 256-entry LUT from `layer.min_val` and
  `layer.max_val`.
- Graph clamp is out-of-place; `src->addr == dst->addr` is rejected.

Limits:

- The C wrapper currently loads LUT entries through AFU MMIO before each op.
  Host/DMA LUT preload is possible but not the default wrapper contract yet.
- AFU config is written to shadow registers before the LUT fill loop and the
  op is launched with one `START` write after LUT contents are ready.

Reference:

- `sw/lib/spatz_ops.c::npu_logistic_i8()`
- `sw/lib/spatz_ops.c::npu_clamp_i8()`
- `sw/test/afu_ops`
- `hw/rtl/cluster/tb/tests/test_spatz_operator_library.py`

### Add and Mul

Graph ops:

- `NPU_OP_ADD_I8`
- `NPU_OP_MUL_I8`

Contract:

- Inputs and output are i8 byte tensors.
- Add graph path requires same H/W/C for `src`, `aux`, and `dst`.
- `MUL_I8` uses AFU `MUL_Q7` only when:

```text
multiplier = 1
shift      = 7
min_val    = -128
max_val    = 127
```

- `ADD_I8` uses AFU `ADD_I8` only when the clamp range is full i8.

Limits:

- Non-fast-path add/mul falls back to scalar/Spatz helper code in
  `sw/lib/spatz_ops.c`; avoid that in optimized model graphs.
- No in-place graph contract is guaranteed.

Reference:

- `sw/lib/spatz_ops.c::npu_mul_q7_i8()`
- `sw/lib/spatz_ops.c::npu_add_i8()`
- `sw/test/afu_ops`

### DFL Softmax4 Q8

Graph op:

- `NPU_OP_DFL_SOFTMAX4_I8_Q8`

Execution block:

- AFU fused DFL mode `NPU_AFU_MODE_DFL4_ROW32_Q8`.
- C wrapper: `sw/lib/spatz_ops.c::npu_dfl_softmax4_row32_i8_q8()`.

Contract:

- Source is `ROW32` i8, one row per location.
- Channels 0..15 encode four sides, four bins per side.
- Channels 16..31 are ignored by the DFL datapath.
- Output is `locations * 4` u16 values in Q8.
- `aux` holds the 256-entry exp LUT as u32 words.
- `aux2` holds the 256-entry reciprocal LUT as u32 words.

Limits:

- Reg-max is fixed to 4 bins per side.
- Output format is u16 Q8, not i8.
- LUTs must be available in memory; current wrapper writes them into AFU LUT
  windows through MMIO before dispatch.

Reference:

- `sw/test/afu_ops`
- `sw/test/micro_yolo/main.c`
- `hw/rtl/cluster/tb/tests/test_micro_yolo_e2e.py`

### Class Sigmoid Row32 High16

Graph op:

- `NPU_OP_CLASS_SIGMOID_ROW32_HIGH16_I8`

Contract:

- Source is `ROW32` i8.
- For each location, lanes 16..31 are transformed through a 256-entry u8 LUT.
- Output is a compact `locations * 16` i8 tensor.

Limits:

- Fixed to the high 16 lanes of a 32-lane raw-head row.
- The lower 16 lanes are not emitted.

Reference:

- `sw/lib/spatz_ops.c::npu_class_sigmoid_row32_high16_i8()`
- `sw/test/afu_ops`
- `sw/test/micro_yolo/main.c`

### GlobalAvgPool C32 Reduce

Graph op:

- `NPU_OP_GLOBAL_AVGPOOL_C32_REDUCE`

Contract:

- Source and destination are `C32_BLOCKED` i8.
- Destination must be 1x1 with the same logical channel count.
- AFU receives `spatial_count = H * W` through `SRC2_PTR`.

Limits:

- Intended for global average pooling only, not arbitrary pooling window sizes.
- Channels can be non-multiple-of-32 in metadata, but padded lanes exist in the
  C32 layout.

Reference:

- `sw/lib/spatz_ops.c::npu_global_avgpool_c32_i8()`
- `sw/test/afu_ops`
- `docs/micro_mobilenet_integration_test_plan.md`

## Pooling, Resize, and Concat Helpers

### MaxPool 5x5 S1 P2

Graph op:

- `NPU_OP_MAXPOOL2D5X5S1P2_I8`

Fast path:

- If C=32 and input/output are out-of-place, graph dispatches
  `systolic_maxpool5x5s1p2_c32_linebuf()`.
- Otherwise it falls back to `spatz_maxpool2d_i8()`.

Limits:

- Graph op shape is fixed to K=5x5, stride=1, pad=2.
- C32 linebuffer fast path does not support in-place source/destination.
- Generic fallback is functional but not the target high-performance path.

Reference:

- `sw/lib/npu_graph.c`
- `sw/lib/hal_systolic.c`
- `sw/lib/spatz_ops.c`
- `sw/test/spatz_ops`

### Nearest Upsample 2x

Graph op:

- `NPU_OP_UPSAMPLE_NEAREST2X_I8`

Contract:

- Source and destination are i8 tensors with same channel count.
- Destination H/W must be exactly 2x source H/W.
- Specialized C32 helper is selected when C=32 and tensors are out-of-place.

Limits:

- Graph op supports only 2x scale.
- Generic fallback is functional but not the preferred high-performance path.

Reference:

- `sw/lib/spatz_ops.c::spatz_upsample_nearest_i8()`
- `sw/test/spatz_ops`

### Logical C32 Concat Consumer

There is no standalone graph op for concat. The optimized YOLO path avoids
materializing concat by using:

- `NPU_OP_CONV2D3X3S1P1_C32X2_LINEBUF_REQUANT_L2`

This op computes:

```text
Conv([src0, src1], W) = Conv(src0, W0) + Conv(src1, W1)
```

and requants after accumulation.

Limits:

- Exactly two C32 sources are supported in this graph op.
- Output is one C32 group.
- The op is specialized for the YOLO head/concat consumer pattern, not a
  general N-way concat API.

Reference:

- `sw/test/micro_yolo/README.md`
- `sw/test/micro_yolo/main.c`

## Legacy / Avoid For New Optimized Graphs

The following graph ops remain available but should not be used for optimized
YOLO/MobileNet paths unless there is no native path yet:

- `NPU_OP_IM2COL3X3S1P1_C3_PAD32`
- `NPU_OP_IM2COL3X3S2P1_C3_PAD32`
- Scalar fallback inside `spatz_add_i8()` / `spatz_mul_i8()` for non-fast-path
  clamp/requant parameters.
- Generic `spatz_maxpool2d_i8()` and generic `spatz_upsample_nearest_i8()` when
  C32-specialized paths are available.

These paths are useful as bring-up references, but they add CPU setup cycles and
should not be part of performance-critical E2E graphs.

## Current Design Limits

1. Most high-performance tensor paths assume 32-lane channel groups.
2. Standard Conv graph ops are specialized for 3x3/pad1 and OC32 chunks.
3. Pointwise multi-C32 is implemented as one graph operator but still issues
   multiple systolic starts internally; there is no RTL single-start IC/OC loop
   yet.
4. Depthwise supports channel tails, but only for 3x3/pad1 stride 1 or 2.
5. Graph-level requant metadata is uniform across 32 lanes. The RTL supports
   per-channel tables; graph descriptors do not yet carry those tables per op.
6. AFU LUT wrappers currently load LUT registers through firmware MMIO. A host
   DMA-loaded LUT preload ABI can reduce setup cycles, but it is not the default
   graph wrapper contract yet.
7. In-place operation is generally not guaranteed. Several graph ops explicitly
   reject `src->addr == dst->addr`.
8. Host/Python is expected to own layout selection, C32 packing, linebuffer
   descriptor generation, and golden-result validation. Firmware should dispatch
   valid descriptors rather than perform expensive shape planning or output
   checking.

## Where To Add New Operators

When adding a new graph-level operator, update these locations:

1. `sw/lib/npu_graph.h`: add the `NPU_OP_*` enum value and fields needed in
   `npu_layer_t`.
2. `sw/lib/npu_graph.c`: add tensor validation and dispatch.
3. `sw/lib/hal_systolic.c` or `sw/lib/spatz_ops.c`: add the low-level wrapper.
4. RTL block, if needed:
   - Systolic/linebuffer: `hw/rtl/systolic/`
   - AFU: `hw/rtl/afu/`
   - Spatz/RVV integration: `hw/rtl/cluster/riscv_instr_npu.sv`
5. `sw/test/<operator>` or `sw/test/afu_ops`: add firmware fixture.
6. `hw/rtl/cluster/tb/tests/`: add Python golden checking. Firmware should
   report dispatch/control failure only; Python owns numerical correctness.
