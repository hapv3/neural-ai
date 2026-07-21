# Micro-LLM Integration Test Plan

## Objective

Prove that the Neural AI hardware can run the operator pattern that defines a
small decoder-only LLM block, not just isolated GEMM tests:

```text
token embedding -> RMSNorm -> QKV Linear -> RoPE -> causal attention
                -> output Linear -> residual
                -> RMSNorm -> MLP gate/up/down -> residual
                -> LM head
```

First end-to-end target: a compact LLaMA-style decoder block using fully
quantized INT8 activations and weights, with INT32 accumulation and explicit
requantization at operator boundaries.

Initial micro target:

| Parameter | Value | Reason |
|---|---:|---|
| Batch | 1 | Current graph/runtime tests are batch-1. |
| Sequence length `T` | 16 | Small enough for cluster simulation, large enough to test causal attention. |
| Hidden size `H` | 128 | Four C32 groups. |
| Attention heads | 4 | One 32-lane head per C32 group. |
| Head dimension | 32 | Exactly one systolic/ROW32 lane group. |
| MLP intermediate | 256 | Eight C32 groups. |
| Vocabulary | 256 | Eight C32 output groups for LM head. |
| Layers | 1 decoder block | First target is operator coverage and scheduler correctness. |

Current status: **planning only**. Existing Micro-YOLO and Micro-MobileNet
infrastructure already provides iDMA movement, graph tests, host-generated
fixtures, systolic GEMM32, AFU Add/Mul/LUT, and C32-blocked tensor utilities.
Micro-LLM requires new native support for row-wise normalization, general
attention softmax, RoPE, and KV-cache scheduling before it can be considered a
native accelerator graph.

The target implementation should stay on native accelerator paths. Scalar CPU
loops are allowed only for fixture setup, graph descriptor validation, or
golden-model checks. Scalar firmware loops are not acceptable as the planned
implementation for Linear, RMSNorm/LayerNorm, RoPE, attention softmax,
attention value matmul, MLP activation, residual Add, or LM head.

Planning rule: **functional milestones are for byte-exact coverage first; pure
performance work belongs to the optimization backlog.** Earlier milestones may
reuse existing native paths, but they should not add performance-only RTL unless
that RTL is required to avoid a scalar hot path.

---

## 1. Topology

The topology intentionally includes the operator families needed by a
LLaMA-style decoder block while keeping dimensions C32-aligned.

Logical tensor notation below uses `T x C` for token-major activations. The
physical implementation should prefer C32-blocked storage:

```text
activation[c_group][token][lane32]
```

This is the same physical idea as `NPU_LAYOUT_C32_BLOCKED`, with token index
serving the role of spatial pixel index. Attention score/probability rows use
`ROW32`, where sequence length is padded to 32 lanes.

| # | Layer | Op | Config | Output logical shape | Hardware Unit |
|---|---|---|---|---|---|
| 0 | InputTokens | Token IDs | `T=16` uint16/u32 ids | `16` | L2 fixture |
| 1 | Embed | Embedding gather | `vocab=256, H=128` | `16x128` INT8 | Spatz/iDMA gather or host-preloaded activation for first bring-up |
| 2 | RMSNorm0 | **RMSNorm** | per token over `H=128` | `16x128` INT8 | new AFU row-wise reduce/scale mode |
| 3 | QKV_Linear | **Linear** | `H=128 -> 3H=384` | `16x384` INT8 | systolic GEMM32 multi-IC/multi-OC + requant |
| 4 | Split_QKV | **Split/View** | Q, K, V each `128` | `3 x 16x128` views | host/Python descriptor view |
| 5 | RoPE_QK | **RoPE** | 4 heads, `head_dim=32` | Q/K `16x128` INT8 | new AFU or Spatz C32 pair-rotate kernel |
| 6 | KV_Write | **KV cache write** | append K/V for 16 tokens | cache `[layer][head][T][32]` | iDMA/TCDM/L2 descriptor copy |
| 7 | QK_Scores | **Attention score GEMM** | per head `Q[T,32] x K[T,32]^T` | `4 x 16x32` INT8/INT16 score rows | systolic GEMM32, N padded to 32 |
| 8 | CausalMask | **Mask/View/Add** | upper triangle masked | `4 x 16x32` | fused into softmax input handling |
| 9 | AttnSoftmax | **Softmax** | row-wise over valid causal positions | `4 x 16x32` INT8 probs | new AFU row softmax mode |
| 10 | PV_Context | **Attention value GEMM** | per head `P[T,T] x V[T,32]` | `16x128` INT8 | systolic GEMM32 or fused attention value kernel |
| 11 | Out_Linear | **Linear** | `H=128 -> H=128` | `16x128` INT8 | systolic GEMM32 + requant |
| 12 | Residual0_Add | **Add** | Embed + attention output | `16x128` INT8 | AFU Add |
| 13 | RMSNorm1 | **RMSNorm** | per token over `H=128` | `16x128` INT8 | new AFU row-wise reduce/scale mode |
| 14 | MLP_GateUp | **Linear** | `H=128 -> 2I=512` | gate/up `16x256` each | systolic GEMM32 + requant |
| 15 | SiLU_Gate | **SiLU / sigmoid+mul** | gate activation | `16x256` INT8 | AFU LUT + AFU Mul, or fused AFU mode |
| 16 | SwiGLU_Mul | **Mul** | `SiLU(gate) * up` | `16x256` INT8 | AFU Mul_Q7 |
| 17 | MLP_Down | **Linear** | `I=256 -> H=128` | `16x128` INT8 | systolic GEMM32 + requant |
| 18 | Residual1_Add | **Add** | Residual0 + MLP output | `16x128` INT8 | AFU Add |
| 19 | LM_Head | **Linear** | `H=128 -> vocab=256` | `16x256` logits | systolic GEMM32 + requant or INT32 logits |
| 20 | Top1 | **Argmax / TopK1** | next-token candidate | `16` token ids | Spatz reduction or optional AFU mode |
| 21 | Output_DMA | **DMA_OUT** | final logits and/or top1 ids | test output | iDMA |

Coverage counts:

| Operator class | Layers | Count |
|---|---|---:|
| Linear / GEMM | `QKV_Linear`, `Out_Linear`, `MLP_GateUp`, `MLP_Down`, `LM_Head` | 5 |
| RMSNorm | `RMSNorm0`, `RMSNorm1` | 2 |
| RoPE | `RoPE_QK` | 1 |
| Attention score/value GEMM | `QK_Scores`, `PV_Context` | 2 |
| Softmax | `AttnSoftmax` | 1 |
| Element-wise Add | `Residual0_Add`, `Residual1_Add` | 2 |
| Element-wise Mul / SiLU | `SiLU_Gate`, `SwiGLU_Mul` | 2 |
| Shape/view ops | `Split_QKV`, causal valid-range views | >= 2 |
| KV cache movement | `KV_Write`, later decode KV read | >= 1 |
| DMA_IN / DMA_OUT | weights, input activation, output logits | >= 1 |

The first E2E graph may start from a host-preloaded embedding activation to
focus on transformer compute. A later milestone adds native embedding gather.
This keeps the first functional target from being blocked by a gather operator
that is not representative of the main LLM compute path.

---

## 2. Native Datapath Assessment

### 2.1 Reuse Existing RTL/Accelerator Paths

| Op | Reuse path | Required planner/SW work | Notes |
|---|---|---|---|
| Linear `M x K` by `K x N` | Existing `SYSTOLIC_GEMM32` / `SYSTOLIC_GEMM32_REQUANT` | Add graph/HAL wrapper for generic `LINEAR_C32_REQUANT`; pack weights as `OCG -> ICG -> 32x32`; loop IC/OC C32 groups like pointwise Conv1x1 | Main LLM path. No linebuffer, no im2col. |
| QKV fused projection | Same systolic Linear wrapper | Treat as one Linear with `N=3H=384`; output can be split by C32 group views | Avoid three separate graph ops if possible. |
| Output projection / MLP / LM head | Same systolic Linear wrapper | Use shared psum scratch for `IC_groups > 1`; support optional INT32 logits for LM head comparison | Compute-dense and C32-aligned. |
| Residual Add | AFU `ADD_I8` | Reuse `npu_add_i8()` fast path with full i8 clamp | Already implemented for same-shape tensors. |
| SiLU/SwiGLU pieces | AFU E8 LUT logistic + AFU `MUL_Q7` | Reuse LUT sigmoid and Q7 multiply for first functional path; a fused SiLU mode is optimization | Current AFU LUT reload overhead should be measured but does not block functionality. |
| Shape-only split/reshape | Host/Python tensor descriptor views | Emit no firmware op for Q/K/V split, head split, reshape, and flatten when byte order is unchanged | Same policy as YOLO/MobileNet docs. |
| DMA movement | iDMA 1D/2D/3D helpers | Load weights/LUTs/KV cache slices from L2 into TCDM scratch; write results back | Needed for large weights and cache tensors. |

### 2.2 Required Native Support

| Item | Required block | Why existing path is insufficient | Category |
|---|---|---|---|
| Row-wise RMSNorm | New AFU mode or Spatz+AFU reduce pipeline | Current `GLOBAL_AVGPOOL_C32_REDUCE` reduces spatial positions per channel; RMSNorm reduces hidden lanes per token and needs sum-of-squares + reciprocal sqrt + scale | Functional |
| General row Softmax | New AFU mode | Current DFL softmax is fixed to four bins per side and ROW32 low16 semantics; attention needs variable valid length, causal mask, max/sub/exp/sum/reciprocal over up to 32 or more positions | Functional |
| RoPE | New AFU mode or Spatz C32 specialized kernel | RoPE is pairwise rotate using sin/cos; implementing with generic Mul/Add would require multiple full tensor passes | Functional for LLaMA-style block |
| KV cache layout and scheduler | Host/Python descriptor + firmware dispatch | K/V tensors are dynamic, per-layer, per-head, and sequence-indexed; they need predictable C32/ROW32 slices for QK/PV GEMM | Functional |
| Embedding gather | Spatz gather or host-preloaded activation for first E2E | iDMA can copy regular ranges, but token gather is indexed; first E2E may bypass this with preloaded embeddings | Functional, but can be deferred |
| Argmax/TopK1 | Spatz reduction or small AFU mode | Final token selection is a reduction across vocab logits; not currently a graph op | Optional for first E2E if final logits are compared directly |

### 2.3 Quantization Contract

The first Micro-LLM target should use static quantization to avoid adding
dynamic scale estimation before the graph is stable.

Contract:

- activations and weights are INT8;
- systolic accumulation is INT32;
- Linear outputs are requantized to INT8 unless the consumer explicitly expects
  score/logit precision;
- RMSNorm, Softmax, RoPE, and SiLU use host-generated LUTs and fixed-point
  reference math;
- qparams are generated by the Python test/golden and carried in descriptors;
- per-tensor qparams are allowed for the first graph;
- per-channel/per-group qparams are an optimization/accuracy extension.

The Python golden must own the exact fixed-point formulas. Firmware should only
dispatch descriptors and report control failures; cocotb owns byte/word result
checking.

---

## 3. Operator Requirements

### 3.1 Linear / GEMM

Requirement: at least five Linear instances.

Planned instances:

1. `QKV_Linear`: `128 -> 384`, validates large OC fanout and output view split.
2. `Out_Linear`: `128 -> 128`, validates attention projection.
3. `MLP_GateUp`: `128 -> 512`, validates fused gate/up projection.
4. `MLP_Down`: `256 -> 128`, validates larger IC accumulation.
5. `LM_Head`: `128 -> 256`, validates vocab projection.

Implementation expectation:

- create a generic graph op such as `LINEAR_C32_REQUANT`;
- use the existing systolic direct GEMM32 path;
- avoid linebuffer/window states;
- keep input/output tensors C32-blocked by token;
- for `IC_groups > 1`, use one reusable INT32 psum tile;
- pack weights as:

```text
weight[oc_group][ic_group][k_lane][n_lane]
```

Useful systolic vector-cycle estimate for the first target:

| Linear | M | IC groups | OC groups | Useful vector cycles |
|---|---:|---:|---:|---:|
| QKV `128->384` | 16 | 4 | 12 | 768 |
| Out `128->128` | 16 | 4 | 4 | 256 |
| MLP gate/up `128->512` | 16 | 4 | 16 | 1,024 |
| MLP down `256->128` | 16 | 8 | 4 | 512 |
| LM head `128->256` | 16 | 4 | 8 | 512 |

These are lower-bound compute cycles only. Current software-controlled
multi-IC/multi-OC starts, psum traffic, DMA, and AFU phases will add overhead.

### 3.2 RMSNorm

Requirement: at least two RMSNorm instances.

RMSNorm per token:

```text
rms = sqrt(mean(x_i^2) + eps)
y_i = x_i * gamma_i / rms
```

Native datapath target:

```text
for each token:
  pass A: read C32 groups, accumulate sumsq in int32/int48
  compute reciprocal sqrt using LUT/Newton step
  pass B: read C32 groups again, multiply by scale/gamma, requant to INT8
```

First hardware option:

- AFU `RMSNORM_ROW32_I8` mode;
- one primary read port for input;
- optional second read port for gamma;
- output write through AFU backend;
- row descriptor fields: token count, hidden groups, epsilon, input scale,
  gamma pointer, output scale, clamp.

This is a functional blocker for LLaMA-style graphs. A scalar firmware RMSNorm
would dominate small simulations and should not be used as the planned path.

### 3.3 RoPE

Requirement: one RoPE application to Q and K.

RoPE rotates each even/odd pair in every head:

```text
q_even' = q_even * cos - q_odd * sin
q_odd'  = q_even * sin + q_odd * cos
```

Native datapath target:

- consume C32 head vectors;
- process 16 lane pairs per C32 row;
- read sin/cos from host-generated LUT/table indexed by token and pair;
- write rotated Q/K out-of-place or in-place only if the backend safely supports
  read-before-write.

First implementation can be a Spatz C32-specialized kernel if that is faster to
bring up, but a fused AFU mode is preferred for predictable throughput and
reuse of the existing fixed-point multiplier pipeline.

### 3.4 Attention Score GEMM

Requirement: compute causal attention scores for 4 heads.

For prefill:

```text
scores[h] = Q[h][T, 32] x K[h][T, 32]^T
```

For the first target, `T=16` is padded to one 32-lane ROW32 score row. Each
head can map to one GEMM32 output group:

```text
M = T
K = head_dim = 32
N = padded_seq = 32
```

The QK path may output INT8 scores after requant or a wider temporary format if
softmax accuracy requires it. The first milestone should choose one fixed
contract and make the Python golden match it bit-exactly.

### 3.5 Attention Softmax

Requirement: one general causal row softmax.

Softmax per attention row:

```text
masked_scores = scores + causal_mask
row_max       = max(masked_scores)
exp_i         = exp(masked_scores_i - row_max)
sum_exp       = sum(exp_i)
prob_i        = exp_i / sum_exp
```

Native datapath target:

- AFU `SOFTMAX_ROW_I8` or `ATTN_SOFTMAX_ROW32` mode;
- valid length per row for causal masking;
- max-reduce, exp LUT, sum-reduce, reciprocal LUT/Newton correction;
- output INT8 probabilities in ROW32.

Current DFL softmax cannot be reused directly because it is specialized to
four-bin groups and ignores most lanes. The exp and reciprocal LUT ideas can be
reused, but the control flow must be generalized to row length and valid mask.

### 3.6 Attention Value GEMM

Requirement: multiply attention probabilities by V cache:

```text
context[h] = probs[h][T, padded_T] x V[h][T, 32]
```

For the first target:

```text
M = T
K = padded_seq = 32
N = head_dim = 32
```

This can use the systolic GEMM32 path if V cache is presented as the dynamic
weight tile. The scheduler must make K/V cache slices available in TCDM and
pack them in the native K-major/N-lane order expected by GEMM32.

### 3.7 MLP Activation

Requirement: SiLU/SwiGLU path.

Planned first path:

```text
gate_sigmoid = LOGISTIC_LUT_I8(gate)
silu_gate    = MUL_I8(gate, gate_sigmoid)
mlp_mid      = MUL_I8(silu_gate, up)
```

This reuses AFU LUT and AFU Mul_Q7. It costs multiple tensor passes but is
acceptable for the first functional graph. A fused `SILU_I8` or `SWIGLU_I8`
AFU mode belongs to the optimization backlog.

### 3.8 Embedding and LM Head

Embedding gather is not representative of the main accelerator compute but is
needed for a full token-to-token LLM flow.

Bring-up policy:

1. First E2E graph may start from host-preloaded embedded activations.
2. Add `EMBEDDING_GATHER_I8` after the core block is stable.
3. Compare LM head logits byte-exactly before adding TopK/Argmax.
4. Add `ARGMAX_I8` or `TOPK1_I8` only after logits match.

---

## 4. Implementation Roadmap

### Milestone 1: Golden, Layout, and Fixtures

Objective: define the exact graph and fixed-point math before RTL/SW changes.

Tasks:

| Step | Task |
|---|---|
| 1a | Create Python golden for one decoder block with static INT8 qparams |
| 1b | Define C32-blocked token activation layout and ROW32 attention score layout |
| 1c | Generate deterministic input tokens, embeddings, weights, RMSNorm gamma, RoPE sin/cos, and LUTs |
| 1d | Emit layer descriptors and tensor metadata in the same style as Micro-YOLO/MobileNet |
| 1e | Add coverage assertions: at least five Linear ops, two RMSNorm ops, one Softmax, one RoPE, two residual Adds |

Acceptance: Python golden runs standalone and emits every intermediate tensor
listed in Section 1.

### Milestone 2: Generic Linear C32 Path

Objective: support LLM Linear layers through systolic GEMM32 without linebuffer
or scalar prepare.

Tasks:

| Step | Task |
|---|---|
| 2a | Add graph op `LINEAR_C32_REQUANT` or reuse a generic GEMM descriptor op |
| 2b | Add HAL wrapper for `M x K` by `K x N` where `K` and `N` are multiples of 32 |
| 2c | Reuse INT32 psum scratch for multi-IC-group accumulation |
| 2d | Add unit tests for `M=1`, `M=16`, `M=17`, `K=32/128/256`, `N=32/128/384/512` |
| 2e | Cover QKV packed output and view split without materializing copies |

Acceptance: all Linear unit tests match Python golden byte-exactly. No im2col
or scalar matrix multiply appears in the hot path.

### Milestone 3: RMSNorm Native Path

Objective: implement row-wise RMSNorm without scalar firmware reduction.

Tasks:

| Step | Task |
|---|---|
| 3a | Add block-level Python golden for RMSNorm fixed-point math |
| 3b | Add AFU `RMSNORM_ROW32_I8` mode or equivalent native vector reduce path |
| 3c | Support hidden groups `1, 2, 4, 8` and token counts `1, 16, 17` |
| 3d | Load gamma from TCDM/L2 in C32 groups; support gamma all-ones shortcut only as an optional mode |
| 3e | Add block-level AFU test first, then a small cluster wrapper test |

Acceptance: RMSNorm outputs match Python golden byte-exactly for the target
`T=16,H=128` and for edge cases `M=1` decode and `H=32`.

### Milestone 4: RoPE and KV Cache Layout

Objective: rotate Q/K vectors and store K/V in a layout that attention GEMMs can
consume efficiently.

Tasks:

| Step | Task |
|---|---|
| 4a | Define K/V cache physical layout: `layer, head, seq_block, token, lane32` |
| 4b | Define RoPE sin/cos table format generated by Python host |
| 4c | Add native `ROPE_ROW32_I8` AFU/Spatz path for Q and K |
| 4d | Add KV write descriptor and decode read descriptor |
| 4e | Add unit tests for prefill `T=16` and decode append/read for one token |

Acceptance: rotated Q/K and K/V cache contents match Python golden. Decode can
read a previously written cache entry without repacking on CPU.

### Milestone 5: Attention Softmax and Value Path

Objective: run causal attention using native GEMM + native softmax.

Tasks:

| Step | Task |
|---|---|
| 5a | Add QK score GEMM test for `heads=4,T=16,head_dim=32` |
| 5b | Add AFU general row softmax with causal valid length |
| 5c | Add softmax block-level tests for valid lengths `1..16` and padded length 32 |
| 5d | Add PV value GEMM test using V cache as dynamic weight tiles |
| 5e | Add a fused attention subgraph test: Q/K/V -> scores -> softmax -> context |

Acceptance: attention context matches Python golden byte-exactly for prefill
`T=16` and decode `T_cache=16, query_tokens=1`.

### Milestone 6: MLP and Residual Subgraph

Objective: wire the feed-forward block using native Linear, AFU logistic, AFU
Mul, and AFU Add.

Tasks:

| Step | Task |
|---|---|
| 6a | Wire `MLP_GateUp` as one Linear output with two C32-group views |
| 6b | Reuse AFU logistic + Mul for SiLU gate |
| 6c | Reuse AFU Mul_Q7 for SwiGLU multiply |
| 6d | Wire `MLP_Down` Linear |
| 6e | Wire residual Add and compare full MLP subgraph |

Acceptance: `RMSNorm1 -> MLP -> Residual1_Add` matches Python golden and uses
no scalar activation/reduction path.

### Milestone 7: LM Head and Optional Token Selection

Objective: produce next-token logits and optionally top-1 ids.

Tasks:

| Step | Task |
|---|---|
| 7a | Wire LM head as `LINEAR_C32_REQUANT` or INT32-output GEMM if logits need wider precision |
| 7b | Compare final logits byte/word-exactly against Python golden |
| 7c | Add `ARGMAX_I8` / `TOPK1_I8` using Spatz reduction or AFU mode only after logits pass |
| 7d | Add final output DMA for logits and token ids |

Acceptance: final logits match golden. Top1 token id is optional for the first
E2E pass but required for a complete autoregressive demo.

### Milestone 8: Micro-LLM E2E

Objective: run the complete topology through one prefill pass and one decode
step.

Tasks:

| Step | Task |
|---|---|
| 8a | Add `sw/test/micro_llm` firmware entrypoint |
| 8b | Add `hw/rtl/cluster/tb/tests/test_micro_llm_e2e.py` |
| 8c | Load host-generated descriptor blobs from L2, matching current runtime descriptor flow |
| 8d | Run prefill `T=16` through the full decoder block |
| 8e | Run decode `M=1` using K/V cache produced by prefill |
| 8f | Compare selected intermediate checkpoints plus final logits |
| 8g | Print PMU per layer and aggregate by Linear, RMSNorm, RoPE, Softmax, AFU element-wise, KV/DMA |

Acceptance: prefill and decode outputs match Python golden with zero mismatch.
No E2E operator uses scalar firmware as its planned compute path.

---

## 5. Verification Strategy

1. **Golden Model**
   - Deterministic Python INT8/fixed-point decoder block.
   - Exact LUT contents for sigmoid, exp, reciprocal, reciprocal sqrt, and RoPE.
   - Golden owns all qparams and descriptor generation.

2. **Operator Unit Tests**
   - Linear: C32-aligned `M/K/N` matrix shapes, including `M=1` decode.
   - RMSNorm: hidden groups `1/2/4/8`, token counts `1/16/17`.
   - RoPE: token positions, head count, pair rotation boundaries.
   - Softmax: valid lengths `1..16`, causal mask, padded lanes forced to zero.
   - Attention: QK and PV GEMM separately before fused attention subgraph.
   - MLP: SiLU/SwiGLU and residual Add.

3. **Block-Level RTL Tests First**
   - AFU RMSNorm, RoPE, and Softmax should have block-level tests before cluster
     firmware tests because cluster simulation with full trace is slow.
   - Cluster tests should be tiny until the block-level datapath is stable.

4. **E2E Cluster Simulation**
   - Cocotb loads firmware and L2 fixtures.
   - Firmware dispatches descriptors only.
   - Cocotb checks final outputs and selected intermediate tensors.
   - PMU is printed per layer and grouped by operator type.

5. **Coverage Rules**
   - At least five Linear layers execute.
   - RMSNorm executes at least twice.
   - RoPE executes on both Q and K.
   - Softmax executes with at least one row for every valid length `1..16`.
   - Both prefill `M=T` and decode `M=1` paths execute.
   - KV cache write and read both execute.
   - No hot-path scalar fallback is accepted for RMSNorm, Softmax, RoPE, or Linear.

---

## 6. Performance Notes

These notes are inputs to the optimization backlog only. They should not block
functional operator coverage or E2E correctness.

### 6.1 Expected Bottlenecks

The first Micro-LLM graph is small, so useful systolic compute is modest. This
means config overhead, psum traffic, AFU passes, and DMA/KV movement may be a
larger fraction of total cycles than in larger CNN layers.

Expected bottlenecks:

| Area | Risk |
|---|---|
| Decode `M=1` Linear | Systolic array is under-utilized when processing one token. |
| RMSNorm | Requires row-wise reductions and reciprocal sqrt; scalar fallback would dominate. |
| Softmax | Requires max/exp/sum/reciprocal per row; current DFL path is too specialized. |
| RoPE | Generic Mul/Add lowering would use too many memory passes. |
| KV cache | Dynamic K/V movement can dominate for longer context lengths. |
| LM head | Vocab projection becomes dominant as vocabulary grows beyond the micro target. |

### 6.2 Optimization Backlog

Entry criteria:

- Micro-LLM E2E passes byte-exactly.
- PMU is printed per layer and grouped by operator type.
- No hot-path scalar operator remains in the E2E graph.

Optimization backlog:

| Priority | Item | Scope | Expected benefit / decision rule |
|---|---|---|---|
| 9a | PMU attribution baseline | Add detailed PMU grouping for Linear, RMSNorm, RoPE, QK, Softmax, PV, MLP activation, KV/DMA, config overhead | Required before optimizing. |
| 9b | Linear single-start IC/OC loop | Controller/HAL descriptor mode that loops all IC/OC groups internally | Reduces config/start overhead for QKV, MLP, and LM head. |
| 9c | Batched decode / multi-token scheduling | Group several decode tokens or requests so `M > 1` | Improves systolic utilization for autoregressive inference. |
| 9d | Fused QKV Linear output split | One systolic Linear op writes Q/K/V views directly without materialized split copies | Avoids extra tensor movement. |
| 9e | Fuse RoPE into Q/K projection drain | Apply pair rotation while Q/K vectors are produced or before cache write | Removes separate RoPE read/write pass. |
| 9f | Fused attention softmax + PV | Stream probabilities into value GEMM without writing full probability tensor | Reduces score/probability memory traffic. |
| 9g | KV cache DMA prefetch/double buffer | Prefetch K/V cache blocks from L2 while previous head computes | Needed for longer context lengths. |
| 9h | Fused SiLU/SwiGLU AFU mode | Combine sigmoid and two multiplies for MLP gate/up | Removes multiple full tensor passes. |
| 9i | Per-channel/per-group qparams | Extend graph metadata and HAL for per-group Linear/RMSNorm qparams | Accuracy improvement for larger models. |
| 9j | TopK sampling support | Add top-k/top-p/temperature only after top1 is stable | Needed for real text generation, not first correctness target. |

### 6.3 Scaling Notes

The micro target is intentionally C32-aligned. Larger LLMs remain structurally
similar but stress different bottlenecks:

- larger hidden sizes improve systolic utilization but increase weight DMA;
- longer context lengths stress KV cache bandwidth and softmax;
- larger vocab sizes make the LM head expensive;
- decode `M=1` remains inefficient unless requests/tokens are batched or the
  controller gains a GEMV-specialized path.

The first implementation should therefore optimize for correctness and
operator coverage first, then use PMU to decide whether Linear scheduling,
Softmax/RMSNorm AFU throughput, or KV-cache movement is the real limiter.

---

## 7. Source Touchpoints

Expected files to add or update during implementation:

| Area | Source files |
|---|---|
| Graph ABI / op enum | `sw/lib/npu_graph.h`, `sw/lib/npu_graph.c` |
| Tensor/layout helpers | `sw/lib/npu_tensor.h`, `sw/lib/npu_tensor.c` |
| Systolic Linear HAL | `sw/lib/hal_systolic.c`, `sw/lib/hal_systolic.h` |
| AFU wrappers | `sw/lib/spatz_ops.c`, `sw/lib/hal_afu.h`, `sw/lib/npu_memory_map.h` |
| AFU RTL | `hw/rtl/afu/afu_core.sv`, `hw/rtl/afu/afu_frontend.sv`, `hw/rtl/afu/afu_backend.sv` |
| Systolic RTL | `hw/rtl/systolic/systolic_controller.sv`, only if generic Linear needs controller changes |
| Firmware fixture | `sw/test/micro_llm/` |
| Cocotb E2E | `hw/rtl/cluster/tb/tests/test_micro_llm_e2e.py` |
| Block-level AFU tests | `hw/rtl/afu/tb/` or current AFU test location |
| Operator docs | `docs/operator_support_matrix.md`, `docs/afu_architecture.md` |

Documentation updates should follow the implementation:

- update `docs/operator_support_matrix.md` when each new graph op becomes stable;
- update `docs/afu_architecture.md` for RMSNorm, Softmax, and RoPE modes;
- keep this plan as the milestone/status tracker, like the Micro-MobileNet plan.
