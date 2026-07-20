# NPU Cluster Operator Gap Analysis for YOLO / CNN / Vision Transformer

**Date:** 2026-06-20
**Target Models:** YOLOv5/v8/v11, general CNNs (ResNet, MobileNet), Vision Transformer (ViT, DeiT)
**Current Hardware:** Snitch (RV32IMAC) + Spatz (RVV INT-only, 2 IPU) + Systolic Array (32x32 INT8) + DMA

---

## 1. Hardware That Can Already Handle Key Operators

| Operator | Models | Hardware path | Notes |
|----------|--------|---------------|-------|
| Dense Conv2D | YOLO, CNN, ViT | **Systolic Array** (im2col + GEMM) | Core INT8 workload |
| Fully Connected / Linear | All | **Systolic Array** (MatMul) | Supported |
| MatMul (Q x K^T, Attn x V) | ViT | **Systolic Array** | Supported |
| Element-wise Add | YOLO (skip), ViT (residual) | **Spatz** (`vadd.vv`) | Supported |
| Element-wise Mul | Requant, fused BN | **Spatz** (`vmul.vv`) | Supported |
| MaxPool | YOLO, CNN | **Spatz** (`vmax.vv` + slide) | Supported, slower than a dedicated engine |
| Concat | YOLO (C2f, neck) | **DMA** (data copy) | Pure data movement |
| Shift/Clamp (Requantization) | All INT8 pipelines | **Spatz** (`vsra` + `vmax/vmin`) | Supported |
| ReLU / Clip | CNN | **Spatz** (`vmax.vx` with x=0) | Trivial |
| BatchNorm (fused into Conv) | CNN, YOLO | **Spatz** (mul + add per-channel) | Typically fused at compile time |
| Depthwise Conv | YOLO (C2f), MobileNet | **Spatz** (element-wise loop) | Functional but slow; does not exploit systolic compute |
| Upsample (Nearest Neighbor) | YOLO (FPN/PAN neck) | **Spatz** or **DMA** | Duplicate pixels |
| Transpose / Reshape | ViT (multi-head reshape) | **DMA** + **Spatz** | Data movement |

---

## 2. Missing or Inefficient Hardware Paths

### 2.1. Activation Function Unit (AFU): Highest Priority

**Problem:** Spatz is integer-only and has no FPU, so it cannot efficiently compute nonlinear functions such as `exp()`, `tanh()`, `1/x`, or `sqrt()`.

| Function | Formula | Models | Importance |
|----------|---------|--------|------------|
| **SiLU / Swish** | x x sigma(x) = x x 1/(1+exp(-x)) | YOLOv5/v8/v11 Conv blocks | Critical |
| **Sigmoid** | 1 / (1 + exp(-x)) | YOLO detection head output | Critical |
| **Softmax** | exp(x_i) / sum(exp(x_j)) | ViT attention layers | Critical for ViT |
| **GELU** | 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))) | ViT MLP blocks | Critical for ViT |
| **LayerNorm** | (x - mean) / sqrt(var + eps) | ViT layers | Needs sqrt + div |
| **Tanh** | (exp(x) - exp(-x)) / (exp(x) + exp(-x)) | Some CNNs | Less common |

This is the largest operator gap. Without AFU support, YOLO and ViT cannot run efficiently on the hardware.

#### Common NPU Approaches

| Approach | Description | Advantages | Disadvantages |
|----------|-------------|------------|---------------|
| **LUT-based** | SRAM lookup table with linear interpolation between points | Simplest, deterministic, very small area | Precision limited by table size; 256-1024 entries is usually enough for INT8 |
| **Piecewise Linear (PWL)** | Split input range into N segments, each modeled as y = ax + b | More accurate than LUT and still compact | Needs multiplier, adder, and segment lookup |
| **Polynomial Approximation** | Taylor/Chebyshev polynomial of degree 2-3 | Very accurate | Needs multiple multipliers and higher latency |
| **CORDIC** | Iterative algorithm for sin/cos/exp/sqrt | General purpose and high precision | Large area, many cycles, overkill for INT8 |
| **Snitch firmware** | Scalar C code computes each element | No new hardware | Extremely slow and not scalable |

#### Recommendation for This NPU

Use **LUT + linear interpolation** because:

- The target is INT8 inference, so input/output only need 8-bit precision.
- LUT 256 entries x 8-bit = 256 bytes per function, which fits in a small local table.
- Throughput of 1 element/cycle is realistic, with multiple lanes possible through pipelining.
- Any function can be supported by changing LUT contents.

#### Suggested AFU Architecture

```text
                    +--------------------------+
  TCDM ------------>|   Activation Function    |------------> TCDM
  (input buffer)    |        Unit (AFU)        |   (output buffer)
                    |                          |
                    |  +-----+   +----------+  |
                    |  | LUT |-->| Interp.  |  |
                    |  |256x8|   | a*x + b  |  |
                    |  +-----+   +----------+  |
                    |                          |
                    |  Config: func_select,    |
                    |          src_ptr,         |
                    |          dst_ptr,         |
                    |          length           |
                    +--------------------------+
```

- Firmware (Snitch) configures function type, source pointer, destination pointer, and length.
- AFU autonomously reads data from TCDM, performs LUT lookup, and writes results back to TCDM.
- Add one TCDM-interconnect master, or multiplex with an existing compatible master.

---

### 2.2. Requantization Pipeline Fused into Systolic Output

**Problem:** After each Conv/FC layer (`INT8 x INT8 = INT32 accumulator`), the output must be rescaled to INT8 for the next layer:

```text
output_int8 = clamp((acc_int32 x scale) >> shift + zero_point, 0, 255)
```

The older path writes raw INT32 accumulators to TCDM, then Spatz reads them back, requantizes, and writes the INT8 result. This:

- Uses 2x TCDM bandwidth (read INT32 + write INT8).
- Adds latency from firmware dispatch and Spatz processing.
- Uses 4x storage compared with INT8.

**Solution:** Add a post-processing pipeline to `systolic_controller.sv`:

```text
Systolic Array INT32 output
         |
         v
    +---------+
    | x scale |  (per-channel scale, loaded from TCDM)
    | >> shift|  (per-channel shift)
    | + zp    |  (per-channel zero point)
    | clamp   |  (saturate to [0, 255])
    +---------+
         |
         v
    INT8 output -> write TCDM (1/4 bandwidth versus INT32)
```

**Impact:** Reduces TCDM write bandwidth by 4x and significantly improves throughput.

**Priority:** Important, but Spatz can serve as a temporary fallback.

---

### 2.3. Dedicated Pooling Engine (Optional)

MaxPool 2x2 / 3x3 is common in older YOLO/CNN models. Spatz can implement it, but throughput is low because it must:

1. Load multiple rows from TCDM.
2. Vector compare and select.
3. Store strided output.

A dedicated pooling engine would be faster, but:

- Recent YOLO variants replace MaxPool with stride-2 Conv.
- MobileNetV2+ uses stride-2 Depthwise Conv instead of pooling.
- ViT does not use pooling except the final Global Average Pool in some variants.

**Priority:** Nice-to-have. Spatz can handle it functionally, so it does not block inference.

---

## 3. Operator Coverage Map

```text
                          Systolic   Spatz    DMA    Missing
                          (MatMul)   (RVV)   (Xfer)  (Need HW)
-----------------------------------------------------------------
Conv2D (dense)              ######
Depthwise Conv                       ####
Fully Connected             ######
MatMul (QxK^T, AxV)         ######
-----------------------------------------------------------------
SiLU / Swish                                          ###### AFU
Sigmoid                                               ###### AFU
GELU                                                  ###### AFU
Softmax                                               ###### AFU
LayerNorm                            ##               #### AFU
-----------------------------------------------------------------
BatchNorm (fused)                    ####
ReLU / Clip                          ######
MaxPool                              ####
AvgPool / GAP                        ####
Add (residual)                       ######
-----------------------------------------------------------------
Requantization                       ####              ## (fuse)
Concat                                        ######
Upsample (nearest)                   ####     ##
Transpose/Reshape                    ##       ####
-----------------------------------------------------------------

###### = Primary handler     #### = Can handle     ## = Partial
```

---

## 4. Priority Recommendation

To run all three target families (YOLO, CNN, ViT), implement in this order:

| Priority | Hardware Block | Rationale | Impact |
|----------|----------------|-----------|--------|
| **1** | **Activation Function Unit (AFU)** | Without it, SiLU/Sigmoid/Softmax/GELU cannot run efficiently | Blocks YOLO + ViT efficiency |
| **2** | **Requantization Pipeline** (fused into systolic output) | Reduces TCDM write bandwidth by 4x and improves throughput | Performance critical |
| **3** | **Pooling Engine** (dedicated) | Faster MaxPool than Spatz, but Spatz is a functional fallback | Nice-to-have |

---

## 5. Model-Specific Operator Breakdown

### YOLOv8 Detection

```text
Input -> [Conv2d + BN + SiLU] x N -> C2f blocks ->
  SPPF (MaxPool 5x5) -> FPN/PAN (Concat + Upsample) ->
  Detection Head (Conv + Sigmoid)
```

- **Critical missing operators:** SiLU and Sigmoid need AFU.
- Systolic handles all Conv2D.
- Spatz handles BN (fused), Add, and MaxPool, though MaxPool is slower.
- DMA handles Concat and data movement.

### Vision Transformer (ViT)

```text
Input -> Patch Embed (Conv2d) ->
  [LayerNorm -> MHSA(QxK^T -> Softmax -> xV) -> Add ->
   LayerNorm -> MLP(Linear -> GELU -> Linear) -> Add] x L ->
  LayerNorm -> Classification Head
```

- **Critical missing operators:** Softmax, GELU, and LayerNorm (sqrt+div) need AFU.
- Systolic handles all Linear/MatMul work (Q, K, V projections, attention, MLP).
- Spatz handles Add (residual) and part of LayerNorm (mean, variance).

### ResNet / MobileNet (CNN)

```text
Input -> [Conv2d + BN + ReLU] x N -> MaxPool ->
  Residual blocks -> Global AvgPool -> FC -> Softmax
```

- ReLU = `max(x, 0)` and maps cleanly to Spatz.
- Softmax appears only at the final layer, so Snitch firmware can handle it once for small tensors.
- These models can mostly run on the current hardware if ReLU is used instead of SiLU.
- MaxPool can run on Spatz, slower but functional.

---

## 6. Conclusion

The main required hardware block is the **Activation Function Unit (AFU)**, supporting SiLU, Sigmoid, GELU, and Softmax through LUT + linear interpolation. It is the single largest gap blocking efficient execution of all three target model families.

The requantization pipeline is the second most important optimization, but it can initially fall back to Spatz.
