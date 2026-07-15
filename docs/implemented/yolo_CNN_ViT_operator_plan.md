# NPU Cluster — Operator Gap Analysis for YOLO / CNN / Vision Transformer

**Date:** 2026-06-20  
**Target Models:** YOLOv5/v8/v11, General CNN (ResNet, MobileNet), Vision Transformer (ViT, DeiT)  
**Current Hardware:** Snitch (RV32IMAC) + Spatz (RVV INT-only, 2 IPU) + Systolic Array (32×32 INT8) + DMA

---

## 1. Hardware hiện tại có thể xử lý

| Operator | Model sử dụng | Hardware xử lý | Ghi chú |
|----------|---------------|----------------|---------|
| Dense Conv2D | YOLO, CNN, ViT | **Systolic Array** (im2col + GEMM) | ✅ Core workload, INT8 |
| Fully Connected / Linear | All | **Systolic Array** (là MatMul) | ✅ |
| MatMul (Q×K^T, Attn×V) | ViT | **Systolic Array** | ✅ |
| Element-wise Add | YOLO (skip), ViT (residual) | **Spatz** (`vadd.vv`) | ✅ |
| Element-wise Mul | Requant, BN fused | **Spatz** (`vmul.vv`) | ✅ |
| MaxPool | YOLO, CNN | **Spatz** (`vmax.vv` + slide) | ✅ Chậm hơn dedicated |
| Concat | YOLO (C2f, neck) | **DMA** (data copy) | ✅ Chỉ là di chuyển data |
| Shift/Clamp (Requantization) | All (INT8 pipeline) | **Spatz** (`vsra` + `vmax/vmin`) | ✅ |
| ReLU / Clip | CNN | **Spatz** (`vmax.vx` with x=0) | ✅ Trivial |
| BatchNorm (fused into Conv) | CNN, YOLO | **Spatz** (mul + add per-channel) | ✅ Typically fused at compile |
| Depthwise Conv | YOLO (C2f), MobileNet | **Spatz** (element-wise loop) | ⚠️ Functional nhưng chậm, không tận dụng systolic |
| Upsample (Nearest Neighbor) | YOLO (FPN/PAN neck) | **Spatz** hoặc **DMA** | ✅ Duplicate pixels |
| Transpose / Reshape | ViT (multi-head reshape) | **DMA** + **Spatz** | ✅ Data movement |

---

## 2. Hardware THIẾU — Operators chưa có cách xử lý hiệu quả

### 2.1. 🔴 Activation Function Unit (AFU) — QUAN TRỌNG NHẤT

**Vấn đề:** Spatz (INT-only, không có FPU) **không thể tính** các hàm phi tuyến: `exp()`, `tanh()`, `1/x`, `sqrt()`.

| Function | Công thức | Model sử dụng | Mức độ cần thiết |
|----------|-----------|---------------|-------------------|
| **SiLU / Swish** | x × σ(x) = x × 1/(1+exp(-x)) | YOLOv5/v8/v11 (mọi Conv block) | 🔴 Critical |
| **Sigmoid** | 1 / (1 + exp(-x)) | YOLO (detection head output) | 🔴 Critical |
| **Softmax** | exp(xᵢ) / Σexp(xⱼ) | ViT (mọi Attention layer) | 🔴 Critical cho ViT |
| **GELU** | 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))) | ViT (MLP block) | 🔴 Critical cho ViT |
| **LayerNorm** | (x - μ) / √(σ² + ε) | ViT (mọi layer) | 🟠 Cần sqrt + div |
| **Tanh** | (exp(x) - exp(-x)) / (exp(x) + exp(-x)) | Một số CNN | 🟡 Ít phổ biến |

**Đây là gap lớn nhất.** Không có AFU thì YOLO và ViT đều không chạy được trên hardware.

#### Giải pháp phổ biến trong NPU industry:

| Approach | Mô tả | Ưu điểm | Nhược điểm |
|----------|-------|---------|------------|
| **LUT-based** | Bảng tra cứu trong SRAM, nội suy tuyến tính giữa 2 điểm | Đơn giản nhất, deterministic, area rất nhỏ | Precision giới hạn bởi table size (256-1024 entries thường đủ cho INT8) |
| **Piecewise Linear (PWL)** | Chia input range thành N segment, mỗi segment là y = ax + b | Chính xác hơn LUT, vẫn nhỏ gọn | Cần multiplier + adder + segment lookup |
| **Polynomial Approximation** | Dùng Taylor/Chebyshev polynomial bậc 2-3 | Rất chính xác | Cần multiple multipliers, latency cao hơn |
| **CORDIC** | Iterative algorithm cho sin/cos/exp/sqrt | General purpose, high precision | Area lớn, nhiều cycle, overkill cho INT8 |
| **Firmware trên Snitch** | Scalar C code tính từng element | Không cần HW mới | Cực chậm, hoàn toàn không scalable |

#### Recommendation cho NPU này:

Dùng **LUT + Linear Interpolation** vì:
- Target là INT8 inference → input/output chỉ cần 8-bit precision
- LUT 256 entries × 8-bit = 256 bytes/function → fit trong registers
- Throughput: 1 element/cycle dễ đạt, có thể pipeline nhiều lanes
- Hỗ trợ mọi function chỉ bằng cách thay LUT content

#### AFU Architecture gợi ý:

```
                    ┌──────────────────────────┐
  TCDM ────────────►│   Activation Function    │────────────► TCDM
  (input buffer)    │        Unit (AFU)        │   (output buffer)
                    │                          │
                    │  ┌─────┐   ┌──────────┐  │
                    │  │ LUT │──►│ Interp.  │  │
                    │  │256×8│   │ a×x + b  │  │
                    │  └─────┘   └──────────┘  │
                    │                          │
                    │  Config: func_select,    │
                    │          src_ptr,         │
                    │          dst_ptr,         │
                    │          length           │
                    └──────────────────────────┘
```

- Firmware (Snitch) cấu hình: function type, source pointer, dest pointer, length
- AFU tự đọc data từ TCDM, tra LUT, ghi kết quả lại TCDM
- Cần thêm 1 Master trên TCDM interconnect (Master 9 hoặc multiplex)

---

### 2.2. 🟠 Requantization Pipeline (Fuse vào Systolic Output)

**Vấn đề:** Sau mỗi layer Conv/FC (INT8 × INT8 = INT32 accumulator), cần rescale về INT8 cho layer tiếp theo:

```
output_int8 = clamp((acc_int32 × scale) >> shift + zero_point, 0, 255)
```

Hiện tại `systolic_controller` ghi INT32 raw accumulator ra TCDM, rồi Spatz phải đọc lại, requant, ghi lại. Điều này:
- Tốn 2× bandwidth TCDM (đọc INT32 + ghi INT8)
- Tốn latency (firmware dispatch + Spatz processing)
- Tốn 4× storage (INT32 vs INT8)

**Giải pháp:** Thêm **post-processing pipeline** vào `systolic_controller.sv`:

```
Systolic Array INT32 output
         │
         ▼
    ┌──────────┐
    │ × scale  │  (per-channel scale, loaded from TCDM)
    │ >> shift  │  (per-channel shift)
    │ + zp     │  (per-channel zero point)
    │ clamp    │  (saturate to [0, 255])
    └──────────┘
         │
         ▼
    INT8 output → ghi TCDM (1/4 bandwidth so với INT32)
```

**Impact:** Giảm 4× TCDM write bandwidth, tăng throughput đáng kể.

**Priority:** 🟠 Nên có nhưng có thể dùng Spatz làm workaround tạm.

---

### 2.3. 🟡 Dedicated Pooling Engine (Optional)

MaxPool 2×2 / 3×3 phổ biến trong YOLO/CNN cũ. Spatz *có thể* làm nhưng throughput thấp vì phải:
1. Load multiple rows từ TCDM
2. Vector compare + select
3. Stride output

Dedicated pooling engine sẽ nhanh hơn, nhưng:
- Trend gần đây (YOLOv8+) thay MaxPool bằng stride-2 Conv
- MobileNetV2+ dùng stride-2 Depthwise Conv thay pooling
- ViT không dùng pooling (chỉ có Global Average Pool cuối)

**Priority:** 🟡 Nice-to-have. Spatz có thể xử lý, không block inference.

---

## 3. Operator Coverage Map (Tổng hợp)

```
                          Systolic   Spatz    DMA    ❌ Missing
                          (MatMul)   (RVV)   (Xfer)  (Need HW)
─────────────────────────────────────────────────────────────────
Conv2D (dense)              ██████                    
Depthwise Conv                       ████             
Fully Connected             ██████                    
MatMul (Q×K^T, A×V)        ██████                    
─────────────────────────────────────────────────────────────────
SiLU / Swish                                          ██████ AFU
Sigmoid                                               ██████ AFU
GELU                                                  ██████ AFU
Softmax                                               ██████ AFU
LayerNorm                            ██               ████ AFU
─────────────────────────────────────────────────────────────────
BatchNorm (fused)                    ████              
ReLU / Clip                          ██████            
MaxPool                              ████              
AvgPool / GAP                        ████              
Add (residual)                       ██████            
─────────────────────────────────────────────────────────────────
Requantization                       ████              ██ (fuse)
Concat                                        ██████   
Upsample (nearest)                   ████     ██       
Transpose/Reshape                    ██       ████     
─────────────────────────────────────────────────────────────────

██████ = Primary handler     ████ = Can handle     ██ = Partial
```

---

## 4. Priority Recommendation

Để chạy được cả 3 target (YOLO, CNN, ViT), implement theo thứ tự:

| Priority | Hardware Block | Lý do | Impact |
|----------|---------------|-------|--------|
| 🔴 **1** | **Activation Function Unit (AFU)** | Không có thì SiLU/Sigmoid/Softmax/GELU đều không chạy được | Blocks YOLO + ViT hoàn toàn |
| 🟠 **2** | **Requantization Pipeline** (fuse vào systolic output) | Giảm 4× TCDM write bandwidth, tăng throughput lớn | Performance critical |
| 🟡 **3** | **Pooling Engine** (dedicated) | MaxPool nhanh hơn Spatz, nhưng Spatz có thể workaround | Nice-to-have |

---

## 5. Model-specific Operator Breakdown

### YOLOv8 (Detection)
```
Input → [Conv2d + BN + SiLU] × N → C2f blocks → 
  SPPF (MaxPool 5×5) → FPN/PAN (Concat + Upsample) →
  Detection Head (Conv + Sigmoid)
```
- **Critical missing:** SiLU, Sigmoid → cần AFU
- Systolic handles: tất cả Conv2D
- Spatz handles: BN (fused), Add, MaxPool (chậm)
- DMA handles: Concat, data movement

### Vision Transformer (ViT)
```
Input → Patch Embed (Conv2d) → 
  [LayerNorm → MHSA(Q×K^T → Softmax → ×V) → Add → 
   LayerNorm → MLP(Linear → GELU → Linear) → Add] × L →
  LayerNorm → Classification Head
```
- **Critical missing:** Softmax, GELU, LayerNorm (sqrt+div) → cần AFU
- Systolic handles: tất cả Linear/MatMul (Q, K, V projections, attention, MLP)
- Spatz handles: Add (residual), partial LayerNorm (mean, variance)

### ResNet / MobileNet (CNN)
```
Input → [Conv2d + BN + ReLU] × N → MaxPool → 
  Residual blocks → Global AvgPool → FC → Softmax
```
- ReLU = `max(x, 0)` → Spatz handles ✅
- Softmax chỉ ở layer cuối → firmware Snitch có thể xử lý (1 lần, nhỏ)
- **Gần như chạy được** trên HW hiện tại nếu dùng ReLU thay SiLU
- MaxPool → Spatz (chậm nhưng functional)

---

## 6. Kết luận

**Chỉ cần thêm 1 hardware block chính: Activation Function Unit (AFU)** — hỗ trợ SiLU, Sigmoid, GELU, Softmax thông qua LUT + linear interpolation. Đây là gap duy nhất thực sự ngăn cả 3 target model chạy trên NPU.

Requantization pipeline là optimization quan trọng thứ hai, nhưng có thể workaround bằng Spatz trước.
