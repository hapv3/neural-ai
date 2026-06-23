# Kiến trúc AFU (Activation Function Unit) & Mapping Operators

Tài liệu này quy hoạch thiết kế vi kiến trúc của khối AFU để xử lý các hàm phi tuyến (SiLU, Sigmoid, GELU, Softmax, LayerNorm) trên NPU, bù đắp cho điểm yếu thiếu FPU của Spatz Vector Engine.

## 1. Triết lý Thiết kế (Design Philosophy)

Mô hình học máy lượng tử hóa (INT8) có một đặc điểm cực kỳ lợi hại: **Dữ liệu đầu vào chỉ có 256 giá trị khả dĩ (-128 đến 127)**.
Do đó, thay vì xây dựng các khối tính toán Taylor/Chebyshev phức tạp, chúng ta sẽ thiết kế AFU dưới dạng một **Streaming LUT Processor**. 

- **Không cần nội suy (No Interpolation):** Ánh xạ 1-1 (1-to-1 mapping). Bất kỳ hàm $f(x)$ nào (SiLU, Sigmoid, Tanh) đều có thể tính trước (pre-computed) thành một mảng 256 bytes bởi trình biên dịch (Compiler/Firmware) và nạp vào SRAM của AFU.
- **Tốc độ:** Xử lý 1 byte / 1 chu kỳ. Pipeline 4-lane thì xử lý 4 bytes/chu kỳ (tương đương tốc độ bus TCDM 32-bit).
- **Phân công phần cứng:** 
  - AFU chỉ lo **Vector Element-wise Non-linear** (bảng tra).
  - Spatz Vector Engine lo **Vector Reductions & Arithmetic** (max, sum, mul, add).
  - Snitch Scalar Core lo **Scalar Non-linear** (tính 1 giá trị 1/sqrt hoặc 1/x).

---

## 2. AFU Micro-Architecture

AFU sẽ được gắn vào Data TCDM Interconnect như một **DMA Master** độc lập.

```text
                      ┌────────────────────────────────────────┐
                      │              AFU Hardware              │
                      │                                        │
                      │  ┌──────────────────────────────────┐  │
                      │  │  Config & Control Registers      │  │
  TCDM (L1)           │  │  (Start, Len, Src, Dst, Mode)    │  │
  Interconnect        │  └──────────────────────────────────┘  │
  (Master Port)       │                                        │
        ▲             │  ┌──────────────────────────────────┐  │
        │ 32-bit      │  │  Dual-Mode LUT SRAM (1 KB)       │  │<-- Firmware nạp
        │ R/W         │  │  (Hỗ trợ 256x8, 256x16, 256x32)  │  │    bảng tra vào đây
        ▼             │  └──────────────────────────────────┘  │    khi chuyển layer
                      │                                        │
                      │  ┌────────┐  Address   ┌────────┐      │
                      │  │ Read   │----------->│ Lookup │      │
                      │  │ Engine │            │ Engine │      │
                      │  └────────┘            └────────┘      │
                      │                             │          │
                      │  ┌────────┐                 │          │
                      │  │ Write  │<----------------┘          │
                      │  │ Engine │ Data (8/16/32-bit)         │
                      │  └────────┘                            │
                      └────────────────────────────────────────┘
```

**Workflow cơ bản của AFU (Dual-Mode):**
1. Firmware Snitch tính trước hàm $f(x)$ và ghi bảng kết quả vào LUT SRAM của AFU (Kích thước mảng tùy thuộc vào Output Precision cần thiết).
2. Firmware ghi cấu hình `Src_Ptr`, `Dst_Ptr`, `Length`, và **`Mode`** (8-bit, 16-bit, hoặc 32-bit output).
3. Kích hoạt AFU. Tùy theo Mode:
   - **Mode 8-bit (SiLU, Sigmoid):** AFU đọc 1 word 32-bit từ TCDM, tách thành 4 bytes, lấy 4 kết quả 8-bit từ LUT, gộp lại thành 32-bit và ghi ra TCDM. Tốc độ: 4 elements/cycle.
   - **Mode 16/32-bit (exp, x^2 cho Softmax/Norm):** AFU đọc từng byte, tra ra kết quả 16-bit hoặc 32-bit để giữ độ chính xác tối đa, sau đó ghi các từ 16/32-bit này ra TCDM để Spatz xử lý cộng dồn tiếp theo. Tốc độ: 1-2 elements/cycle.

---

## 3. Mapping Cụ Thể Từng Function

### 3.1. Các hàm Element-wise Đơn Thuần (SiLU, Sigmoid, Tanh, GELU)
**Model:** YOLO (SiLU/Sigmoid), ViT (GELU), CNN (Tanh).
- **Cách chạy:** Đẩy 100% vào AFU.
- **Quy trình:**
  1. Trình biên dịch (Compiler) offline sinh ra mảng tĩnh `const uint8_t silu_lut[256] = {...}`.
  2. Snitch copy mảng này vào `AFU_LUT_RAM`.
  3. AFU chạy một lèo hết tensor. Tốc độ cao nhất (100% utilization).

### 3.2. Softmax (ViT Attention)
Công thức: $y_i = \frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$
- **Cách chạy:** Kết hợp Spatz + AFU.
- **Quy trình:**
  1. **Spatz:** Chạy lệnh `vmax` tìm giá trị lớn nhất của hàng (vector) -> $M$.
  2. **Spatz:** Chạy lệnh `vsub` trừ đi $M$: $x' = x - M$.
  3. **Snitch:** Cấu hình AFU với bảng LUT `exp()`.
  4. **AFU:** Tra bảng biến mảng $x'$ thành mảng $E = e^{x'}$.
  5. **Spatz:** Chạy lệnh `vsum` trên mảng $E$ để ra tổng $S$.
  6. **Snitch:** Dùng C code tính giá trị vô hướng (scalar) `inv_S = 1 / S`.
  7. **Spatz:** Chạy lệnh `vmul` nhân toàn bộ mảng $E$ với `inv_S` để ra kết quả cuối.

### 3.3. LayerNorm (ViT Layer)
Công thức: $y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta$
- **Cách chạy:** Kết hợp Spatz + Snitch (Thực tế **không cần dùng AFU**).
- **Quy trình:**
  1. **Spatz:** Chạy `vsum` tìm tổng -> Tính Mean $\mu$.
  2. **Spatz:** Chạy `vsub` -> $(x - \mu)$, `vmul` -> $(x - \mu)^2$, `vsum` -> Variance $\sigma^2$.
  3. **Snitch:** Tính hàm `1/sqrt(sigma^2 + eps)` bằng thư viện C chuẩn trên scalar core. Vì đây chỉ là **1 giá trị vô hướng (scalar)** cho cả 1 hàng vector, Snitch tốn ~50 cycles là xong, không ảnh hưởng tổng thời gian. Lưu kết quả vào biến `inv_std`.
  4. **Spatz:** Dùng `vmul` nhân mảng $(x-\mu)$ với `inv_std` và $\gamma$, dùng `vadd` cộng $\beta$.

## Verification Plan

### Automated Security Check
- **Security Scanner**: Quét các đoạn firmware cấu hình AFU để phát hiện lồi tràn bộ đệm (đảm bảo Src_Ptr, Dst_Ptr và Length không ghi đè ra ngoài vùng nhớ TCDM an toàn).
- **Security Audit**: Kiểm tra luồng cấp quyền truy cập vùng nhớ DMA của AFU.

### Functional Verification
- Viết testbench RTL cho AFU: Nạp 1 bảng LUT bất kỳ, đưa mảng 1024 bytes vào và kiểm tra kết quả ngõ ra có khớp với mô hình Python hay không.
