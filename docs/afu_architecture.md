# Kiến trúc AFU (Activation Function Unit) & Mapping Operators

Tài liệu này quy hoạch thiết kế vi kiến trúc của khối AFU để xử lý các hàm phi tuyến (SiLU, Sigmoid, GELU, Softmax, LayerNorm) trên NPU, bù đắp cho điểm yếu thiếu FPU của Spatz Vector Engine. Bản cập nhật này bao gồm các thay đổi mới nhất về Dataflow, OBI Compliance và Pipeline Optimization.

## 1. Triết lý Thiết kế (Design Philosophy)

Mô hình học máy lượng tử hóa (INT8) có một đặc điểm cực kỳ lợi hại: **Dữ liệu đầu vào chỉ có 256 giá trị khả dĩ (-128 đến 127)**.
Do đó, thay vì xây dựng các khối tính toán Taylor/Chebyshev phức tạp, chúng ta sẽ thiết kế AFU dưới dạng một **Streaming LUT Processor**. 

- **Không cần nội suy (No Interpolation):** Ánh xạ 1-1 (1-to-1 mapping). Bất kỳ hàm $f(x)$ nào (SiLU, Sigmoid, Tanh) đều có thể tính trước (pre-computed) thành một mảng 256 bytes bởi trình biên dịch (Compiler/Firmware) và nạp vào SRAM của AFU.
- **Tốc độ:** Xử lý 1 byte / 1 chu kỳ. Pipeline 4-lane thì xử lý 4 bytes/chu kỳ (tương đương tốc độ bus TCDM 32-bit).
- **Phân công phần cứng:** 
  - AFU lo **Vector Element-wise Non-linear** (bảng tra).
  - Spatz Vector Engine lo **Vector Reductions & Arithmetic** (max, sum, mul, add).
  - Snitch Scalar Core lo **Scalar Non-linear** (tính 1 giá trị 1/sqrt hoặc 1/x).
- **Quy tắc Căn lề Bộ nhớ (Memory Alignment Rule):** Bắt buộc các địa chỉ nguồn (`src_ptr`) và đích (`dst_ptr`) phải được căn lề 32-byte (Aligned to 32 bytes). Việc xử lý mảng có kích thước lẻ (odd lengths) được thực hiện hoàn toàn thông qua Byte Enables (Mặt nạ Ghi) thay vì dùng dịch bit (shift-and-or) để tiết kiệm diện tích (Area).

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
                      │  ┌──────────┐  Address   ┌──────────┐  │
                      │  │ Read     │----------->│ 2-Stage  │  │
                      │  │ Backend  │            │ Pipeline │  │
                      │  └──────────┘            └──────────┘  │
                      │  (OBI Compliant)              │        │
                      │  ┌──────────┐                 │        │
                      │  │ Write    │<----------------┘        │
                      │  │ Backend  │ Data + Byte Enables      │
                      │  └──────────┘                          │
                      └────────────────────────────────────────┘
```

### 2.1. Thiết kế RTL Chi tiết (RTL Components)

Hệ thống AFU được chia thành 5 module chính để đảm bảo dễ bảo trì và phân tách rõ ràng trách nhiệm (Separation of Concerns):

1. **`afu.sv` (Top-level Wrapper):** 
   - Đóng vai trò là cầu nối liên kết tất cả các module con. 
   - Khai báo các giao diện OBI (Slave cho config, Master cho memory access).
   - Instantiations: Gọi `afu_frontend`, `afu_backend`, `afu_core`, và hai bộ `afu_fifo_ff` (rfifo và wfifo).

2. **`afu_frontend.sv` (Control & CSRs):**
   - Đóng vai trò là OBI Slave nhận các lệnh cấu hình từ Snitch Core.
   - Quản lý các thanh ghi CSR: `src_ptr`, `dst_ptr`, `length`, `mode`.
   - Sinh ra xung `start` để đánh thức các khối khác khi firmware ghi vào thanh ghi `START`.
   - Phơi bày dải địa chỉ của SRAM LUT (từ 0x000 đến 0x3FC) để firmware có thể nạp bảng tra hàm.

3. **`afu_backend.sv` (Memory Access Engine):**
   - Đóng vai trò là OBI Master, tự động hóa việc đọc dữ liệu chưa xử lý (từ `src_ptr`) và ghi dữ liệu đã xử lý (tới `dst_ptr`).
   - Tuân thủ chặt chẽ OBI Handshake (`req`, `gnt`, `rvalid`). Các tín hiệu yêu cầu sẽ được chốt (latch) thông qua thanh ghi `hold_req_q` cho tới khi có tín hiệu `gnt` trả về, đảm bảo an toàn và không bị mất request (protocol violations).
   - Quản lý vòng lặp địa chỉ: Mỗi lần đọc/ghi đều kiểm tra xem đã chạm đến địa chỉ kết thúc (`end_addr`) chưa để dừng phát req.

4. **`afu_core.sv` (2-Stage Compute Pipeline):**
   - **Trái tim của AFU**, nơi thực hiện việc tra bảng (LUT Lookup) với độ trễ thấp nhất.
   - **Stage 1 (S1):** Rút dữ liệu (pop) từ Read FIFO, tính toán số lượng phần tử hơp lệ dựa trên `elem_cnt`, phân tách dữ liệu 32-bit thành 4 khối 8-bit và đưa vào 4 port địa chỉ truy vấn LUT SRAM (`tc_sram`).
   - **Stage 2 (S2):** Nhận dữ liệu kết quả từ SRAM (trễ 1 chu kỳ), tổng hợp lại thành một block hoàn chỉnh. Tính toán mặt nạ Byte Enables (`s2_out_be_comb`) để đảm bảo không ghi lạm vào vùng nhớ khác khi `length` là số lẻ. Đẩy kết quả vào Write FIFO.
   - **Stall Protection:** Nếu S2 bị kẹt do Write FIFO đầy (`s2_stall`), mạch sẽ dùng thanh ghi đệm `s2_lut_rdata_saved_q` để chốt cứng lại kết quả từ SRAM, tránh việc dữ liệu bị hỏng khi S1 lỡ cập nhật địa chỉ mới.

5. **`afu_fifo_ff.sv` (Flip-Flop Based FIFOs):**
   - Hàng đợi dữ liệu trung gian giữa Backend và Core, giúp hấp thụ (absorb) độ trễ của giao thức OBI.
   - Vì độ sâu chỉ cần `DEPTH=2`, việc thiết kế bằng Flip-Flop thay vì gọi SRAM Macro giúp tiết kiệm đáng kể diện tích (Area), giảm đường trễ (Routing delay), và tránh phải dùng bộ khởi tạo bộ nhớ phức tạp.

### 2.2. Workflow cơ bản của AFU (Dual-Mode)
1. Firmware Snitch tính trước hàm $f(x)$ và ghi bảng kết quả vào LUT SRAM của AFU.
2. Firmware ghi cấu hình `Src_Ptr`, `Dst_Ptr`, `Length`, và **`Mode`** (8-bit, 16-bit, hoặc 32-bit output).
3. Kích hoạt AFU. Tùy theo Mode:
   - **Mode 8-bit (SiLU, Sigmoid):** AFU xử lý 4 bytes/cycle (hoặc theo độ rộng bus nếu cấu hình rộng hơn), tra 4 kết quả 8-bit và đẩy vào Write FIFO. Cực kỳ nhanh.
   - **Mode 16/32-bit (exp, x^2 cho Softmax/Norm):** AFU đọc từng byte, tra ra kết quả 16-bit hoặc 32-bit để giữ độ chính xác tối đa, sau đó ghi các từ 16/32-bit này ra TCDM để Spatz xử lý cộng dồn tiếp theo.

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
  4. **AFU:** Tra bảng biến mảng $x'$ thành mảng $E = e^{x'}$. (Lưu ý: Mảng Output được yêu cầu phải có độ chính xác 16/32-bit để chống sai số - dùng Mode 16/32).
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

---

## 4. Tích hợp AFU vào Hệ thống (Integration Guide)

Để lắp ráp khối IP `afu.sv` vào hệ thống NPU thực tế, cần thực hiện các bước cấu hình sau trên Top-level của NPU (như `npu_cluster.sv`):

### 4.1. Kết nối Mạng lưới (Interconnect)
- **Gắn AFU vào Peripheral/System Interconnect (Cổng OBI Slave):**
  - Mở thêm 1 port Slave trên Address Decoder/Crossbar để kết nối từ Snitch Core.
  - Phân bổ dải địa chỉ Memory-Mapped (VD: `0x1004_0000`) để Snitch cấu hình và nạp LUT.
- **Gắn AFU vào TCDM Interconnect (Cổng OBI Master):**
  - Tăng tham số `NUM_MASTERS` của TCDM Crossbar lên 1.
  - Gắn cổng `obi_m_*` của AFU vào port Master mới để AFU tự động đọc/ghi bộ nhớ L1.

### 4.2. Kết nối Tín hiệu Điều khiển
- **Clock và Reset:** Đồng bộ chung `clk_i` và `rst_ni` với toàn cụm Cluster.
- **Tín hiệu Ngắt (Interrupt - `done_o`):**
  - Nối cổng `done_o` vào Interrupt Controller (PLIC/CLINT) hoặc Fast Interrupt của lõi Snitch Core.
  - Điều này giúp NPU có thể gọi AFU chạy rồi vào chế độ ngủ (WFI), khi AFU tính xong sẽ đánh thức Snitch chạy layer tiếp theo.

### 4.3. Cập nhật Firmware (Software C)
Bổ sung định nghĩa cấu trúc phần cứng vào SDK của Firmware để điều khiển bằng mã C:

```c
// Địa chỉ cơ sở AFU (Ví dụ)
#define AFU_BASE_ADDR 0x10040000

// Cấu trúc thanh ghi AFU
typedef struct {
    volatile uint32_t LUT_SRAM[256];  // 0x0000 - 0x03FC
    uint32_t _reserved[768];          // Đệm cho đủ 0x1000
    volatile uint32_t START;          // 0x1000
    volatile uint32_t SRC_PTR;        // 0x1004
    volatile uint32_t DST_PTR;        // 0x1008
    volatile uint32_t LENGTH;         // 0x100C
    volatile uint32_t MODE;           // 0x1010
} afu_regs_t;

#define AFU_REGS ((afu_regs_t*) AFU_BASE_ADDR)
```
