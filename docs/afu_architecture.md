# Kiến trúc AFU (Activation Function Unit) & Mapping Operators

Tài liệu này quy hoạch thiết kế vi kiến trúc của khối AFU để xử lý các hàm phi tuyến (SiLU, Sigmoid, GELU, Softmax, LayerNorm) trên NPU, bù đắp cho điểm yếu thiếu FPU của Spatz Vector Engine. Bản cập nhật này bao gồm các thay đổi mới nhất về Dataflow, OBI Compliance và Pipeline Optimization.

## 1. Triết lý Thiết kế (Design Philosophy)

Mô hình học máy lượng tử hóa (INT8) có một đặc điểm cực kỳ lợi hại: **Dữ liệu đầu vào chỉ có 256 giá trị khả dĩ (-128 đến 127)**.
Do đó, thay vì xây dựng các khối tính toán Taylor/Chebyshev phức tạp, chúng ta sẽ thiết kế AFU dưới dạng một **Streaming LUT Processor**. 

- **Không cần nội suy (No Interpolation):** Ánh xạ 1-1 (1-to-1 mapping). Bất kỳ hàm $f(x)$ nào (SiLU, Sigmoid, Tanh) đều có thể tính trước (pre-computed) thành một mảng 256 bytes bởi trình biên dịch (Compiler/Firmware) và nạp vào SRAM của AFU.
- **Tốc độ:** Phase hiện tại dùng 4 LUT lanes nội bộ. Data path AFU nối vào Shared Data TCDM 256-bit; core tiêu thụ theo nhóm byte từ read FIFO và phát write beat có byte-enable theo tail.
- **Phân công phần cứng:** 
  - AFU lo **Vector Element-wise Non-linear** (bảng tra).
  - Spatz Vector Engine lo **Vector Reductions & Arithmetic** (max, sum, mul, add).
  - Snitch Scalar Core lo **Scalar Non-linear** (tính 1 giá trị 1/sqrt hoặc 1/x).
- **Quy tắc Căn lề Bộ nhớ (Memory Alignment Rule):** Firmware/HAL phase hiện tại yêu cầu `src_ptr` và `dst_ptr` căn lề 32-byte khi tích hợp cluster. Mảng có số phần tử không chia hết beat vẫn được hỗ trợ bằng byte-enable tail; arbitrary unaligned e16/e32 destination chưa là contract scheduler.

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
        │ 256-bit     │  │  Dual-Mode LUT SRAM (1 KB)       │  │<-- Firmware nạp
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
   - Tuân thủ chặt chẽ OBI Handshake (`req`, `gnt`, `rvalid`). Các tín hiệu request được chốt trong pending transaction register cho tới khi có `gnt`, đồng thời read side giới hạn một outstanding read để tránh ghi đè dữ liệu trả về.
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

## 3. Đánh giá Performance Hiện Tại

Đánh giá này áp dụng cho RTL hiện tại: `MEM_DATA_WIDTH=256`, `LUT_LANES=4`, RFIFO/WFIFO depth 2, AFU là một TCDM master riêng và LUT SRAM có latency 1 cycle. Đây là đánh giá **AFU active path**, không tính thời gian host AXI nạp firmware, firmware tự check output, hoặc cocotb readback.

### 3.1. Throughput lý thuyết

Core có 4 LUT lanes nên upper bound compute là **4 input elements/cycle** cho cả 3 mode. Khác biệt giữa các mode nằm ở output bandwidth:

| Mode | Input element | Output element | Core throughput | Output bandwidth @ 1 GHz | Total TCDM bandwidth cần |
|------|---------------|----------------|-----------------|--------------------------|--------------------------|
| `e8` | 8-bit | 8-bit | 4 elems/cycle | 4 GB/s | 8 GB/s = 4 GB/s read + 4 GB/s write |
| `e16` | 8-bit | 16-bit | 4 elems/cycle | 8 GB/s | 12 GB/s = 4 GB/s read + 8 GB/s write |
| `e32` | 8-bit | 32-bit | 4 elems/cycle | 16 GB/s | 20 GB/s = 4 GB/s read + 16 GB/s write |

Shared Data TCDM port của AFU là 256-bit, nên peak một beat/cycle tương đương **32 GB/s @ 1 GHz** nếu không bị arbitration stall. Vì vậy với cấu hình 4 lanes hiện tại, AFU chủ yếu **lane-limited**, chưa phải TCDM-bandwidth-limited trong điều kiện không tranh chấp.

### 3.2. Cycle model dùng để estimate

Với tensor dài `N` phần tử 8-bit input và output width `B ∈ {1,2,4}` byte:

```text
compute_cycles       = ceil(N / 4)
read_beats_256b      = ceil(N / 32)
write_beats_256b     = ceil(N * B / 32)
active_cycles_lower  ≈ max(compute_cycles, read_beats_256b + write_beats_256b)
active_cycles_upper  ≈ compute_cycles + read_beats_256b + write_beats_256b + small_pipeline_drain
```

Trong workload dài, backend read/write và core lookup overlap một phần, nên số cycle thực tế nên nằm gần lower bound nếu TCDM grant đều. Trong workload nhỏ, overhead start, read-first latency, write drain, IRQ/polling, và LUT programming sẽ chiếm tỷ lệ lớn.

### 3.3. Ví dụ workload

| Tensor | Mode | Elements | Ideal compute cycles | Traffic | Nhận xét |
|--------|------|----------|----------------------|---------|----------|
| Micro-YOLO activation `32×32×32` | `e8` | 32,768 | 8,192 | 32 KB read + 32 KB write | AFU active khoảng vài microsecond ở 1 GHz; firmware/LUT setup có thể đáng kể nếu chỉ chạy một tensor nhỏ. |
| ViT/Softmax exp row batch 32k elems | `e16` | 32,768 | 8,192 | 32 KB read + 64 KB write | Vẫn lane-limited; output traffic tăng nhưng dưới peak TCDM port. |
| Precision staging 32k elems | `e32` | 32,768 | 8,192 | 32 KB read + 128 KB write | Gần bandwidth hơn nhưng vẫn chỉ cần ~20/32 GB/s ở 1 GHz nếu không tranh chấp. |
| YOLO feature map `80×80×64` | `e8` | 409,600 | 102,400 | 400 KB read + 400 KB write | Phù hợp AFU streaming; LUT programming amortize tốt trên tensor lớn. |

### 3.4. Bottleneck thực tế hiện tại

- **LUT programming cost:** mỗi bảng LUT cần 256 MMIO writes. Với nhiều layer dùng cùng activation/qparam, firmware nên cache/reuse LUT và chỉ nạp lại khi bảng đổi.
- **TCDM arbitration:** AFU hiện đi chung Shared Data TCDM với DMA, Systolic và Spatz. Khi chạy overlap thật, AFU có thể bị stall bởi HWPE traffic có priority cao hơn; chưa có PMU counter riêng để đo stall.
- **Backend policy:** backend hiện ưu tiên write hơn read và giới hạn một read outstanding để giữ OBI response đơn giản. Cấu hình này đúng cho correctness, nhưng chưa tối ưu bandwidth tuyệt đối.
- **FIFO depth:** RFIFO/WFIFO depth 2 đủ cho regression hiện tại; nếu TCDM grant jitter cao, tăng FIFO hoặc thêm outstanding read/write queue có thể cải thiện utilization.
- **Firmware wait:** test hiện dùng polling `afu_wait_done`; event-driven `wfi`/trap sẽ giảm năng lượng và scalar busy cycles, nhưng không thay đổi AFU active throughput.

### 3.5. Số liệu regression hiện tại

`test_afu_basic` chạy pass ở mức cluster và kết thúc ở khoảng `24,532 ns` simulation time với clock 1 ns. Con số này **không phải latency thuần AFU**, vì bao gồm AXI boot load, firmware seed data, 3 lần nạp LUT, AFU run, firmware self-check và host notify. Nó chỉ chứng minh rằng path tích hợp hiện tại chạy đúng và không timeout. Để có số liệu performance thật, cần thêm counter đo:

- `afu_active_cycles`: từ `start` tới `done_o`.
- `afu_core_stall_cycles`: `s2_stall` hoặc core chờ RFIFO.
- `afu_tcdm_wait_cycles`: `obi_m_req_o && !obi_m_gnt_i`.
- `afu_read_beats` / `afu_write_beats`: số beat TCDM thực tế.

Các counter này nên đi vào PMU hoặc debug CSR trước khi dùng số liệu AFU để tối ưu scheduler/performance.

---

## 4. Mapping Cụ Thể Từng Function

### 4.1. Các hàm Element-wise Đơn Thuần (SiLU, Sigmoid, Tanh, GELU)
**Model:** YOLO (SiLU/Sigmoid), ViT (GELU), CNN (Tanh).
- **Cách chạy:** Đẩy 100% vào AFU.
- **Quy trình:**
  1. Trình biên dịch (Compiler) offline sinh ra mảng tĩnh `const uint8_t silu_lut[256] = {...}`.
  2. Snitch copy mảng này vào `AFU_LUT_RAM`.
  3. AFU chạy một lèo hết tensor. Tốc độ cao nhất (100% utilization).

### 4.2. Softmax (ViT Attention)
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

### 4.3. LayerNorm (ViT Layer)
Công thức: $y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta$
- **Cách chạy:** Kết hợp Spatz + Snitch (Thực tế **không cần dùng AFU**).
- **Quy trình:**
  1. **Spatz:** Chạy `vsum` tìm tổng -> Tính Mean $\mu$.
  2. **Spatz:** Chạy `vsub` -> $(x - \mu)$, `vmul` -> $(x - \mu)^2$, `vsum` -> Variance $\sigma^2$.
  3. **Snitch:** Tính hàm `1/sqrt(sigma^2 + eps)` bằng thư viện C chuẩn trên scalar core. Vì đây chỉ là **1 giá trị vô hướng (scalar)** cho cả 1 hàng vector, Snitch tốn ~50 cycles là xong, không ảnh hưởng tổng thời gian. Lưu kết quả vào biến `inv_std`.
  4. **Spatz:** Dùng `vmul` nhân mảng $(x-\mu)$ với `inv_std` và $\gamma$, dùng `vadd` cộng $\beta$.

---

## 5. Tích hợp AFU vào Hệ thống (Integration Guide)

Để lắp ráp khối IP `afu.sv` vào hệ thống NPU thực tế, cần thực hiện các bước cấu hình sau trên Top-level của NPU (như `npu_cluster.sv`):

### 5.1. Kết nối Mạng lưới (Interconnect)
- **Gắn AFU vào Peripheral/System Interconnect (Cổng OBI Slave):**
  - AFU là MMIO slave 32-bit sau D-side demux của Snitch.
  - Dải địa chỉ hiện tại là `0x2000_3000 – 0x2000_3fff`. LUT ở offset `0x000..0x3ff`; CSR bắt đầu tại offset `0x400`.
- **Gắn AFU vào TCDM Interconnect (Cổng OBI Master):**
  - `NUM_MASTERS` của Shared Data TCDM hiện là 11.
  - AFU dùng master port riêng trên Shared Data TCDM 256-bit để tự động đọc/ghi tensor L1.

### 5.2. Kết nối Tín hiệu Điều khiển
- **Clock và Reset:** Đồng bộ chung `clk_i` và `rst_ni` với toàn cụm Cluster.
- **Tín hiệu Ngắt (Interrupt - `done_o`):**
  - `done_o` đã nối vào `npu_interrupt_ctrl` bit `NPU_IRQ_SRC_AFU`.
  - Firmware có thể enable/clear `INT_PENDING` để xác nhận AFU done; trap/WFI handler đầy đủ vẫn là phase sau.

### 5.3. Cập nhật Firmware (Software C)
Bổ sung định nghĩa cấu trúc phần cứng vào SDK của Firmware để điều khiển bằng mã C:

```c
// Địa chỉ cơ sở AFU trong cluster MMIO aperture
#define AFU_BASE_ADDR 0x20003000

// Cấu trúc thanh ghi AFU
typedef struct {
    volatile uint32_t LUT_SRAM[256];  // 0x0000 - 0x03FC
    volatile uint32_t STATUS_START;   // 0x0400: read done/busy/error, write start pulse
    volatile uint32_t SRC_PTR;        // 0x0404
    volatile uint32_t DST_PTR;        // 0x0408
    volatile uint32_t LENGTH;         // 0x040C
    volatile uint32_t MODE;           // 0x0410
} afu_regs_t;

#define AFU_REGS ((afu_regs_t*) AFU_BASE_ADDR)
```
