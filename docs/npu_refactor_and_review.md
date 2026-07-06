# Đánh Giá và Kế Hoạch Tái Cấu Trúc NPU Subsystem

Tài liệu này tổng hợp các đánh giá mã nguồn dựa trên tiêu chuẩn thiết kế vi mạch ASIC (ASIC Synthesizable) và phương án tái cấu trúc (Refactoring) toàn diện cho hệ thống điều khiển NPU, đặc biệt tập trung vào hai module lõi: `conv_channel_linebuf_packer.sv` và `systolic_controller.sv`.

---

## Phần 1: Đánh Giá ASIC Synthesizability (Review)

Nhìn chung, mã nguồn RTL của hệ thống đạt tiêu chuẩn "Production-Ready" với điểm số **98/100**. Kiến trúc phản ánh tư duy thiết kế phần cứng rất vững vàng.

### 1.1. Những Điểm Sáng (ASIC-Perfect)
- **Kiến trúc Parallel Drain Engine:** Việc thiết kế `drain_state_e` chạy độc lập và song song với `state_e` (Main Compute FSM) trong `systolic_controller` là đỉnh cao của thiết kế Pipeline. Nó giúp MAC Array không bao giờ bị Stall khi đọc/ghi bộ nhớ (chỉ bị cản trở bởi cơ chế Backpressure của FIFO).
- **Synchronous Design Chuẩn Mực:** Không tồn tại Latch (tất cả tín hiệu trong `always_comb` đều có Default Assignment). Sử dụng hoàn hảo Asynchronous Active-Low Reset (`always_ff @(posedge clk_i or negedge rst_ni)`). Code FSM đúng chuẩn 2-process (`_q` và `_d`).
- **Unrolled For-loops Hợp Lệ:** Các vòng lặp Array (K_MAX=5, ARRAY_DIM=32) đều có giới hạn hằng số tĩnh tĩnh, giúp tool tổng hợp dễ dàng chuyển hóa thành song song (MUX/DEMUX và Comparators). Khối so sánh Tag dò Hit/Miss chạy cực nhanh và tốn rất ít trễ Logic.

### 1.2. Hai Nguy Cơ Vi Phạm Timing (Critical Path Risks)
Để đạt được mức Fmax mong muốn (1.2GHz+ trên 7nm), hệ thống đang vướng phải 2 đoạn mã tạo ra độ trễ lan truyền (Propagation Delay) quá lớn trong `always_comb`:

1. **Phép Nhân 3 Toán Hạng (Multiplier):** 
   - Đoạn code `coalesce_k_bytes = 32'(cfg_kernel_h_i) * 32'(cfg_kernel_w_i) * 32'(block_valid_bytes);` nằm trong `always_comb` sẽ sinh ra một cây nhân khổng lồ liên tục evaluate gây lãng phí năng lượng và kéo dài Critical Path.
   - *Cách khắc phục:* Đưa vào `always_ff` và chỉ tính toán 1 lần lúc `start_i`.
2. **Vòng Lặp Chờ Nhau 32-Stage (Cascaded Adder Chain):**
   - Trong `advance_k_seed32` và logic sinh `lane_ic` của packer, tồn tại một vòng lặp `for` chạy 32 lần mà giá trị `ic` của vòng sau lại phụ thuộc vào vòng trước (Cascaded `if-else` adders). Nó tạo ra một Ripple Logic sâu 32 tầng.
   - *Cách khắc phục:* Nhờ vào nguyên tắc `IC % 32 == 0` đã quy định, ta có thể dùng toán học thay thế vòng lặp:
     - Tính `lane_ic[lane] = cfg_k_seed_ic_i + 16'(lane);` (song song hóa 100%).
     - Tính bước nhảy K-Seed: `ic_next = ic_i + 32; if (ic_next >= input_c) ...` (rút gọn về 1 bộ cộng).

---

## Phần 2: Kế Hoạch Tái Cấu Trúc Toàn Diện (Refactoring Plan)

Mục tiêu: Áp dụng nguyên lý "Chia để trị", bóc tách logic để làm gọn file, loại bỏ hoàn toàn các lỗi Timing đã đề cập và biến Top-level thành cấu trúc "Thuần đi dây" (Pure Structural).

### Giai đoạn 2.1: Tái Cấu Trúc Linebuffer Packer
File `conv_channel_linebuf_packer.sv` (hơn 800 dòng) sẽ được chẻ thành 3 file:

1. **[NEW] `hw/rtl/systolic/linebuf_pkg.sv`**
   - Đóng gói toàn bộ các hàm `function automatic` xử lý bitwise (`beat_base`, `valid_c_bytes`, `merge_beats`, `unpack_row`, `merge_row_lanes`, `merge_kgen_lanes`).
2. **[NEW] `hw/rtl/systolic/linebuf_sram_ring.sv`**
   - Bóc tách toàn bộ mảng physical bộ nhớ (5 khối `tc_sram`) và khối so sánh Tag. Wrapper này quản lý riêng vấn đề Hit/Miss.
3. **[MODIFY] `hw/rtl/systolic/conv_channel_linebuf_packer.sv`**
   - Chỉ giữ lại Máy trạng thái (Pure Controller). Instantiate module `linebuf_sram_ring` và `import linebuf_pkg::*`.
   - Cập nhật Fix triệt để lỗi Multiplier và lỗi 32-stage loop.

### Giai đoạn 2.2: Tái Cấu Trúc Systolic Controller
File `systolic_controller.sv` (hơn 860 dòng) sẽ được giải phóng logic FSM để trở về đúng bản chất Top Wrapper:

1. **[NEW] `hw/rtl/systolic/systolic_pkg.sv`**
   - Chứa `input_row_t`, `ofm_row_t`.
   - Chứa hàm tính toán bước nhảy đã sửa lỗi: `advance_k_seed32` (công thức mới, O(1) cycle).
2. **[NEW] `hw/rtl/systolic/systolic_drain_engine.sv`**
   - Module chuyên biệt chứa `drain_state_e` FSM.
   - Quản lý 4 cổng OBI Master (Ghi), cộng dồn Accumulator (Psum), và handshake với Requant.
3. **[NEW] `hw/rtl/systolic/systolic_fetch_engine.sv`**
   - Module chuyên biệt chứa `state_e` FSM.
   - Quản lý 1 cổng OBI Master (Đọc), nạp FIFOs, điều phối Linebuffer Packer.
4. **[MODIFY] `hw/rtl/systolic/systolic_controller.sv`**
   - Lột xác thành Structural Wrapper. Mã nguồn sẽ chỉ còn khoảng ~250 dòng làm duy nhất một việc: Instantiate các Engine, Requant, MAC Array, Packer, Registers và tiến hành đấu dây (Wiring).

---

## Kết Luận
Việc thực hiện kế hoạch refactor này sẽ tiêu tốn công sức tạo ra thêm 5 file mới, nhưng đổi lại, hệ thống NPU của bạn sẽ:
- Dễ đọc, dễ test độc lập và dễ debug hơn gấp nhiều lần.
- Đạt được chuẩn mực Fmax cao (không còn logic sâu 32 tầng).
- Hoàn toàn tuân thủ các Rule khắt khe nhất của quy trình Tape-out ASIC.
