# Báo cáo Code Review: Khối AFU & Requantization

Xin chúc mừng! Bạn đã tích hợp thành công hai mảnh ghép phần cứng quan trọng nhất để gỡ nút thắt cổ chai cho hệ thống `neural-ai`. Việc tách bạch **Requantization (tính toán tuyến tính)** cắm thẳng vào Systolic Array và **AFU (tính toán phi tuyến tính qua LUT)** thành một bộ tăng tốc riêng là một quyết định kiến trúc cực kỳ chuẩn xác và chuyên nghiệp.

Dưới đây là phần Review chi tiết mã nguồn RTL (`requant_pipeline.sv` và `afu_core.sv`) cùng với các đề xuất tối ưu (Performance/Synthesis):

---

## 1. Module `requant_pipeline.sv`

> [!SUCCESS]
> **Điểm sáng:** 
> - Logic tính toán chuẩn xác cho luồng Requantization chuẩn của TensorFlow Lite (cộng Bias, nhân Multiplier, dịch phải có làm tròn Rounding, cộng Zero Point và Kẹp Clamp).
> - Sử dụng vòng lặp `for` để bung (unroll) phần cứng song song cho cả 32 cột của Systolic Array.

> [!WARNING]
> **Vấn đề Timing & Diện tích (Critical Path & Area):**
> Mã nguồn hiện tại đang viết dưới dạng `always_comb` (tổ hợp hoàn toàn). Trong 1 chu kỳ xung nhịp, mạch phải thực hiện 32 phép cộng 64-bit, 32 phép nhân 64-bit, 32 phép dịch bit (barrel shifter), và 32 cụm so sánh kẹp giá trị. 
> 
> **Hậu quả khi tổng hợp (Synthesis):**
> - Không thể nào đạt được xung nhịp cao (ví dụ 500MHz) vì đường trễ (Critical Path) quá dài.
> - Phép nhân `scaled = biased * 64'($signed(multiplier_i[i]))` ép Synthesizer đúc ra các khối nhân 64x64 siêu to khổng lồ, làm "cháy" tài nguyên DSP của chip.

> [!TIP]
> **Đề xuất sửa đổi (Action Items):**
> 1. **Cắt nhỏ Pipeline (Pipelining):** Phải nhét thêm ít nhất 2 đến 3 tầng thanh ghi (`always_ff`) vào giữa các bước. Ví dụ: Tầng 1 (Add Bias + Multiply) -> Tầng 2 (Shift + Add Zero Point) -> Tầng 3 (Clamp).
> 2. **Ép kiểu phép nhân (Multiplication Casting):** Vì `biased` chỉ khoảng 33-bit và `multiplier` là 32-bit. Hãy ép kiểu phép nhân thành `34-bit * 32-bit` thay vì `64 * 64` để tiết kiệm DSP: 
> ```verilog
> logic signed [33:0] biased_34;
> biased_34 = 34'($signed(acc_i[i])) + 34'($signed(bias_i[i]));
> scaled = 64'(biased_34) * 64'($signed(multiplier_i[i])); // Công cụ tổng hợp sẽ tối ưu tốt hơn
> ```

---

## 2. Module `afu_core.sv`

> [!SUCCESS]
> **Điểm sáng:** 
> - Thiết kế State Machine (FSM) phân chia 2 Pipeline Stages (`ST_PROCESS` và Stage 2) khá rõ ràng.
> - Kỹ thuật `Pipeline hazard fix` (lưu lại rdata của SRAM khi S2 bị stall) được implement rất cứng cáp, giải quyết đúng bản chất độ trễ 1-cycle của `tc_sram`.
> - Hỗ trợ tốt các mode giải nén dữ liệu `MODE_8BIT`, `16BIT`, `32BIT`.

> [!CAUTION]
> **Vấn đề diện tích (Logic Area) ở đoạn dịch bit dữ liệu vào:**
> Dòng số 86: `assign shift_in  = in_buf_q >> {in_off_s1, 3'd0};`
> Bạn đang yêu cầu chip thiết kế một bộ **256-bit Barrel Shifter** (dịch phải mảng 256 bit với độ dịch tùy biến lên tới 31 bytes). Bộ Shifter này tốn một lượng khổng lồ các cổng logic (MUX) và làm chậm đáng kể Critical Path của Stage 1.

> [!TIP]
> **Đề xuất sửa đổi (Action Items):**
> Mặc dù logic chạy đúng, nhưng để tối ưu trên chip Edge, thay vì dùng một bộ 256-bit Shifter, bạn có thể cân nhắc dùng một mảng MUX (Multiplexer) đơn giản chỉ chắt lọc ra đúng `LUT_LANES` (ví dụ 4 byte) thay vì dịch toàn bộ 256-bit.
> ```verilog
> // Ví dụ tối ưu thay vì shift toàn bộ 256 bit
> for (genvar i = 0; i < LUT_LANES; i++) begin
>     logic [4:0] byte_idx;
>     assign byte_idx = in_off_s1 + 5'(i);
>     assign lut_idx_s1[i] = in_buf_q[byte_idx * 8 +: 8]; // Dùng array index
> end
> ```
> *Cú pháp array indexing với biến ở trên có thể phải sửa lại một chút tùy tool tổng hợp, nhưng ý tưởng cốt lõi là chỉ MUX đúng 4 bytes cần thiết thay vì dịch 32 bytes.*

## Tóm tắt Review
Code viết logic cực tốt, không thấy xuất hiện Bug logic nghiêm trọng nào gây sai lệch dữ liệu. Bạn có thể chốt phương án kiến trúc này. Tuy nhiên, nếu mang bộ RTL này đi tổng hợp (Synthesis) thành silicon thực tế, bạn **bắt buộc phải chia thêm Pipeline Stage** cho khối `requant_pipeline` để tránh rớt xung nhịp.
