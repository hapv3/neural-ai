# Kiến trúc Linebuffer NPU (Channel Block Mode)

Tài liệu này ghi lại contract hiện tại của `conv_channel_linebuf_packer` và các
điểm cần giữ rõ khi compiler/scheduler phát lệnh Conv2D trực tiếp cho systolic
array.

## 1. Contract RTL Hiện Tại

| Thông số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `K_MAX` | `5` | Native support kernel height/width `1..5`. Kernel `7x7/9x9` phải được compiler/scheduler decompose trước khi dùng linebuffer. |
| `MAX_INPUT_W` | `640` | SRAM line depth theo input width thật, không tính padding. Tile/stripe có `input_w > 640` phải split trước. |
| `DATA_WIDTH` | `256-bit` | Một SRAM word chứa một C-block `32 x INT8`, khớp IFM row width của systolic array. |
| `C_BLOCK` | `32` kênh | Mỗi run xử lý một `c_base`/C-block. Đổi `c_base` thì linebuffer flush ở `start_i`. |

`conv_channel_linebuf_packer` support padding bằng zero-injection, không
materialize padding vào SRAM. Vì vậy `MAX_INPUT_W=640` là width input thật, còn
`pad_h/pad_w` chỉ ảnh hưởng bounds check và output shape.

## 2. SRAM Sizing

Với cấu hình hiện tại:

- Dung lượng một line: `640 words x 32B = 20 KiB`.
- Dung lượng ring buffer: `5 lines x 20 KiB = 100 KiB`.
- Mỗi SRAM word là một vector 32 byte tại một tọa độ `x` cho một `c_base`.

Thiết kế hiện tại không tag theo `c_base/cblk`; row tag chỉ theo `ih`. Điều này
đúng với rule mỗi run chỉ dùng một `c_base` và flush khi start. Nếu cần giữ
nhiều C-block trong cùng ring buffer, tag phải mở rộng thành `{ih, cblk}`.

## 3. Dataflow

Luồng dữ liệu hiện tại:

1. Host/scheduler cấu hình shape, stride, padding, `c_base`, và byte strides.
   `input_base` là origin của padded output row: firmware đặt bằng
   `input_addr - pad_h * row_stride_bytes`; padding ngang vẫn được xử lý bằng
   `pad_w` và bounds check, không materialize vào SRAM.
2. Packer bảo đảm các input row cần thiết nằm trong ring SRAM.
3. Với mỗi output spatial/kernal position, packer đọc `C_BLOCK=32` byte từ row
   SRAM hoặc phát zero nếu vị trí nằm ngoài input bounds.
4. `1x1` không padding dùng bypass path riêng: đọc trực tiếp OBI/TCDM và bỏ qua
   ring SRAM.
5. Output row được feed vào systolic IFM stream bằng ready/valid.

Với `KH*KW*IC <= 32`, RTL hỗ trợ `coalesce` mode: mỗi output spatial phát một
IFM row duy nhất chứa toàn bộ window theo thứ tự lane `{kh, kw, ic}`, khớp
layout weight K-major của systolic array.

Với `K > 32`, RTL có KGEN v0 cho micro-tile:

1. Host/Python tính layer/tile config, seed `{kh,kw,ic}` và `k_tile_count`.
2. Snitch ghi `REG_LB_K_SEED`, `REG_LB_K_TILES`, bật `REG_LB_CTRL_KGEN`.
3. Snitch start systolic một lần và wait một lần.
4. `systolic_controller` tự lặp qua các K tile, tăng lane descriptor 32 bước
   mỗi tile theo thứ tự `{kh,kw,ic}`.
5. Tile đầu ghi OFM INT32; các tile sau đọc psum và accumulate vào cùng OFM.

Gate hiện tại của KGEN v0: `M <= SYSTOLIC_GEMM32_ACCUM_TILE_M` (hiện là
`16`), `IC <= 32` hoặc `IC % 32 == 0`. Ngoài gate này scheduler dùng per-slice
linebuffer accumulation hoặc packed prepare. Giới hạn `M` hiện là gate
firmware/test conservative, không còn bị khóa bởi default OFM FIFO depth `8`:
controller đã có drain engine song song với main compute FSM để đọc psum từ
O-TCDM port 0, accumulate OFM FIFO row, writeback, và chỉ stall compute qua
backpressure bình thường nếu drain engine không theo kịp.

Unaligned vector được xử lý bằng `merge_beats`: nếu vector 32 byte cắt qua biên
256-bit beat, packer phát hai read rồi ghép lại.

## 4. Compiler/Scheduler Rules

- Native linebuffer path: `kernel_h <= 5`, `kernel_w <= 5`, `input_w <= 640`.
- Coalesced fast path: chỉ dùng khi `KH*KW*IC <= 32` và output spatial tile
  không vượt giới hạn một GEMM linebuffer launch hiện tại.
- KGEN fast path: chỉ dùng khi `KH*KW*IC > 32`,
  `M <= SYSTOLIC_GEMM32_ACCUM_TILE_M` (hiện là `16`), và channel shape không
  cắt ngang C-block (`IC <= 32` hoặc `IC % 32 == 0`).
- `7x7`, `9x9`, hoặc tile width lớn hơn `640` phải đi qua một trong hai hướng:
  decompose thành sub-kernel/tile nhỏ hơn, hoặc fallback packed prepare bằng
  iDMA/Spatz.
- `1x1` pad0 nên dùng bypass path; `1x1` có padding vẫn dùng ring/zero-injection
  fallback để giữ correctness.
- Conv2D scheduler ưu tiên linebuffer trước khi dùng packed prepare. iDMA/Spatz
  prepare chỉ là backup khi shape/address không nằm trong contract linebuffer
  hoặc khi cần debug comparison.
- Halo retention/cascade giữa stripe chưa được implement; `start_i` hiện clear
  valid state của ring.

## 5. Verification

Regression trực tiếp cho contract này nằm ở
`hw/rtl/systolic/tb/test_conv_channel_linebuf_packer.py`:

- Positive: `1x1`, `3x3`, `5x5`, coalesced `3x3/C3`, stride `1/2`, padding,
  tail channel, unaligned/cross-beat, width boundary `640`.
- Sweep: toàn bộ kernel `1x1..5x5`.
- Negative: reject `7x7`, `9x9`, và `input_w=641`.
