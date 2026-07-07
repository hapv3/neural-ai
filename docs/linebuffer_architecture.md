# Kiến trúc Linebuffer NPU

Tài liệu này ghi lại quá trình kiểm tra performance của
`conv_channel_linebuf_packer`, lý do không tiếp tục dùng block này làm
performance path, và kiến trúc linebuffer mới cần implement để feed systolic
array ở mục tiêu `1 vector/cycle`.

Quyết định hiện tại:

- `conv_channel_linebuf_packer` chỉ được xem là RTL tham chiếu/legacy cho
  correctness. Không dùng làm default performance path.
- Linebuffer mới là `conv_linebuf_stream_packer`, thiết kế lại theo window-cache
  + segment prefetch để phát IFM vector trực tiếp cho systolic.
- Mục tiêu steady-state của linebuffer mới là `1 x 256-bit vector/cycle` vào
  systolic IFM stream cho các shape được support.
- Native stride tập trung vào `stride=1`, `stride=2`. Stride lớn hơn
  phải do compiler/scheduler decompose hoặc đi qua packed prepare backup.

## 1. Contract RTL Legacy Đã Đo

| Thông số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `K_MAX` | `5` | Native support kernel height/width `1..5`. Kernel `7x7/9x9` phải được compiler/scheduler decompose trước khi dùng linebuffer. |
| `MAX_INPUT_W` | `640` | SRAM line depth theo input width thật, không tính padding. Tile/stripe có `input_w > 640` phải split trước. |
| `DATA_WIDTH` | `256-bit` | Một SRAM word chứa một C-block `32 x INT8`, khớp IFM row width của systolic array. |
| `C_BLOCK` | `32` kênh | Mỗi run xử lý một `c_base`/C-block. Đổi `c_base` thì linebuffer flush ở `start_i`. |

`conv_channel_linebuf_packer` support padding bằng zero-injection, không
materialize padding vào SRAM. Vì vậy `MAX_INPUT_W=640` là width input thật, còn
`pad_h/pad_w` chỉ ảnh hưởng bounds check và output shape. Contract này hữu ích
để giữ semantics, nhưng không đủ làm performance target.

## 2. SRAM Sizing

Với cấu hình hiện tại:

- Dung lượng một line: `640 words x 32B = 20 KiB`.
- Dung lượng ring buffer: `5 lines x 20 KiB = 100 KiB`.
- Mỗi SRAM word là một vector 32 byte tại một tọa độ `x` cho một `c_base`.

Thiết kế hiện tại không tag theo `c_base/cblk`; row tag chỉ theo `ih`. Điều này
đúng với rule mỗi run chỉ dùng một `c_base` và flush khi start. Nếu cần giữ
nhiều C-block trong cùng ring buffer, tag phải mở rộng thành `{ih, cblk}`.

## 3. Dataflow Legacy

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

## 4. Compiler/Scheduler Rules Legacy

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
- Conv2D scheduler đã từng ưu tiên linebuffer này trước packed prepare. Sau
  performance review bên dưới, rule này bị thay thế: chỉ linebuffer mới
  `conv_linebuf_stream_packer` được ưu tiên làm performance path.
- Halo retention/cascade giữa stripe chưa được implement; `start_i` hiện clear
  valid state của ring.

## 5. Verification Legacy

Regression trực tiếp cho contract này nằm ở
`hw/rtl/systolic/tb/test_conv_channel_linebuf_packer.py`:

- Positive: `1x1`, `3x3`, `5x5`, coalesced `3x3/C3`, stride `1/2`, padding,
  tail channel, unaligned/cross-beat, width boundary `640`.
- Sweep: toàn bộ kernel `1x1..5x5`.
- Negative: reject `7x7`, `9x9`, và `input_w=641`.

## 6. Quá Trình Check Performance

Mục tiêu của check là xác định linebuffer hiện tại có thể thay thế packed
im2col prepare cho Conv2D hay không. Case quan trọng nhất là deep Conv2D có
`IC > 32`, vì đây là phần chiếm phần lớn workload YOLO/CNN sau first layer.

### 6.1 Baseline Trước Khi Có Linebuffer

Các số đo trước đó cho thấy packed prepare bằng scalar software không thể là
performance path:

- `Conv1x1 IC=33`: `prepare=108592`, `gemm=662`, `total=109382`.
- `Conv3x3 IC=3`: `prepare=95822`, `gemm=200`, `total=96096`.

Sau khi tối ưu `Conv1x1 IC=33` bằng iDMA contiguous/2D pack, path pointwise đã
gần như giải quyết được prepare:

- `Conv1x1 IC=33`: `gemm=666`, `idma=2`, `spatz=0`, `scalar=0`.

Kết luận từ giai đoạn này: vấn đề chính còn lại là spatial Conv2D
`KH x KW > 1`, đặc biệt `3x3/5x5` với `IC` lớn. Vì vậy linebuffer được thêm để
tránh materialize im2col trong TCDM.

### 6.2 Case Đo Chính: Conv3x3 IC120

Command dùng để đo:

```bash
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  CONV_PERF_CASE=20 \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf
```

Test pass:

```text
TESTS=1 PASS=1 FAIL=0
```

PMU tổng:

```text
cycles=24170
systolic: compute=544 (2.25%) ifm_req=2038 ofm_req=4288 ofm_stall=0
tcdm: req=7588 gnt=7562 stall=26 read=4216 write=3372
```

Python monitor:

```text
conv_perf state monitor events:
cycles=24448
compute=544
weight_load=1088
ofm_valid=544
linebuf_row_valid=544
linebuf_row_ready=544
linebuf_obi_req=950
linebuf_obi_stall=26
ofm_fifo_empty=20736
ofm_fifo_full=0
```

Systolic state cycles:

```text
total=24448 active=18700
COMPUTE      15202 total=62.18% active=81.29%
IDLE          5748 total=23.51%
WAIT_DRAIN    2374 total=9.71%  active=12.70%
LOAD_WEIGHTS  1122 total=4.59%  active=6.00%
DONE             2 total=0.01%
```

OFM drain state cycles:

```text
DRAIN_IDLE         21280
DRAIN_ACCUM_READ    2640
DRAIN_ACCUM_WRITE    528
DRAIN_ACCUM_REQUANT    0
```

Linebuffer state cycles:

```text
total=24448 active=15202
CH_IDLE           9246
CH_COAL_PREP      4896 active=32.21%
CH_COAL_READ_REQ  3400 active=22.37%
CH_COAL_READ_WAIT 3400 active=22.37%
CH_FILL_REQ0       570
CH_ENSURE          544
CH_FILL_WAIT0      544
CH_FILL_WRITE      544
CH_COAL_EMIT       544
CH_FILL_REQ1       380
CH_FILL_WAIT1      380
```

Firmware stats:

```text
linebuffer split conv3x3 IC120 stats:
rows=16 k_tiles=34 prepare=0 gemm=19418 total=19490
last_prepare=0 last_gemm=4162 idma=0 idma_tx=0 spatz=0 scalar=0
```

Ý nghĩa của case:

- `M=16` output spatial rows trong micro-tile.
- `k_tiles=34`; mỗi K tile tạo một IFM vector cho mỗi output row.
- Tổng IFM vector đúng là `16 * 34 = 544`.
- `systolic.compute=544` đúng bằng số vector thật vào MAC array.
- `linebuf_row_valid=544` và `linebuf_row_ready=544`, nên không mất dữ liệu và
  không bị backpressure ở output linebuffer.

### 6.3 Phân Tích Bottleneck

`conv_channel_linebuf_packer` không bị nghẽn bởi TCDM grant:

- `linebuf_obi_stall=26` trên tổng `950` OBI request.
- `tcdm stall=26` trên tổng `7588` request.
- OFM FIFO không full: `ofm_fifo_full=0`.
- Linebuffer output luôn được nhận: `row_valid=row_ready=544`.

Bottleneck nằm trong cách FSM tạo mỗi IFM vector:

- MAC active thật chỉ `544` cycles.
- Controller ở state `COMPUTE` tới `15202` cycles.
- Bubbles trong compute state là `15202 - 544 = 14658` cycles.
- Tốc độ hiệu dụng trong compute state là `544 / 15202 = 0.0358`
  vector/cycle, tương đương khoảng `27.9` cycles/vector.

Ba state linebuffer lớn nhất:

- `CH_COAL_PREP=4896`;
- `CH_COAL_READ_REQ=3400`;
- `CH_COAL_READ_WAIT=3400`;
- tổng `11696 / 15202 = 76.9%` active linebuffer cycles.

Nguyên nhân kiến trúc:

- Coalesce/KGEN mixed tap hiện đi theo kiểu scan `kh/kw/ic` tuần tự.
- Mỗi output vector cần nhiều bước prepare, request, wait, merge trước khi
  `CH_COAL_EMIT`.
- Row SRAM đang được dùng như nguồn đọc tap trực tiếp cho từng output vector,
  không phải backing store cho một window cache đã sẵn sàng.
- Vì vậy dù correctness đúng, design không thể đạt `1 vector/cycle`.

### 6.4 Tác Động Của Split IC Tail

Trước khi split IC tail tốt hơn, case IC120 từng rơi vào path tệ:

```text
gemm=85008 cycles=89316
```

Sau khi scheduler split IC thành các command hợp lý hơn:

```text
gemm=19418 cycles=24170
```

Split command giải quyết fallback/tail quá chậm, nhưng không sửa bottleneck
gốc của linebuffer: chỉ phát `544` vector sau khoảng `15202` cycles compute
state. Vì vậy đây là tối ưu cần thiết cho correctness/perf bước đầu, nhưng
không thay đổi quyết định phải viết lại linebuffer.

## 7. Quyết Định Thiết Kế Lại

`conv_channel_linebuf_packer` bị loại khỏi roadmap performance vì không có
đường nâng cấp nhỏ nào đạt `1 vector/cycle`:

- Thêm FIFO chỉ che backpressure downstream, không giảm `CH_COAL_PREP` và
  `CH_COAL_READ_*`.
- Tăng SRAM depth không giúp vì stall không đến từ capacity.
- Tối ưu scheduler không loại được scan tuần tự trong RTL.
- Giữ row SRAM 1-read làm nguồn tap trực tiếp không đủ cho coalesce/KGEN mixed
  tap khi cần emit đều mỗi cycle.

Linebuffer mới phải đổi dataflow:

```text
Shared TCDM / stripe in TCDM
        │
        ▼
row SRAM banks / segment prefetch
        │
        ▼
window cache: K_MAX x K_MAX x 32B
        │
        ▼
lane descriptor generator
        │
        ▼
32-lane byte mux / zero inject
        │
        ▼
skid FIFO
        │
        ▼
systolic IFM stream
```

Vai trò từng khối:

- Row SRAM banks giữ input rows hoặc prefetched segments.
- Window cache giữ window đang emit, không đọc tap trực tiếp từ SRAM cho từng
  lane trong mỗi output.
- Lane descriptor generator tạo mapping `{kh,kw,ic}` cho từng lane, bao gồm
  KGEN seed và K tile increment.
- Lane mux chọn byte từ window cache hoặc zero padding.
- Skid FIFO tách timing giữa mux lớn và systolic ready/valid.

## 8. Target `conv_linebuf_stream_packer`

Feature bắt buộc:

- Non-KGEN/non-coalesce: phát vector theo descriptor `{oh, ow, kh, kw, c_base}`.
- Coalesce non-KGEN: khi `KH*KW*valid_channels <= 32`, phát một vector chứa
  toàn bộ window theo lane order `{kh, kw, ic}`.
- KGEN mixed tap: host/Python chỉ đưa seed `{kh,kw,ic}` và `k_tile_count`; RTL
  tự sinh lane descriptor 32 bước mỗi K tile.
- Padding: zero-injection, không materialize padding vào SRAM.
- `1x1` pad0: bypass path riêng, không đi qua window cache.
- Native stride: `1`, `2`.

Performance contract:

- Với window cache hit và systolic ready, emit path phải phát `1 x 256-bit`
  vector/cycle.
- Refill/prefetch không được nằm nối tiếp trên critical emit path trong steady
  state.
- Counter/PMU phải đo riêng `window_refill`, `emit_valid`, `emit_ready`,
  `emit_stall`, `k_tile_transition`, và `padding_zero`.

Storage estimate:

- `K_MAX=5`, `C_BLOCK=32B` tạo window cache `5*5*32B = 800B = 6.4Kb`.
- Dung lượng này không lớn cho ASIC; rủi ro chính là mux/timing của
  `25 x 32B` vào 32 lanes, nên cần pipeline hoặc chia mux theo lane group.

Stride policy:

- `stride=1`: với 5 row banks, mỗi bank đọc một column/cycle là đủ để cập nhật
  window `5x5` khi trượt sang output kế tiếp.
- `stride=2`: không implement bằng scan stride-1 rồi bỏ output. Cần prefetch
  segment rộng hơn hoặc x-banking để window cache có sẵn các column kế tiếp
  trước khi emit.
- `stride>=3`: không là native performance target trong phase này. Compiler
  phải decompose/rewrite hoặc dùng packed prepare backup.

## 9. Verification Cho Linebuffer Mới

Unit tests bắt buộc trước khi nối cluster:

- `1x1` bypass: pad0, stride1, tail width, unaligned address.
- Non-KGEN single tap: `3x3`, `5x5`, pad0/pad1, stride1/2.
- Coalesce: `3x3 IC=3`, `5x5 IC=1`, tail lanes zero.
- KGEN mixed tap: `3x3 IC=32`, `3x3 IC=96`, `3x3 IC=120`, `5x5 IC=32`.
- Padding: top/bottom/left/right/asymmetric.
- Read alignment: same-beat and cross-beat 256-bit reads.
- Backpressure: random `ifm_ready_i` stalls while output data remains exact.
- Performance assertion: after warm-up, accepted vectors must be one per cycle
  for supported steady-state regions.

Cluster tests:

- Run existing `CONV_PERF_CASE=17/18/20`.
- Add large-shape first-layer YOLO case `3x3/s2/p1/IC=3`.
- Collect PMU after every run and fail the perf test if linebuffer active
  cycles/vector exceeds the configured threshold.
