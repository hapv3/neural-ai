# NPU Systolic + Linebuffer Review và Refactor Plan

Tài liệu này review cấu trúc hiện tại của `conv_linebuf_stream_packer.sv`,
`systolic_controller.sv`, và `npu_systolic_array.sv`, sau các tối ưu linebuffer
gần đây. Mục tiêu là tách rõ:

- điều gì đã đúng về kiến trúc;
- bottleneck nào đã được đo bằng perf case;
- rủi ro timing/synthesis nào cần xử lý;
- refactor nào nên làm trước để không phá correctness.

## 1. Tóm tắt hiện trạng

Linebuffer hiện đã chuyển từ mô hình đọc tap tuần tự sang mô hình gần đúng cần
cho conv streaming:

- `K_MAX=5`, `STRIDE_MAX=2`, `BANKS=K_MAX*STRIDE_MAX=10` logical banks.
- Có `window_q`/`slide_window` để giữ window sẵn sàng.
- Có lane descriptor cho KGEN/coalesce: `lane_kh`, `lane_kw`, `lane_ic`.
- Có beat FIFO cho OBI response để phát request fill liên tục hơn.
- Có output pipeline stage để `CH_STREAM_EMIT` có thể chạy 1 vector/cycle khi
  window đã sẵn sàng.

Systolic array là datapath 32x32 weight-stationary, input skew + output deskew,
và valid delay cố định. Structure này phù hợp cho MAC throughput cao, nhưng hiện
bị backpressure bởi OFM drain/accumulation khi FIFO đầy.

## 2. Kết quả perf gần nhất

Các số dưới đây lấy từ sim cluster sau khi fix `conv_linebuf_stream_packer.sv`.

### Case 18: conv3x3 IC32, 16x16

- PASS.
- PMU cycles: 4824.
- K tiles: 9.
- `linebuf_row_valid = linebuf_row_ready = 144`.
- `CH_STREAM_EMIT = 144`.
- Bottleneck còn lại chủ yếu là `CH_STREAM_DONE = 808`.

Nhận xét: steady emit đã đúng 1 vector/cycle. Overhead chủ yếu là tile boundary.

### Case 19: conv3x3 IC96, 16x16

- PASS.
- PMU cycles: 9168.
- K tiles: 27.
- `linebuf_row_valid = linebuf_row_ready = 432`.
- `CH_STREAM_EMIT = 432`.
- `CH_STREAM_DONE = 2626`.
- `linebuf_obi_req = 476`, `linebuf_obi_stall = 44`.

Nhận xét: khi nhiều K tile hơn, `CH_STREAM_DONE` tăng mạnh. Bottleneck không còn
là emit trong window, mà là chuyển tile + refill/prefetch + drain.

### Case 20: split conv3x3 IC120, 16x16

- PASS.
- PMU cycles: 68968 sau commit `b074a68`.
- K tiles: 204.
- `linebuf_row_valid = linebuf_row_ready = 8704`.
- `CH_STREAM_EMIT = 8704`.
- Systolic states:
  - `COMPUTE = 14236`
  - `WAIT_DRAIN = 31101`
  - `IDLE = 23297`
  - `LOAD_WEIGHTS = 600`
- Linebuffer states:
  - `CH_IDLE = 24485`
  - `CH_FILL_REQ0 = 11648`
  - `CH_STREAM_DONE = 9617`
  - `CH_STREAM_EMIT = 8704`
- `linebuf_obi_req = 20280`, `linebuf_obi_stall = 0`.
- `ofm_fifo_full = 25`.

Nhận xét: đây là bằng chứng rõ nhất rằng linebuffer emit đã không còn là điểm
nghẽn chính. Điểm nghẽn hiện tại là controller/drain, firmware spatial tiling,
và linebuffer refill theo từng `c_base`.

## 3. Đánh giá kiến trúc

### 3.1. Điểm đúng

- 10 logical banks là lựa chọn hiện tại cho target `K<=5`, `stride<=2`. Với
  mapping `(row_slot, x % stride)`, mỗi chu kỳ có thể đọc các cột mới cho slide
  window mà không biến row SRAM thành nguồn tap tuần tự.
- `window_q` giúp tách storage fill khỏi emit. Đây là thay đổi cốt lõi để đạt
  1 vector/cycle trong `CH_STREAM_EMIT`.
- Beat FIFO/response engine giúp fill row không còn bắt buộc đi theo nhịp
  request-wait-write cứng từng beat.
- Psum FIFO đã tách reader và adder/writer, đúng hướng để giảm bubble trong
  accumulation.
- Weight preload và linebuffer prefetch trong `WAIT_DRAIN` đã bắt đầu overlap
  được một phần công việc cho K tile kế tiếp.

### 3.2. Điểm còn yếu

- Main controller vẫn vận hành theo chuỗi:
  `LOAD_WEIGHTS -> COMPUTE -> WAIT_DRAIN -> LOAD_WEIGHTS -> COMPUTE`.
  Tile kế tiếp chỉ bắt đầu khi `drain_cnt_q == 0 && ofm_fifo_empty`.
- `CH_STREAM_DONE` đang là proxy cho tile-boundary wait. Case 20 có
  `CH_STREAM_DONE=9617`, còn `WAIT_DRAIN=31101` và `IDLE=23297`.
- Accumulation vẫn read psum cũ từ TCDM rồi write lại cho nhiều K tile. Với nhiều
  K tile, traffic này tăng tuyến tính và đẩy ngược backpressure lên array.
- `ofm_fifo_full` xuất hiện trong case 20, nghĩa là drain path chưa đủ nhanh để
  hấp thụ output, làm `array_pipe_ready` kéo chậm toàn bộ systolic array.
- `obi_i` vẫn là một cổng đọc dùng chung cho weight preload và linebuffer
  prefetch. Hiện overlap có giới hạn vì linebuffer prefetch chỉ chạy sau khi
  weight preload fetch done.
- Test monitor đang label enum value 6 là `CH_FILL_WRITE`, trong khi RTL hiện là
  `CH_FILL_DRAIN`. Đây là lỗi reporting, không phải lỗi datapath.

## 4. Rủi ro timing/synthesis cần xử lý

Các điểm dưới đây là risk cần giảm trước khi synth nghiêm túc. Không nên gọi là
"vi phạm ASIC chí mạng" nếu chưa có report synthesis/STA, nhưng chúng có khả
năng tạo critical path hoặc area không cần thiết.

### 4.1. Chia/modulo theo `STRIDE_MAX`

`bank_index()` dùng `% STRIDE_MAX`, `bank_word_addr()` dùng `/ STRIDE_MAX`.
Với default mới `STRIDE_MAX=2`, tool có thể reduce mapping thành bit select và
shift. Nếu sau này bật lại `STRIDE_MAX=3`, cần xử lý `/3` và `%3` bằng helper
chuyên biệt hoặc LUT nhỏ.

Không nên đổi mặc định sang `STRIDE_MAX=4` chỉ để dùng bit mask, vì như vậy đổi
kiến trúc từ 10 banks sang 20 banks. Hướng hiện tại:

- Giữ native support ở stride 1/2.
- Stride 3 không là native performance target; compiler/scheduler phải fallback
  sang packed prepare hoặc decompose thành nhiều bước nội bộ.
- Chỉ cân nhắc cấu hình build-time `STRIDE_MAX=3` hoặc `4` nếu có layer bắt buộc
  và synthesis chấp nhận cost.

### 4.2. Lane KGEN loop phụ thuộc tuần tự

Logic sinh `lane_kh/lane_kw/lane_ic` hiện advance từng lane nối tiếp. Vì
`ARRAY_DIM=32`, đây có thể là path sâu trong `always_comb`.

Hướng tối ưu:

- Với phần lớn K tile, `lane_ic = seed_ic + lane` khi chưa wrap qua `input_c`.
- Tách fast path không wrap và slow path wrap boundary.
- Hoặc precompute 32 lane descriptors vào register khi nhận `start_i` /
  `next_tile_i`, thay vì tính lại liên tục mỗi cycle.

### 4.3. `coalesce_k_bytes` multiplier

`coalesce_k_bytes = kernel_h * kernel_w * block_valid_bytes` đang nằm trong
`always_comb`. Vì `kernel_h/kernel_w <= 5`, có thể thay bằng LUT/case nhỏ hoặc
register hoá khi config/start.

### 4.4. `build_emit_row` quá lớn

`build_emit_row()` vừa xử lý KGEN, coalesce thường, và non-coalesce. Nó có nhiều
loop lồng nhau và phụ thuộc nhiều config. Dù đã pipeline stage output, logic này
vẫn là candidate critical path.

Hướng tối ưu:

- Tách `linebuf_lane_mapper` riêng cho KGEN/coalesce.
- Register lane descriptor trước khi vào mux data.
- Giữ data mux trong một stage riêng, tránh trộn coordinate generation với data
  selection.

## 5. Refactor đề xuất

Refactor nên đi theo thứ tự giữ behavior ổn định. Không nên chẻ module lớn ngay
khi còn đang thay đổi thuật toán scheduler, vì dễ làm diff lớn và khó debug.

### Phase A: Correctness + instrumentation cleanup

Mục tiêu: làm perf report và debug state đáng tin trước.

- Sửa label monitor: enum value 6 là `CH_FILL_DRAIN`, không phải
  `CH_FILL_WRITE`.
- Thêm counter rõ cho:
  - linebuffer prefetch cycles;
  - next-tile wait cycles;
  - psum read stall;
  - psum write stall;
  - OFM FIFO backpressure.
- Tách perf assert cho `CH_STREAM_EMIT == linebuf_row_ready` ở các case KGEN
  nhỏ để bắt regression stream throughput.

### Phase B: Timing-risk cleanup trong linebuffer

Mục tiêu: giảm logic sâu mà không đổi microarchitecture.

- Tạo `linebuf_math_pkg.sv` hoặc helper module nhỏ cho:
  - `div2/mod2` bank mapping mặc định, và optional `div3/mod3` nếu bật lại
    stride 3;
  - `valid_c_bytes`;
  - `advance_lane_descriptor`.
- Register `block_valid_bytes`, `coalesce_k_bytes`, và lane descriptors ở
  boundary `start_i/next_tile_i`.
- Tách `build_emit_row` thành 2 stage:
  - descriptor stage;
  - data mux stage.

### Phase C: Structural split vừa phải

Mục tiêu: giảm kích thước file nhưng không tạo quá nhiều abstraction giả.

Đề xuất file:

- `linebuf_math_pkg.sv`: pure functions, type aliases nhỏ.
- `linebuf_bank_file.sv`: 10-bank SRAM wrapper + read/write arbitration.
- `linebuf_window_engine.sv`: window request/wait/slide + data mux.
- `conv_linebuf_stream_packer.sv`: top FSM + OBI fill/prefetch orchestration.

Không nên ép `conv_linebuf_stream_packer.sv` thành "pure controller" tuyệt đối
ngay lập tức. Module này vẫn cần giữ ownership của stream protocol và K tile
state; nếu tách quá tay, handshake giữa các submodule sẽ phức tạp hơn logic gốc.

### Phase D: Controller/drain refactor

Mục tiêu: xử lý bottleneck thật của case 20.

Tách theo ownership thực tế:

- `systolic_drain_engine.sv`
  - OFM FIFO pop;
  - psum read prefetch;
  - accumulation add;
  - writeback/requant.
- `systolic_input_engine.sv`
  - weight load/preload;
  - IFM/linebuffer input arbitration;
  - next K tile sequencing.
- `systolic_controller.sv`
  - register block;
  - array instance;
  - linebuffer instance;
  - top-level scheduling handshake.

Điểm quan trọng: tách file chưa đủ tăng performance. Performance chỉ tăng nếu
controller cho phép tile kế tiếp overlap sâu hơn với drain hoặc giảm writeback
psum trung gian.

## 6. Optimize kiến trúc tiếp theo

### 6.1. Row-level overlap giữa compute và drain

Hiện tile kế tiếp đợi drain hết tile trước. Hướng tốt hơn là cho tile kế tiếp
bắt đầu khi còn drain tail, miễn là không đụng cùng output row/psum row chưa an
toàn. Cần thêm scoreboard nhỏ theo row hoặc theo FIFO occupancy.

Điều kiện tối thiểu:

- OFM FIFO còn đủ headroom cho array flush của tile kế tiếp.
- Psum reader/writer không overwrite row mà tile kế tiếp còn cần đọc.
- Linebuffer đã prefetch/window-ready cho tile kế tiếp.

### 6.2. On-chip partial sum across K tiles

Với nhiều K tile, write psum ra TCDM sau mỗi tile rồi đọc lại ở tile sau là rất
đắt. Nếu tile spatial đủ nhỏ, giữ partial sum trong on-chip accumulator buffer
qua nhiều K tile rồi chỉ write cuối cùng sẽ giảm mạnh TCDM traffic.

Tradeoff:

- Tốn SRAM/register file cho psum tile.
- Cần scheduler phức tạp hơn.
- Nhưng đây là hướng có impact lớn nhất cho các layer YOLO nhiều IC.

### 6.3. Separate IFM/weight read arbitration

Một cổng `obi_i` vẫn là giới hạn. Nếu backend memory cho phép, tách weight read
và IFM/linebuffer read thành 2 master/port sẽ đơn giản hoá overlap. Nếu chỉ có
1 port, cần arbiter có policy rõ:

- weight preload có deadline trước compute;
- linebuffer prefetch tranh thủ idle slot;
- không để prefetch làm trễ weight critical path.

### 6.4. Spatial/stripe-stationary linebuffer cho target 2x compute

`Spatial/stripe-stationary` nghĩa là giữ một stripe không gian nhỏ của IFM
resident trong linebuffer, rồi chạy hết các K tile cần cho stripe đó trước khi
chuyển sang stripe kế tiếp. Dữ liệu đứng yên là spatial window, không phải một
`c_base` đơn lẻ. Systolic vẫn weight-stationary trong từng K tile.

Với case 20 (`16x16x120`, `3x3`, stride 1), target lý tưởng đang xét là:

- Compute thật: `8 stripes * 34 K tiles * 32 M rows = 8704` cycles.
- Weight load tối thiểu: `8 * 34 * 32 = 8704` cycles.
- Tổng target gần đúng: `17408` cycles, tức `2 * compute`.

Để đạt mô hình này, stripe nên là `tile_oh=2`:

- `M = tile_oh * output_w = 2 * 16 = 32`, khớp đúng một systolic M tile.
- Mỗi stripe cần input rows resident là `tile_oh + K - 1 = 4` rows.
- Cần giữ toàn bộ `ceil(120/32)=4` cblocks để không refill khi KGEN quay vòng
  qua `kh/kw/ic`.
- IFM data per context:
  `4 rows * 16 cols * 4 cblocks * 32B = 8192B`.
- Ping-pong hai stripe context đúng `16 KiB` IFM data.

Điều này khác full-row-cache. Full-row-cache giữ 16 row cho một cbase nên vẫn
thất bại khi KGEN quay lại cbase cũ sau khi đã đổi `kh/kw`; còn
stripe-stationary giữ ít row hơn nhưng giữ đủ các cblock của stripe hiện tại.

Dataflow mong muốn:

```text
context A resident stripe 0     context B filling stripe 1
        │                               ▲
        ▼                               │
KGEN lane descriptor {kh,kw,ic}         │
        │                               │
per-lane cblock/row/col mux             │
        │                               │
1 x 256-bit IFM vector/cycle ──► systolic compute
```

Các thay đổi cần có:

- Firmware không tách `IC=96 + 24` cho path này; chạy `IC=120` như một KGEN job
  34 tile, với tail lane zero-injection ở tile cuối.
- Linebuffer có hai resident contexts, mỗi context giữ `ROWS=4`, `W=16`,
  `CBLOCKS=4`, `32B` mỗi pixel-block.
- Lane mux không dùng một `cfg_c_base_i` duy nhất; mỗi lane tự tính
  `cblock = lane_ic / 32`, `lane = lane_ic % 32`.
- Scheduler chạy hết 34 K tiles cho cùng stripe trước khi chuyển stripe, và fill
  context kế tiếp dưới compute/drain của context hiện tại.
- Controller chỉ nên thấy 8 stripe jobs lớn, không phải 6 vòng `oh_base` nhân
  hai pass `96 + 24`.

Nếu chỉ thêm IFM FIFO hoặc tăng depth SRAM mà vẫn giữ `cfg_c_base_i` một giá trị,
thiết kế vẫn refill khi KGEN đổi cblock. Đó là lý do hướng này cần multi-cblock
resident cache và lane mux, không chỉ thêm prefetch.

## 7. Kết luận

Linebuffer refactor gần đây đã giải quyết đúng vấn đề ban đầu: `CH_STREAM_EMIT`
có thể chạy 1 vector/cycle khi window sẵn sàng. Bottleneck còn lại không phải
là "thiếu SRAM bank" đơn thuần, mà là tile scheduler + accumulation/drain path.

Thứ tự làm tiếp nên là:

1. Cleanup instrumentation và timing-risk nhỏ.
2. Implement mode `tile_oh=2` spatial/stripe-stationary cho case 20: resident
   4-row x 16-col x 4-cblock ping-pong cache và lane mux theo `lane_ic`.
3. Cho firmware chạy `IC=120` một job KGEN 34 tile thay vì split `96 + 24`.
4. Tách drain/input engine trong controller nếu sau đó `WAIT_DRAIN` vẫn còn
   vượt phần không thể overlap.

Nếu mục tiêu là YOLO 640x640, phase 4 mới là phần tạo khác biệt lớn nhất về
performance tổng thể.
