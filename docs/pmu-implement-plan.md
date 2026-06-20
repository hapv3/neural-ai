# Performance Management Unit (PMU) Design Plan

Mục tiêu: Xây dựng một khối PMU (Performance Management Unit) phần cứng giúp thu thập các số liệu hiệu năng (Hardware Performance Counters - HPC) của từng thành phần trong NPU Cluster theo thời gian thực. Từ đó, firmware hoặc host CPU có thể profile, tìm ra nút thắt cổ chai (bottleneck) và tính toán các chỉ số như TOPS, Memory Bandwidth, Hardware Utilization.

## 1. Cơ Chế Hoạt Động Của PMU

- **Kiến trúc:** PMU là một tập hợp các thanh ghi bộ đếm (Counters) 32-bit hoặc 64-bit.
- **Giao tiếp:** Được map vào một dải địa chỉ MMIO riêng (ví dụ: `0x2000_2000`). Firmware Snitch có thể ghi để Xóa (Reset), Bật/Tắt (Start/Stop) và Đọc (Read) các counter này.
- **Routing:** Mọi thành phần (DMA, Systolic, TCDM) sẽ xuất ra các cờ tín hiệu (Event Wires) như `is_active`, `is_stalled`, `conflict_pulse`. Các tín hiệu này được nối trực tiếp vào ngõ vào của khối PMU để kích hoạt tăng (increment) counter tương ứng trong mỗi chu kỳ xung nhịp (clock cycle).

---

## 2. Các Thông Số Cần Đo Lường Theo Từng Thành Phần

> [!TIP]
> **Quy tắc vàng trong Profiling:** Để biết NPU đang bị bottleneck ở đâu, ta cần đo 3 trạng thái cơ bản của mọi module: **Active** (đang làm việc), **Idle** (đang nghỉ chờ việc), và **Stalled** (muốn làm nhưng bị nghẽn I/O).

### 2.1. Systolic Array (Matrix Engine)
Đặc tính: Là cỗ máy ngốn dữ liệu, cần tính toán số lượng phép nhân cộng (MAC) và tỷ lệ đói dữ liệu (Data Starvation).

> [!NOTE]
> **Lưu ý về ảnh hưởng hiệu năng phần cứng:** Việc thiết kế và tích hợp bộ PMU để đếm sự kiện cho Systolic Array **hoàn toàn không làm giảm hiệu năng** (throughput) hay tăng critical path delay của khối này. Mạch PMU chỉ đóng vai trò "nghe lén" (snoop) các dây tín hiệu điều khiển có sẵn (như `valid`, `ready`, trạng thái FSM) và đếm bằng các bộ accumulator độc lập bên ngoài datapath. Do đó, timing của Systolic Array được bảo toàn 100%.

- **`SYS_ACTIVE_CYCLES`**: Số chu kỳ Systolic Array đang thực sự thực hiện phép tính MAC.
- **`SYS_STALL_CYCLES`**: Số chu kỳ Systolic Array đang tính nhưng phải dừng lại do TCDM Interconnect bị nghẽn (không nạp kịp Weight/IFM hoặc không ghi kịp OFM ra).
- **`SYS_IDLE_CYCLES`**: Số chu kỳ mảng rảnh rỗi (chờ Snitch cấu hình layer mới).
- **`SYS_TOTAL_MACS`**: (Tùy chọn) Tính tổng số phép toán MAC thực tế đã làm (hoặc có thể tự suy ra từ cấu hình M, N, K).
*=> Phân tích: Hiệu suất (Utilization) = `SYS_ACTIVE_CYCLES` / Tổng số chu kỳ. Nếu `SYS_STALL_CYCLES` quá cao, I/O TCDM đang là nút thắt cổ chai.*

### 2.2. iDMA (Data Movement Engine)
Đặc tính: Bộ di chuyển dữ liệu bất đồng bộ.
> **Tích hợp Native:** Rất may mắn là thư viện iDMA đã cung cấp sẵn module `idma_inst64_events` sinh ra các xung sự kiện (Events) dạng struct `dma_events_t`. Chúng ta sẽ nối thẳng các cờ này vào bộ PMU của NPU để đếm mà không cần tự viết logic phân tích AXI bus.

Các counter sẽ map trực tiếp từ cờ của iDMA:
- **`DMA_ACTIVE_CYCLES`**: Nối từ cờ `dma_busy`. Đếm số chu kỳ iDMA đang có transfer in-flight.
- **`DMA_L2_STALL_CYCLES`**: Nối từ cờ `ar_stall` và `aw_stall`. Đếm số chu kỳ bị nghẽn do L2/AXI Interconnect chậm (AXI `ar_ready` hoặc `aw_ready` bị low).
- **`DMA_L1_STALL_CYCLES`**: Nối từ cờ `w_stall` và `r_stall`. Đếm số chu kỳ bị nghẽn do L1/TCDM Interconnect chậm.
- **`DMA_BYTES_TRANSFERRED`**: Cộng dồn từ tín hiệu `num_bytes_written` và `r_bw` của iDMA.
*=> Phân tích: Bandwidth thực tế (GB/s) = `DMA_BYTES_TRANSFERRED` / (`DMA_ACTIVE_CYCLES` * 1/Freq).*

### 2.3. Snitch Core (Control Core)
Đặc tính: Quản lý luồng (Control flow). Đa số thời gian nên ở trạng thái ngủ tiết kiệm điện (WFI - Wait For Interrupt) chờ DMA và Systolic xong việc.
> **Tích hợp Native:** Snitch **có sẵn** bộ Performance Monitor rất mạnh. Khi bật macro `SNITCH_ENABLE_PERF`, Snitch tự động đếm `mcycle` và `minstret` (truy cập qua lệnh CSR). Đồng thời, core còn xuất ra một struct `core_events_t` chứa sẵn các xung sự kiện như `retired_instr`, `retired_load`, `retired_acc`.

Các counter sẽ map trực tiếp:
- **`CORE_ACTIVE_CYCLES`**: Có thể đọc trực tiếp từ thanh ghi CSR `mcycle`.
- **`CORE_INSTR_RETIRED`**: Có thể đọc trực tiếp từ thanh ghi CSR `minstret`.
- **`CORE_WFI_CYCLES`**: Có thể suy ra từ sự chênh lệch giữa mcycle và số lệnh thực thi, hoặc đếm dựa trên xung tín hiệu nội bộ khi core đang sleep.
*=> Phân tích: NPU thiết kế tốt thì CPU phải dành > 90% thời gian cho `CORE_WFI_CYCLES`.*

### 2.4. TCDM Interconnect (Memory Subsystem)
Đặc tính: Crossbar điều hướng dữ liệu trọng yếu nhất trong NPU. Nơi dễ xảy ra nút thắt I/O do va chạm (Bank Conflict).
> **Tích hợp Native:** Tương tự thư viện của PULP/Spatz, ở mỗi cổng gắn vào SRAM Bank, ta sẽ sử dụng mạch `popcount` để lấy 2 thông số: số request chạm đến bank (accessed) và số request bị từ chối do xung đột (congested).

Các counter sẽ map:
- **`TCDM_BANK_CONFLICTS`**: Đếm tổng số lần các Master muốn truy cập nhưng bị từ chối (stall) do vướng priority.
- **`TCDM_TOTAL_REQ`**: Tổng số request thành công được đẩy xuống các bank SRAM.
*=> Phân tích: Tỷ lệ Conflict = `TCDM_BANK_CONFLICTS` / `TCDM_TOTAL_REQ`. Nếu > 5-10%, firmware cần tối ưu hóa lại địa chỉ lưu trữ (memory layout) để tản đều ma trận ra các bank, tránh việc nhiều port cùng dồn vào đọc/ghi 1 bank gây bottleneck.*

---

## 3. Kiến Trúc Hardware PMU

```text
                                              +-----------------------------------+
[Systolic Array] ---- (active, stall) ------> |                                   |
[iDMA] -------------- (active, stall) ------> |         Performance               |
[TCDM Arbiter] ------ (conflict) -----------> |         Management                |
[Snitch Core] ------- (sleep, active) ------> |         Unit (PMU)                |
                                              |                                   |
                                              |  - Counter 0: SYS_ACTIVE (32b)    |
   MMIO Bus (0x2000_2000)                     |  - Counter 1: SYS_STALL  (32b)    |
   (Đọc kết quả / Reset Counters)  ---------> |  - Counter N: ...                 |
                                              +-----------------------------------+
```
