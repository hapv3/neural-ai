# Kế hoạch Thiết kế Hệ thống Ngắt (Interrupt Controller Plan) cho NPU Cluster

Mục tiêu: Thay thế hoàn toàn cơ chế Polling (hỏi vòng liên tục qua MMIO) tốn điện và kém hiệu quả hiện tại bằng cơ chế điều khiển hướng sự kiện (Event-driven) dựa trên ngắt (Interrupt). Điều này cho phép Snitch Core vào trạng thái ngủ `WFI` (Wait For Interrupt) trong lúc Systolic, AFU, hoặc DMA đang làm việc nặng, giúp tiết kiệm năng lượng và tăng khả năng phản hồi.

## Open Questions

> [!WARNING]
> **Độ rộng của `irq_i` trên Snitch Core?**
> RISC-V thường chuẩn hóa ngắt thành M-mode External/Timer/Software (MEIP, MTIP, MSIP) hoặc một mảng (vector) Fast Local Interrupts `[15:0]`. Tôi giả định Snitch sử dụng một mảng bit cho các Fast Interrupts nội bộ. Nếu không, chúng ta sẽ cần thiết kế một thanh ghi MMIO `IRQ_CAUSE` để Snitch tự đọc xem ai vừa gửi ngắt. Cần user confirm kiến trúc ngắt cụ thể của lõi Snitch hiện tại.

## Proposed Changes

## Implementation Status

Phase-1 interrupt support is implemented for verification and firmware completion:

- `hw/rtl/cluster/npu_interrupt_ctrl.sv` provides internal pending/enable/clear registers plus external host pending/enable/clear registers.
- Hardware done events from DMA, Systolic, AFU, and Spatz are latched into the internal interrupt domain.
- `snitch_irq_o.mcip` is driven when an enabled internal event is pending.
- Firmware reports test completion by writing `NPU_IRQ_HOST_NOTIFY`; this latches `HOST_STATUS` internally and asserts cluster `irq_o`.
- Cocotb tests now boot firmware through AXI I-TCM, release Snitch with `fetch_enable_i`, wait on `irq_o`, and validate correctness through L2/TCDM output data. Current host AXI path intentionally reaches I-TCM only, so it does not read IRQ MMIO.
- D-TCM is no longer used as a testbench preload/start/status path for the active boot, memory, systolic, Spatz, operator, and matmul gates.

Remaining work is the true low-power/event-driven firmware path: install a Snitch trap vector, enable M-mode interrupts, replace selected `REG_DONE` polling loops with `wfi`, and clear `IRQ_INT_PENDING` in the handler.

### 1. Tạo Khối `npu_interrupt_ctrl.sv` (Event Unit)

Khối này đóng vai trò như một bộ tập hợp ngắt (Interrupt Aggregator / PLIC-lite) nằm bên trong Cluster. Nó nhận tín hiệu từ các hardware engines và định tuyến theo 2 hướng hoàn toàn biệt lập:

- **Inputs (Các nguồn sinh ngắt):**
  - `dma_done_i`: Ngắt từ DMA (A2O hoặc O2A báo xong).
  - `sys_done_i`: Ngắt từ Systolic Controller.
  - `afu_done_i`: Ngắt từ khối AFU.
  - `spatz_done_i`: Ngắt từ Vector Engine.

---

### Hướng 1: Internal Interrupt (Hướng vào Snitch Core)
**Mục đích:** Đồng bộ hóa các luồng công việc bên trong Cluster, cho phép Snitch Core thoát khỏi việc Polling vòng lặp và đi vào trạng thái ngủ `WFI` để tiết kiệm năng lượng.
- **Tín hiệu Output:** `snitch_irq_o` (Nối trực tiếp vào cổng `irq_i` của lõi Snitch).
- **Cơ chế Hoạt động:**
  - Có một dải thanh ghi MMIO riêng (VD: `IRQ_ENABLE_INTERNAL`, `IRQ_PENDING_INTERNAL`, `IRQ_CLEAR_INTERNAL`).
  - Khi DMA hoặc Systolic làm xong 1 tile dữ liệu nhỏ, nó báo về khối này. Khối này "đánh thức" Snitch.
  - Snitch sẽ chạy Trap Handler, ghi đè cờ cấu hình và cấp việc mới cho Systolic/DMA xử lý tile tiếp theo, rồi lại ngủ `WFI`.

### Hướng 2: External Interrupt (Hướng ra ngoài Host CPU)
**Mục đích:** Báo cáo tiến độ của cả một luồng công việc lớn (ví dụ: chạy xong toàn bộ 1 Layer của ResNet hoặc YOLO) ra cho CPU bên ngoài (Host) để Host ra quyết định tiếp theo.
- **Tín hiệu Output:** `host_irq_o` (Nối thẳng ra cổng `irq_o` của NPU Cluster).
- **Cơ chế Hoạt động:**
  - Có một dải thanh ghi MMIO độc lập khác (VD: `IRQ_ENABLE_EXTERNAL`, `IRQ_PENDING_EXTERNAL`, `IRQ_CLEAR_EXTERNAL`).
  - Các khối phần cứng (DMA, Systolic) KHÔNG tự động kích hoạt External Interrupt. 
  - Chỉ khi Snitch đếm đủ số lượng vòng lặp (ví dụ đã xử lý đủ 100 tiles của 1 Layer), Firmware của Snitch mới ghi một lệnh vào thanh ghi `HOST_NOTIFY` của khối Event Unit. Khối này sau đó sẽ dội ngắt `host_irq_o` ra ngoài để gọi Host CPU.
  - Host CPU sẽ phục vụ ngắt, đọc trạng thái của Cluster, và nạp Firmware/Data cho Layer mới.

---

### 2. Sửa đổi ở Tầng Giao Tiếp (RTL Modifications)

#### [MODIFY] `hw/rtl/cluster/npu_cluster.sv`
- Instantiate khối `npu_interrupt_ctrl` mới.
- Kết nối các dây `done` đang bị bỏ lửng của DMA, Systolic, AFU vào khối này.
- **Quan trọng:** Sửa dòng `snitch_core` instantiation. Đổi từ `.irq_i ('0)` thành `.irq_i (snitch_irq_o)`.
- Gán đầu ra `irq_o` của module `npu_cluster` bằng tín hiệu `host_irq_o` sinh ra từ khối điều khiển.

#### [MODIFY] `hw/rtl/cluster/snitch_core.sv`
- Nếu cần thiết, bóc tách cổng `irq_i` (nếu nó đang bị hardcode ẩn bên trong các wrapper) để nối trực tiếp vào lõi `i_snitch`.

---

### 3. Cập nhật Firmware (Software Impact)

Cơ chế WFI (Wait For Interrupt) sẽ thay đổi hoàn toàn vòng lặp của Firmware:

1. **Khởi tạo:** Trong hàm `main()`, bật cờ `MIE` (Machine Interrupt Enable) trong thanh ghi `mstatus` của RISC-V. Bật `IRQ_ENABLE` tương ứng trên khối `npu_interrupt_ctrl`.
2. **Kích hoạt Hardware:** Ghi lệnh `start` xuống Systolic Controller.
3. **Ngủ (Sleep):** Chạy lệnh asm `wfi;`. Snitch core sẽ tắt clock và ngủ say.
4. **Đánh thức (Wakeup):** Khi Systolic chạy xong, nó kéo chân `sys_done` lên 1. `npu_interrupt_ctrl` kéo chân `irq` của Snitch.
5. **Trap Handler:** PC nhảy vào hàm `trap_vector`. Hàm này đọc `IRQ_PENDING`, nhận ra Systolic vừa xong. Nó cập nhật state nội bộ của firmware, sau đó ghi `1` vào `IRQ_CLEAR` để tắt ngắt. Quay trở lại hàm `main()`.

## Verification Plan

### Automated Tests
- Viết một kịch bản testbench `test_interrupt.py` trên Cocotb:
  1. Nạp một firmware nhỏ có chứa `trap_vector`. Firmware in ra `A`, gửi lệnh chạy Systolic, gọi lệnh `wfi`.
  2. Testbench theo dõi, đảm bảo rằng trong lúc Systolic chạy, lệnh `wfi` có hiệu lực (PC của Snitch không đổi, hoặc cờ sleep bật).
  3. Khi Systolic bắn ngắt `done`, Testbench kiểm tra xem Snitch có thoát khỏi `wfi` và in ra chữ `B` (từ trong ngắt) hay không.
  4. Nếu Snitch tiếp tục in ra chữ `B` và chạy tiếp, cơ chế Interrupt đã pass!
