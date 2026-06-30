# Neural-AI YOLO NPU - Project Summary

**Date**: 2026-06-19
**Current Version**: Phase 3B (Matrix Engine Integration)

## 1. Mục tiêu dự án (Project Goals)
Xây dựng một kiến trúc NPU (Neural Processing Unit) xử lý hỗn hợp để đạt hiệu năng **10 TOPS**. Dự án được tối ưu hóa sâu cho họ mô hình **YOLO**, đặc biệt là hỗ trợ quá trình suy luận (inference) với dữ liệu lượng tử hóa INT8.

Kiến trúc bao gồm 5 **NPU Clusters** chạy song song. Mỗi cluster có:
- **Control Core (Snitch)**: RISC-V RV32IMAC điều khiển luồng và điều phối các đơn vị tính toán.
- **Matrix Engine**: Systolic Array (32x32) chuyên cho Dense MatMul / Conv2D (INT8).
- **Vector Engine**: Spatz RVV co-processor xử lý Depthwise Conv, Activations (SiLU, Softmax).
- **Shared Data TCDM**: Bộ nhớ L1 (SRAM chia sẻ 256-bit) cho phép truy cập đồng thời từ DMA, Snitch, Systolic Array và Vector Engine.
- **DMA Engine**: Di chuyển dữ liệu giữa L2 (DRAM/AXI) và L1 (TCDM) một cách tự động, không cần sự can thiệp liên tục của CPU.

## 2. Tiến độ công việc (Progress)

### Các Phase đã hoàn thành (✅ Done):
- **Phase 1**: DMA Engine + giao tiếp AXI.
- **Phase 2**: Data TCDM Interconnect (Crossbar đa bank).
- **Phase 2.5**: AXI-to-OBI bridge, test truyền dữ liệu DMA-to-TCM.
- **Phase 3A**: Tích hợp Snitch Core (cô lập I-TCM, D-TCM, Boot firmware thành công qua AXI).
- **Phase 3B - Phần 1 (Matrix Engine)**: 
  - Đã tích hợp thành công `systolic_controller` và `npu_systolic_array` vào `npu_cluster`.
  - Nâng cấp TCDM Interconnect lên 8 Master để hỗ trợ 4 cổng ghi và 1 cổng đọc từ Systolic Array.
  - Sửa các lỗi kiến trúc nghiêm trọng: Bug địa chỉ TCDM, Lỗi Mux/Demux, Backpressure cho OFM (Output Feature Map), DMA L1→L1, thứ tự nạp weight cho systolic array, và thiết lập lại các hằng số không nhất quán giữa RTL và Firmware.
  - Scoreboard của `test_systolic.py` đã assert thật trên số lượng kết quả và số mismatch; standalone Systolic Array pass 32/32 kết quả.
  - `test_snitch_boot.py` pass với firmware signature `0xDEADBEEF`, xác nhận boot path và DMA L1→L1 không phá luồng firmware.
  - `test_matmul.py` pass 10 vòng lặp ngẫu nhiên end-to-end trong cluster: Snitch firmware → DMA → Systolic Array → OFM writeback.

## 3. Công việc còn lại (Remaining Work / Next Steps)

- **Phase 3B - Phần 2 (Vector Engine)**: Tích hợp Spatz Vector Engine vào hệ thống để xử lý các phép toán phi tuyến (non-linear) và element-wise.
- **Phase 4**: Tích hợp Top-Level (5 Cluster). Đưa 5 cluster kết nối với một Manager Snitch để phân phối tải (tiling & scheduling).
- **Phase 5**: Chạy mô phỏng toàn bộ một layer YOLO (End-to-End Simulation) từ External Memory -> DMA -> TCDM -> Compute -> Writeback. Đánh giá hiệu năng (TOPS thực tế so với mục tiêu).

## 4. Các quy tắc quan trọng đã thống nhất (Design Decisions)
1. **Snitch làm Master**: Chỉ đạo và điều phối mọi block khác qua firmware.
2. **Loại bỏ APB**: Không dùng APB, mọi giao tiếp Memory-Mapped I/O được định tuyến qua OBI.
3. **Repository Management**: Không commit các file binary, hex hoặc object sinh ra trong quá trình biên dịch (dùng `.gitignore`).
4. Sử dụng biến `$(REPO_ROOT)` cho đường dẫn include trong các file cấu hình.
