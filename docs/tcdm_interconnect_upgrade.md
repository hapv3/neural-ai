# NPU Cluster TCDM Interconnect Upgrade

**Date:** 2026-06-20  
**Target:** NPU Cluster Shared Data TCDM  
**Inspiration:** PULP MAGIA Project (`local_interconnect.sv`)

---

## 1. Vấn Đề Hiện Tại (Flat Architecture)

Kiến trúc TCDM Interconnect cũ (`tcdm_interconnect.sv`) sử dụng mô hình 1 Crossbar nguyên khối (9 masters × 16 banks) với cơ chế phân xử **Fixed Priority nối tiếp** (Daisy-chain).

- **Khuyết điểm nghiêm trọng:** 
  1. Priority đảo ngược: Core (M0) và DMA (M2) có priority cao hơn compute engines (Systolic Array), dễ gây Starvation làm giảm throughput.
  2. Timing & Area: 9-to-1 daisy-chain arbiter per bank tạo ra critical path dài $O(N)$.
  3. Contention cao: Mọi truy cập đều đụng độ trên cùng một vòng phân xử.

## 2. Giải Pháp: Mô Hình Phân Nhóm (Grouped Tree Topology)

Học hỏi từ kiến trúc của dự án **MAGIA**, chúng ta sẽ tái cấu trúc TCDM Interconnect thành mô hình phân nhóm theo loại Traffic (HWPE, DMA, Core). 

**Số lượng bank được giữ nguyên là 16 banks (512 KB physical).**

### Phân Nhóm Priority

| Nhóm | Priority | Masters | Lý do |
|------|----------|---------|-------|
| **HWPE** (Compute) | Cao nhất (High) | Systolic Array (1 R, 4 W) <br> Spatz Vector (2 R/W) <br> *[Tương lai: MAGIA AFU]* | Hardware Processing Elements chạy theo pipeline khắt khe. Nếu bị stall sẽ mất hàng chục MAC/cycle. |
| **DMA** (Data Move) | Trung bình (Med) | iDMA (1 R/W) | Di chuyển dữ liệu nền (background transfer). Burst dài nhưng có FIFO đệm, có thể stall vài cycle. |
| **CORE** (Scalar) | Thấp nhất (Low) | Snitch D-Bus (1 R/W) | Đọc/ghi cấu hình hoặc scalar data lác đác. Stall vài cycle không ảnh hưởng tổng throughput. |

---

## 3. Kiến Trúc Cây Phân Xử (Arbiter Tree Architecture)

Thay vì 1 crossbar khổng lồ, dữ liệu sẽ được route theo các tầng (Hierarchical Routing & Arbitration):

```text
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ HWPE Masters    │       │ DMA Master(s)   │       │ CORE Master(s)  │
  │ (7 Ports)       │       │ (1 Port)        │       │ (1 Port)        │
  └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
           │                         │                         │
           ▼                         ▼                         ▼
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ HWPE Router     │       │ DMA Router      │       │ CORE Router     │
  │ (1-to-16 Demux) │       │ (1-to-16 Demux) │       │ (1-to-16 Demux) │
  └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
           │                         │                         │
           ▼                         ▼                         ▼
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ HWPE Arbiter    │       │ DMA Arbiter     │       │ CORE Arbiter    │
  │ (Round-Robin)   │       │ (Round-Robin)   │       │ (Round-Robin)   │
  └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
           │ (High)                  │ (Med)                   │ (Low)
           │                         │                         │
           ▼                         ▼                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        Final Bank Arbiter                           │
  │                  (Strict Priority: HWPE > DMA > CORE)               │
  │                         [Per Bank 0 -> 15]                          │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │    16 x TCDM SRAM Banks      │
                      │    (32KB each = 512KB)       │
                      └──────────────────────────────┘
```

### Chi tiết các tầng:

1. **Router Layer (Định tuyến):** Mỗi Master có một bộ Demux để tính toán `target_bank = (addr >> 5) % 16` và đẩy request đến đúng bank.
2. **Group Arbiter Layer (Phân xử nội bộ nhóm):** 
   - Tại mỗi bank, nếu có nhiều HWPE cùng truy cập (ví dụ: Systolic Write và Spatz cùng ghi), sử dụng `rr_arb_tree` (Round-Robin) để đảm bảo không HWPE nào bị starve vĩnh viễn.
   - Tương tự cho DMA và CORE (nếu có >1 master trong nhóm).
3. **Final Bank Arbiter Layer (Phân xử cuối cùng):**
   - Bộ phân xử 3 ngõ vào (HWPE, DMA, CORE) dùng **Strict Priority**.
   - Nếu HWPE có request, luôn luôn cho phép HWPE qua. 
   - DMA chỉ được ghi/đọc khi không có HWPE truy cập vào bank đó. 
   - CORE chỉ được phép truy cập khi cả HWPE và DMA đều đang rảnh tại bank đó.

---

## 4. Lợi ích của Kiến trúc Mới

1. **Zero HWPE Starvation:** Systolic Array và Spatz luôn được ưu tiên cao nhất, đảm bảo chạy full 100% throughput của pipeline tính toán mà không bị DMA tranh chấp.
2. **Timing Tốt Hơn:** Critical path chia thành các bộ arbiter nhỏ (Round-Robin riêng, Priority riêng) thay vì 1 bộ arbiter khổng lồ.
3. **Dễ Scale:** Khi thêm **MAGIA AFU**, chỉ cần cắm vào cổng HWPE Router, bộ Round-Robin nội bộ sẽ tự chia sẻ băng thông mà không phải viết lại code phân xử cứng.
4. **Giữ nguyên Flexibility:** 16 banks vẫn là một dải địa chỉ liên tục (Unified Address Space). Software (Snitch) vẫn có thể quyết định đặt Weight, IFM, OFM ở đâu mà không bị giới hạn vật lý.

---

## 5. Kế hoạch Implement (Next Steps)

1. Tích hợp thư viện `common_cells` để lấy module `rr_arb_tree`.
2. Đập bỏ `hw/rtl/interconnect/tcdm_interconnect.sv` hiện tại.
3. Viết lại module `tcdm_interconnect_grouped.sv` dựa theo sơ đồ cây.
4. Cập nhật `npu_cluster.sv` kết nối lại các cổng (chuyển iDMA vào nhóm DMA, Systolic + Spatz vào nhóm HWPE).
