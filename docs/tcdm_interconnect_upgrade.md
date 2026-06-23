# NPU Cluster TCDM Interconnect Upgrade

**Date:** 2026-06-22
**Target:** NPU Cluster Shared Data TCDM  
**Inspiration:** PULP MAGIA Project (`local_interconnect.sv`)

---

## 1. Vấn Đề Hiện Tại (Flat Architecture)

Kiến trúc TCDM Interconnect cũ (`tcdm_interconnect.sv`) sử dụng mô hình 1 crossbar nguyên khối (**10 masters × 16 banks**) với cơ chế phân xử **Fixed Priority nối tiếp** (Daisy-chain).

- **Khuyết điểm nghiêm trọng:** 
  1. Priority đảo ngược: Core (M0) và DMA (M2) có priority cao hơn compute engines (Systolic Array), dễ gây Starvation làm giảm throughput.
  2. Timing & Area: 10-to-1 daisy-chain arbiter per bank tạo ra critical path dài $O(N)$.
  3. Contention cao: Mọi truy cập đều đụng độ trên cùng một vòng phân xử.

## 2. Giải Pháp: Mô Hình Phân Nhóm (Grouped Tree Topology)

Học hỏi từ kiến trúc của dự án **MAGIA**, chúng ta sẽ tái cấu trúc TCDM Interconnect thành mô hình phân nhóm theo loại Traffic (HWPE, DMA, Core). 

**Số lượng bank được giữ nguyên là 16 banks (512 KB physical).**

### Phân Nhóm Priority

| Nhóm | Priority | Masters | Lý do |
|------|----------|---------|-------|
| **HWPE** (Compute) | Cao nhất (High) | Systolic Array (1 R, 4 W) <br> Spatz Vector (2 R/W) <br> *[Tương lai: MAGIA AFU]* | Hardware Processing Elements chạy theo pipeline khắt khe. Nếu bị stall sẽ mất hàng chục MAC/cycle. |
| **DMA** (Data Move) | Trung bình (Med) | iDMA AXI2OBI write port <br> iDMA OBI2AXI read port | Di chuyển dữ liệu nền (background transfer). Burst dài nhưng có FIFO đệm, có thể stall vài cycle. |
| **CORE** (Scalar) | Thấp nhất (Low) | Snitch D-Bus (1 R/W) | Đọc/ghi cấu hình hoặc scalar data lác đác. Stall vài cycle không ảnh hưởng tổng throughput. |

### Mapping Master Hiện Tại

| Master ID | Source | Group |
|-----------|--------|-------|
| `M0` | Snitch D-Bus | CORE |
| `M1` | Spatz VLSU port 0 | HWPE |
| `M2` | iDMA AXI2OBI write port | DMA |
| `M3` | Systolic controller read port | HWPE |
| `M4` | Systolic controller write port 0 | HWPE |
| `M5` | Systolic controller write port 1 | HWPE |
| `M6` | Systolic controller write port 2 | HWPE |
| `M7` | Systolic controller write port 3 | HWPE |
| `M8` | Spatz VLSU port 1 | HWPE |
| `M9` | iDMA OBI2AXI read port | DMA |

---

## 3. Kiến Trúc Cây Phân Xử (Arbiter Tree Architecture)

Thay vì 1 crossbar khổng lồ, dữ liệu sẽ được route theo các tầng (Hierarchical Routing & Arbitration):

```text
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ HWPE Masters    │       │ DMA Master(s)   │       │ CORE Master(s)  │
  │ (7 Ports)       │       │ (2 Ports)       │       │ (1 Port)        │
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

1. Dùng grouped round-robin arbiter nhỏ ngay trong `tcdm_interconnect.sv`; `rr_arb_tree` của `common_cells` vẫn là reference nhưng không instantiate hàng loạt để tránh tăng thời gian Verilator build.
2. Giữ nguyên public interface của `hw/rtl/interconnect/tcdm_interconnect.sv` để tránh phá wiring `npu_cluster.sv`.
3. Thay internals bằng grouped arbitration: HWPE round-robin, DMA round-robin, CORE round-robin, final strict priority `HWPE > DMA > CORE`.
4. Cập nhật `npu_cluster.sv` truyền master masks: HWPE=`M1,M3,M4,M5,M6,M7,M8`, DMA=`M2,M9`, CORE=`M0`.
5. Thêm contention test độc lập cho arbitration policy trước khi chạy cluster regressions.

## 6. Rủi Ro / Giới Hạn Cần Chấp Nhận

1. **Strict priority có thể starve DMA/CORE** nếu HWPE request liên tục vào cùng bank. Đây là tradeoff có chủ ý cho baseline compute-first. Nếu cần bounded fairness, thêm ageing/credit ở phase sau.
2. **Response routing phải giữ contract 1-cycle SRAM latency** như interconnect cũ. Nếu thêm pipeline, phải thêm FIFO route response theo master.
3. **I-TCDM/O-TCDM vẫn là logical windows** trên cùng 16-bank physical TCDM. Upgrade này chưa thay đổi memory map hay bank capacity.
4. **Không thay đổi firmware-visible address contract** trong phase này; mọi test hiện tại phải tiếp tục pass.
