# iDMA Integration Architecture Plan

**Date:** 2026-06-20  
**Target:** Replace the custom `dma_engine` with PULP `iDMA` in `npu_cluster.sv`

---

## 1. Mục tiêu (Goals)

Thay thế khối `dma_engine.sv` (FSM tự code, chạy từng beat rất chậm) bằng submodule `iDMA` của PULP Platform. `iDMA` hỗ trợ 16 burst transfer tối đa, pipelining, queueing và 2D strided transfer, giúp tăng tốc độ di chuyển dữ liệu (Weight, IFM, OFM) giữa L2 và L1 (TCDM) gấp nhiều lần.

> [!IMPORTANT]
> **Lý do cốt lõi (The Killer Feature): On-the-fly Im2Col cho 3x3 CONV2D**
> Bản thân Systolic Array 32x32 rất kém trong việc xử lý convolution 2D vì nó yêu cầu dữ liệu phải được làm phẳng (Im2Col). Nếu để Snitch Core tự làm Im2Col bằng phần mềm thì cực kỳ chậm và lãng phí bộ nhớ TCDM. 
> iDMA giải quyết triệt để vấn đề này nhờ tính năng **2D strided transfer** (`idma_nd_midend`). Nó có thể đọc nhảy bước (strided read) 3x3 patches từ L2 Memory và ghi tuần tự (sequential write) vào L1 TCDM. Nghĩa là iDMA thực hiện **On-the-fly Im2Col** hoàn toàn bằng phần cứng. Kết hợp với chiến thuật Double-Buffering (Ping-Pong), iDMA hoàn toàn che giấu được độ trễ di chuyển dữ liệu, giúp Systolic Array đạt 100% hiệu suất cho các mô hình YOLO/ResNet!
## 2. Lựa chọn Frontend & Backend

iDMA có thiết kế rất linh hoạt. Dựa trên source code từ `hw/magia/hw/tile/idma_ctrl_mm.sv`, chúng ta sẽ sử dụng cấu hình **Memory-Mapped (MM)**. 

Tuy nhiên, dải địa chỉ MMIO hiện tại của NPU Cluster (`0x2000_0000`) đang được dùng chung cho cả khối điều khiển Systolic Array (`cluster_ctrl_regs`). Do đó, ta sẽ sử dụng một bộ **MMIO Demux** để chia không gian này:
- `0x2000_0000 -> 0x2000_00FF`: Dành cho `cluster_ctrl_regs` (điều khiển Systolic Array).
- `0x2000_1000 -> 0x2000_1FFF`: Map thẳng vào Config Frontend của `iDMA` (idma_ctrl_mm).

Về Backend:
- **Data Backend 1 (AXI2OBI):** Chuyên fetch dữ liệu từ L2 (AXI Read) ghi vào L1 TCDM (OBI Write).
- **Data Backend 2 (OBI2AXI):** Chuyên đẩy kết quả từ L1 TCDM (OBI Read) ra ngoài L2 (AXI Write).

## 3. Detailed Architecture Diagram

```text
                                +-------------------------------------------+
                                |               Snitch Core                 |
                                | (Firmware MMIO writes to 0x2000_xxxx)     |
                                +-------------------------------------------+
                                                      | OBI (D-Bus)
                                                      v
                                +-------------------------------------------+
                                |               OBI Demux (1-to-4)          |
                                +-------------------------------------------+
                                     | (Demux Port 3: 0x2000_0000)
                                     v
                        +----------------------------------------+
                        |           MMIO Sub-Demux               |
                        +----------------------------------------+
                          | (0x2000_0000)                  | (0x2000_1000)
                          v                                v
+------------------------------------+   +-------------------------------------------------------------+
|        cluster_ctrl_regs           |   |                 idma_ctrl_mm (iDMA Wrapper)                 |
| (Điều khiển Systolic Array, v.v)   |   |                                                             |
+------------------------------------+   |  +-------------------------------------------------------+  |
                                         |  |                 idma_obi_ctrl_decoder                 |  |
                                         |  +---------------------------+---------------------------+  |
                                         |                              |                              |
                                         |                 [Config]     v                              |
                                         |   +-----------------------------------------+               |
                                         |   |         Channel AXI2OBI (L2 -> L1)      |               |
                                         |   |         (Fetch Weight, IFM)             |               |
                                         |   |                                         |               |
                                         |   |  +---------------+  +----------------+  |               |
                                         |   |  | AXI4 Master   |  | OBI Master     |  |               |
                                         |   |  | (Read Port)   |  | (Write Port)   |  |               |
|   |  +---------------+  +----------------+  |   |  +---------------+   +---------------+   |   |
|   +----------|-------------------|----------+   +----------|-------------------|-----------+   |
|              |                   |                         |                   |               |
+--------------|-------------------|-------------------------|-------------------|---------------+
               | AXI Read          | OBI Write               | OBI Read          | AXI Write
               v                   v                         v                   v
        +-------------+    +-----------------------------------------+    +-------------+
        | L2 / DRAM   |    |         Grouped TCDM Interconnect       |    | L2 / DRAM   |
        | (AXI XBAR)  |    |         (DMA Router Group)              |    | (AXI XBAR)  |
        +-------------+    +-----------------------------------------+    +-------------+
```

## 4. Các thay đổi cụ thể trên RTL (`npu_cluster.sv`)

### 4.1. Xóa bỏ `dma_engine.sv`
- Xóa instantiation của `u_dma_engine`.
- Xóa các logic MMIO cũ tự code (`cfg_dma_src_addr`, `cfg_dma_length`,...).

### 4.2. Khởi tạo `idma_ctrl_mm`
Sử dụng `idma_ctrl_mm` (lấy từ MAGIA hoặc tạo một wrapper tương tự cho NPU) với các parameter tương ứng với kiến trúc NPU (AXI 256-bit, OBI 256-bit).

```systemverilog
idma_ctrl_mm #(
    // Parameters map to 256-bit data width and 32-bit addr
) u_idma (
    .clk_i            (clk_i),
    .rst_ni           (rst_ni),
    
    // MMIO Config (từ OBI Demux)
    .obi_req_i        (mmio_obi_req),
    .obi_rsp_o        (mmio_obi_rsp),
    
    // L2 AXI Interfaces
    .axi_read_req_o   (idma_axi_read_req),
    .axi_read_rsp_i   (idma_axi_read_rsp),
    .axi_write_req_o  (idma_axi_write_req),
    .axi_write_rsp_i  (idma_axi_write_rsp),
    
    // L1 OBI Interfaces (vào TCDM Interconnect)
    .obi_read_req_o   (idma_obi_read_req),
    .obi_read_rsp_i   (idma_obi_read_rsp),
    .obi_write_req_o  (idma_obi_write_req),
    .obi_write_rsp_i  (idma_obi_write_rsp)
);
```

### 4.3. Cập nhật AXI Interconnect (L2)
- iDMA xuất ra 2 kênh AXI riêng biệt (một cho Read từ L2->L1, một cho Write từ L1->L2).
- Nếu NPU Cluster hiện tại chỉ phơi (expose) 1 cổng AXI Master ra ngoài, ta cần sử dụng một **AXI Multiplexer** (ví dụ: `axi_demux` hoặc `axi_mux` tùy chiều) để gộp 2 kênh Read/Write này lại trước khi đẩy ra cổng `axi_aw_...`, `axi_ar_...` của cụm. 
- *May mắn là AXI chia sẵn kênh AR (Address Read) và AW (Address Write) hoàn toàn độc lập, nên ta chỉ cần nối trực tiếp idma_axi_read vào các tín hiệu AR/R và idma_axi_write vào các tín hiệu AW/W/B của NPU Cluster!*

### 4.4. Tương tác với Firmware
Vì iDMA sử dụng một Register File phức tạp hơn `dma_engine` cũ (có channel config, status, transfer IDs, 2D stride), chúng ta sẽ cần mang file header C `idma_mm_utils.h` (hoặc tương đương) từ thư viện phần mềm của PULP sang để firmware Snitch có thể gọi API điều khiển.

---

## 5. Đánh giá (Review)

Anh hãy review xem kiến trúc thay thế này đã hợp lý chưa. Nếu OK, em sẽ lập **Implementation Plan** cho Phase 4 bao gồm cả nâng cấp TCDM (theo bài học từ MAGIA) và thay máu DMA bằng iDMA này.
