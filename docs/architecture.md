# YOLO NPU Architecture Specification

**Version**: Phase 3B-A — Matrix Engine Integrated in Cluster  
**Last Updated**: 2026-06-19

---

## 1. Architectural Overview

The YOLO NPU is a heterogeneous compute architecture designed to achieve 10 TOPS. Deeply optimized for the YOLO family of models, it supports INT8 quantized inference.

To achieve high-performance matrix and vector operations, the architecture integrates 5 parallel **NPU Clusters**. Each cluster features:

1. **Control Core (Snitch)**: A RISC-V RV32IMAC scalar core. Boots from local I-TCM firmware, controls and dispatches compute commands to the Matrix and Vector engines via OBI.
2. **Matrix Engine** *(Phase 3B-A)*: A 32×32 INT8 Systolic Array optimized for Dense MatMul / Conv2D.
3. **Vector Engine** *(Phase 3B-B)*: A Spatz RVV co-processor for Depthwise Conv, Element-wise ops, and Non-linear Activations (SiLU, Softmax).
4. **Shared Data TCDM**: A 256-bit wide L1 data memory shared between the Systolic Array, Vector Engine, and DMA.
5. **DMA Engine**: Handles burst data movement between AXI external memory (L2/DRAM) and Data TCDM autonomously, without involving Snitch.

### L1 vs. L2 Memory Model

- **L1** là toàn bộ memory local trong cluster: I-TCM, Snitch D-TCM, Shared Data TCDM SRAM, và MMIO/CSR aperture.
- **L2** là external memory phía ngoài cluster, được truy cập qua AXI4 master của DMA hoặc AXI4-Lite host/testbench port.
- Snitch chỉ điều phối bằng firmware và CSR; đường dữ liệu tile lớn đi qua DMA để tránh scalar core phải copy từng word.
- DMA hiện hỗ trợ các hướng copy chính: **L2→L1** để load weights/IFM, **L1→L2** để writeback OFM, và **L1→L1** để copy/rearrange dữ liệu trong local memories.

---

## 2. Cluster Micro-Architecture & Interfaces

```text
+----------------------------------------------------------------------------------------------------+
|                                      Host CPU / External L2 DRAM                                   |
+----------------------------------------------------------------------------------------------------+
                                           | (AXI4-Lite)
                                           v
+----------------------------------------------------------------------------------------------------+
|                                    NPU Cluster (1 of 5)                                            |
|                                                                                                    |
|  +--------------------+                                                                            |
|  | Bootloader (AXI4)  |                                                                            |
|  +---------+----------+                                                                            |
|            | (AXI-to-OBI)                                                                          |
|            v                                                                                       |
|  +-------------------------------------------------------------+                                   |
|  |                        I-TCM Arbiter                        |                                   |
|  +---------+------------------------------------------+--------+                                   |
|            |                                          ^ (I-Fetch)                                  |
|            v                                          |                                            |
|  +--------------------+                     +---------+----------+                                 |
|  |   I-TCM (32 KB)    |<---(M0 0x10000000)--|  Snitch Core       |                                 |
|  +--------------------+                     |   (RV32IMAC)       |                                 |
|                                             +---------+----------+                                 |
|                                                       | (D-Bus)                                    |
|                                                       v                                            |
|                                             +--------------------+                                 |
|                                             |  OBI Demux (1-to-4)|                                 |
|                                             +--------------------+                                 |
|                                              /      |          \                                   |
|                     (M1 0x10008000)---------+       |           +-------(M3 0x20000000)            |
|                    /                                |                               \              |
|          +--------v---------+             (M2 0x10100000)                   +-------v--+           |
|          | D-TCM (32 KB)    |                       |                       | MMIO/CSR |           |
|          +------------------+                       |                       +----------+           |
|                                                     v                                              |
|  +----------------------------------------------------------------------------------------------+  |
|  |                          Shared Data TCDM Interconnect (256-bit)                             |  |
|  +-------+--------------------+---------------------+-----------------------+-------------------+  |
|          ^                    ^                     ^                       |                      |
|          |                    |                     |                       v                      |
| +--------+--------+  +--------+--------+  +---------+---------+   +-------------------+            |
| | Systolic Array  |  | Spatz Vector    |  | DMA Engine        |<--|  Data TCDM SRAM   |            |
| | (Matrix Engine) |  | (Vector Engine) |  | (AXI4 Master to L2)|  | (12 I-TCDM Banks) |            |
| +-----------------+  +-----------------+  +-------------------+   | (4 O-TCDM Banks)  |            |
|                                                                   +-------------------+            |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Interface Reference

| ID | Name | Protocol | Width | Description |
|----|------|----------|-------|-------------|
| A  | I-Fetch | OBI | 256-bit | Snitch instruction fetch từ I-TCM (native width) |
| B  | AXI4-Lite Slave | AXI4-Lite | 32-bit | Host/bootloader ghi firmware vào I-TCM qua AXI |
| C  | I-TCM Arbiter | OBI Arbiter 2→1 | 256-bit | Phân giải I-TCM giữa Bootloader và Snitch |
| D  | Snitch D-Bus | OBI | 256-bit | Bus dữ liệu chính của Snitch Core |
| E  | OBI Demux 1→4 | OBI | 256-bit | Định tuyến data bus: I-TCM / D-TCM / Shared TCDM / MMIO |
| F  | Data TCDM Port | OBI | 256-bit | Master Port từ Demux vào Shared Data TCDM |
| G  | DMA AXI Master | AXI4 | 256-bit | DMA tự load dữ liệu từ L2/DRAM vào Data TCDM |
| H  | Host AXI4-Lite | AXI4-Lite | 32-bit | Kết nối từ Host xuống Cluster để nạp boot firmware |

---

## 3. Systolic Array Micro-Architecture

```text
+------------------------------------------------------------------------------------------------+
|                                    Systolic Array (32x32)                                      |
|                                                                                                |
|                                       [Weights (W) loaded sequentially downwards]              |
|                                            |     |     |           |                           |
|  +-----------------+                       v     v     v           v                           |
|  |                 |    IFM_Row[0]      +-----+-----+-----+     +-----+                        |
|  | Input Skewing   |------------------->|PE0,0|PE0,1|PE0,2| ... |P0,31| (Delay 0 for IFM)      |
|  | (Triangle       |                    +-----+-----+-----+     +-----+                        |
|  |  Delay Regs)    |                       |     |     |           |                           |
|  |                 |    IFM_Row[1] (d=1)+-----+-----+-----+     +-----+                        |
|  |   IFM           |------------------->|PE1,0|PE1,1|PE1,2| ... |P1,31|                        |
|  |  ------>        |                    +-----+-----+-----+     +-----+                        |
|  |                 |                       |     |     |           |                           |
|  |                 |    IFM_Row[2] (d=2)+-----+-----+-----+     +-----+                        |
|  |                 |------------------->|PE2,0|PE2,1|PE2,2| ... |P2,31|                        |
|  |                 |                    +-----+-----+-----+     +-----+                        |
|  |                 |                       |     |     |           |                           |
|  |                 |                      ...   ...   ...  ...    ...                          |
|  |                 |                       |     |     |           |                           |
|  |                 |    IFM_Row[31]     +-----+-----+-----+     +-----+                        |
|  |                 |------------------->|P31,0|P31,1|P31,2| ... |31,31| (Delay 31 for IFM)     |
|  +-----------------+                    +-----+-----+-----+     +-----+                        |
|                                            |     |     |           |                           |
|                                            v     v     v           v  Psums                    |
|  +------------------------------------------------------------------------------------------+  |
|  |                                  Output Deskewing (Reverse Triangle)                     |  |
|  |                                 (Delay 31) (Delay 30)      (Delay 0)                     |  |
|  +------------------------------------------------------------------------------------------+  |
|                                            |     |     |           |                           |
|                                            v     v     v           v                           |
## 4. Data TCDM SRAM Micro-Architecture (Detailed)

> **Lưu ý Kiến trúc (Architecture Update):**
> Kiến trúc TCDM Interconnect hiện tại đang được lên kế hoạch nâng cấp sang mô hình phân nhóm (Grouped Tree Topology) học hỏi từ dự án MAGIA để giải quyết triệt để các vấn đề về Priority Starvation và Bank Conflict. 
> Chi tiết thiết kế và kế hoạch nâng cấp xem tại: [tcdm_interconnect_upgrade.md](file:///home/dev01/neural-ai/docs/tcdm_interconnect_upgrade.md).

Shared Data TCDM là L1 data scratchpad chính cho compute path. Nó không phải cache: mọi tile được firmware/DMA đặt vào địa chỉ rõ ràng, deterministic latency, và arbitration được xử lý bởi TCDM interconnect.

*(Chi tiết về kiến trúc TCDM hiện tại và phương án nâng cấp sang mô hình phân nhóm Grouped Tree Topology đã được chuyển sang tài liệu chuyên đề: [tcdm_interconnect_upgrade.md](file:///home/dev01/neural-ai/docs/tcdm_interconnect_upgrade.md))*

#### Current Logical Buffers

| Buffer | Address | Region | Vai trò |
|--------|---------|--------|---------|
| `WEIGHT_PING_ADDR` | `0x1011_0000` | I-TCDM | 32×32 INT8 weight tile |
| `IFM_PING_ADDR` | `0x1012_0000` | I-TCDM | M×32 INT8 IFM tile |
| `OFM_PING_ADDR` | `0x1020_0000` | O-TCDM | M×32 INT32 output tile |

|--------------|------|--------|---------|
| `0x1000_0000 – 0x1000_7FFF` | 32 KB | **I-TCM** | Firmware Snitch (instruction). Đọc/ghi qua D-Bus M0. |
| `0x1000_8000 – 0x1000_FFFF` | 32 KB | **Snitch D-TCM** | Private data (stack, scalars) qua D-Bus M1. |
| `0x1010_0000 – 0x1015_FFFF` | 384 KB | **I-TCDM logical window** | Weights và IFM tiles cho compute engines. |
| `0x1020_0000 – 0x1021_FFFF` | 128 KB | **O-TCDM logical window** | OFM / INT32 accumulator writeback. |
| `0x2000_0000 – 0x2000_FFFF` | 64 KB | **MMIO / CSR** | `cluster_ctrl_regs`, cấu hình DMA. |
| `0x8000_0000+` | External | **L2 / AXI sim memory** | Testbench/external memory chứa input/output buffers. |

> **Tại sao tách I-TCM và D-TCM?**  
> Harvard Architecture: Snitch fetch lệnh qua I-Fetch (không cạnh tranh băng thông với D-Bus). D-TCM private đảm bảo 1-cycle latency cố định cho stack/local vars bất kể DMA hay Systolic Array đang chiếm Shared Data TCDM.

---

## 6. L1/L2 Data Movement

```text
L2 / External Memory
  |  AXI4 read/write
  v
DMA Engine
  |  OBI read/write
  v
L1 Shared Data TCDM
  |  Systolic read/write ports
  v
Matrix Engine
```

### Supported DMA Directions

| Direction | Example | Use case |
|-----------|---------|----------|
| **L2→L1** | `0x8000_0000 → 0x1011_0000` | Load weight tile into I-TCDM |
| **L2→L1** | `0x8000_1000 → 0x1012_0000` | Load IFM tile into I-TCDM |
| **L1→L2** | `0x1020_0000 → 0x8000_2000` | Write OFM tile back to external memory |
| **L1→L1** | `0x101x_xxxx → 0x101y_yyyy` | Local copy/repacking path for future tiling |

### Matrix Engine Dataflow

1. Firmware programs DMA registers through MMIO.
2. DMA copies weights and IFM from L2 into L1 I-TCDM.
3. Firmware programs Systolic Controller CSR with weight, IFM, OFM pointers.
4. Systolic Controller reads weights/IFM from I-TCDM and streams them into `npu_systolic_array`.
5. Systolic Controller writes OFM rows into O-TCDM through 4 parallel OBI write ports.
6. DMA copies OFM from O-TCDM back to L2.

---

## 7. Boot Flow

Detailed walkthrough: [Boot Flow](boot_flow.md).

```
1. Host ghi firmware vào I-TCM qua AXI4-Lite Slave (cổng B/H)
   → AXI-to-OBI bridge → I-TCM Arbiter → I-TCM SRAM
   
2. Host de-assert rst_ni

3. Snitch core reset → PC = 0x1000_0000 (BootAddr)

4. Snitch fetch instruction từ I-TCM (cổng A, qua Arbiter)

5. Snitch thực thi firmware:
   - Khởi tạo D-TCM (stack setup)
   - Cấu hình cluster_ctrl_regs qua MMIO (cổng M3)
   - (Phase 3B-A) Trigger DMA để load weight/IFM vào Data TCDM
   - (Phase 3B-A) Dispatch lệnh tới Systolic Array
   - (Phase 3B-B) Dispatch lệnh RVV tới Spatz

6. Snitch ghi 0xDEADBEEF vào địa chỉ MMIO signature
   → Host đọc để xác nhận boot thành công
```

---

## 8. Test Flow

Detailed walkthrough: [Test Flow](test_flow.md).

### Current Verified Tests

| Test | Flow | Pass criteria |
|------|------|---------------|
| `test_systolic.py` | Direct standalone Systolic Array stimulus | Scoreboard captures OFM and all 32 columns match golden model |
| `test_snitch_boot.py` | Host loads `boot.bin` into I-TCM and releases Snitch | Firmware writes signature `0xDEADBEEF` |
| `test_matmul.py` | Host prepares random tensors in L2, firmware runs DMA + Systolic + writeback | 10 randomized MatMul iterations match NumPy golden |

### Cluster MatMul Test Flow

```text
Testbench L2 buffers
  ├─ weights @ 0x8000_0000
  ├─ IFM     @ 0x8000_1000
  └─ OFM     @ 0x8000_2000

Host AXI-Lite loads matmul firmware into I-TCM
Host writes dim/start flag into D-TCM
Snitch firmware:
  1. DMA weights L2→I-TCDM
  2. DMA IFM L2→I-TCDM
  3. Start Systolic Controller
  4. Wait SYS_DONE
  5. DMA OFM O-TCDM→L2
  6. Set done flag in D-TCM
Testbench reads L2 OFM and compares against NumPy golden
```

---

## 9. SRAM Allocation Analysis

Mục tiêu ban đầu của top-level là **2.5 MB SRAM on-chip**. Tuy nhiên, cấu hình Phase 3B-A hiện tại ưu tiên tính đúng architecture và verification trong 1 cluster trước: mỗi cluster đang dùng Data TCDM physical 512 KB cộng với local I/D-TCM. Khi lên Phase 4, cần re-balance lại SRAM để khớp budget cuối.

### A. Current Implemented Per-Cluster SRAM

| Bank | Size | Address / Region | Vai trò |
|------|------|------------------|---------|
| I-TCM | 32 KB | `0x1000_0000` | Cluster firmware |
| Snitch D-TCM | 32 KB | `0x1000_8000` | Private scalar data |
| I-TCDM | 384 KB | `0x1010_0000` logical window | Weights + IFM tiles |
| O-TCDM | 128 KB | `0x1020_0000` logical window | OFM / INT32 accumulator |
| **Total per cluster** | **576 KB** | 32 + 32 + 512 KB | Current Phase 3B-A implementation |

### B. Logical Buffer Plan

| Buffer | Size target | Region | Vai trò |
|--------|-------------|--------|---------|
| Weight Buffer | 2 × 128 KB | I-TCDM | Ping-pong weight stationary |
| IFM Buffer | 2 × 50 KB | I-TCDM | Input feature map tiles |
| OFM Buffer | 2 × 50 KB | O-TCDM | Output / INT32 accumulator |

**Memory Latency Hiding**: Double-buffering (Ping-Pong) cho phép DMA prefetch tile kế tiếp vào Bank 1 trong khi Systolic Array xử lý Bank 0 → hardware utilization ~100%.

> **Phase 4 sizing note**: Nếu giữ 5 cluster với 576 KB/cluster thì riêng cluster-local SRAM là 2880 KB, vượt mục tiêu 2.5 MB. Vì vậy Phase 4 cần chọn một trong các hướng: giảm TCDM per cluster, giảm số cluster, hoặc cập nhật lại SRAM budget mục tiêu.

---

## 10. Development Phases

| Phase | Nội dung | Trạng thái |
|-------|----------|-----------|
| 1 | DMA Engine + AXI interface | ✅ Done |
| 2 | Data TCDM Interconnect (N-bank crossbar) | ✅ Done |
| 2.5 | AXI→OBI bridge, DMA-to-TCM test | ✅ Done |
| **3A** | **Snitch Core Integration: I-TCM, D-TCM isolation, Boot via AXI** | **✅ Done** |
| 3B-A | Systolic Array + Matrix Engine cluster integration | ✅ Done |
| 3B-B | Spatz Vector Engine integration (1 GHz cluster) | ⬜ Planned |
| 4 | Top-Level: 5-cluster integration + Manager Snitch | ⬜ Planned |
| 5 | Full YOLO layer end-to-end simulation | ⬜ Planned |

---

## 11. Hardware Verification Plan

### Unit Testing (Block-Level)
- **Snitch Boot TB** (`test_snitch_boot`): Nạp firmware qua AXI4-Lite, release reset, xác nhận Snitch viết signature `0xDEADBEEF` vào MMIO. *(Passed)*
- **I-TCM Arbiter TB**: Kiểm tra ưu tiên AXI vs Snitch, không có collision. *(Passed)*
- **OBI Demux TB**: Kiểm tra address decoding chính xác (I-TCM / D-TCM / Data TCDM / MMIO). *(Passed)*
- **Matrix Engine TB** *(Phase 3B-A)*: Verify 32×32 Systolic Array vs Python golden model. *(Passed)*
- **Cluster MatMul TB** *(Phase 3B-A)*: Snitch firmware trigger DMA, Systolic Array compute, OFM writeback; 10 randomized MatMul iterations. *(Passed)*
- **Spatz Vector TB** *(Phase 3B-B)*: Test RVV instructions.

### Cluster-Level Verification
- **TCDM Arbitration**: Snitch, Spatz, Systolic Array đồng thời access Data TCDM → không deadlock.
- **Firmware Dispatch**: Snitch firmware trigger DMA, DMA load data, Systolic Array compute.

### Top-Level Integration
- **5-Cluster TB**: Tất cả 5 cluster chạy đồng thời, Manager Snitch phân chia tiling.
- **End-to-End**: Mô phỏng full YOLO layer (Conv2D 3×3): External Memory → DMA → TCDM → Compute → Writeback.
- **Performance Profiling**: Đo cycle count thực tế → tính TOPS thực tế vs mục tiêu 10 TOPS.
