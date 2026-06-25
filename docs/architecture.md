# YOLO NPU Architecture Specification

**Version**: Phase 3B-A — Matrix Engine Integrated in Cluster  
**Last Updated**: 2026-06-25

---

## 1. Architectural Overview

The YOLO NPU is a heterogeneous compute architecture designed to achieve 10 TOPS. Deeply optimized for the YOLO family of models, it supports INT8 quantized inference.

To achieve high-performance matrix and vector operations, the architecture integrates 5 parallel **NPU Clusters**. Each cluster features:

1. **Control Core (Snitch)**: A RISC-V RV32IMAC scalar core. Boots from local I-TCM firmware, controls and dispatches compute commands to the Matrix and Vector engines via OBI.
2. **Matrix Engine** *(Phase 3B-A)*: A 32×32 INT8 Systolic Array optimized for Dense MatMul / Conv2D.
3. **Vector Engine** *(Phase 3B-B)*: A Spatz RVV co-processor for Depthwise Conv, Element-wise arithmetic, reductions, and vectorized post-processing.
4. **Shared Data TCDM**: A 256-bit wide L1 data memory shared between the Systolic Array, Vector Engine, and DMA.
5. **DMA Engine**: Handles burst data movement between AXI external memory (L2/DRAM) and Data TCDM autonomously, without involving Snitch.
6. **AFU**: A LUT-based activation/function unit for tensor-wide nonlinear lookup transforms.
7. **Interrupt Controller**: Aggregates internal done events and provides firmware-driven host completion through `irq_o`.

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
|  | Bootloader / Host  |                                                                            |
|  | AXI4-Lite Slave    |                                                                            |
|  +---------+----------+                                                                            |
|            | AXI-to-OBI                                                                            |
|            v                                                                                       |
|  +-----------------------------+        +----------------------+                                   |
|  | u_itcm_arbiter              |<-------| Snitch Core          |                                   |
|  | I-TCM Arbiter               |        | RV32IMAC             |                                   |
|  | host boot vs I-fetch        |        +----------+-----------+                                   |
|  +-------------+---------------+                   |                                               |
|                |                                   | Snitch D-Bus                                  |
|                v                                   v                                               |
|  +-----------------------------+        +----------------------+                                   |
|  | I-TCM (32 KB)               |        | D-side OBI Demux     |                                   |
|  | firmware only               |        | D-TCM / TCDM / MMIO  |                                   |
|  +-----------------------------+        +----+-----+------+---+                                    |
|                                             |     |      |                                         |
|                                             |     |      +------> +----------------------+         |
|                                             |     |               | MMIO/CSR             |         |
|                                             |     |               | cluster/idma/irq/AFU |         |
|                                             |     |               +----------+-----------+         |
|                                             |     |                          | irq_o               |
|                                             |     |                          v                     |
|                                             |     |               +----------------------+         |
|                                             |     |               | npu_interrupt_ctrl   |         |
|                                             |     |               | internal + host IRQ  |         |
|                                             |     |               +----------------------+         |
|                                             |     +------ Shared TCDM window access ------+        |
|                                             |                                             |        |
|                                             v                                             |        |
|                                      +------------------+                                 |        |
|                                      | Snitch D-TCM     |                                 |        |
|                                      | private 32 KB    |                                 |        |
|                                      | TB backdoor dbg  |                                 |        |
|                                      +------------------+                                 |        |
|                                                                                           v        |
|  +----------------------------------------------------------------------------------------------+  |
|  |                          Shared Data TCDM Interconnect (256-bit)                             |  |
|  +-------+--------------------+---------------------+-----------------------+-------------------+  |
|          ^                    ^                     ^                       |                      |
|          |                    |                     |                       v                      |
| +--------+--------+  +--------+--------+  +---------+---------+   +-------------------+            |
| | Systolic Array  |  | Spatz Vector    |  | DMA Engine        |<--|  Data TCDM SRAM   |            |
| | (Matrix Engine) |  | (Vector Engine) |  | (AXI4 Master to L2)|  | (12 I-TCDM Banks) |            |
| +-----------------+  +-----------------+  +-------------------+   | (4 O-TCDM Banks)  |            |
|          ^                    ^                     ^             +---------^---------+            |
|          |                    |                     |                       |                      |
|          +--------------------+---------------------+---------------+ +-----+------+               |
|                                                                  AFU | | LUT Func |               |
|                                                              256-bit | | Unit     |               |
|                                                               master | +----------+               |
|                                                                   +-------------------+            |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Interface Reference

| ID | Name | Protocol | Width | Description |
|----|------|----------|-------|-------------|
| A  | Snitch I-Fetch | OBI | 32-bit | Instruction fetch path từ Snitch tới `u_itcm_arbiter`. |
| B  | Host AXI4-Lite Slave | AXI4-Lite | 32-bit | Host/testbench chỉ nạp firmware vào I-TCM. Không dùng host frontend path để access D-TCM/MMIO. |
| C  | AXI-to-OBI Boot Bridge | AXI4-Lite→OBI | 32-bit I-TCM-only | Convert host AXI-Lite boot writes sang OBI request đi trực tiếp tới `u_itcm_arbiter`. |
| D  | Snitch D-Bus | OBI | 32-bit | Bus dữ liệu của Snitch tới private D-TCM, Shared Data TCDM window và MMIO. Không arbitrate với host AXI path. |
| E  | D-side OBI Demux | OBI Demux | 32-bit control side | Decode Snitch D-Bus access tới D-TCM, Shared Data TCDM window và MMIO; không decode I-TCM. |
| F  | `u_itcm_arbiter` | OBI Arbiter 2→1 | 32-bit | Phân giải giữa host AXI boot bridge access vào I-TCM và Snitch I-fetch. |
| G  | DMA AXI Master | AXI4 | 256-bit | DMA tự load dữ liệu từ L2/DRAM vào Data TCDM |
| H  | Shared Data TCDM Interconnect | OBI/TCDM | 256-bit | Data path lớn cho DMA, Systolic, Spatz và shared SRAM; không kéo xuống 32-bit. |
| I  | Interrupt Controller | OBI MMIO + pins | 32-bit regs | Snitch writes `HOST_NOTIFY`; block asserts `irq_o`. Hardware done events can wake Snitch through `snitch_irq_o.mcip`. |
| J  | AFU | OBI MMIO + OBI/TCDM master | 32-bit control, 256-bit data | LUT activation/function unit. Snitch programs LUT/CSR; AFU reads/writes Shared Data TCDM autonomously. |

### Arbiter Naming Clarification

RTL hiện tại tách rõ boot instruction path và Snitch data path:

- **`u_itcm_arbiter`** là I-TCM arbiter. `m0` nhận host AXI boot bridge; `m1` nhận Snitch I-fetch.
- **D-side demux** nhận trực tiếp Snitch D-Bus 32-bit và decode D-TCM, Shared Data TCDM window, MMIO, plus error sink.
- **Legacy `u_sys_arbiter` path đã bỏ**: host không còn frontend access vào D-TCM/MMIO/TCDM; debug/readback dùng TB backdoor.

### Width Partitioning Direction

Kiến trúc hiện tại tách hai miền width:

- **Address width**: giữ thống nhất 32-bit physical address trên AXI-Lite/OBI nội bộ của cluster. Width refactor bên dưới là **data bus width**, không đổi memory map.
- **Boot side 32-bit**: host AXI-Lite physical, AXI-to-OBI boot bridge, `u_itcm_arbiter`, và I-TCM SRAM.
- **Snitch control side 32-bit**: Snitch D-Bus, D-TCM và MMIO đều native 32-bit vì Snitch chỉ làm firmware control/scheduler.
- **Compute/data side 256-bit**: Shared Data TCDM interconnect, DMA data path, Systolic ports, Spatz vector data movement, và SRAM banking cho tensor tiles.
- **Boundary adapter**: chỉ dùng adapter tại boundary `Snitch D-Bus 32-bit → Shared Data TCDM 256-bit`. Không kéo Shared Data TCDM xuống 32-bit.
- **Snitch D-TCM private**: host AXI-Lite không cần frontend access vào D-TCM. Debug/readback D-TCM trong verification dùng testbench backdoor function, không dùng memory-mapped host path.

---

## 3. Interrupt and Completion Architecture

`npu_interrupt_ctrl` là block MMIO 32-bit nằm trong aperture control của cluster. Nó tách hai miền event:

1. **Internal interrupt domain**: DMA/Systolic/AFU/Spatz done events được latch vào `INT_PENDING`. Nếu bit tương ứng được bật trong `INT_ENABLE`, controller kéo `snitch_irq_o.mcip` để đánh thức Snitch. Đây là nền tảng cho firmware `wfi`/trap handler ở phase sau.
2. **External host completion domain**: firmware ghi `NPU_IRQ_HOST_NOTIFY`; controller latch `HOST_STATUS` nội bộ, set external pending bit và assert `irq_o` ra ngoài cluster. Verification hiện dùng `irq_o` làm completion event và kiểm tra data output ở L2/TCDM.

### Interrupt Register Map

| Offset | Register | Direction | Role |
|--------|----------|-----------|------|
| `0x00` | `NPU_IRQ_INT_ENABLE` | Snitch RW | Enable internal event bits vào Snitch IRQ. |
| `0x04` | `NPU_IRQ_INT_PENDING` | Snitch R | Latched DMA/Systolic/AFU/Spatz done events. |
| `0x08` | `NPU_IRQ_INT_CLEAR` | Snitch W1C | Clear internal pending bits sau trap/handler. |
| `0x0c` | `NPU_IRQ_EXT_ENABLE` | Snitch RW | Enable host completion IRQ. Reset default enables host done. |
| `0x10` | `NPU_IRQ_EXT_PENDING` | Snitch R | Latched host completion pending bit. |
| `0x14` | `NPU_IRQ_EXT_CLEAR` | Snitch W1C | Clear external pending bit if firmware/host-control path needs reuse. |
| `0x18` | `NPU_IRQ_HOST_NOTIFY` | Snitch W | Firmware writes pass/fail/progress code; asserts `irq_o`. |
| `0x1c` | `NPU_IRQ_HOST_STATUS` | Snitch R/W | Internal status latch for future host-control/MMIO visibility. |

### Event Bits

| Bit | Internal source | Meaning |
|-----|-----------------|---------|
| `0` | DMA | iDMA A2O/O2A transfer done. |
| `1` | Systolic | Matrix controller done. |
| `2` | AFU | AFU operation done. Connected to AFU `done_o`. |
| `3` | Spatz | Spatz accelerator response valid. |

> Current host AXI-Lite frontend intentionally reaches **I-TCM only**. Therefore cocotb does not read interrupt MMIO through the host path today. Completion is observed via `irq_o`; correctness is proved by exact output data checks.

---

## 4. Systolic Array Micro-Architecture

```text
+------------------------------------------------------------------------------------------------+
|                                    Systolic Array (32x32)                                      |
|                                                                                                |
|   OBI I-TCDM Mux ---> [Weight FIFO] ---> [Weights (W) loaded sequentially downwards]           |
|                                            |     |     |           |                           |
|  +-----------------+                       v     v     v           v                           |
|  | [IFM FIFO]      |    IFM_Row[0]      +-----+-----+-----+     +-----+                        |
|  |     |           |------------------->|PE0,0|PE0,1|PE0,2| ... |P0,31| (Delay 0 for IFM)      |
|  |     v           |                    +-----+-----+-----+     +-----+                        |
|  | Input Skewing   |                       |     |     |           |                           |
|  | (Triangle       |    IFM_Row[1] (d=1)+-----+-----+-----+     +-----+                        |
|  |  Delay Regs)    |------------------->|PE1,0|PE1,1|PE1,2| ... |P1,31|                        |
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
|                                            |                                                   |
|                                            v                                                   |
|                                       [OFM FIFO] ---> OBI O-TCDM Demux                         |
|                                                                                                |
+------------------------------------------------------------------------------------------------+
```

### 4.1 Cấu trúc Data FIFOs
Mạch điều khiển Systolic Array (`systolic_controller.sv`) sử dụng kiến trúc bất đồng bộ một phần (Decoupled I/O) thông qua các hàng đợi FIFO:
- **IFM & Weight FIFOs (Input):** Nạp dữ liệu song song từ giao tiếp OBI I-TCDM. Chức năng chính là che giấu độ trễ (latency hiding) của mạng Interconnect, đảm bảo dữ liệu đưa vào Input Skewing luôn có sẵn mà không làm đình trệ pipeline tính toán bên trong mảng PE.
- **OFM FIFO (Output):** Các giá trị PSum sau khi đi qua Output Deskewing sẽ được đẩy vào OFM FIFO trước khi ghi ngược ra O-TCDM. Bộ đệm này sử dụng cơ chế Backpressure (`almost_full`) để cho phép Hệ thống Writeback có thể chủ động stall (khi băng thông TCDM nghẽn) mà không làm mất dữ liệu đầu ra của Array.

## 5. Data TCDM SRAM Micro-Architecture (Detailed)

> **Lưu ý Kiến trúc (Architecture Update):**
> Kiến trúc TCDM Interconnect đã được nâng cấp sang mô hình phân nhóm (Grouped Tree Topology) học hỏi từ dự án MAGIA để giảm Priority Starvation và Bank Conflict.
> Chi tiết thiết kế đã implemented xem tại: [implemented/tcdm_interconnect_upgrade.md](implemented/tcdm_interconnect_upgrade.md).

Shared Data TCDM là L1 data scratchpad chính cho compute path. Nó không phải cache: mọi tile được firmware/DMA đặt vào địa chỉ rõ ràng, deterministic latency, và arbitration được xử lý bởi TCDM interconnect.

*(Chi tiết về kiến trúc TCDM hiện tại và phương án nâng cấp sang mô hình phân nhóm Grouped Tree Topology đã được chuyển sang tài liệu chuyên đề: [implemented/tcdm_interconnect_upgrade.md](implemented/tcdm_interconnect_upgrade.md))*

#### Current Logical Buffers

| Buffer | Address | Region | Vai trò |
|--------|---------|--------|---------|
| `WEIGHT_PING_ADDR` | `0x1011_0000` | I-TCDM | 32×32 INT8 weight tile |
| `IFM_PING_ADDR` | `0x1012_0000` | I-TCDM | M×32 INT8 IFM tile |
| `OFM_PING_ADDR` | `0x1020_0000` | O-TCDM | M×32 INT32 output tile |

---

## 6. Memory Map

| Address Range | Size | Region | Role |
|---------------|------|--------|------|
| `0x1000_0000 – 0x1000_7FFF` | 32 KB | **I-TCM** | Firmware Snitch instruction memory. Target host AXI-Lite boot path đi trực tiếp tới `u_itcm_arbiter`; Snitch fetch đi qua I-fetch port. |
| `0x1000_8000 – 0x1000_FFFF` | 32 KB | **Snitch D-TCM** | Private data của Snitch: stack, `.data`, `.bss`, scalar state. Không expose trên AXI-Lite host path sau refactor; debug dùng TB backdoor. |
| `0x1010_0000 – 0x1015_FFFF` | 384 KB | **I-TCDM logical window** | Weights và IFM tiles cho compute engines. |
| `0x1020_0000 – 0x1021_FFFF` | 128 KB | **O-TCDM logical window** | OFM / INT32 accumulator writeback. |
| `0x2000_0000 – 0x2000_FFFF` | 64 KB | **MMIO / CSR** | `cluster_ctrl_regs`, iDMA, interrupt controller, AFU/accelerator control. |
| `0x8000_0000+` | External | **L2 / AXI sim memory** | Testbench/external memory chứa input/output buffers. |

> **Tại sao tách I-TCM và D-TCM?**  
> Harvard Architecture: Snitch fetch lệnh qua I-Fetch (không cạnh tranh băng thông với D-Bus). D-TCM private đảm bảo latency cố định cho stack/local vars và không cần host frontend access trong normal boot/inference path.

### MMIO Sub-Map

| Address | Block | Role |
|---------|-------|------|
| `0x2000_0000` | `cluster_ctrl_regs` | Legacy DMA and Systolic control/status registers. |
| `0x2000_1000` | `npu_idma_ctrl_mm` | iDMA-compatible 1D/2D/3D transfer configuration. |
| `0x2000_2000` | `npu_interrupt_ctrl` | Internal done-event IRQ and firmware-driven host completion. |
| `0x2000_3000` | `afu` | LUT activation unit. Offset `0x000..0x3ff` is LUT SRAM; offset `0x400+` is status/src/dst/length/mode CSR. |

---

## 7. L1/L2 Data Movement

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

## 8. Boot Flow

Detailed walkthrough: [Boot Flow](boot_flow.md).

```
1. Host ghi firmware vào I-TCM qua AXI4-Lite Slave (cổng B/H)
   → AXI-to-OBI bridge
   → u_itcm_arbiter
   → I-TCM SRAM
   
2. Host de-assert reset and then asserts `fetch_enable_i`

3. Snitch core reset → PC = 0x1000_0000 (BootAddr)

4. Snitch fetch instruction từ I-TCM:
   Snitch I-Fetch → u_itcm_arbiter → I-TCM SRAM

5. Snitch thực thi firmware:
   - Khởi tạo private D-TCM (stack setup, `.data`, `.bss`)
   - Cấu hình cluster_ctrl_regs qua Snitch D-Bus → D-side demux → MMIO
   - (Phase 3B-A) Trigger DMA để load weight/IFM vào Data TCDM
   - (Phase 3B-A) Dispatch lệnh tới Systolic Array
   - (Phase 3B-B) Dispatch lệnh RVV tới Spatz
   - Dispatch AFU LUT transforms through MMIO and internal AFU done interrupt

6. Firmware ghi completion status vào `NPU_IRQ_HOST_NOTIFY`
   → `npu_interrupt_ctrl` latch `HOST_STATUS` nội bộ và assert `irq_o` cho host/testbench
```

---

## 9. Test Flow

Detailed walkthrough: [Test Flow](test_flow.md).

### Current Verified Tests

| Test | Flow | Pass criteria |
|------|------|---------------|
| `test_systolic.py` | Direct standalone Systolic Array stimulus | Scoreboard captures OFM and all 32 columns match golden model |
| `test_snitch_boot.py` | Host loads `boot.bin` into I-TCM and releases Snitch fetch | `irq_o` asserts |
| `test_matmul.py` | Host prepares M=64 tensors in L2, firmware runs DMA + Systolic + writeback | Host IRQ plus full OFM match against NumPy golden |
| `test_afu_basic.py` | Firmware seeds TCDM tensors, programs AFU LUT/CSR, waits AFU IRQ, checks e8/e16/e32 output | Host IRQ plus exact TCDM output compare |

### Cluster MatMul Test Flow

```text
Testbench L2 buffers
  ├─ weights @ 0x8000_0000
  ├─ IFM     @ 0x8000_1000
  └─ OFM     @ 0x8000_2000

Host AXI-Lite loads matmul firmware into I-TCM
Testbench prepares L2 fixture buffers
Testbench releases fetch_enable_i
Snitch firmware:
  1. DMA weights L2→I-TCDM
  2. DMA IFM L2→I-TCDM
  3. Start Systolic Controller
  4. Wait SYS_DONE
  5. DMA OFM O-TCDM→L2
  6. Write pass/fail status into NPU_IRQ_HOST_NOTIFY
Testbench reads L2 OFM and compares against NumPy golden
Testbench waits `irq_o`; current host AXI path remains I-TCM-only
```

---

## 10. SRAM Allocation Analysis

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

## 11. Development Phases

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

## 12. Hardware Verification Plan

### Unit Testing (Block-Level)
- **Snitch Boot TB** (`test_snitch_boot`): Nạp firmware qua AXI4-Lite vào I-TCM, release fetch, xác nhận host IRQ. *(Passed)*
- **I-TCM Arbiter TB**: Kiểm tra ưu tiên AXI vs Snitch, không có collision. *(Passed)*
- **D-side OBI Demux TB**: Kiểm tra address decoding chính xác (D-TCM / Data TCDM / MMIO / error sink). *(Passed)*
- **Matrix Engine TB** *(Phase 3B-A)*: Verify 32×32 Systolic Array vs Python golden model. *(Passed)*
- **Cluster MatMul TB** *(Phase 3B-A)*: Snitch firmware trigger DMA, Systolic Array compute, OFM writeback; M=64 raw-register regression. *(Passed)*
- **AFU Cluster TB**: Snitch firmware programs AFU LUT/control, verifies e8/e16/e32 output and AFU internal interrupt. *(Passed)*
- **Spatz Vector TB** *(Phase 3B-B)*: Test RVV instructions.

### Cluster-Level Verification
- **TCDM Arbitration**: Snitch, Spatz, Systolic Array đồng thời access Data TCDM → không deadlock.
- **Firmware Dispatch**: Snitch firmware trigger DMA, DMA load data, Systolic Array compute.

### Top-Level Integration
- **5-Cluster TB**: Tất cả 5 cluster chạy đồng thời, Manager Snitch phân chia tiling.
- **End-to-End**: Mô phỏng full YOLO layer (Conv2D 3×3): External Memory → DMA → TCDM → Compute → Writeback.
- **Performance Profiling**: Đo cycle count thực tế → tính TOPS thực tế vs mục tiêu 10 TOPS.
