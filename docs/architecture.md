# YOLO NPU Architecture Specification


## 1. Architectural Overview

The YOLO NPU is a heterogeneous compute architecture designed to achieve 10 TOPS. Deeply optimized for the YOLO family of models, it supports INT8 quantized inference.

To achieve high-performance matrix and vector operations, the architecture integrates 5 parallel **NPU Clusters**. Each cluster features:

1. **Control Core (Snitch)**: A RISC-V RV32IMAC scalar core. Boots from local I-TCM firmware, controls and dispatches compute commands to the Matrix and Vector engines via OBI.
2. **Matrix Engine**: A 32×32 INT8 Systolic Array optimized for Dense MatMul / Conv2D.
3. **Vector Engine**: A Spatz RVV co-processor for Depthwise Conv, element-wise arithmetic, reductions, and vectorized post-processing.
4. **Shared Data TCDM**: A 256-bit wide L1 data memory shared between the Systolic Array, Vector Engine, and DMA.
5. **DMA Engine**: Handles burst data movement between AXI external memory (L2/DRAM) and Data TCDM autonomously, without involving Snitch.
6. **AFU**: A LUT-based activation/function unit for tensor-wide nonlinear lookup transforms.
7. **Interrupt Controller**: Aggregates internal done events and provides firmware-driven host completion through `irq_o`.

### L1 vs. L2 Memory Model

- **L1** is all cluster-local memory: I-TCM, Snitch D-TCM, Shared Data TCDM SRAM, and the MMIO/CSR aperture.
- **L2** is external memory outside the cluster, accessed through the DMA AXI4 master or the AXI4-Lite host/testbench port.
- Snitch only orchestrates work through firmware and CSRs; large tile data paths go through DMA so the scalar core does not copy individual words.
- DMA currently supports the main copy directions: **L2 -> L1** for weights/IFM, **L1 -> L2** for OFM writeback, and **L1 -> L1** for local copy/repacking inside local memories.

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
|                                             |     |               | sys/idma/irq/afu/pmu |         |
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
|  |          Shared Data TCDM Interconnect (256-bit) (M0:Snitch, M1-M10:Engines)                 |  |
|  +----+-----------------+------------------+-----------------+------------------+---------------+  |
|       ^                 ^                  ^                 ^                  |                  |
|       |                 |                  |                 |                  v                  |
| +-----+-----+   +-------+-------+  +-------+-------+  +------+------+  +-------------------+       |
| | Systolic  |   | Spatz Vector  |  | iDMA Engine   |  | AFU LUT     |  |  Data TCDM SRAM   |       |
| | Array     |   | Engine        |  | (AXI Master)  |  | Engine      |  | (12 I-TCDM Banks) |       |
| | (M3-M7)   |   | (M1, M8)      |  | (M2, M9)      |  | (M10)       |  | (4 O-TCDM Banks)  |       |
| +-----------+   +---------------+  +---------------+  +-------------+  +-------------------+       |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Interface Reference

| ID | Name | Protocol | Width | Description |
|----|------|----------|-------|-------------|
| A  | Snitch I-Fetch | OBI | 32-bit | Instruction fetch path from Snitch to `u_itcm_arbiter`. |
| B  | Host AXI4-Lite Slave | AXI4-Lite | 32-bit | Host/testbench loads firmware into I-TCM and accesses explicit host-visible status/control windows such as PMU or command-control. The host frontend path does not directly access D-TCM or Shared TCDM. |
| C  | AXI-to-OBI Host Bridge | AXI4-Lite→OBI | 32-bit | Converts host AXI-Lite transactions into OBI requests, then decodes to I-TCM or explicitly exposed host-visible status/control windows. |
| D  | Snitch D-Bus | OBI | 32-bit | Snitch data bus to private D-TCM, the Shared Data TCDM window, and MMIO. It does not arbitrate with the host AXI path. |
| E  | D-side OBI Demux | OBI Demux | 32-bit control side | Decodes Snitch D-Bus accesses to D-TCM, the Shared Data TCDM window, and MMIO; it does not decode I-TCM. |
| F  | `u_itcm_arbiter` | OBI Arbiter 2->1 | 32-bit | Arbitrates between host AXI boot-bridge access to I-TCM and Snitch I-fetch. |
| G  | DMA AXI Master | AXI4 | 256-bit | DMA autonomously loads data from L2/DRAM into Data TCDM. |
| H  | Shared Data TCDM Interconnect | OBI/TCDM | 256-bit | Wide data path for DMA, Systolic, Spatz, and shared SRAM; it is not reduced to 32-bit. |
| I  | Interrupt Controller | OBI MMIO + pins | 32-bit regs | Snitch writes `HOST_NOTIFY`; block asserts `irq_o`. Hardware done events can wake Snitch through `snitch_irq_o.mcip`. |
| J  | AFU | OBI MMIO + OBI/TCDM master | 32-bit control, 256-bit data | LUT activation/function unit. Snitch programs LUT/CSR; AFU reads/writes Shared Data TCDM autonomously. |

### Arbiter Naming Clarification

The current RTL clearly separates the boot instruction path from the Snitch data path:

- **`u_itcm_arbiter`** is the I-TCM arbiter. `m0` receives host AXI transactions decoded to the I-TCM boot window; `m1` receives Snitch I-fetch.
- **D-side demux** receives the 32-bit Snitch D-Bus directly and decodes D-TCM, the Shared Data TCDM window, MMIO, plus the error sink.
- **Legacy `u_sys_arbiter` path removed**: the host no longer has frontend access to D-TCM or Shared TCDM; debug/readback uses testbench backdoor access. Host-visible MMIO should only cover explicit control/status blocks such as PMU or command-control.

### Width Partitioning Direction

The current architecture separates two width domains:

- **Address width**: keep one 32-bit physical address convention across AXI-Lite/OBI inside the cluster. The width partitioning below refers to **data bus width**, not memory-map changes.
- **Host side 32-bit**: host AXI-Lite physical, AXI-to-OBI host bridge, I-TCM boot window, and selected host-visible status/control windows.
- **Snitch control side 32-bit**: Snitch D-Bus, D-TCM, and MMIO are native 32-bit because Snitch acts only as firmware control/scheduler.
- **Compute/data side 256-bit**: Shared Data TCDM interconnect, DMA data path, Systolic ports, Spatz vector data movement, and SRAM banking for tensor tiles.
- **Boundary adapter**: use an adapter only at the `Snitch D-Bus 32-bit -> Shared Data TCDM 256-bit` boundary. Do not reduce Shared Data TCDM to 32-bit.
- **Snitch D-TCM private**: host AXI-Lite does not need frontend access to D-TCM. D-TCM debug/readback in verification uses a testbench backdoor function, not a memory-mapped host path.
- **Command stream placement**: host writes command streams to L2/DRAM, then programs a small host-visible command-control register block. Firmware uses a fixed 4 KB Shared TCDM staging window and refills it from L2 with iDMA while dispatching descriptors.

---

## 3. Interrupt and Completion Architecture

`npu_interrupt_ctrl` is a 32-bit MMIO block inside the cluster control aperture. It separates two event domains:

1. **Internal interrupt domain**: DMA/Systolic/AFU/Spatz done events are latched into `INT_PENDING`. If the corresponding bit is enabled in `INT_ENABLE`, the controller asserts `snitch_irq_o.mcip` to wake Snitch. This is the foundation for firmware `wfi`/trap-handler support.
2. **External host completion domain**: firmware writes `NPU_IRQ_HOST_NOTIFY`; the controller latches `HOST_STATUS` internally, sets the external pending bit, and asserts cluster `irq_o`. Verification currently uses `irq_o` as the completion event and checks output data in L2/TCDM.

### Interrupt Register Map

| Offset | Register | Direction | Role |
|--------|----------|-----------|------|
| `0x00` | `NPU_IRQ_INT_ENABLE` | Snitch RW | Enable internal event bits into Snitch IRQ. |
| `0x04` | `NPU_IRQ_INT_PENDING` | Snitch R | Latched DMA/Systolic/AFU/Spatz done events. |
| `0x08` | `NPU_IRQ_INT_CLEAR` | Snitch W1C | Clear internal pending bits after trap/handler service. |
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

> Current host AXI-Lite frontend reaches I-TCM and selected host-visible
> status/control windows such as PMU. It must not expose D-TCM or the full
> Shared TCDM aperture. Completion remains observed via `irq_o`; correctness is
> proved by exact output data checks.

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

### 4.1 Data FIFO Structure
The Systolic Array controller (`systolic_controller.sv`) uses a partially decoupled I/O architecture through FIFO queues:
- **IFM & Weight FIFOs (Input):** Load data in parallel from the OBI I-TCDM interface. Their main role is latency hiding for the interconnect, ensuring that data entering input skewing is available without stalling the PE-array compute pipeline.
- **OFM FIFO (Output):** PSum values that pass through output deskewing are pushed into the OFM FIFO before being written back to O-TCDM. This buffer uses backpressure (`almost_full`) so the writeback system can intentionally stall when TCDM bandwidth is congested without losing array output data.

## 5. Data TCDM SRAM Micro-Architecture (Detailed)

> **Architecture Update:**
> The TCDM interconnect has been upgraded to a grouped tree topology inspired by the MAGIA project to reduce priority starvation and bank conflicts.
> The detailed design is tracked in [implemented/tcdm_interconnect_upgrade.md](implemented/tcdm_interconnect_upgrade.md).

Shared Data TCDM is the main L1 data scratchpad for the compute path. It is not a cache: firmware/DMA places every tile at an explicit address, latency is deterministic, and arbitration is handled by the TCDM interconnect.

*(Details on the current TCDM architecture and the grouped tree topology upgrade live in the dedicated document: [implemented/tcdm_interconnect_upgrade.md](implemented/tcdm_interconnect_upgrade.md).)*

#### Current Logical Buffers

| Buffer | Address | Region | Role |
|--------|---------|--------|---------|
| `WEIGHT_PING_ADDR` | `0x1011_0000` | I-TCDM | 32×32 INT8 weight tile |
| `IFM_PING_ADDR` | `0x1012_0000` | I-TCDM | M×32 INT8 IFM tile |
| `OFM_PING_ADDR` | `0x1020_0000` | O-TCDM | M×32 INT32 output tile |

---

## 6. Memory Map

| Address Range | Size | Region | Role |
|---------------|------|--------|------|
| `0x1000_0000 – 0x1000_7FFF` | 32 KB | **I-TCM** | Snitch firmware instruction memory. The target host AXI-Lite boot path goes directly to `u_itcm_arbiter`; Snitch fetch uses the I-fetch port. |
| `0x1000_8000 – 0x1000_FFFF` | 32 KB | **Snitch D-TCM** | Snitch private data: stack, `.data`, `.bss`, and scalar state. It is not exposed on the AXI-Lite host path after the refactor; debug uses a testbench backdoor. |
| `0x1010_0000 – 0x1015_FFFF` | 384 KB | **I-TCDM logical window** | Weights and IFM tiles for compute engines. |
| `0x1017_F000 – 0x1017_FFFF` | 4 KB | **TCDM command staging window** | Reserved local command-stream staging buffer. Host does not write this directly; Snitch refills this window from L2 through iDMA as descriptor offsets advance. |
| `0x1020_0000 – 0x1021_FFFF` | 128 KB | **O-TCDM logical window** | OFM / INT32 accumulator writeback. |
| `0x2000_0000 – 0x2000_FFFF` | 64 KB | **MMIO / CSR** | Systolic control, iDMA, interrupt controller, AFU/accelerator control. |
| `0x8000_0000+` | External | **L2 / AXI sim memory** | Testbench/external memory that contains input/output buffers. |

> **Why separate I-TCM and D-TCM?**
> Harvard architecture: Snitch fetches instructions through I-Fetch, which does not compete with D-Bus bandwidth. Private D-TCM gives fixed latency for stack/local variables and does not need host frontend access during the normal boot/inference path.

### MMIO Sub-Map

| Address | Block | Role |
|---------|-------|------|
| `0x2000_0000` | `systolic_ctrl_regs` | Systolic control/status, linebuffer, and requant registers. Unused low offsets return zero. |
| `0x2000_1000` | `npu_idma_ctrl_mm` | iDMA-compatible 1D/2D/3D transfer configuration. |
| `0x2000_2000` | `npu_interrupt_ctrl` | Internal done-event IRQ and firmware-driven host completion. |
| `0x2000_3000` | `afu` | LUT activation unit. Offset `0x000..0x3ff` is LUT SRAM; offset `0x400+` is status/src/dst/length/mode CSR. |
| `0x2000_5000` | `npu_cmd_ctrl` | Host-visible command bootstrap/status block: command L2 base, byte length, TCDM command staging base/size, start/status/fail registers. |

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
5. Systolic Controller drains `M x 32` INT32 rows from the OFM FIFO through a drain engine that runs independently from the main compute FSM.
6. In raw mode, the drain engine writes each row as 32 INT32 values through 4 parallel OBI write ports.
7. In accumulated mode, the drain engine reads the previous psum row from O-TCDM port 0, adds the new OFM row, writes back, and lets compute continue until normal OFM FIFO backpressure is required.
8. In requant mode, the drain engine applies per-channel bias/scale/shift/zero-point/clamp and writes one packed 32-byte INT8 row through output port 0.
9. DMA copies OFM/activation buffers from O-TCDM back to L2.

### Direct Conv2D Stream Linebuffer Path

The default high-performance Conv2D path for supported TCDM-resident tensors is
now `conv_linebuf_stream_packer` inside `systolic_controller`. The older
`conv_channel_linebuf_packer` is retained only as a legacy/reference block in
the documentation and tests; it is not the Micro-YOLO performance path. Packed
`M x 32` IFM materialization through software+iDMA+Spatz remains available only
as a backup for unsupported shapes, L2-only inputs, wider kernels, or debug
comparison.

The stream linebuffer supports `C_BLOCK=32`, native `1x1..5x5` window
generation, `input_w <= 640`, padding zero-injection, row-ring/window caching,
background fill, and a dedicated `1x1` bypass. For small kernels where
`KH*KW*IC <= 32`, firmware enables coalesced window-pack mode: the linebuffer
emits one 32-byte IFM vector per output spatial position, with all kernel
taps/channels packed into systolic lanes. This is the default fast path for
YOLO/CNN first-layer style `3x3/C3` cases and avoids materializing im2col rows
in TCDM.

For larger `K`, KGEN mode is the main deep-layer path. Host/Python provides
the linebuffer register image, `{kh,kw,ic}` seed, and `k_tile_count`; Snitch
starts systolic once per spatial micro-tile; `systolic_controller` loops over K
tiles internally. RTL increments the 32 lane descriptors in `{kh,kw,ic}` order
without Snitch hot-path div/mod, and intermediate K tiles accumulate through
the controller psum path. Current firmware uses `M_tile=256`/`16x16` spatial
micro-tiles for Micro-YOLO C32 convs; `NPU_CONV2D_LINEBUF_KGEN_MAX_M=1024`
remains the broader software limit for supported linebuffer launches.

For C32-blocked YOLO activations, the host-planned descriptor enables the
`C32_FAST` RTL path: `input_base`, `pixel_stride_bytes`, `row_stride_bytes`,
and `channel_addr_offset` are 32-byte aligned; `block_valid_bytes=32`;
`lane_base=0`; `input_c_base=0`. In this path the linebuffer can consume the
256-bit beat directly and bypass byte-level `merge_beats` on the main data
path. `merge_beats` remains implemented for RGB stem, sub-C32/tail,
`lane_base != 0`, raw/NHWC fallback, and any access crossing a 32-byte beat.

To reduce Snitch workload, Micro-YOLO now uses host-generated
`npu_conv2d_linebuf_job_desc_t` arrays delivered through a runtime L2 descriptor
manifest plus descriptor blobs. `hw/rtl/cluster/tb/npu_linebuf_precompute.py` emits each
descriptor with the full `systolic_linebuf_cfg_t`, `systolic_gemm32_req_t`, row
count, and K-tile count. The host writes the manifest and blobs to L2, firmware
DMA-copies them into scratch/TCDM, stores descriptor pointer/count in
`npu_layer_t`, and calls
`npu_conv2d_packed_run_linebuf_job_descs()`. That runner preloads the next
linebuffer/GEMM shadow registers while the current job is running, so Snitch
does not rebuild tile config in the critical graph path. The more general
L2-resident command-stream flow through `npu_cmd_ctrl` remains the intended
model-level host interface; Micro-YOLO now exercises the same host-owned
descriptor delivery model at operator-descriptor granularity.

Detailed architecture, requirements, and pseudo-code are tracked in
[`conv2d_packed_systolic_plan.md`](conv2d_packed_systolic_plan.md).

### Systolic Requant Path

The systolic controller contains an optional fused requant path for Conv/GEMM layers whose next consumer expects INT8 activations:

```text
npu_systolic_array
  -> output deskew
  -> OFM FIFO row: 32 x INT32
  -> requant_pipeline: per-channel INT32 -> INT8
  -> packed row write: 32 bytes to O-TCDM
```

Default reset and the raw HAL path keep requant disabled, so debug and accumulator tests still observe full INT32 results. Firmware enables requant only through `systolic_gemm32_requant()` after programming the qparam arrays in `systolic_ctrl_regs`.

| Mode | Row size | Write ports | Use case |
|------|----------|-------------|----------|
| Raw accumulator | 128 bytes | 4 × 256-bit OBI | Debug, full INT32 compare, layers needing accumulation output |
| Requant INT8 | 32 bytes | 1 × 256-bit OBI | Conv/GEMM activation output for next INT8 layer |

---

## 8. Boot Flow

Detailed walkthrough: [Boot Flow](boot_flow.md).

```
1. Host writes firmware into I-TCM through the AXI4-Lite slave (port B/H)
   → AXI-to-OBI bridge
   → u_itcm_arbiter
   → I-TCM SRAM

2. Host de-assert reset and then asserts `fetch_enable_i`

3. Snitch core reset → PC = 0x1000_0000 (BootAddr)

4. Snitch fetches instructions from I-TCM:
   Snitch I-Fetch → u_itcm_arbiter → I-TCM SRAM

5. Snitch executes firmware:
   - Initializes private D-TCM (stack setup, `.data`, `.bss`)
   - Configure MMIO CSRs through Snitch D-Bus → D-side demux → MMIO
   - Triggers DMA to load weights/IFM into Data TCDM
   - Dispatches commands to the Systolic Array
   - Dispatches RVV commands to Spatz
   - Dispatch AFU LUT transforms through MMIO and internal AFU done interrupt

6. Firmware writes completion status to `NPU_IRQ_HOST_NOTIFY`
   → `npu_interrupt_ctrl` latches `HOST_STATUS` internally and asserts `irq_o` for host/testbench
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
Testbench waits `irq_o`; host AXI path is used only for I-TCM boot and explicit host-visible status/control windows
```

---

## 10. SRAM Allocation Analysis

The original top-level target was **2.5 MB of on-chip SRAM**. The current cluster baseline prioritizes architectural correctness and single-cluster verification first: each cluster currently uses a physical 512 KB Data TCDM plus local I/D-TCM. The top-level integration work must rebalance SRAM to match the final budget.

### A. Current Implemented Per-Cluster SRAM

| Bank | Size | Address / Region | Role |
|------|------|------------------|---------|
| I-TCM | 32 KB | `0x1000_0000` | Cluster firmware |
| Snitch D-TCM | 32 KB | `0x1000_8000` | Private scalar data |
| I-TCDM | 384 KB | `0x1010_0000` logical window | Weights + IFM tiles |
| O-TCDM | 128 KB | `0x1020_0000` logical window | OFM / INT32 accumulator |
| **Total per cluster** | **576 KB** | 32 + 32 + 512 KB | Current implemented cluster baseline |

### B. Logical Buffer Plan

| Buffer | Size target | Region | Role |
|--------|-------------|--------|---------|
| Weight Buffer | 2 × 128 KB | I-TCDM | Ping-pong weight stationary |
| IFM Buffer | 2 × 50 KB | I-TCDM | Input feature map tiles |
| OFM Buffer | 2 × 50 KB | O-TCDM | Output / INT32 accumulator |

**Memory Latency Hiding**: Double-buffering (ping-pong) allows DMA to prefetch the next tile into Bank 1 while the Systolic Array processes Bank 0, targeting near-full hardware utilization.

> **Top-level sizing note**: With 5 clusters at 576 KB per cluster, cluster-local SRAM alone is 2880 KB, above the 2.5 MB target. Top-level integration must choose one of these directions: reduce TCDM per cluster, reduce cluster count, or update the target SRAM budget.

---

## 11. Development Status

| Area | Content | Status |
|------|---------|--------|
| DMA and AXI | DMA Engine + AXI interface | Done |
| Shared memory fabric | Data TCDM Interconnect (N-bank crossbar) | Done |
| Host bridge | AXI-to-OBI bridge, DMA-to-TCM test | Done |
| Snitch cluster control | Snitch Core integration: I-TCM, D-TCM isolation, boot via AXI | Done |
| Matrix compute | Systolic Array + Matrix Engine cluster integration | Done |
| Vector compute | Spatz Vector Engine integration (1 GHz cluster) | Planned |
| Top-level scaling | 5-cluster integration + Manager Snitch | Planned |
| Model E2E | Full YOLO layer end-to-end simulation | Planned |

---

## 12. Hardware Verification Plan

### Unit Testing (Block-Level)
- **Snitch Boot TB** (`test_snitch_boot`): Load firmware through AXI4-Lite into I-TCM, release fetch, and confirm host IRQ. *(Passed)*
- **I-TCM Arbiter TB**: Check AXI vs. Snitch priority with no collisions. *(Passed)*
- **D-side OBI Demux TB**: Check correct address decoding for D-TCM / Data TCDM / MMIO / error sink. *(Passed)*
- **Matrix Engine TB**: Verify the 32×32 Systolic Array against the Python golden model. *(Passed)*
- **Cluster MatMul TB**: Snitch firmware triggers DMA, Systolic Array compute, and OFM writeback; M=64 raw-register regression. *(Passed)*
- **AFU Cluster TB**: Snitch firmware programs AFU LUT/control, verifies e8/e16/e32 output and AFU internal interrupt. *(Passed)*
- **Spatz Vector TB**: Test RVV instructions.

### Cluster-Level Verification
- **TCDM Arbitration**: Snitch, Spatz, and Systolic Array access Data TCDM concurrently with no deadlock.
- **Firmware Dispatch**: Snitch firmware triggers DMA, DMA loads data, and the Systolic Array computes.

### Top-Level Integration
- **5-Cluster TB**: All 5 clusters run concurrently, with Manager Snitch distributing tiling.
- **End-to-End**: Simulate a full YOLO layer (Conv2D 3×3): External Memory → DMA → TCDM → Compute → Writeback.
- **Performance Profiling**: Measure real cycle counts and compute actual TOPS against the 10 TOPS target.
