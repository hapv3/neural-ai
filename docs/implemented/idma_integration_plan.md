# iDMA Integration Architecture Plan

**Date:** 2026-06-20
**Target:** Replace the custom `dma_engine` with PULP `iDMA` in `npu_cluster.sv`

---

## 1. Goals

Replace the custom `dma_engine.sv` block, a hand-written FSM that transfers one beat at a time and is very slow, with the PULP Platform `iDMA` submodule. `iDMA` supports burst transfers up to 16 beats, pipelining, queueing, and 2D strided transfers, greatly improving movement of weights, IFM, and OFM between L2 and L1/TCDM.

> [!IMPORTANT]
> **Core reason: on-the-fly Im2Col for 3x3 Conv2D**
> The 32x32 Systolic Array is inefficient for direct 2D convolution unless the data is flattened into Im2Col form. If Snitch performs Im2Col in software, it is extremely slow and wastes TCDM memory.
>
> `iDMA` solves this with **2D strided transfer** support (`idma_nd_midend`). It can perform strided reads of 3x3 patches from L2 memory and sequential writes into L1 TCDM. In effect, iDMA performs on-the-fly Im2Col in hardware. Combined with ping-pong double buffering, iDMA can hide data-movement latency and let the Systolic Array approach full utilization for YOLO/ResNet-style workloads.

## 2. Frontend and Backend Choice

iDMA is flexible. Based on source code from `hw/magia/hw/tile/idma_ctrl_mm.sv`, this design uses the **memory-mapped (MM)** configuration.

The current NPU cluster MMIO aperture (`0x2000_0000`) is shared with the Systolic Array control block (`systolic_ctrl_regs`). A local **MMIO demux** splits this space:

- `0x2000_0000 -> 0x2000_0FFF`: `systolic_ctrl_regs` for Systolic Array, linebuffer, and requant control. Unused low offsets return zero.
- `0x2000_1000 -> 0x2000_1FFF`: Directly mapped to the `iDMA` config frontend (`idma_ctrl_mm`).

Backend split:

- **Data Backend 1 (AXI2OBI):** Fetches data from L2 through AXI read and writes it into L1 TCDM through OBI.
- **Data Backend 2 (OBI2AXI):** Reads results from L1 TCDM through OBI and writes them back to L2 through AXI write.

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
|        systolic_ctrl_regs          |   |                 idma_ctrl_mm (iDMA Wrapper)                 |
| (Systolic/linebuffer/requant CSRs) |   |                                                             |
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

## 4. Required RTL Changes (`npu_cluster.sv`)

### 4.1. Remove `dma_engine.sv`

- Remove the `u_dma_engine` instantiation.
- Remove the old hand-written MMIO logic such as `cfg_dma_src_addr`, `cfg_dma_length`, and related signals.

### 4.2. Instantiate `idma_ctrl_mm`

Use `idma_ctrl_mm`, either imported from MAGIA or wrapped similarly for this NPU, with parameters matching the NPU architecture: 256-bit AXI, 256-bit OBI, and 32-bit addresses.

```systemverilog
idma_ctrl_mm #(
    // Parameters map to 256-bit data width and 32-bit addr
) u_idma (
    .clk_i            (clk_i),
    .rst_ni           (rst_ni),

    // MMIO Config (from OBI Demux)
    .obi_req_i        (mmio_obi_req),
    .obi_rsp_o        (mmio_obi_rsp),

    // L2 AXI Interfaces
    .axi_read_req_o   (idma_axi_read_req),
    .axi_read_rsp_i   (idma_axi_read_rsp),
    .axi_write_req_o  (idma_axi_write_req),
    .axi_write_rsp_i  (idma_axi_write_rsp),

    // L1 OBI Interfaces (into TCDM Interconnect)
    .obi_read_req_o   (idma_obi_read_req),
    .obi_read_rsp_i   (idma_obi_read_rsp),
    .obi_write_req_o  (idma_obi_write_req),
    .obi_write_rsp_i  (idma_obi_write_rsp)
);
```

### 4.3. Update the L2 AXI Interconnect

- iDMA exports two separate AXI channels: one read channel for L2 -> L1 and one write channel for L1 -> L2.
- If the current NPU cluster exposes only one AXI master port externally, an **AXI multiplexer** such as `axi_demux` or `axi_mux` may be needed to combine the read/write paths before the cluster `axi_aw_*` and `axi_ar_*` pins.
- AXI already separates AR (address read) and AW (address write), so the simplest wiring connects `idma_axi_read` directly to AR/R signals and `idma_axi_write` directly to AW/W/B signals.

### 4.4. Firmware Interaction

Because iDMA uses a more complex register file than the old `dma_engine` (channel config, status, transfer IDs, 2D strides), firmware needs a C helper header such as `idma_mm_utils.h` from the PULP software library, or an equivalent local wrapper, so Snitch firmware can program transfers safely.

---

## 5. Review Notes

This replacement architecture is reasonable if the following are preserved:

- The MMIO sub-map remains explicit and does not alias Systolic control registers.
- The two iDMA backends are connected to the correct L2 and TCDM directions.
- Firmware gets a typed helper API instead of open-coded register pokes for every transfer shape.
- TCDM arbitration is upgraded or tuned so the faster DMA does not starve compute ports.
