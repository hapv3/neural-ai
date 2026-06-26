# AFU (Activation Function Unit) Architecture & Operators Mapping

This document outlines the micro-architecture design of the AFU block for processing non-linear functions (SiLU, Sigmoid, GELU, Softmax, LayerNorm) on the NPU, compensating for the Spatz Vector Engine's lack of an FPU. This update includes the latest changes on Dataflow, OBI Compliance, and Pipeline Optimization.

## 1. Design Philosophy

Quantized (INT8) machine learning models have a highly advantageous characteristic: **The input data only has 256 possible values (-128 to 127)**.
Therefore, instead of building complex Taylor/Chebyshev compute blocks, we will design the AFU as a **Streaming LUT Processor**.

- **No Interpolation:** 1-to-1 mapping. Any function $f(x)$ (SiLU, Sigmoid, Tanh) can be pre-computed into a 256-byte array by the compiler/firmware and loaded into the AFU's SRAM.
- **Speed:** The current phase uses 4 internal LUT lanes. The AFU data path connects to the Shared Data TCDM 256-bit; the core consumes byte groups from the read FIFO and issues write beats with byte-enables based on the tail.
- **Hardware distribution:**
  - AFU handles **Vector Element-wise Non-linear** (lookup tables).
  - Spatz Vector Engine handles **Vector Reductions & Arithmetic** (max, sum, mul, add).
  - Snitch Scalar Core handles **Scalar Non-linear** (computing 1 value like 1/sqrt or 1/x).
- **Memory Alignment Rule:** The firmware/HAL in the current phase requires `src_ptr` and `dst_ptr` to be 32-byte aligned when integrated into the cluster. Arrays with element counts not divisible by the beat size are still supported via tail byte-enables; arbitrary unaligned e16/e32 destination is not yet a scheduler contract.

---

## 2. AFU Micro-Architecture

The AFU will be attached to the Data TCDM Interconnect as an independent **DMA Master**.

```text
                      ┌────────────────────────────────────────┐
                      │              AFU Hardware              │
                      │                                        │
                      │  ┌──────────────────────────────────┐  │
                      │  │  Config & Control Registers      │  │
  TCDM (L1)           │  │  (Start, Len, Src, Dst, Mode)    │  │
  Interconnect        │  └──────────────────────────────────┘  │
  (Master Port)       │                                        │
        ▲             │  ┌──────────────────────────────────┐  │
        │ 256-bit     │  │  Dual-Mode LUT SRAM (1 KB)       │  │<-- Firmware loads
        │ R/W         │  │  (Supports 256x8, 256x16, 256x32)│  │    lookup table here
        ▼             │  └──────────────────────────────────┘  │    when switching layers
                      │                                        │
                      │  ┌──────────┐  Address   ┌──────────┐  │
                      │  │ Read     │----------->│ 2-Stage  │  │
                      │  │ Backend  │            │ Pipeline │  │
                      │  └──────────┘            └──────────┘  │
                      │  (OBI Compliant)              │        │
                      │  ┌──────────┐                 │        │
                      │  │ Write    │<----------------┘        │
                      │  │ Backend  │ Data + Byte Enables      │
                      │  └──────────┘                          │
                      └────────────────────────────────────────┘
```

### 2.1. Detailed RTL Components

The AFU system is divided into 5 main modules to ensure maintainability and clear Separation of Concerns:

1. **`afu.sv` (Top-level Wrapper):**
   - Acts as the bridge connecting all sub-modules.
   - Declares OBI interfaces (Slave for config, Master for memory access).
   - Instantiations: Calls `afu_frontend`, `afu_backend`, `afu_core`, and two `afu_fifo_ff` instances (rfifo and wfifo).

2. **`afu_frontend.sv` (Control & CSRs):**
   - Acts as an OBI Slave receiving configuration commands from the Snitch Core.
   - Manages CSR registers: `src_ptr`, `dst_ptr`, `length`, `mode`.
   - Generates the `start` pulse to wake up other blocks when firmware writes to the `START` register.
   - Exposes the SRAM LUT address range (from 0x000 to 0x3FC) so firmware can load the lookup tables.

3. **`afu_backend.sv` (Memory Access Engine):**
   - Acts as an OBI Master, automating the reading of unprocessed data (from `src_ptr`) and writing processed data (to `dst_ptr`).
   - Strictly adheres to the OBI Handshake (`req`, `gnt`, `rvalid`). Request signals are latched in a pending transaction register until `gnt` is received, while the read side is limited to one outstanding read to prevent overwriting returning data.
   - Address loop management: Every read/write checks if the `end_addr` has been reached to stop issuing reqs.

4. **`afu_core.sv` (2-Stage Compute Pipeline):**
   - **The heart of the AFU**, where LUT Lookups are performed with minimal latency.
   - **Stage 1 (S1):** Pops data from the Read FIFO, calculates the number of valid elements based on `elem_cnt`, splits the 32-bit data into 4 8-bit blocks, and feeds them into the 4 address ports for LUT SRAM query (`tc_sram`).
   - **Stage 2 (S2):** Receives the result data from SRAM (1 cycle latency), synthesizes it into a complete block. Calculates the Byte Enables mask (`s2_out_be_comb`) to ensure no out-of-bounds memory writes when `length` is odd. Pushes the result to the Write FIFO.
   - **Stall Protection:** If S2 stalls because the Write FIFO is full (`s2_stall`), the circuit will use the `s2_lut_rdata_saved_q` buffer register to freeze the result from SRAM, preventing data corruption if S1 accidentally updates to a new address.

5. **`afu_fifo_ff.sv` (Flip-Flop Based FIFOs):**
   - Intermediate data queues between the Backend and Core, helping to absorb OBI protocol latency.
   - Since the depth only requires `DEPTH=2`, designing it with Flip-Flops instead of calling an SRAM Macro saves significant Area, reduces routing delay, and avoids using complex memory initializers.

### 2.2. Basic AFU Workflow (Dual-Mode)
1. Snitch Firmware pre-calculates the function $f(x)$ and writes the result table into the AFU's LUT SRAM.
2. Firmware writes configuration: `Src_Ptr`, `Dst_Ptr`, `Length`, and **`Mode`** (8-bit, 16-bit, or 32-bit output).
3. Activate the AFU. Depending on the Mode:
   - **8-bit Mode (SiLU, Sigmoid):** The AFU processes 4 bytes/cycle (or based on bus width if configured wider), looks up 4 8-bit results, and pushes them to the Write FIFO. Extremely fast.
   - **16/32-bit Mode (exp, x^2 for Softmax/Norm):** The AFU reads byte by byte, looks up 16-bit or 32-bit results to maintain maximum precision, then writes these 16/32-bit words to TCDM for Spatz to handle subsequent accumulations.

---

## 3. Current Performance Evaluation

This evaluation applies to the current RTL: `MEM_DATA_WIDTH=256`, `LUT_LANES=4`, RFIFO/WFIFO depth 2, the AFU is a dedicated TCDM master, and the LUT SRAM has a 1-cycle latency. This is an **AFU active path** evaluation, not counting the time for host AXI firmware loading, firmware self-checks, or cocotb readbacks.

### 3.1. Theoretical Throughput

The Core has 4 LUT lanes, so the upper compute bound is **4 input elements/cycle** for all 3 modes. The difference between modes lies in the output bandwidth:

| Mode | Input element | Output element | Core throughput | Output bandwidth @ 1 GHz | Total required TCDM bandwidth |
|------|---------------|----------------|-----------------|--------------------------|--------------------------|
| `e8` | 8-bit | 8-bit | 4 elems/cycle | 4 GB/s | 8 GB/s = 4 GB/s read + 4 GB/s write |
| `e16` | 8-bit | 16-bit | 4 elems/cycle | 8 GB/s | 12 GB/s = 4 GB/s read + 8 GB/s write |
| `e32` | 8-bit | 32-bit | 4 elems/cycle | 16 GB/s | 20 GB/s = 4 GB/s read + 16 GB/s write |

The AFU's Shared Data TCDM port is 256-bit, so a peak of one beat/cycle equals **32 GB/s @ 1 GHz** if there are no arbitration stalls. Thus, with the current 4-lane configuration, the AFU is mainly **lane-limited**, not yet TCDM-bandwidth-limited under non-contention conditions.

### 3.2. Cycle model used for estimation

With a tensor of length `N` 8-bit input elements and output width `B ∈ {1,2,4}` bytes:

```text
compute_cycles       = ceil(N / 4)
read_beats_256b      = ceil(N / 32)
write_beats_256b     = ceil(N * B / 32)
active_cycles_lower  ≈ max(compute_cycles, read_beats_256b + write_beats_256b)
active_cycles_upper  ≈ compute_cycles + read_beats_256b + write_beats_256b + small_pipeline_drain
```

In a long workload, backend read/write and core lookup partially overlap, so the actual cycle count should be closer to the lower bound if TCDM grants are steady. In small workloads, overheads for start, read-first latency, write drain, IRQ/polling, and LUT programming will take up a larger proportion.

### 3.3. Workload Examples

| Tensor | Mode | Elements | Ideal compute cycles | Traffic | Remarks |
|--------|------|----------|----------------------|---------|----------|
| Micro-YOLO activation `32×32×32` | `e8` | 32,768 | 8,192 | 32 KB read + 32 KB write | AFU active for a few microseconds at 1 GHz; firmware/LUT setup can be significant if only running one small tensor. |
| ViT/Softmax exp row batch 32k elems | `e16` | 32,768 | 8,192 | 32 KB read + 64 KB write | Still lane-limited; output traffic increases but remains under peak TCDM port bandwidth. |
| Precision staging 32k elems | `e32` | 32,768 | 8,192 | 32 KB read + 128 KB write | Closer to bandwidth limits but still only needs ~20/32 GB/s at 1 GHz if no contention. |
| YOLO feature map `80×80×64` | `e8` | 409,600 | 102,400 | 400 KB read + 400 KB write | Highly suitable for AFU streaming; LUT programming amortizes well over large tensors. |

### 3.4. Current Practical Bottlenecks

- **LUT programming cost:** Each LUT table requires 256 MMIO writes. For multiple layers using the same activation/qparam, the firmware should cache/reuse the LUT and only reload when the table changes.
- **TCDM arbitration:** The AFU currently shares the Shared Data TCDM with DMA, Systolic, and Spatz. During true overlapping execution, the AFU might stall due to higher-priority HWPE traffic; there is no dedicated PMU counter to measure these stalls yet.
- **Backend policy:** The backend currently prioritizes writes over reads and limits reads to one outstanding request to keep OBI responses simple. This configuration is correct for correctness, but not yet optimized for absolute bandwidth.
- **FIFO depth:** RFIFO/WFIFO depth 2 is sufficient for the current regression; if TCDM grant jitter is high, increasing FIFO depth or adding an outstanding read/write queue could improve utilization.
- **Firmware wait:** The test currently polls `afu_wait_done`; an event-driven `wfi`/trap approach would reduce energy and scalar busy cycles, though it won't change AFU active throughput.

### 3.5. Current Regression Metrics

`test_afu_basic` passes at the cluster level and finishes at around `24,532 ns` simulation time with a 1 ns clock. This number **is not pure AFU latency**, as it includes AXI boot load, firmware seed data generation, 3 LUT programming phases, AFU run, firmware self-check, and host notification. It only proves that the current integration path is functionally correct and does not timeout. To get real performance metrics, we need additional counters:

- `afu_active_cycles`: From `start` to `done_o`.
- `afu_core_stall_cycles`: `s2_stall` or core waiting for RFIFO.
- `afu_tcdm_wait_cycles`: `obi_m_req_o && !obi_m_gnt_i`.
- `afu_read_beats` / `afu_write_beats`: Actual TCDM beats.

These counters should be added to the PMU or debug CSRs before using AFU metrics to optimize the scheduler/performance.

---

## 4. Specific Function Mapping

### 4.1. Pure Element-wise Functions (SiLU, Sigmoid, Tanh, GELU)
**Models:** YOLO (SiLU/Sigmoid), ViT (GELU), CNN (Tanh).
- **Execution:** 100% offloaded to AFU.
- **Process:**
  1. The offline Compiler generates a static array `const uint8_t silu_lut[256] = {...}`.
  2. Snitch copies this array into `AFU_LUT_RAM`.
  3. The AFU streams through the entire tensor. Maximum speed (100% utilization).

### 4.2. Softmax (ViT Attention)
Formula: $y_i = \frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$
- **Execution:** Combined Spatz + AFU.
- **Process:**
  1. **Spatz:** Runs `vmax` to find the maximum value of the row (vector) -> $M$.
  2. **Spatz:** Runs `vsub` to subtract $M$: $x' = x - M$.
  3. **Snitch:** Configures AFU with the `exp()` LUT.
  4. **AFU:** Looks up the table to convert array $x'$ into array $E = e^{x'}$. (Note: The Output array is required to have 16/32-bit precision to prevent quantization loss - uses Mode 16/32).
  5. **Spatz:** Runs `vsum` on array $E$ to get the sum $S$.
  6. **Snitch:** Uses C code to compute the scalar value `inv_S = 1 / S`.
  7. **Spatz:** Runs `vmul` multiplying the entire array $E$ by `inv_S` to yield the final result.

### 4.3. LayerNorm (ViT Layer)
Formula: $y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta$
- **Execution:** Combined Spatz + Snitch (**no AFU needed** in practice).
- **Process:**
  1. **Spatz:** Runs `vsum` to find the sum -> Computes Mean $\mu$.
  2. **Spatz:** Runs `vsub` -> $(x - \mu)$, `vmul` -> $(x - \mu)^2$, `vsum` -> Variance $\sigma^2$.
  3. **Snitch:** Computes the function `1/sqrt(sigma^2 + eps)` using standard C libraries on the scalar core. Because this is just **1 scalar value** for an entire vector row, it takes Snitch ~50 cycles and doesn't impact total time. Saves the result in `inv_std`.
  4. **Spatz:** Uses `vmul` to multiply array $(x-\mu)$ with `inv_std` and $\gamma$, uses `vadd` to add $\beta$.

---

## 5. Integrating AFU into the System (Integration Guide)

To assemble the `afu.sv` IP block into the actual NPU system, the following configuration steps must be performed at the NPU Top-level (e.g., `npu_cluster.sv`):

### 5.1. Interconnect Integration
- **Attaching AFU to the Peripheral/System Interconnect (OBI Slave Port):**
  - The AFU is a 32-bit MMIO slave behind the Snitch D-side demux.
  - The current address range is `0x2000_3000 – 0x2000_3fff`. LUT is at offset `0x000..0x3ff`; CSRs start at offset `0x400`.
- **Attaching AFU to the TCDM Interconnect (OBI Master Port):**
  - `NUM_MASTERS` of the Shared Data TCDM is currently 11.
  - The AFU uses a dedicated master port on the 256-bit Shared Data TCDM to automatically read/write L1 tensors.

### 5.2. Control Signals Integration
- **Clock and Reset:** Synchronized `clk_i` and `rst_ni` with the entire Cluster.
- **Interrupt Signal (`done_o`):**
  - `done_o` is connected to the `npu_interrupt_ctrl` bit `NPU_IRQ_SRC_AFU`.
  - Firmware can enable/clear `INT_PENDING` to acknowledge AFU done; full trap/WFI handlers are planned for a later phase.

### 5.3. Firmware Update (Software C)
Add hardware structure definitions to the Firmware SDK to control via C code:

```c
// AFU base address in cluster MMIO aperture
#define AFU_BASE_ADDR 0x20003000

// AFU register structure
typedef struct {
    volatile uint32_t LUT_SRAM[256];  // 0x0000 - 0x03FC
    volatile uint32_t STATUS_START;   // 0x0400: read done/busy/error, write start pulse
    volatile uint32_t SRC_PTR;        // 0x0404
    volatile uint32_t DST_PTR;        // 0x0408
    volatile uint32_t LENGTH;         // 0x040C
    volatile uint32_t MODE;           // 0x0410
} afu_regs_t;

#define AFU_REGS ((afu_regs_t*) AFU_BASE_ADDR)
```
