# NPU Cluster TCDM Interconnect Upgrade

**Date:** 2026-06-22
**Target:** NPU Cluster Shared Data TCDM
**Inspiration:** PULP MAGIA Project (`local_interconnect.sv`)

---

## 1. Current Problem: Flat Architecture

The previous TCDM interconnect (`tcdm_interconnect.sv`) used one monolithic crossbar (**10 masters x 16 banks**) with a daisy-chained fixed-priority arbitration policy.

Critical drawbacks:

1. **Priority inversion:** Core (M0) and DMA (M2) had higher priority than compute engines such as the Systolic Array, making starvation likely and reducing throughput.
2. **Timing and area:** A 10-to-1 daisy-chain arbiter per bank creates a long $O(N)$ critical path.
3. **High contention:** Every access competes on the same arbitration ring.

## 2. Solution: Grouped Tree Topology

Inspired by the **MAGIA** architecture, the TCDM interconnect is restructured into groups by traffic type: HWPE, DMA, and Core.

The bank count remains unchanged: **16 banks, 512 KB physical capacity**.

### Priority Groups

| Group | Priority | Masters | Rationale |
|-------|----------|---------|-----------|
| **HWPE** (Compute) | Highest | Systolic Array (1 R, 4 W) <br> Spatz Vector (2 R/W) <br> *[Future: MAGIA AFU]* | Hardware processing elements run strict pipelines. A stall can waste tens of MACs per cycle. |
| **DMA** (Data Move) | Medium | iDMA AXI2OBI write port <br> iDMA OBI2AXI read port | Background data movement. Bursts are long but buffered and can tolerate a few stall cycles. |
| **CORE** (Scalar) | Lowest | Snitch D-Bus (1 R/W) | Sparse configuration and scalar data accesses. A few stall cycles do not materially affect total throughput. |

### Current Master Mapping

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

## 3. Arbiter Tree Architecture

Instead of one giant crossbar, data routes through hierarchical routing and arbitration layers:

```text
  +-----------------+       +-----------------+       +-----------------+
  | HWPE Masters    |       | DMA Master(s)   |       | CORE Master(s)  |
  | (7 Ports)       |       | (2 Ports)       |       | (1 Port)        |
  +--------+--------+       +--------+--------+       +--------+--------+
           |                         |                         |
           v                         v                         v
  +-----------------+       +-----------------+       +-----------------+
  | HWPE Router     |       | DMA Router      |       | CORE Router     |
  | (1-to-16 Demux) |       | (1-to-16 Demux) |       | (1-to-16 Demux) |
  +--------+--------+       +--------+--------+       +--------+--------+
           |                         |                         |
           v                         v                         v
  +-----------------+       +-----------------+       +-----------------+
  | HWPE Arbiter    |       | DMA Arbiter     |       | CORE Arbiter    |
  | (Round-Robin)   |       | (Round-Robin)   |       | (Round-Robin)   |
  +--------+--------+       +--------+--------+       +--------+--------+
           | (High)                  | (Med)                   | (Low)
           |                         |                         |
           v                         v                         v
  +---------------------------------------------------------------------+
  |                        Final Bank Arbiter                           |
  |                  (Strict Priority: HWPE > DMA > CORE)               |
  |                         [Per Bank 0 -> 15]                          |
  +----------------------------------+----------------------------------+
                                     |
                                     v
                      +------------------------------+
                      |    16 x TCDM SRAM Banks      |
                      |    (32KB each = 512KB)       |
                      +------------------------------+
```

### Layer Details

1. **Router Layer:** Each master has a demux that calculates `target_bank = (addr >> 5) % 16` and sends the request to the correct bank.
2. **Group Arbiter Layer:**
   - At each bank, if multiple HWPE ports access the same bank, `rr_arb_tree` (round-robin) prevents permanent starvation.
   - DMA and CORE use the same pattern when a group has more than one master.
3. **Final Bank Arbiter Layer:**
   - A three-input arbiter (HWPE, DMA, CORE) uses **strict priority**.
   - If HWPE has a request, HWPE always wins.
   - DMA can read/write only when there is no HWPE access to that bank.
   - CORE can access only when both HWPE and DMA are idle for that bank.

---

## 4. Benefits

1. **Zero HWPE starvation:** Systolic Array and Spatz have top priority, preserving compute-pipeline throughput when DMA contends.
2. **Better timing:** The critical path is split into smaller arbiters instead of one large daisy-chain arbiter.
3. **Easier scaling:** A future AFU can plug into the HWPE router and share bandwidth through the internal round-robin arbiter.
4. **Preserved flexibility:** The 16 banks remain one contiguous unified address space. Software can still choose where to place Weight, IFM, and OFM tensors.

---

## 5. Implementation Steps

1. Use small grouped round-robin arbiters inside `tcdm_interconnect.sv`; `common_cells` `rr_arb_tree` remains a reference, but is not mass-instantiated to avoid longer Verilator builds.
2. Preserve the public interface of `hw/rtl/interconnect/tcdm_interconnect.sv` so `npu_cluster.sv` wiring is not broken.
3. Replace internals with grouped arbitration: HWPE round-robin, DMA round-robin, CORE round-robin, final strict priority `HWPE > DMA > CORE`.
4. Update `npu_cluster.sv` to pass master masks: HWPE=`M1,M3,M4,M5,M6,M7,M8`, DMA=`M2,M9`, CORE=`M0`.
5. Add an independent contention test for the arbitration policy before running cluster regressions.

## 6. Risks and Accepted Limits

1. **Strict priority can starve DMA/CORE** if HWPE continuously requests the same bank. This is an intentional compute-first baseline tradeoff. If bounded fairness is needed, add aging or credits in follow-up work.
2. **Response routing must preserve the 1-cycle SRAM latency contract** from the old interconnect. If pipeline stages are added, response routing must use FIFOs to preserve the target master.
3. **I-TCDM/O-TCDM remain logical windows** on the same 16-bank physical TCDM. This upgrade does not change the memory map or bank capacity.
4. **Firmware-visible address contracts do not change** in this work; all existing tests must keep passing.
