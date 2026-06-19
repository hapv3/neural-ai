# NPU Cluster — Architecture Review Status

**Scope**: RTL, firmware, and testbench review items  
**Last updated**: 2026-06-19  
**Current baseline**: Phase 3B-A, Matrix Engine integrated into `npu_cluster`

---

## Summary

| Bucket | Count | Current decision |
|--------|-------|------------------|
| Resolved / no further update | 16 | Already reflected in RTL, firmware, tests, or docs |
| Deferred improvement | 2 | Not required for current passing Matrix Engine path |
| Code update required now | 0 | No additional RTL/firmware patch needed from this review pass |

The original review was written before the Phase 3B-A fixes landed. Most critical items are now stale: the Systolic Array is instantiated, TCDM addressing has been corrected, DMA supports L1↔L1/L1↔L2/L2↔L1 paths, and the randomized cluster MatMul test passes.

---

## Critical Issues

| ID | Original item | Current status | Decision |
|----|---------------|----------------|----------|
| C1 | Systolic Array not instantiated | **Resolved** | `systolic_controller` and `npu_systolic_array` are instantiated in `hw/rtl/cluster/npu_cluster.sv`; no update needed. |
| C2 | MMIO register spacing vs. 256-bit OBI width | **Deferred improvement** | Current 64-bit Snitch D-bus adapter aligns to 32-byte beats and byte-enables isolate each 32-bit CSR lane. Tests pass. Keep 4-byte system CSR spacing for now; revisit only if CSR writes become wider/batched. |
| C3 | DMA address never advances / misleading repeated logs | **Resolved** | DMA FSM now advances addresses and no longer relies on noisy combinational debug prints. No update needed. |
| C4 | `cfg_dma_start` level causes re-triggers | **Resolved / false positive** | Start signals are self-clearing pulses in `cluster_ctrl_regs`; repeated logs were from combinational debug output. No update needed. |

---

## Architectural Issues

| ID | Original item | Current status | Decision |
|----|---------------|----------------|----------|
| A1 | D-TCM base mismatch | **Resolved** | `DTCM_BASE_ADDR` is `0x1000_8000`; RTL and linker agree. No update needed. |
| A2 | TCDM bank address double-shift | **Resolved** | `tcdm_interconnect` de-interleaves to a byte address and `npu_cluster` converts once to 256-bit word index. Cluster MatMul passes. No update needed. |
| A3 | I-TCM address shift | **Valid as-is** | Existing byte-to-256-bit-word conversion is correct. No update needed. |
| A4 | Empty block in `obi_arbiter_2to1` | **Resolved** | Dead empty block has been removed. No update needed. |
| A5 | Empty block in `obi_demux_1to4` | **Resolved** | Dead empty block has been removed. No update needed. |

---

## Design Improvements

| ID | Original item | Current status | Decision |
|----|---------------|----------------|----------|
| I1 | Gate `weight_data_o` and `ifm_data_o` by state | **Resolved** | Controller drives only `weight_data_o` during `LOAD_WEIGHTS` and only `ifm_data_o` during `COMPUTE`. No update needed. |
| I2 | OFM packing ambiguity | **Valid as-is** | Packed array slices intentionally select 8 × 32-bit elements per 256-bit OBI write. Comments now clarify element ranges. No update needed. |
| I3 | Missing OFM write backpressure | **Partially resolved / deferred** | A one-entry OFM buffer retries writes until all four OBI ports grant. This is sufficient for the current non-concurrent Matrix Engine test. For future concurrent DMA/Vector traffic, replace it with a deeper FIFO or a stallable systolic output protocol. |
| I4 | DMA lacks L1→L1 support | **Resolved** | DMA now routes AXI read/write and OBI read/write based on L1 source/destination classification. No update needed. |
| I5 | Firmware hardcoded addresses inconsistent with package | **Resolved** | Firmware and package now agree on `WEIGHT_PING_ADDR`, `IFM_PING_ADDR`, and `OFM_PING_ADDR`. No update needed. |
| I6 | Instruction response could select stale data | **Valid as-is** | Snitch request/response handshake keeps the request address stable until response. No update needed. |

---

## Minor Issues

| ID | Original item | Current status | Decision |
|----|---------------|----------------|----------|
| M1 | `main.c` indentation | **Resolved** | Firmware control loop indentation is consistent. No update needed. |
| M2 | D-TCM size mismatch | **Resolved** | D-TCM is 32 KB in package, RTL SRAM instantiation, linker script, and architecture doc. No update needed. |
| M3 | Duplicate `import struct` in `test_matmul.py` | **Resolved / false positive** | `struct` is imported once at file top. No update needed. |

---

## Current Required Follow-Up

No code update is required from the old review items for the current Phase 3B-A baseline.

Recommended next work remains architectural progression rather than review cleanup:

1. Integrate Spatz Vector Engine as Phase 3B-B.
2. Add mixed Matrix + Vector + DMA arbitration tests.
3. Revisit I3 with a deeper OFM FIFO before enabling heavy concurrent writers to TCDM.
4. Consider 32-byte CSR spacing only if firmware starts issuing wider or batched MMIO writes.

---

## Validation Baseline

The review status above should be considered valid only if these tests pass:

- `make -C hw/rtl/systolic`
- `make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot`
- `make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul`
