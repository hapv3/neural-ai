# Neural-AI YOLO NPU - Project Summary

**Date**: 2026-06-19
**Current Version**: Matrix Engine Integration Baseline

## 1. Project Goals

Build a heterogeneous NPU architecture targeting **10 TOPS**. The project is deeply optimized for the **YOLO** model family, especially INT8 quantized inference.

The architecture contains 5 parallel **NPU Clusters**. Each cluster includes:

- **Control Core (Snitch)**: RISC-V RV32IMAC control core that orchestrates flow and dispatches compute units.
- **Matrix Engine**: 32x32 Systolic Array for dense MatMul / Conv2D (INT8).
- **Vector Engine**: Spatz RVV coprocessor for Depthwise Conv and activation/post-processing work.
- **Shared Data TCDM**: 256-bit shared L1 SRAM that supports concurrent access from DMA, Snitch, Systolic Array, and Vector Engine.
- **DMA Engine**: Moves data between L2 (DRAM/AXI) and L1 (TCDM) autonomously without continuous CPU intervention.

## 2. Progress

Completed work:

- **DMA and AXI:** DMA Engine + AXI interface.
- **Shared memory fabric:** Data TCDM Interconnect with a multi-bank crossbar.
- **AXI-to-OBI bridge:** DMA-to-TCM data-transfer test.
- **Snitch integration:** I-TCM and D-TCM isolation, plus successful firmware boot through AXI.
- **Matrix Engine integration:**
  - Successfully integrated `systolic_controller` and `npu_systolic_array` into `npu_cluster`.
  - Upgraded the TCDM interconnect to 8 masters to support 4 write ports and 1 read port from the Systolic Array.
  - Fixed critical architectural issues: TCDM address bug, mux/demux issue, OFM backpressure, DMA L1-to-L1 path, systolic weight-load ordering, and inconsistent constants between RTL and firmware.
  - `test_systolic.py` scoreboard now asserts real result counts and mismatch counts; standalone Systolic Array passes 32/32 outputs.
  - `test_snitch_boot.py` passes with firmware signature `0xDEADBEEF`, confirming that the boot path works and DMA L1-to-L1 does not break firmware flow.
  - `test_matmul.py` passes 10 randomized end-to-end cluster iterations: Snitch firmware -> DMA -> Systolic Array -> OFM writeback.

## 3. Remaining Work / Next Steps

- **Vector Engine integration:** Integrate Spatz Vector Engine into the system for nonlinear and element-wise operations.
- **Top-level integration:** Connect 5 clusters to a Manager Snitch for tiling and scheduling.
- **Full YOLO layer simulation:** Run an end-to-end simulation from External Memory -> DMA -> TCDM -> Compute -> Writeback, then evaluate actual TOPS against the target.

## 4. Agreed Design Decisions

1. **Snitch is the master:** Snitch directs and orchestrates all other blocks through firmware.
2. **APB is removed:** Do not use APB; route all memory-mapped I/O through OBI.
3. **Repository management:** Do not commit generated binary, hex, or object files; keep them covered by `.gitignore`.
4. Use `$(REPO_ROOT)` for include paths in configuration files.
