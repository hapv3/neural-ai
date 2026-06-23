# Spatz Vector Engine Integration into NPU Cluster

## Background

The NPU Cluster currently has a functional Snitch core, DMA engine, and Systolic Array. Master 1 on the TCDM interconnect is reserved for the Spatz Vector Engine but is currently a placeholder (tied to `1'b0`).

Spatz is a RISC-V Vector (RVV) coprocessor designed to pair with the Snitch scalar core. In the reference `spatz_cc.sv`, Snitch offloads RVV instructions to Spatz via the **accelerator interface** (`acc_req_o` / `acc_rsp_i`). Spatz then accesses memory through its **VLSU** (Vector Load/Store Unit), which generates TCDM-style memory requests.

### Key Design Insight

Our NPU uses OBI for the shared TCDM, while Spatz natively speaks the **TCDM protocol** (a simpler valid/ready + data interface). We need a **TCDM-to-OBI bridge** to connect Spatz's VLSU memory ports to our TCDM interconnect.

---

## User Review Required

> [!IMPORTANT]
> **Spatz Configuration**: The default `spatz_pkg.sv` has `N_FPU = 4` and `N_IPU = 1`, which means `N_FU = 4`. This creates **4 VLSU memory ports** (`NrMemPorts = N_FU = 4`). Adding 4 new masters to the TCDM (on top of 8 existing) would make `NUM_MASTERS = 12`. This is getting large.
>
> **Option A' (Implemented)**: Override `spatz_pkg` to use `N_FPU = 0` and `N_IPU = 2` (integer-only Spatz). This gives `N_FU = 2` → **2 VLSU memory ports** → `NUM_MASTERS = 9` by using Master 1 and Master 8 for Spatz. We don't need FP for INT8 inference, and the current Spatz VLSU RTL does not elaborate cleanly with a single FU/port because some `$clog2(N_FU/NrMemPorts)` paths become zero-width.
>
> **Option B**: Keep `N_FPU = 4`, accept 12 masters. Gives full FP vector capability but heavier interconnect.

> [!IMPORTANT]
> **FPU Sequencer**: Spatz has an optional FPU sequencer that muxes Snitch's scalar D-Bus with the FP LSU. Since we disable FPU (`N_FPU = 0`), this path is bypassed automatically (`gen_no_fpu_sequencer`). No extra complexity.

> [!WARNING]
> **`snitch_core.sv` Wrapper**: The current wrapper ties off the accelerator interface (`acc_rsp = '0`). We need to expose `acc_req_o` / `acc_rsp_i` / `acc_qvalid` / `acc_qready` as module ports so `npu_cluster.sv` can wire them to Spatz. This requires modifying the wrapper.

## Open Questions

1. **Integer-only Spatz (Option A') vs Full Spatz (Option B)?** — Implemented Option A' for now (INT8 inference focus, 2 INT lanes, no FP). FP can be added later by changing the pkg parameter and wiring additional memory ports.
2. **Snitch D-Bus routing for Spatz TCDM access**: In the reference design, Spatz's scalar FP LSU requests go through a `reqrsp_mux` that merges them with Snitch's D-Bus. With `N_FPU = 0`, this path is dead, so we skip it entirely.

---

## Proposed Changes

### snitch_core.sv — Expose Accelerator Interface

#### [MODIFY] [snitch_core.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/snitch_core.sv)

1. Add output/input ports for the accelerator offload interface:
   - `acc_qvalid_o`, `acc_qready_i` — handshake
   - `acc_qreq_o` (type `acc_req_t`) — instruction + operands from Snitch → Spatz
   - `acc_prsp_i` (type `acc_rsp_t`) — writeback result from Spatz → Snitch
   - `acc_pvalid_i`, `acc_pready_o` — response handshake
   - `acc_mem_finished_i`, `acc_mem_str_finished_i` — memory ordering signals
   - `fpu_rnd_mode_o`, `fpu_fmt_mode_o`, `fpu_status_i` — FPU side-channel

2. Remove the `acc_rsp = '0` tie-off.

3. Wire Snitch's `acc_req_o` internal signals to the new ports.

---

### npu_cluster.sv — Instantiate Spatz + Bridge

#### [MODIFY] [npu_cluster.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/npu_cluster.sv)

1. **Increase `NUM_MASTERS` from 8 to 9** (Master 1 and Master 8 are Spatz VLSU ports).

2. **Instantiate Spatz** (`spatz`) with:
   - `NrMemPorts = 2` (integer-only, 2 VLSU ports for the 2-lane Spatz config)
   - `RegisterRsp = 0`
   - Wire `issue_valid_i` / `issue_ready_o` / `issue_req_i` / `issue_rsp_o` to the acc interface from snitch_core.
   - Wire `rsp_valid_o` / `rsp_ready_i` / `rsp_o` for writeback.
   - FPU memory interface tied off (no FP LSU).

3. **Add TCDM-to-OBI bridge** (`tcdm_to_obi_bridge`):
   - A new small module that converts Spatz's TCDM memory channel (`tcdm_req_chan_t` + valid/ready) to OBI protocol (req/gnt/rvalid/rdata).
   - Maps: `tcdm_req.addr → obi_addr`, `tcdm_req.write → obi_we`, `tcdm_req.strb → obi_be`, `tcdm_req.data → obi_wdata`.
   - Response: `obi_rdata → tcdm_rsp.data`, `obi_rvalid → tcdm_rsp.p_valid`, `obi_gnt → tcdm_rsp.q_ready`.

4. **Wire bridge outputs to Master 1 and Master 8** of the TCDM interconnect.

---

### tcdm_to_obi_bridge.sv — New Bridge Module

#### [NEW] [tcdm_to_obi_bridge.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/tcdm_to_obi_bridge.sv)

Simple combinational bridge (no buffering needed — TCDM is already single-cycle):

```
TCDM Side:                        OBI Side:
  tcdm_req_valid  ─────────────►  obi_req_o
  tcdm_req.addr   ─────────────►  obi_addr_o
  tcdm_req.write  ─────────────►  obi_we_o
  tcdm_req.strb   ─────────────►  obi_be_o
  tcdm_req.data   ─────────────►  obi_wdata_o
  tcdm_rsp_ready  ◄─────────────  obi_gnt_i
  tcdm_rsp.data   ◄─────────────  obi_rdata_i
  tcdm_rsp_valid  ◄─────────────  obi_rvalid_i
```

> [!NOTE]
> The OBI protocol has a 2-phase handshake (req/gnt for address phase, rvalid for response phase). TCDM is simpler (1-phase: valid/ready). The bridge must track outstanding requests: after `obi_req && obi_gnt`, wait for `obi_rvalid` before allowing the next request. This is a simple 2-state FSM (IDLE → WAIT_RESP).

---

### snitch_minimal.F — Add New Files

#### [MODIFY] [snitch_minimal.F](file:///home/dev01/neural-ai/hw/rtl/cluster/snitch_minimal.F)

Add:
```
// Spatz Vector Engine
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/generated/spatz_pkg.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/rvv_pkg.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/vregfile.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vrf.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_decoder.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_controller.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_simd_lane.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_ipu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vfu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_serdiv.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vlsu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vsldu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/reorder_buffer.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_fpu_sequencer.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz.sv
// Bridge
$(REPO_ROOT)/hw/rtl/cluster/tcdm_to_obi_bridge.sv
```

---

### spatz_pkg.sv — Override for INT8 NPU

#### [MODIFY] [spatz_pkg.sv](file:///home/dev01/neural-ai/hw/spatz/hw/ip/spatz/src/generated/spatz_pkg.sv) (or create a local override)

> [!WARNING]
> Modifying the Spatz submodule directly is risky for upstream compatibility. Alternative: Create a local `npu_spatz_pkg.sv` that overrides `N_FPU = 0`. However, `spatz_pkg` is a package, not parameterizable at instantiation time. We would need to either:
> - **Option 1**: Modify the generated `spatz_pkg.sv` in-place (quick, dirty)
> - **Option 2**: Regenerate from `spatz_pkg.sv.tpl` with `N_FPU = 0`
> - **Option 3**: Keep `N_FPU = 4` and accept 4 VLSU ports (Option B)
>
> **Recommendation**: Option 1 for now. Change `N_FPU = 4` → `N_FPU = 0` in the generated file.

Changes:
- `N_FPU = 0` (no floating-point units)
- `RVF = 0`, `RVD = 0` (disable FP extensions)
- Set `N_IPU = 2`, keep `VLEN = 512`, `NRVREG = 32`

---

## Verification Plan

### Automated Tests
1. **Compile test**: `make -C hw/rtl/cluster clean && make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot` — Verify Verilator compile passes with Spatz included.
2. **Existing tests**: Re-run `test_snitch_boot` and `test_matmul` to verify no regressions.

### Manual Verification
1. Write a simple firmware that executes a RISC-V Vector instruction (e.g., `vsetvli`, `vle8.v`, `vadd.vv`) and verify Spatz processes it correctly.
2. Check that Spatz can load/store vectors via TCDM through the bridge.

### Future Work
- Write a full activation function (e.g., SiLU) using RVV instructions as a firmware test.
- Integrate Spatz into the matmul pipeline for post-processing (quantization, activation).


## Current Status

- Task 1: `spatz_pkg.sv` is configured for integer-only Spatz (`N_FPU = 0`, `RVF = 0`, `RVD = 0`, `N_IPU = 2`, `VLEN = 512`).
- Task 2: `tcdm_to_obi_bridge.sv` is implemented and converts each 32-bit Spatz VLSU channel into the 256-bit OBI TCDM fabric lane.
- Task 3: `snitch_core.sv` exposes the accelerator request/response interface and removes the old local tie-off.
- Task 4: `npu_cluster.sv` instantiates Spatz, wires issue/writeback to Snitch, and connects two VLSU bridges to TCDM Master 1 and Master 8.
- Filelist: `snitch_minimal.F` includes Spatz RTL, local merged `riscv_instr_npu.sv`, Spatz reqrsp package, and required CommonCells/TechCells dependencies.
- Verification: `make -C hw/rtl/cluster clean && env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot` passes.
- Verification: `env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul` passes 10/10 randomized matmul iterations.
- Dedicated RVV firmware suite: `sw/test/spatz_vector` now builds standalone Spatz-vector firmware with the Spatz-local LLVM toolchain (`hw/spatz/install/llvm/bin/clang`) and RVV mnemonics under `-march=rv32im_zve32x -mabi=ilp32`.
- Dedicated RVV test: `sw/test/spatz_vector/tests/basic_mem_arith.S` covers `vsetvli`, `vle32.v`, `vse32.v`, `vadd.vv`, `vsub.vv`, `vand.vv`, `vor.vv`, `vxor.vv`, `vsll.vi`, and `vsrl.vi`.
- Verification: `test_spatz_vector_basic` checks both firmware signature/debug words and the full output vectors by reading TCDM SRAM banks directly.
- Verification: `env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_vector_basic` passes with `pass_count = 7`, output data match, and success signature `0xDEADBEEF`.

## Remaining Work

- Expand the Spatz-vector firmware suite from the current baseline test into full configured-instruction coverage: memory widths/strided/indexed access, min/max, multiply, divide/remainder, compare/mask, slide/move, and reduction groups.
- Add negative tests for disabled FP vector paths in the current integer-only Spatz config (`N_FPU = 0`, `RVF = 0`, `RVD = 0`).
- Run `test_dma_tcm` after the RVV test exists to close regression coverage across Snitch, DMA, systolic, and Spatz.
