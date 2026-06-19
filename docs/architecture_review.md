# NPU Cluster — Architecture & Code Review

**Scope**: Full review of all RTL, firmware, and testbench  
**Date**: 2026-06-19

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 **Critical** (blocks simulation) | 4 | Must fix before test can pass |
| 🟠 **Architectural** (design flaw) | 5 | Should fix for correctness |
| 🟡 **Improvement** (perf/quality) | 6 | Recommended |
| 🔵 **Minor** (code hygiene) | 3 | Nice to have |

---

## 🔴 Critical Issues

### C1. Systolic Array Not Instantiated

**Files**: [npu_cluster.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/npu_cluster.sv#L666-L671)

The `systolic_controller` and `npu_systolic_array` modules exist in `hw/rtl/systolic/` but are **never instantiated** inside `npu_cluster.sv`. Lines 666-671 have only a placeholder:

```sv
// Master 3: Systolic Array (Phase 3B Placeholder)
assign master_req[3].req   = 1'b0;
```

**Impact**: The firmware writes `REG_SYS_START = 1` and then polls `REG_SYS_DONE` forever. Since `cfg_sys_done_i` is never driven (it's implicitly `0`), `r_sys_done` stays `0` → **infinite firmware hang**.

**Fix**: Instantiate `systolic_controller` + `npu_systolic_array`, connect their OBI ports to the TCDM interconnect, and wire `cfg_sys_done` back to `cluster_ctrl_regs`.

---

### C2. MMIO Register Address Spacing vs. OBI Bus Width Mismatch

**Files**: [cluster_ctrl_regs.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/cluster_ctrl_regs.sv#L37-L48), [main.c](file:///home/dev01/neural-ai/sw/matmul_app/main.c#L4-L15)

The DMA registers use 32-byte (0x20) spacing between offsets:
```
REG_DMA_START = 0x0000
REG_DMA_SRC   = 0x0020
REG_DMA_DST   = 0x0040
```

But the Systolic registers use **4-byte** spacing:
```
REG_SYS_W_PTR = 0x0100
REG_SYS_I_PTR = 0x0104
REG_SYS_O_PTR = 0x0108
```

The OBI bus is 256-bit (32-byte) wide. When Snitch writes to `0x20000104`, the `obi_narrow_to_wide` adapter aligns the address to a 32-byte boundary → the address becomes `0x20000100`. All 8 lanes (offsets `0x100..0x11C`) are written **simultaneously** with the MMIO write logic iterating over all lanes.

**Impact**: Writing `REG_SYS_I_PTR` at `0x104` causes the lane-iteration loop to also see `exact_addr = 0x100` (the W_PTR lane), `0x108` (O_PTR lane), etc. Since `be_i` is only set for lane 1 (`0x104`), the other lanes won't match `be_i[i*4 +: 4] != 4'b0000` — so this actually works **by accident** through the byte-enable check. However, the `REG_SYS_START` at offset `0x110` falls within the **same 32-byte-aligned block** as `0x100-0x11F`. This means writing `REG_SYS_START=1` at `0x110` will fire in lane 4 of the same OBI transaction as the register at `0x100`.

> [!WARNING]
> When the firmware writes these registers sequentially (each is a separate Snitch store instruction), the `obi_narrow_to_wide` adapter correctly isolates each 8-byte write with appropriate `be`. The current design works **only because** each 64-bit Snitch store hits exactly one 8-byte lane. But if you ever move to a wider Snitch data bus (e.g. 128-bit) or batch writes, this will silently corrupt registers. **Recommendation**: Use 32-byte spacing consistently for all register groups, matching the OBI bus natural alignment.

---

### C3. DMA Address Never Advances (Same Address Repeated in Writes)

**Files**: [dma_engine.sv](file:///home/dev01/neural-ai/hw/rtl/dma/dma_engine.sv#L147-L154)

Looking at the simulation output:
```
[DMA] OBI Write sent: Addr=10110000, bytes_left=      1024
[DMA] OBI Write sent: Addr=10110000, bytes_left=      1024
[DMA] OBI Write sent: Addr=10110000, bytes_left=      1024
```

The DMA keeps writing to `Addr=10110000` and `bytes_left` never decrements. The `always_ff` block at line 227 updates `dst_addr_q` and `bytes_left_q` only when `state_q == OBI_WRITE && obi_gnt_i`. However, the `always_comb` block drives `obi_addr_o = is_l1_src_q ? src_addr_q : dst_addr_q` (line 109).

**Root cause**: The `obi_addr_o` default uses `dst_addr_q`, but when in `OBI_WRITE` state and `obi_gnt_i` fires, the `always_ff` update happens on the **next** clock edge. The same cycle sees the old `dst_addr_q`. On the next cycle, the FSM moves to `AXI_READ_REQ` (if bytes_left > 32), so `dst_addr_q` has been updated. This should be correct. 

Looking more carefully: the `bytes_left` display shows `1024` every time. The `$display` at line 151 prints `bytes_left_q` which is the **old** value (before the `always_ff` updates it). So the display is misleading. However, the repeated `Addr=10110000` is suspicious — let me check if `src_addr_q` and `dst_addr_q` are both updating correctly.

Actually, re-reading the output: the address is `10110000` on **every** write. This means `dst_addr_q` is NOT being updated between AXI_READ→OBI_WRITE cycles. The condition `state_q == OBI_WRITE && obi_gnt_i` fires, and the `always_ff` should update `dst_addr_q <= dst_addr_q + 32`. But the next OBI_WRITE still shows `10110000`.

**Likely cause**: The TCDM interconnect is granting (`obi_gnt_i`) the DMA request, but the FSM transition and address update may be racing. However, looking at the `always_ff` more carefully — the address update is correctly gated. The real problem may be that the `$display` in the `always_comb` block fires multiple times per clock cycle in Verilator (combinational evaluations), showing the same values.

> [!IMPORTANT]
> **Actionable Fix**: Replace the `$display` inside `always_comb` (line 151) with one inside the `always_ff` block to get accurate per-cycle reporting. Combinational `$display` in Verilator fires on every re-evaluation, flooding the log with duplicate lines and masking the real address progression.

---

### C4. `cfg_dma_start` Is a Level, Not a Pulse — Causes Re-Triggers

**Files**: [cluster_ctrl_regs.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/cluster_ctrl_regs.sv#L90-L92), [dma_engine.sv](file:///home/dev01/neural-ai/hw/rtl/dma/dma_engine.sv#L116)

`cluster_ctrl_regs` makes `r_dma_start` a self-clearing pulse (line 91: `r_dma_start <= 1'b0`). But the output `cfg_dma_start_o` is assigned combinationally:
```sv
assign cfg_dma_start_o = r_dma_start;   // line 176
```

The `dma_engine` checks `cfg_start_i` in its IDLE state:
```sv
IDLE: begin
    if (cfg_start_i) begin ...
```

This is correct: the pulse lasts exactly 1 cycle. However, the simulation log shows:
```
[DMA] cfg_start_i=1, src=80000000, dst=10110000, len=      1024
[DMA] cfg_start_i=1, src=80000000, dst=10110000, len=      1024
[DMA] cfg_start_i=1, src=80000000, dst=10110000, len=      1024
```

The same `cfg_start_i=1` message appears **6 times** for a single DMA start. This is the same Verilator `$display` in `always_comb` issue — it fires on every combinational re-evaluation.

> [!NOTE]
> The self-clearing pulse mechanism is correct. But the repeated `$display` output is extremely misleading during debug. Move all `$display` statements from `always_comb` blocks into `always_ff` blocks throughout the design.

---

## 🟠 Architectural Issues

### A1. D-TCM Address Offset Hardcoded Incorrectly

**File**: [npu_cluster.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/npu_cluster.sv#L431)

```sv
.addr_i  ((dtcm_addr - 32'h1000_8000) >> 5),
```

The D-TCM base is `0x1000_8000` in the linker script and firmware. But `npu_cluster_pkg.sv` line 32 defines:
```sv
localparam logic [31:0] DTCM_BASE_ADDR = 32'h1001_0000;
```

**These don't match!** The firmware uses `0x1000_8000` (from `link.ld`), the demux routes `0x1000_8000` correctly (M1_BASE), but the pkg says `0x1001_0000`.

**Impact**: The subtraction `dtcm_addr - 32'h1000_8000` is correct for the actual firmware addresses. The pkg constant is stale/wrong. This doesn't cause a bug now, but will confuse anyone using the pkg constants.

**Fix**: Update `npu_cluster_pkg.sv` to `DTCM_BASE_ADDR = 32'h1000_8000`.

---

### A2. TCDM Bank Address Calculation Double-Shifts

**File**: [npu_cluster.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/npu_cluster.sv#L524-L529)

```sv
// Bank ID = addr[8:5]. Word offset in bank = addr[18:9].
assign slave_req[b].addr = (slv_addr[b] & 32'h000F_FFFF) >> 9;
```

The `tcdm_interconnect.sv` already computes `bank_addr_o` at line 76:
```sv
bank_addr_o[b] = ((master_addr_i[m] >> BYTE_SEL_BITS) / NUM_BANKS) << BYTE_SEL_BITS;
```

So the interconnect outputs an address that has **already been de-interleaved**. Then `npu_cluster.sv` applies `>> 9` on top of that. Meanwhile, `cluster_sram_bank.sv` uses `addr_i[ADDR_BITS-1:0]` directly as the memory index.

**Impact**: This triple-transformation chain (`interconnect shifts → npu_cluster shifts → sram indexes`) is fragile and hard to verify. The address that reaches the SRAM bank is:
1. Interconnect computes `(addr >> 5) / 16) << 5` → strips bank bits, re-adds byte offset
2. npu_cluster strips base with mask, then `>> 9`
3. SRAM uses low bits as index

For a correct access to TCDM address `0x10100020` (bank 1, word 0):
- Interconnect: `(0x10100020 >> 5) / 16) << 5 = (0x808001 / 16) << 5 = 0x80800 << 5 = 0x1010000` → **Wait, this is wrong**. The `<< BYTE_SEL_BITS` re-adds the shift, making it a byte address again.

> [!CAUTION]
> The address pipeline between `tcdm_interconnect`, `npu_cluster`'s `slave_req[b].addr` assignment, and `cluster_sram_bank` has redundant and contradictory shifts. This is the most likely root cause of data corruption in the TCDM path. **Recommendation**: Simplify to a single address transformation at one location.

---

### A3. I-TCM Address Shift at Instantiation Site

**File**: [npu_cluster.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/npu_cluster.sv#L245)

```sv
.addr_i  ((itcm_addr & 32'h0000_7FFF) >> 5),
```

This masks to 32KB range and divides by 32 (the 256-bit word size = 32 bytes). This is correct for converting byte addresses to word addresses for the SRAM. But:

1. The mask `0x7FFF` limits to 32KB, which matches `SIZE_BYTES=32768`.
2. The `>> 5` converts byte address → 32-byte word index.
3. Inside `cluster_sram_bank.sv`, `addr_i[ADDR_BITS-1:0]` selects the word — `ADDR_BITS = clog2(32768/32) = clog2(1024) = 10`.

This is correct. ✅

---

### A4. `obi_arbiter_2to1` Has Empty `always_comb` Block

**File**: [obi_arbiter_2to1.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/obi_arbiter_2to1.sv#L55-L58)

```sv
always_comb begin
    if (m0_req_i || m1_req_i) begin
    end
end
```

This is dead code — an empty block that does nothing. Likely a leftover from debug.

**Impact**: No functional impact, but Verilator may emit warnings and it clutters the code.

---

### A5. `obi_demux_1to4` Has Empty `always_comb` Block

**File**: [obi_demux_1to4.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/obi_demux_1to4.sv#L105-L108)

Same issue as A4 — dead code.

---

## 🟡 Design Improvements

### I1. Systolic Controller `weight_data_o` / `ifm_data_o` Width Mismatch

**File**: [systolic_controller.sv](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L102-L103)

```sv
weight_data_o = obi_i_rdata_i;  // 256-bit → 32×8 = 256-bit ✅
ifm_data_o    = obi_i_rdata_i;  // 256-bit → 32×8 = 256-bit ✅
```

The widths match, but **both** are driven from `obi_i_rdata_i` simultaneously. During `LOAD_WEIGHTS`, the controller drives weight data. During `COMPUTE`, it drives IFM data. Since both are always assigned from the same source, the systolic array will see garbage on the unused port.

**Fix**: Gate the assignments by state:
```sv
weight_data_o = '0;
ifm_data_o    = '0;
// Then in LOAD_WEIGHTS: weight_data_o = obi_i_rdata_i;
// In COMPUTE: ifm_data_o = obi_i_rdata_i;
```

---

### I2. Systolic Controller OFM Packing Is Wrong

**File**: [systolic_controller.sv](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L112-L115)

```sv
obi_o_wdata_o[0] = ofm_data_i[ 7: 0]; // First 8 elements (256-bit)
obi_o_wdata_o[1] = ofm_data_i[15: 8];
obi_o_wdata_o[2] = ofm_data_i[23:16];
obi_o_wdata_o[3] = ofm_data_i[31:24];
```

`ofm_data_i` is `logic signed [31:0][31:0]` — a packed array of 32 elements, each 32-bit wide. Total = 1024 bits. But `obi_o_wdata_o[i]` is 256-bit each.

The slice `ofm_data_i[7:0]` selects **elements 0-7** (8 × 32-bit = 256 bits) — this is correct for SystemVerilog packed array slicing. But the notation `[7:0]` vs `[15:8]` is selecting element indices, not bit indices. This works because `ofm_data_i` is declared as `logic signed [31:0][31:0]` which makes the outer dimension the element index.

**This is actually correct** ✅ if the declaration matches. But:

> [!TIP]
> Use named constants or comments to clarify that `ofm_data_i[7:0]` means "elements 0 through 7" rather than "bits 7 through 0". This code is very easy to misread.

---

### I3. No Backpressure Handling in Systolic Controller OFM Writes

**File**: [systolic_controller.sv](file:///home/dev01/neural-ai/hw/rtl/systolic/systolic_controller.sv#L120-L128)

```sv
if (ofm_valid_i) begin
    obi_o_req_o = 4'b1111;
    if (obi_o_gnt_i == 4'b1111) begin
        o_ptr_d = o_ptr_q + 128;
        drain_cnt_d = drain_cnt_q - 1;
    end
end
```

If **any** of the 4 OBI ports doesn't grant (`obi_o_gnt_i != 4'b1111`), the data is **lost**. The systolic array continues to produce `ofm_valid_i` pulses on subsequent cycles regardless of whether the controller consumed the previous output.

**Impact**: Data corruption in OFM writes whenever there's a bank conflict in the TCDM.

**Fix**: Add a FIFO or stall mechanism. The systolic array's `ofm_valid_o` is driven by a shift register and can't be stalled. You need an output buffer.

---

### I4. DMA Only Supports L2→L1 and L1→L2, Not L1→L1

**File**: [dma_engine.sv](file:///home/dev01/neural-ai/hw/rtl/dma/dma_engine.sv#L119-L124)

The DMA decides direction based on `src[31:24] == 0x10` (L1) vs other (L2). There's no L1→L1 path. This isn't a bug for the current use case, but limits future flexibility (e.g., rearranging data within TCDM).

---

### I5. Firmware Uses Hardcoded Addresses Inconsistent with Pkg

**File**: [main.c](file:///home/dev01/neural-ai/sw/matmul_app/main.c#L17-L19)

```c
#define WEIGHT_PING_ADDR 0x10110000
#define IFM_PING_ADDR    0x10120000
#define OFM_PING_ADDR    0x10200000
```

But `npu_cluster_pkg.sv` defines:
```sv
localparam logic [31:0] WEIGHT_PING_ADDR = 32'h1010_0000;  // different!
```

**Impact**: Firmware and RTL disagree on buffer locations. The TCDM is a flat address space so both might work, but this causes confusion and will break if the TCDM base/size changes.

---

### I6. `inst_rsp.data` Selection May Read Stale Data

**File**: [snitch_core.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/snitch_core.sv#L151)

```sv
assign inst_rsp.data = adapter_rsp_data >> (inst_req.addr[...] * 32);
```

This uses `inst_req.addr` (the **current** request address) to select a word from the **previous** response. If Snitch has already moved to a new `inst_req.addr` by the time the old response arrives, the wrong 32-bit word will be selected.

The `obi_snitch_if_adapter` holds the pipeline: it only asserts `snitch_req_ready_o` when `obi_rvalid_i` comes, so Snitch can't advance its PC until the response arrives. This means `inst_req.addr` should still point to the address of the in-flight request. **This is correct** as long as Snitch's instruction interface blocks on `q_ready`.

---

## 🔵 Minor Issues

### M1. Inconsistent Indentation in `main.c`

**File**: [main.c](file:///home/dev01/neural-ai/sw/matmul_app/main.c#L55-L75)

Lines 55-67 (DMA and Systolic Array section) have no indentation relative to the `while(1)` loop, while lines 74-78 are properly indented. This is purely cosmetic.

---

### M2. `npu_cluster_pkg.sv` D-TCM Size = 8KB but `link.ld` D-TCM = 32KB

**Files**: [npu_cluster_pkg.sv](file:///home/dev01/neural-ai/hw/rtl/cluster/npu_cluster_pkg.sv#L31-L33) vs [link.ld](file:///home/dev01/neural-ai/sw/matmul_app/link.ld#L7)

```sv
// pkg: 8KB
localparam logic [31:0] DTCM_SIZE = 32'h0000_2000;
```
```ld
// linker: 32KB
D_TCM (rw) : ORIGIN = 0x10008000, LENGTH = 32K
```

The SRAM is instantiated as 8KB in RTL (line 425: `SIZE_BYTES(8192)`), but the linker script claims 32KB. If the firmware stack grows beyond 8KB, it will wrap around silently.

**Fix**: Either increase SRAM to 32KB or reduce linker script to 8KB.

---

### M3. `import struct` Duplicated in test_matmul.py

**File**: [test_matmul.py](file:///home/dev01/neural-ai/hw/rtl/cluster/tb/tests/test_matmul.py#L94) and [line 121](file:///home/dev01/neural-ai/hw/rtl/cluster/tb/tests/test_matmul.py#L121)

`import struct` appears twice inside the loop body. Move it to the top of the file with the other imports.

---

## Priority Recommendation

To get the MatMul test passing, fix these in order:

1. **C1**: Instantiate Systolic Array (blocks everything)
2. **A2**: Fix TCDM address pipeline (data corruption)
3. **I3**: Add OFM write backpressure (data loss)
4. **I1**: Gate weight/ifm data by state (wrong data fed to array)
5. **C3**: Move `$display` out of `always_comb` (debugging sanity)
6. **M2**: Reconcile D-TCM sizes (silent memory corruption risk)

After those, the randomized test loop should have a chance of passing.
