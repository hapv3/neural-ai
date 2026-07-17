# NPU Cluster Test Flow

**Scope**: Current cluster verification after DMA/TCDM and Spatz integration work.  
**Firmware layout**: all standalone test firmware lives under `sw/test/<name>`.

---

## 1. SW Test Layout

Historical suffixes `_app` and `_test` were removed from firmware directory
names. The `sw/test` parent directory is now the semantic marker for regression
firmware.

| Directory | Binary | Primary RTL test | Target |
|-----------|--------|------------------|--------|
| `sw/test/boot` | `boot.bin` | `test_snitch_boot` | Boot, AXI I-TCM load, host IRQ completion, iDMA MMIO smoke |
| `sw/test/pmu` | `pmu.bin` | `test_pmu_basic` | Host AXI-Lite PMU control, snapshot, fixed counters, Snitch/TCDM event smoke |
| `sw/test/conv_perf` | `conv_perf.bin` | `test_conv_perf` | Packed Conv2D scheduler, iDMA/RVV backend checks, exact output compare, cycle stats in L2 |
| `sw/test/independent_memory` | `independent_memory.bin` | `test_independent_memory` | L2 fixture, DMA 1D/2D/3D, TCDM bank/boundary decode |
| `sw/test/independent_memory` | `independent_memory.bin` | `test_dma_tcm` | Legacy DMA/TCDM smoke alias for current iDMA MMIO path |
| `sw/test/independent_systolic` | `independent_systolic.bin` | `test_independent_systolic` | GEMM32 for boundary `M` sizes, full INT32 compare |
| `sw/test/systolic_requant` | `systolic_requant.bin` | `test_systolic_requant` | Systolic GEMM32 with RTL per-channel requant and packed INT8 output |
| `sw/test/matmul` | `matmul.bin` | `test_matmul` | Raw systolic register matmul regression |
| `sw/test/afu` | `afu.bin` | `test_afu_basic` | AFU LUT/CSR, TCDM master path, e8/e16/e32 output, AFU internal IRQ |
| `sw/test/afu_ops` | `afu_ops_*.bin` | `test_afu_op_*` | C-callable AFU-native operator wrappers, with cocotb golden checks |
| `sw/test/spatz_ops` | `spatz_ops_test.bin` | `test_spatz_operator_library` | C-callable non-AFU Spatz/C-wrapper helpers |
| `sw/test/spatz_vector` | `basic_mem_arith.bin`, etc. | `test_spatz_vector_basic` | Direct RVV instruction groups |
| `sw/test/micro_yolo` | `micro_yolo.elf` | `test_micro_yolo_e2e` | 96x96 raw-head Micro-YOLO graph, host-generated linebuffer job descriptors, full output compare |
| `sw/test/micro_mobilenet` | `micro_mobilenet.bin` | `test_micro_mobilenet_e2e` | 96x96 Micro-MobileNet graph using native standard conv, depthwise, pointwise, clamp, add, GlobalAvgPool, and DMA |

`sw/lib` remains shared runtime/HAL code, not a test suite.

---

## 2. Common Pass/Fail Contract

Firmware tests report completion through the interrupt controller:

| Register | Meaning |
|----------|---------|
| `NPU_IRQ_HOST_NOTIFY` / `0x2000_2018` | Firmware writes `0xDEADBEEF` pass or `0xBADxxxxx` fail |
| `NPU_IRQ_HOST_STATUS` / `0x2000_201c` | Internal MMIO status latch for future host-control path |
| `irq_o` | External host interrupt asserted after `HOST_NOTIFY` |

Every new firmware regression should use this IRQ/status path. D-TCM remains
private Snitch memory for stack, `.data`, `.bss`, and optional firmware-local
debug words; cocotb must not use D-TCM backdoor writes as a start mailbox or
preload mechanism for active gates.

The host AXI-Lite boot path reaches I-TCM only. Cocotb holds Snitch with
`fetch_enable_i=0`, loads the binary through AXI, then releases fetch. Cocotb
does not read IRQ MMIO through AXI in the current topology; exact L2/TCDM output
checks are the pass/fail oracle after `irq_o` asserts.

Exception for initialized firmware data: tests that need static `.data` content
may use an ELF-aware loader. `test_micro_yolo_e2e` uses this because the
host-generated linebuffer/GEMM job descriptor arrays are linked into D-TCM
`.data`. The loader writes `.text` to I-TCM through AXI and initializes `.data`
through testbench hierarchy before fetch release. This is not a host-visible
D-TCM frontend and must not be used as a start mailbox or pass/fail polling
mechanism.

The host AXI-Lite boot path also exposes the PMU window at `0x2000_4000`.
Cocotb starts PMU counters before releasing fetch, snapshots/stops them after
`irq_o`, and prints a performance report for each firmware test that passes an
`axi_master` into `release_fetch`/`wait_for_host_irq`.

---

## 3. Independent Suite Order

Run independent suites before micro-model or graph-level work:

1. **Boot**: proves instruction load/execution and signature path.
2. **PMU**: proves performance counters before using cycle/event data for optimization.
3. **Memory**: proves L2 fixtures, DMA paths, and TCDM decode without compute.
4. **RVV**: proves Spatz instruction groups before operator wrappers depend on them.
5. **Operators**: proves reusable C-callable Spatz ops before scheduler use.
5. **Systolic**: proves HAL GEMM32 tiling and full output correctness.
6. **Systolic Requant**: proves fused INT32→INT8 drain path before graph use.
7. **Conv Packed Scheduler**: proves software+iDMA+Spatz packed Conv2D lowering and records cycle stats before graph/model use.
8. **Legacy Matmul**: keeps raw register-level systolic regression alive.

Micro-YOLO or graph scheduler tests should only run after these gates are green.

---

## 4. Test Scenarios

### Boot

```text
cocotb loads sw/test/boot/boot.bin into I-TCM
  -> cocotb releases fetch_enable_i
  -> Snitch executes firmware
  -> firmware seeds TCDM source
  -> firmware checks iDMA-compatible MMIO readback
  -> firmware copies TCDM source to destination
  -> firmware writes NPU_IRQ_HOST_NOTIFY
  -> cocotb waits irq_o
```

Pass criteria: `irq_o` asserts; no timeout.

### Memory

```text
cocotb writes deterministic L2 fixtures
  -> cocotb loads firmware into I-TCM and releases fetch
  -> L2 -> TCDM 1D, 2D, 3D checked in firmware
  -> TCDM -> L2 1D, 2D, 3D checked by cocotb
  -> firmware probes representative low/high addresses for each TCDM bank
```

Pass criteria: firmware pass signature plus exact L2 output bytes for all
output-side copies.

### RVV

```text
firmware assembly test
  -> configure VL with vsetvli
  -> run one RVV instruction group
  -> store vector output to TCDM
  -> scalar-check every lane
  -> firmware writes NPU_IRQ_HOST_NOTIFY
  -> cocotb reads TCDM output buffers
```

Covered groups today:

- `basic_mem_arith`: e32 load/store, add/sub/logic, logical shifts.
- `memory_width`: e8/e16/e32 load-store.
- `strided_indexed`: e8 strided load/store plus unordered/ordered indexed
  load-store with e8 index vectors. This group passes RTL with exact firmware
  and cocotb output checks, so higher-level Spatz gather/scatter code may use
  indexed e8 load/store. It also checks a stride-32 packed-row store pattern
  used by Conv2D `M x 32` im2col tiles, including an LMUL `m8` case.
- `arith_mask`: multiply, min/max, arithmetic shift, compare/merge.
- `reduction`: e32 sum reduction.

### Operators

```text
spatz_ops / afu_ops firmware
  -> initialize deterministic vectors
  -> call C wrapper
  -> firmware writes NPU_IRQ_HOST_NOTIFY
  -> cocotb reads TCDM output buffers for exact data check
```

Covered wrappers today:

- `spatz_vec_copy_i8`
- `spatz_vec_relu_i8`
- `spatz_requant_i32_to_i8`

### Systolic

```text
cocotb writes signed INT8 W and IFM to L2
  -> cocotb loads firmware into I-TCM and releases fetch
  -> firmware DMA-copies fixtures into TCDM
  -> firmware calls systolic_gemm32 for M={1,2,31,32,33,64,128,1024}
  -> firmware runs K=64 as base GEMM32 + accumulated GEMM32 psum block
  -> HAL tiles large M safely
  -> firmware copies all INT32 OFM words back to L2
  -> cocotb compares full tensors with Python golden
```

Pass criteria: all `M x 32` INT32 words and the `K=64` accumulated tensor match golden.

### Systolic Requant

```text
cocotb writes signed INT8 W and IFM to L2
  -> cocotb loads systolic_requant firmware into I-TCM and releases fetch
  -> firmware DMA-copies fixtures into TCDM
  -> firmware programs per-channel requant qparams
  -> firmware calls systolic_gemm32_requant for M={1,2,31,32,33,64}
  -> controller drains each INT32 OFM row through requant_pipeline
  -> firmware runs K=64 as base GEMM32 + accumulated final requant GEMM32
  -> controller writes one packed 32-byte INT8 row to O-TCDM
  -> firmware copies packed output rows back to L2
  -> cocotb compares every byte against Python golden
```

Pass criteria: all `M x 32` INT8 bytes and the fused `K=64` accumulated
requant tensor match the exact requant formula.

### Conv Packed Performance

```text
cocotb writes the same Conv1x1 and Conv3x3 fixtures to L2
  -> firmware keeps L2-resident inputs in L2 so iDMA can pack regular tiles
  -> npu_conv2d_packed_run_oc32 prepares packed M x 32 IFM tiles in TCDM
     using iDMA 2D for contiguous Conv1x1/K-tail tiles, iDMA 3D for
     regular multi-spatial Conv2D segments, and Spatz RVV strided copies for
     L1/TCDM fallback cases
  -> firmware runs systolic GEMM32/accumulate over the packed tile
  -> firmware records mcycle deltas for prepare, GEMM, total, last K tile,
     backend tile counts, and actual iDMA transfer counts
  -> firmware copies full INT32 outputs and stats back to L2
  -> cocotb compares output tensors and checks iDMA/RVV backend selection
```

Pass criteria: Conv outputs match Python golden exactly, scheduler status is
`NPU_CONV2D_PACKED_OK`, scalar prepare tile count is zero, Conv1x1 uses iDMA
for contiguous K tiles, `IC=32/64` pointwise cases prove full and multi-K
tiles, L2-resident regular Conv3x3/RGB tiles use iDMA 3D multi-spatial pack,
fallback L1/irregular cases use Spatz RVV pack, OC64 tiling stores both output
channel tiles at the correct row stride, and final accumulated K-block requant
matches the exact INT32-to-INT8 golden formula. This is the current
performance-path gate. `conv_perf` is the only Conv2D performance-path gate.

`test_conv_perf` can run as one full binary or as shorter focused groups:

```bash
make -C sw/test/conv_perf
env CONV_PERF_GROUP=1 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf

env CONV_PERF_GROUP=2 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf

env CONV_PERF_GROUP=3 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf
```

Group `1` covers pointwise/OC tiling, group `2` covers kernel shapes, and group
`3` covers final-block requant.

For one-case debug, compile `conv_perf.bin` once and pass
`CONV_PERF_CASE=<case_id>` only to cocotb. Cocotb writes the selected group/case
into the runtime L2 config block before Snitch starts, so firmware rebuilds are
not required between cases. Example: `CONV_PERF_CASE=6` runs the L1-source
`Conv5x5 pad2 IC3` Spatz fallback case. Compile with `CONV_PERF_SPATZ_RECT=0`
to compare against the older segmented Spatz fallback.

For broad regression, use the cluster runner. It starts independent cocotb
simulator processes from a shared Verilator binary for each RTL generic set,
and can sweep `conv_perf` cases with one firmware image. Result XMLs and logs
are written separately under `hw/rtl/cluster/tb/sim/<shared-build>/`:

```bash
python3 hw/rtl/cluster/tb/run_cluster_tests.py --build-fw --jobs 4 --tests all
python3 hw/rtl/cluster/tb/run_cluster_tests.py --tests test_conv_perf --conv-perf-cases 0-23
```

`conv_perf` case ID 20 is reserved for the removed raw IC120 slow-path
regression. The runner filters it out from `all` and numeric ranges; case 22 is
the active C32-padded IC120 coverage.

### Matmul

```text
cocotb prepares deterministic M=64 W and IFM
  -> firmware stages data through DMA
  -> firmware drives raw systolic MMIO registers directly
  -> firmware copies OFM to L2
  -> firmware writes NPU_IRQ_HOST_NOTIFY
  -> cocotb compares every INT32 output word
```

This test intentionally bypasses HAL tiling to preserve raw-controller coverage.
Boundary M coverage lives in `test_independent_systolic`.

### AFU

```text
firmware seeds deterministic source tensors in Shared Data TCDM
  -> firmware loads 256-entry LUT through AFU MMIO
  -> firmware enables NPU_IRQ_SRC_AFU
  -> firmware starts AFU for e8, e16, and e32 output modes
  -> firmware waits AFU done status and checks INT_PENDING
  -> firmware compares every output element in TCDM
  -> firmware writes NPU_IRQ_HOST_NOTIFY
  -> cocotb compares the same output buffers against Python golden
```

Current cluster contract uses 32-byte-aligned source/destination buffers with
non-multiple element counts to cover tail byte-enable behavior. Arbitrary
unaligned e16/e32 destinations are not yet a scheduler contract.

---

## 5. Build Gates

```bash
make -C sw/test/boot
make -C sw/test/pmu
make -C sw/test/conv_perf
make -C sw/test/independent_memory
make -C sw/test/independent_systolic
make -C sw/test/systolic_requant
make -C sw/test/spatz_vector
make -C sw/test/spatz_ops
make -C sw/test/afu_ops
make -C sw/test/matmul
make -C sw/test/afu
```

Spatz-related tests use the local toolchain under `hw/spatz/install` by default.

---

## 6. RTL Gates

Use the same Verilator/cocotb cluster target and select modules explicitly:

```bash
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_pmu_basic

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_memory

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_vector_basic

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_systolic

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_systolic_requant

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_afu_basic

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_micro_yolo_e2e
```

Optional diagnostic:

```bash
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_systolic_ofm_fifo_highwater
```

`test_systolic_ofm_fifo_highwater` is a sizing/observability diagnostic, not a
functional gate.

---

## 7. Acceptance Rule

A change that touches DMA, TCDM interconnect, Spatz integration, or systolic
controller behavior should at minimum rebuild all `sw/test` firmware and rerun
the affected RTL gates. If the change is broad or changes shared arbitration,
rerun the full gate list in this document.
