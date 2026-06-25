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
| `sw/test/independent_memory` | `independent_memory.bin` | `test_independent_memory` | L2 fixture, DMA 1D/2D/3D, TCDM bank/boundary decode |
| `sw/test/independent_memory` | `independent_memory.bin` | `test_dma_tcm` | Legacy DMA/TCDM smoke alias for current iDMA MMIO path |
| `sw/test/independent_systolic` | `independent_systolic.bin` | `test_independent_systolic` | GEMM32 for boundary `M` sizes, full INT32 compare |
| `sw/test/matmul` | `matmul.bin` | `test_matmul` | Raw systolic register matmul regression |
| `sw/test/afu` | `afu.bin` | `test_afu_basic` | AFU LUT/CSR, TCDM master path, e8/e16/e32 output, AFU internal IRQ |
| `sw/test/spatz_ops` | `spatz_ops_test.bin` | `test_spatz_operator_library` | C-callable Spatz operator wrappers |
| `sw/test/spatz_vector` | `basic_mem_arith.bin`, etc. | `test_spatz_vector_basic` | Direct RVV instruction groups |

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

---

## 3. Independent Suite Order

Run independent suites before micro-model or graph-level work:

1. **Boot**: proves instruction load/execution and signature path.
2. **Memory**: proves L2 fixtures, DMA paths, and TCDM decode without compute.
3. **RVV**: proves Spatz instruction groups before operator wrappers depend on them.
4. **Operators**: proves reusable C-callable Spatz ops before scheduler use.
5. **Systolic**: proves HAL GEMM32 tiling and full output correctness.
6. **Legacy Matmul**: keeps raw register-level systolic regression alive.

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
- `arith_mask`: multiply, min/max, arithmetic shift, compare/merge.
- `reduction`: e32 sum reduction.

### Operators

```text
spatz_ops firmware
  -> initialize deterministic vectors
  -> call C wrapper
  -> compare every output lane in firmware
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
  -> HAL tiles large M safely
  -> firmware copies all INT32 OFM words back to L2
  -> cocotb compares full tensors with Python golden
```

Pass criteria: all `M x 32` INT32 words match golden.

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
make -C sw/test/independent_memory
make -C sw/test/independent_systolic
make -C sw/test/spatz_vector
make -C sw/test/spatz_ops
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
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_memory

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_vector_basic

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_systolic

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul

env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_afu_basic
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
