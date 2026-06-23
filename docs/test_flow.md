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
| `sw/test/boot` | `boot.bin` | `test_snitch_boot` | Boot, I-TCM load, D-TCM signature, iDMA MMIO smoke |
| `sw/test/independent_memory` | `independent_memory.bin` | `test_independent_memory` | L2 fixture, DMA 1D/2D/3D, TCDM bank/boundary decode |
| `sw/test/independent_systolic` | `independent_systolic.bin` | `test_independent_systolic` | GEMM32 for boundary `M` sizes, full INT32 compare |
| `sw/test/matmul` | `matmul.bin` | `test_matmul` | Legacy raw systolic register matmul regression |
| `sw/test/spatz_ops` | `spatz_ops_test.bin` | `test_spatz_operator_library` | C-callable Spatz operator wrappers |
| `sw/test/spatz_vector` | `basic_mem_arith.bin`, etc. | `test_spatz_vector_basic` | Direct RVV instruction groups |

`sw/lib` remains shared runtime/HAL code, not a test suite.

---

## 2. Common Pass/Fail Contract

Firmware tests publish status at the D-TCM debug page:

| Offset | Meaning |
|--------|---------|
| `0x10008000` | status: `0xDEADBEEF` pass or `0xBADxxxxx` fail |
| `0x10008004` | pass count where implemented |
| `0x10008008` | failing test id |
| `0x1000800c` | failing element/index |
| `0x10008010` | got value |
| `0x10008014` | expected value |
| `0x10008018` | phase where implemented |
| `0x1000801c` | op/sub-op where implemented |

Every new firmware regression should use this page unless the test has an older
explicit cocotb contract.

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
  -> Snitch executes firmware
  -> firmware seeds TCDM source
  -> firmware checks iDMA-compatible MMIO readback
  -> firmware copies TCDM source to destination
  -> cocotb polls D-TCM status
```

Pass criteria: status is `0xDEADBEEF`; no timeout.

### Memory

```text
cocotb writes deterministic L2 fixtures
  -> firmware waits for SIG_START
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
  -> cocotb optionally reads output buffers
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
  -> cocotb reads TCDM output buffers for exact data check
```

Covered wrappers today:

- `spatz_vec_copy_i8`
- `spatz_vec_relu_i8`
- `spatz_requant_i32_to_i8`

### Systolic

```text
cocotb writes signed INT8 W and IFM to L2
  -> firmware DMA-copies fixtures into TCDM
  -> firmware calls systolic_gemm32 for M={1,2,31,32,33,64,128,1024}
  -> HAL tiles large M safely
  -> firmware copies all INT32 OFM words back to L2
  -> cocotb compares full tensors with Python golden
```

Pass criteria: all `M x 32` INT32 words match golden.

### Legacy Matmul

```text
cocotb randomizes M <= 128, W, IFM
  -> firmware stages data through DMA
  -> firmware drives raw systolic MMIO registers directly
  -> firmware copies OFM to L2
  -> cocotb compares every INT32 output word
```

This test intentionally bypasses HAL tiling to preserve raw-controller coverage.

---

## 5. Build Gates

```bash
make -C sw/test/boot
make -C sw/test/independent_memory
make -C sw/test/independent_systolic
make -C sw/test/spatz_vector
make -C sw/test/spatz_ops
make -C sw/test/matmul
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
