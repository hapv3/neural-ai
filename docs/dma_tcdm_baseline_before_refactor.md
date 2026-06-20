# DMA/TCDM Baseline Before Refactor

Date: 2026-06-20

## Scope

This baseline captures the minimum known-good behavior before the planned DMA and
TCDM interconnect refactor. The goal is not full model validation; it is a compact
regression point for boot, L2 fixture handling, DMA movement, TCDM bank addressing,
Spatz RVV bring-up, and Spatz operator wrappers.

## Build Gates

All firmware build gates below completed successfully:

```sh
make -C sw/independent_memory_test
make -C sw/independent_systolic_test
make -C sw/spatz_vector
make -C sw/spatz_ops_test
```

Notes:

- `sw/spatz_vector` builds the existing basic test plus added instruction-group
  binaries: `basic_mem_arith`, `memory_width`, `arith_mask`, and `reduction`.
- Only `basic_mem_arith` is currently covered by the existing
  `test_spatz_vector_basic` RTL test.

## RTL Baseline Results

### Memory Suite

Command:

```sh
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_memory
```

Result: PASS

Observed status/debug page:

```text
status     = 0xDEADBEEF
pass_count = 3
fail_test  = 0
fail_index = 0
got        = 0
expected   = 0
phase      = 4
op         = 3
```

Coverage:

- Firmware boot/start gate after reset.
- L2 fixture is written by cocotb after reset, then consumed by firmware.
- DMA L2 to TCDM exact byte compare.
- DMA TCDM to L2 exact byte compare.
- Representative TCDM bank low/high address checks across 16 banks.

This is the primary baseline for the upcoming DMA/TCDM refactor.

### Spatz RVV Basic Suite

Command:

```sh
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_vector_basic
```

Result: PASS

Observed status/debug page:

```text
status     = 0xDEADBEEF
pass_count = 7
fail_test  = 0
fail_index = 0
got        = 0
expected   = 0
```

Coverage:

- Existing Spatz RVV boot path.
- Basic RVV memory/arithmetic firmware self-check.

Limit:

- The newer instruction-group binaries for memory width, arithmetic/mask, and
  reduction compile, but do not yet have dedicated cocotb loaders in this baseline.

### Spatz Operator Library Suite

Command:

```sh
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library
```

Result: PASS

Coverage:

- C-callable Spatz operator wrappers.
- Data-output checks through cocotb for the current operator library.
- Confirms the status/debug page reservation remains compatible with operator
  firmware after linker-script updates.

## Deferred Baseline

The full independent systolic RTL suite is intentionally not part of this minimum
baseline because it includes large GEMM cases up to `M=1024` and is expected to be
slower. It should be run before changing the systolic data path, but it is not a
blocker for starting the DMA/TCDM refactor.

Deferred command:

```sh
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_systolic
```

Existing `test_matmul` is also useful as a legacy systolic + DMA sanity gate, but
was not rerun for this minimum DMA/TCDM baseline.

## Refactor Gate Recommendation

Before modifying DMA or TCDM interconnect RTL, keep these commands as the required
smoke gate:

```sh
make -C sw/independent_memory_test
make -C sw/spatz_vector
make -C sw/spatz_ops_test
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_memory
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_vector_basic
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library
```

After each DMA/TCDM refactor step, rerun at least `test_independent_memory`. Once
memory movement is stable, add `test_matmul` or `test_independent_systolic` before
continuing with graph/operator integration.
