# Spatz Operator Wrapper Test

## Target

Verify the C-callable Spatz operator library that future graph firmware will
reuse. Scheduler code should only consume wrappers that pass this suite.

## Scenario

1. `spatz_vec_copy_i8()` copies signed INT8 data exactly.
2. `spatz_vec_relu_i8()` clamps negative lanes in-place and preserves positive lanes.
3. `spatz_requant_i32_to_i8()` applies integer multiply, arithmetic shift, and clamp.
4. Firmware self-checks every output lane and records first failing lane.
5. Cocotb additionally reads output TCDM buffers for exact data comparison.

## Command

```sh
make -C sw/test/spatz_ops
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library
```
