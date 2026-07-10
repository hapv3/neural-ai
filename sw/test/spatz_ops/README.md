# Spatz Operator Wrapper Test

## Target

Verify the C-callable Spatz operator library that future graph firmware will
reuse. Scheduler code should only consume wrappers that pass this suite.

## Scenario

Each operator has a standalone firmware binary so cocotb can report PMU counters
per operator while optimization is still local:

| Binary | Operator |
|---|---|
| `spatz_ops_copy.bin` | `spatz_vec_copy_i8()` |
| `spatz_ops_relu.bin` | `spatz_vec_relu_i8()` |
| `spatz_ops_requant.bin` | `spatz_requant_i32_to_i8()` |
| `spatz_ops_add.bin` | `spatz_add_i8()` |
| `spatz_ops_mul.bin` | `spatz_mul_i8()` |
| `spatz_ops_logistic.bin` | `npu_logistic_i8()` AFU LUT path |
| `spatz_ops_maxpool.bin` | `spatz_maxpool2d_i8()` |
| `spatz_ops_upsample.bin` | `spatz_upsample_nearest_i8()` |
| `spatz_ops_concat.bin` | `spatz_concat_c32_i8()` |

`spatz_ops_test.bin` still runs the aggregate suite.

Firmware self-checks every output lane and records the first failing
test/index/got/expected tuple. Cocotb additionally reads output TCDM buffers for
exact data comparison.

## Command

```sh
make -C sw/test/spatz_ops
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library
```
