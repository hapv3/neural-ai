# Spatz/C-Wrapper Operator Test

## Target

Verify the C-callable Spatz/AFU operator library that future graph firmware will
reuse for non-convolution helper paths.

## Scenario

Each operator has a standalone firmware binary so cocotb can report PMU counters
per operator while optimization is still local:

| Binary | Operator |
|---|---|
| `spatz_ops_copy.bin` | `spatz_vec_copy_i8()` |
| `spatz_ops_relu.bin` | `spatz_vec_relu_i8()` |
| `spatz_ops_requant.bin` | `spatz_requant_i32_to_i8()` |
| `spatz_ops_add.bin` | `spatz_add_i8()` via AFU add plus AFU clamp |
| `spatz_ops_mul.bin` | `spatz_mul_i8()` scalar multiply/requant fallback; native AFU currently covers Q7 multiply only |
| `spatz_ops_maxpool.bin` | `spatz_maxpool2d_i8()` |
| `spatz_ops_upsample.bin` | `spatz_upsample_nearest_i8()` |
| `spatz_ops_concat.bin` | `spatz_concat_c32_i8()` |

`spatz_ops_test.bin` runs the aggregate operator suite.

Firmware records only dispatch failures. Cocotb reads output TCDM/L2 buffers
for exact data comparison.

## Command

```sh
make -C sw/test/spatz_ops
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library
```
