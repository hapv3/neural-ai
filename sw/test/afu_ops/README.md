# AFU Operator Wrapper Test

## Target

Build standalone firmware binaries for C-callable `npu_*` wrappers that dispatch
to the AFU native datapaths. Firmware only configures inputs, starts the AFU,
and reports accelerator timeout/error. Golden output comparison belongs in the
cocotb testbench.

## Binaries

| Binary | Operator |
|---|---|
| `afu_ops_add_full.bin` | `npu_add_i8()` AFU add path over a full tensor |
| `afu_ops_mul_q7_full.bin` | `npu_mul_q7_i8()` AFU Q7 multiply path over a full tensor |
| `afu_ops_logistic.bin` | `npu_logistic_i8()` AFU E8 LUT path |
| `afu_ops_logistic_full.bin` | AFU E8 LUT path over a `48x48x32` ROW32 tensor |
| `afu_ops_clamp_relu6.bin` | `npu_clamp_i8()` AFU LUT path for standalone quantized ReLU6/clamp |
| `afu_ops_dfl.bin` | `npu_dfl_softmax4_i8_q8()` and ROW32 AFU-assisted DFL coverage |
| `afu_ops_dfl_fused.bin` | `npu_dfl_softmax4_row32_i8_q8()` fused AFU DFL path |
| `afu_ops_class_sigmoid.bin` | `npu_class_sigmoid_row32_high16_i8()` AFU class sigmoid |
| `afu_ops_global_avgpool.bin` | `npu_global_avgpool_c32_i8()` AFU C32 spatial average |

## Command

```sh
make -C sw/test/afu_ops
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library \
  COCOTB_TESTCASE=test_afu_op_clamp_relu6
```
