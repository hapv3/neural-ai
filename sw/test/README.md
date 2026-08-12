# SW Test Firmware Layout

All standalone firmware regressions live under `sw/test/<name>`. Directory names
avoid historical `_app` and `_test` suffixes; the path itself already marks these
as test firmware.

## Suites

| Directory | Firmware | Target |
|-----------|----------|--------|
| `boot` | `boot.bin` | Snitch boot, AXI I-TCM load, host IRQ signature, iDMA MMIO smoke |
| `afu_ops` | `afu_ops_*.bin` | C-callable AFU-native operator wrappers; cocotb owns golden output checks |
| `conv_perf` | `conv_perf.bin` | P0/P1 packed Conv2D scheduler using iDMA/RVV prepare with cycle stats written to L2 |
| `depthwise_conv` | `depthwise_conv*.bin` | Graph-level DepthwiseConv3x3 linebuffer fast path over C32-blocked input; covers S1/P1 C32/C33-tail/C64/C96 and S2/P1 C32/C65-tail downsample |
| `independent_memory` | `independent_memory.bin` | L2 fixtures, iDMA 1D/2D/3D, TCDM bank/boundary decode |
| `independent_systolic` | `independent_systolic.bin` | HAL GEMM32 for boundary `M` sizes with full INT32 output compare |
| `matmul` | shared `runtime/neural_ai` firmware | V2 package/invocation GEMM32 path with M=64 cocotb regression |
| `micro_yolo` | `micro_yolo.bin` | Current Micro-YOLO raw-head, DFL, and class-sigmoid E2E firmware |
| `pointwise_conv` | `pointwise_conv.bin`, `pointwise_conv_c64_c128.bin` | Graph-level Conv1x1 C32 fast path over C32-blocked tensors; covers single-group `C32->C32` and multi-group `C64->C128` |
| `spatz_ops` | `spatz_ops_test.bin` | C-callable non-AFU Spatz/C-wrapper helpers used by graph firmware |
| `spatz_vector` | `*.bin` per `.S` file | Direct RVV instruction coverage for integrated Spatz |

## Shared Contract

- Firmware writes completion status to `NPU_IRQ_HOST_NOTIFY`.
- Passing tests notify `0xDEADBEEF`.
- Failing tests notify `0xBADxxxxx`; firmware may also keep private D-TCM debug words for local diagnosis.
- Cocotb owns large randomized fixtures and final byte/word comparison. Accelerator operator firmware should only report dispatch/control failures.

## Build

```sh
make -C sw/test/boot
make -C sw/test/afu_ops
make -C sw/test/conv_perf
make -C sw/test/depthwise_conv
make -C sw/test/independent_memory
make -C sw/test/independent_systolic
make -C sw/test/matmul
make -C sw/test/micro_yolo
make -C sw/test/pointwise_conv
make -C sw/test/spatz_ops
make -C sw/test/spatz_vector
```
