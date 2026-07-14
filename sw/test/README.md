# SW Test Firmware Layout

All standalone firmware regressions live under `sw/test/<name>`. Directory names
avoid historical `_app` and `_test` suffixes; the path itself already marks these
as test firmware.

## Suites

| Directory | Firmware | Target |
|-----------|----------|--------|
| `boot` | `boot.bin` | Snitch boot, AXI I-TCM load, host IRQ signature, iDMA MMIO smoke |
| `conv_perf` | `conv_perf.bin` | P0/P1 packed Conv2D scheduler using iDMA/RVV prepare with cycle stats written to L2 |
| `independent_memory` | `independent_memory.bin` | L2 fixtures, iDMA 1D/2D/3D, TCDM bank/boundary decode |
| `independent_systolic` | `independent_systolic.bin` | HAL GEMM32 for boundary `M` sizes with full INT32 output compare |
| `matmul` | `matmul.bin` | Raw systolic register path with M=64 cocotb regression |
| `micro_yolo` | `micro_yolo.bin` | Current Micro-YOLO raw-head, DFL, and class-sigmoid E2E firmware |
| `pointwise_conv` | `pointwise_conv.bin` | Graph-level Conv1x1 C32 fast path over `1x32x48x48` C32-blocked input |
| `spatz_ops` | `spatz_ops_test.bin` | C-callable Spatz/AFU operator wrappers used by graph firmware |
| `spatz_vector` | `*.bin` per `.S` file | Direct RVV instruction coverage for integrated Spatz |

## Shared Contract

- Firmware writes completion status to `NPU_IRQ_HOST_NOTIFY`.
- Passing tests notify `0xDEADBEEF`.
- Failing tests notify `0xBADxxxxx`; firmware may also keep private D-TCM debug words for local diagnosis.
- Cocotb owns large randomized fixtures and final byte/word comparison; firmware performs local checks where self-aliasing is not a risk.

## Build

```sh
make -C sw/test/boot
make -C sw/test/conv_perf
make -C sw/test/independent_memory
make -C sw/test/independent_systolic
make -C sw/test/matmul
make -C sw/test/micro_yolo
make -C sw/test/pointwise_conv
make -C sw/test/spatz_ops
make -C sw/test/spatz_vector
```
