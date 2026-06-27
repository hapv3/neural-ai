# SW Test Firmware Layout

All standalone firmware regressions live under `sw/test/<name>`. Directory names
avoid historical `_app` and `_test` suffixes; the path itself already marks these
as test firmware.

## Suites

| Directory | Firmware | Target |
|-----------|----------|--------|
| `boot` | `boot.bin` | Snitch boot, AXI I-TCM load, host IRQ signature, iDMA MMIO smoke |
| `conv_feeder` | `conv_feeder.bin` | Legacy/debug software im2col reference, Conv1x1 K>32 accumulation, Conv3x3 pad1 |
| `conv_feeder_rtl` | `conv_feeder_rtl.bin` | Legacy/debug RTL Conv2D feeder coverage; not a performance roadmap gate |
| `conv_perf` | `conv_perf.bin` | P0/P1 packed Conv2D scheduler using iDMA/RVV prepare with cycle stats written to L2 |
| `independent_memory` | `independent_memory.bin` | L2 fixtures, iDMA 1D/2D/3D, TCDM bank/boundary decode |
| `independent_systolic` | `independent_systolic.bin` | HAL GEMM32 for boundary `M` sizes with full INT32 output compare |
| `matmul` | `matmul.bin` | Raw systolic register path with M=64 cocotb regression |
| `spatz_ops` | `spatz_ops_test.bin` | C-callable Spatz operator wrappers used by future graph scheduler |
| `spatz_vector` | `*.bin` per `.S` file | Direct RVV instruction coverage for integrated Spatz |

## Shared Contract

- Firmware writes completion status to `NPU_IRQ_HOST_NOTIFY`.
- Passing tests notify `0xDEADBEEF`.
- Failing tests notify `0xBADxxxxx`; firmware may also keep private D-TCM debug words for local diagnosis.
- Cocotb owns large randomized fixtures and final byte/word comparison; firmware performs local checks where self-aliasing is not a risk.

## Build

```sh
make -C sw/test/boot
make -C sw/test/conv_feeder
make -C sw/test/conv_feeder_rtl
make -C sw/test/conv_perf
make -C sw/test/independent_memory
make -C sw/test/independent_systolic
make -C sw/test/matmul
make -C sw/test/spatz_ops
make -C sw/test/spatz_vector
```
