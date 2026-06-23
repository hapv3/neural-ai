# SW Test Firmware Layout

All standalone firmware regressions live under `sw/test/<name>`. Directory names
avoid historical `_app` and `_test` suffixes; the path itself already marks these
as test firmware.

## Suites

| Directory | Firmware | Target |
|-----------|----------|--------|
| `boot` | `boot.bin` | Snitch boot, D-TCM signature, iDMA MMIO smoke, local TCDM copy |
| `independent_memory` | `independent_memory.bin` | L2 fixtures, iDMA 1D/2D/3D, TCDM bank/boundary decode |
| `independent_systolic` | `independent_systolic.bin` | HAL GEMM32 for boundary `M` sizes with full INT32 output compare |
| `matmul` | `matmul.bin` | Legacy raw systolic register path and randomized matmul cocotb regression |
| `spatz_ops` | `spatz_ops_test.bin` | C-callable Spatz operator wrappers used by future graph scheduler |
| `spatz_vector` | `*.bin` per `.S` file | Direct RVV instruction coverage for integrated Spatz |

## Shared Contract

- Firmware writes status/debug words at `0x10008000`.
- Passing tests write `0xDEADBEEF`.
- Failing tests write `0xBADxxxxx` plus failing test id, element index, got, and expected values when supported.
- Cocotb owns large randomized fixtures and final byte/word comparison; firmware performs local checks where self-aliasing is not a risk.

## Build

```sh
make -C sw/test/boot
make -C sw/test/independent_memory
make -C sw/test/independent_systolic
make -C sw/test/matmul
make -C sw/test/spatz_ops
make -C sw/test/spatz_vector
```
