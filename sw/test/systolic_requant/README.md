# Systolic Requant Test

## Scenario

Verify the RTL requant path integrated in the systolic controller output drain.
Firmware loads deterministic INT8 weights/IFM from L2, configures per-channel
requant qparams, runs GEMM32 with requant enabled, and copies packed INT8 output
back to L2.

## Target

- Raw systolic bypass remains covered by `independent_systolic`.
- Requant mode writes one packed 256-bit row per `M x 32` output row.
- Per-channel bias, multiplier, shift, zero-point, and clamp are applied in RTL.

## Pass Criteria

- Firmware reports `0xDEADBEEF` through `NPU_IRQ_HOST_NOTIFY`.
- Cocotb compares every output byte against the Python golden formula.
