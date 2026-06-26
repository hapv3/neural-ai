# Systolic Requant Test

## Scenario

Verify the RTL requant path integrated in the systolic controller output drain.
Firmware loads deterministic INT8 weights/IFM from L2, configures per-channel
requant qparams, runs GEMM32 with requant enabled, and copies packed INT8 output
back to L2. It also runs a two-block `K=64` fixture where the first block stores
INT32 psum and the final accumulated block is requantized directly in RTL.

## Target

- Raw systolic bypass remains covered by `independent_systolic`.
- Requant mode writes one packed 256-bit row per `M x 32` output row.
- Accumulated final K-blocks can feed the same requant pipeline without an
  intermediate INT32 writeback.
- Per-channel bias, multiplier, shift, zero-point, and clamp are applied in RTL.

## Pass Criteria

- Firmware reports `0xDEADBEEF` through `NPU_IRQ_HOST_NOTIFY`.
- Cocotb compares every output byte against the Python golden formula.
