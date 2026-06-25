# AFU Independent Test

## Scenario

This firmware verifies the cluster-integrated Activation Function Unit without
using D-TCM preload or host MMIO access. Snitch seeds source tensors in Shared
Data TCDM, loads deterministic LUT contents through the AFU MMIO window, starts
the AFU, waits for completion, and checks output data in firmware.

## Target

- AFU 32-bit MMIO control and LUT programming at `0x2000_3000`.
- AFU 256-bit TCDM master read/write path.
- AFU modes `e8`, `e16`, and `e32` with 32-byte-aligned buffers and
  non-multiple element counts/tails.
- AFU done event latched into `npu_interrupt_ctrl.INT_PENDING`.
- Host completion through `NPU_IRQ_HOST_NOTIFY` and `irq_o`.

## Pass Criteria

- Firmware writes `0xDEADBEEF` to `NPU_IRQ_HOST_NOTIFY`.
- Cocotb observes `irq_o`.
- Cocotb reads TCDM output buffers and compares exact bytes/words against the
  same LUT transform used by firmware.
