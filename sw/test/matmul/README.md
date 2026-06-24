# Matmul Regression

## Target

Keep a register-level systolic regression for an `M=64` matmul test. Unlike
`independent_systolic`, this firmware bypasses HAL tiling and drives raw
systolic MMIO registers directly.

## Scenario

1. Cocotb prepares deterministic unsigned INT8 `W[32][32]` and `IFM[64][32]`.
2. Cocotb writes fixtures to L2, loads firmware through AXI I-TCM, and releases fetch.
3. Firmware DMA-copies weights and IFM into I-TCDM.
4. Firmware programs raw systolic registers and polls `REG_SYS_DONE`.
5. Firmware DMA-copies INT32 OFM from O-TCDM back to L2.
6. Firmware writes `NPU_IRQ_HOST_NOTIFY`; cocotb compares every output word with NumPy golden.

## Command

```sh
make -C sw/test/matmul
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul
```
