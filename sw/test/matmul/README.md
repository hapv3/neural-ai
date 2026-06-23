# Matmul Regression

## Target

Keep a legacy register-level systolic regression for randomized `M <= 128`
matmul tests. Unlike `independent_systolic`, this firmware bypasses HAL tiling
and drives raw systolic MMIO registers directly.

## Scenario

1. Cocotb randomizes signed INT8 `W[32][32]`, `IFM[M][32]`, and `M`.
2. Cocotb writes fixtures to L2 and signals firmware through D-TCM.
3. Firmware DMA-copies weights and IFM into I-TCDM.
4. Firmware programs raw systolic registers and polls `REG_SYS_DONE`.
5. Firmware DMA-copies INT32 OFM from O-TCDM back to L2.
6. Cocotb compares every output word with NumPy golden.

## Command

```sh
make -C sw/test/matmul
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_matmul
```
