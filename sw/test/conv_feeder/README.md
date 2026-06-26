# Conv Feeder Test

## Scenario

Verify the P1 software Conv2D feeder before adding an RTL address generator.
Firmware materializes tile-local `M x 32` IFM rows in TCDM, then calls the
existing systolic HAL.

## Target

- Conv `1x1` NHWC with `IC=33`, proving `K>32` split and INT32 psum accumulation.
- Conv `3x3` NHWC with stride `1`, pad `1`, `IC=3`, proving padding zero
  injection and spatial kernel address generation.
- Conv1x1 im2col K-block dumps are copied to L2 so cocotb validates the feeder
  separately from the systolic array.
- Full `OH x OW x 32` INT32 outputs are copied back to L2 and checked by
  cocotb against Python golden.

## Addressing Note

All temporary buffers are inside the Shared Data TCDM window. The OFM buffer is
`0x10140000`; avoid `0x10200000` in this suite because it is outside the current
TCDM decode range and can alias or miss the intended SRAM banks.

## Command

```sh
make -C sw/test/conv_feeder
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_feeder
```
