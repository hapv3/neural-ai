# RTL Conv Feeder Test

## Scenario

Verify the P2 RTL Conv2D feeder direct-stream path. Firmware configures the
hardware feeder through MMIO, the feeder streams `M x 32` IFM rows directly into
the systolic IFM input, and the optional materialize mode dumps the same rows
into Shared Data TCDM for debug/verification.

## Target

- Conv `1x1` NHWC with `IC=33`, proving RTL K-block streaming and systolic
  INT32 psum accumulation.
- Conv `3x3` NHWC with stride `1`, pad `1`, `IC=3`, proving hardware padding
  zero injection and spatial kernel address generation.
- Conv1x1 im2col K-block dumps are copied to L2 from materialize mode so cocotb
  validates the RTL feeder separately from the direct stream compute path.

## Command

```sh
make -C sw/test/conv_feeder_rtl
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_feeder_rtl
```
