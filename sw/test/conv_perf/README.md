# Conv2D Packed Performance Test

## Scenario

Benchmark the packed Conv2D performance path.
Firmware prepares `M x 32` IFM tiles in TCDM with iDMA/RVV helpers, runs
systolic GEMM32/accumulate, and writes per-layer cycle counters to L2.

## Target

- Conv `1x1`, `IC=33`, two K-tiles, proving K > 32 packed scheduling.
  Source stays in L2 and the prepare path uses iDMA 2D strided pack
  (`src_stride=IC`, `dst_stride=32`), including K-tail zero-fill.
- Conv `1x1`, `IC=32`, `M=64`, proving exact one full K tile.
- Conv `1x1`, `IC=64`, `M=64`, proving multiple full K tiles and INT32
  psum accumulation.
- Conv `3x3`, `IC=3`, stride `1`, pad `1`, proving first-layer RGB padding
  prepare path through Spatz RVV segment copies with explicit zero-fill.
- P3 functional coverage adds Conv1x1 pointwise tails (`IC=1/3/31`), Conv1x1
  `IC=33, OC=64`, Conv3x3 pad0/pad1 with `IC=32`, Conv5x5, Conv7x7,
  asymmetric kernels, stride2, tail-K `K=9/45`, and final-block requant.
- Output tensors are still compared by cocotb against Python golden.
- Cycle stats report rows, K-tiles, prepare cycles, GEMM cycles, total cycles,
  final tile cycles, and per-backend tile counts.

## Command

```sh
make -C sw/test/conv_perf
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf
```

For faster debug, build and run one group at a time:

```sh
make -C sw/test/conv_perf clean && make -C sw/test/conv_perf CONV_PERF_GROUP=1
env CONV_PERF_GROUP=1 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf

make -C sw/test/conv_perf clean && make -C sw/test/conv_perf CONV_PERF_GROUP=2
env CONV_PERF_GROUP=2 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf

make -C sw/test/conv_perf clean && make -C sw/test/conv_perf CONV_PERF_GROUP=3
env CONV_PERF_GROUP=3 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf
```

Group `1` covers legacy pointwise, IC tails, and OC64 tiling. Group `2` covers
kernel/stride/padding/tail-K shapes. Group `3` covers final-block INT8 requant.
