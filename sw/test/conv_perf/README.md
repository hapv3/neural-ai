# Conv2D Packed Performance Test

## Scenario

Benchmark the packed Conv2D performance path.
Firmware prepares `M x 32` IFM tiles in TCDM with iDMA/RVV helpers, runs
systolic GEMM32/accumulate, and writes per-layer cycle counters to L2.
The Conv2D packed runtime builds a per-layer prepare command list before the
measured K-tile loop, then reuses those commands for iDMA/Spatz prepare.

## Target

- Conv `1x1`, `IC=33`, two K-tiles, proving K > 32 packed scheduling.
  Source stays in L2 and the prepare path uses iDMA 2D strided pack
  (`src_stride=IC`, `dst_stride=32`), including K-tail zero-fill.
- Conv `1x1`, `IC=32`, `M=64`, proving exact one full K tile.
- Conv `1x1`, `IC=64`, `M=64`, proving multiple full K tiles and INT32
  psum accumulation.
- Conv `3x3`, `IC=3`, stride `1`, pad `1`, proving first-layer RGB padding
  prepare path through multi-spatial iDMA 3D segment copies with explicit
  zero-fill.
- YOLO-style first-layer RGB Conv `3x3`, `IC=3`, stride `2`, pad `1`,
  using an L2-to-TCDM input stripe copy followed by Spatz rectangle-strided
  packing. This evaluates the alternative path where source pixels are staged
  locally before packed `M x 32` prepare.
- P3 functional coverage adds Conv1x1 pointwise tails (`IC=1/3/31`), Conv1x1
  `IC=33, OC=64`, Conv3x3 pad0/pad1 with `IC=32`, Conv5x5, Conv7x7,
  asymmetric kernels, stride2, tail-K `K=9/45`, and final-block requant.
- Output tensors are still compared by cocotb against Python golden.
- Cycle stats report rows, K-tiles, prepare cycles, GEMM cycles, total cycles,
  final tile cycles, per-backend tile counts, and iDMA transfer counts.
- Multi-spatial iDMA cases may issue several transfers for one K tile; the
  iDMA job FIFO lets firmware queue those transfers and wait only on the final
  transfer ID.

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
kernel/stride/padding/tail-K shapes, including iDMA 3D multi-spatial and Spatz
fallback coverage. Group `3` covers final-block INT8 requant. Group `4` covers
the YOLO-style RGB first-layer TCDM+Spatz tiled evaluation.

For single-case debug, compile and run with `CONV_PERF_CASE=<case_id>`. This
overrides `CONV_PERF_GROUP` and skips legacy cases:

```sh
make -C sw/test/conv_perf clean && make -C sw/test/conv_perf CONV_PERF_CASE=6
env CONV_PERF_CASE=6 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf
```

For the dedicated YOLO-style first-layer evaluation:

```sh
make -C sw/test/conv_perf clean && make -C sw/test/conv_perf CONV_PERF_GROUP=4
env CONV_PERF_GROUP=4 CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_conv_perf
```

Set `CONV_PERF_SPATZ_RECT=0` while compiling to compare the older segmented
Spatz fallback against the rectangle-strided `vlse8`/`vsse8` path.

Current focused RTL measurement for `CONV_PERF_CASE=6` (`Conv5x5 pad2 IC3`,
L1/TCDM input source):

| Mode | Prepare cycles | GEMM cycles | Total cycles |
| --- | ---: | ---: | ---: |
| Command-list + fused-channel Spatz rectangle path | 11454 | 780 | 12472 |
| Initial 2D rectangle-strided Spatz fallback | 34904 | 780 | 35922 |
| Older segmented Spatz fallback (`CONV_PERF_SPATZ_RECT=0`) | 75560 | 780 | 76578 |

Current YOLO-style RGB first-layer TCDM+Spatz tiled measurement
(`CONV_PERF_GROUP=4`, input `64x64x3`, Conv `3x3/s2/p1`, output
`32x32x32`, four output-height tiles):

| Rows | K tile invocations | Prepare cycles | GEMM cycles | Total cycles | Backend |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 4 | 74124 | 1888 | 76428 | L2 stripe copy + Spatz rect pack |

This case intentionally stages each input stripe from L2 into TCDM first, then
uses the Spatz rectangle-strided pack path. It is not the direct iDMA
L2-to-im2col segment path used by the legacy RGB `3x3` smoke case.
