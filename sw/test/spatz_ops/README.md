# Spatz/C-Wrapper Operator Test

## Target

Verify the C-callable Spatz/AFU operator library that future graph firmware will
reuse for non-convolution helper paths.

## Scenario

Each operator has a standalone firmware binary so cocotb can report PMU counters
per operator while optimization is still local:

| Binary | Operator |
|---|---|
| `spatz_ops_copy.bin` | `spatz_vec_copy_i8()` |
| `spatz_ops_copy_batch.bin` | Affine local copy: word-strided pack/unpack, repeated-source fill and fallback |
| `spatz_ops_relu.bin` | `spatz_vec_relu_i8()` |
| `spatz_ops_requant.bin` | `spatz_requant_i32_to_i8()` |
| `spatz_ops_add.bin` | `spatz_add_i8()` via AFU add plus AFU clamp |
| `spatz_ops_mul.bin` | `spatz_mul_i8()` scalar multiply/requant fallback; native AFU currently covers Q7 multiply only |
| `spatz_ops_maxpool.bin` | `spatz_maxpool2d_i8()` |
| `spatz_ops_upsample.bin` | `spatz_upsample_nearest_i8()` |
| `spatz_ops_concat.bin` | `spatz_concat_c32_i8()` |

`spatz_ops_test.bin` runs the aggregate operator suite.

Firmware records only dispatch failures. Cocotb reads output TCDM/L2 buffers
for exact data comparison.

The operator-suite loader uses the ELF for every Spatz/AFU case. It loads each
PT_LOAD segment to its linked I-TCM or D-TCM address and zero-initializes
NOLOAD/BSS memory. A `.bin` can include the gap to D-TCM plus `.rodata`/`.data`;
such an image must not be streamed as though it were contiguous I-TCM code.
The I-TCM limit remains 32 KiB. The raw-image loader rejects oversized images
before issuing any bus writes; it is not a replacement for ELF segment routing.
Offline loader regressions (including the batch test's real ELF) run with:

```sh
make -C sw/test/compiler_runtime check_firmware_loading
```

The `copy_batch` firmware additionally compares every destination byte after
each of twelve geometries, including untouched gaps and trailing canaries.
It covers nonzero pattern fill, C16/C32 packing in both directions, 3D row gaps,
singleton dimensions, contiguous collapse, unaligned byte tails and a 1,031-run
case crossing the kernel's VL limit. Host dispatch/geometry tests are part of
`make -C sw/test/compiler_runtime check` (1,000 additional geometries). Host
stubs do not validate RVV instructions; the following simulation gate does:

Each case times only the legacy transfer and the batched transfer (including
their final fence), excluding initialization and the independent byte oracle.
`copy_batch_cycles.csv` records these cycle counts beside the simulation results;
performance is reported, not asserted to a fixed RTL-independent threshold.

```sh
make -j10 -C sw/test/spatz_ops spatz_ops_copy_batch.bin
make -j10 -C hw/rtl/cluster sim \
  COCOTB_TEST_MODULES=test_spatz_operator_library \
  COCOTB_TESTCASE=test_spatz_op_copy_batch \
  CLUSTER_SIM_NAME=test_spatz_copy_batch
```

Runtime local DMA2D/3D commands use the batching helper; external transfers
continue using iDMA and its existing queue/completion protocol. Aligned,
non-overlapping blocks of at most 32 bytes use `vlse32`/`vsse32` across pixels,
with a scalar-word broadcast for source stride zero. Byte tails and overlapping
views retain the legacy per-run order; no memmove guarantee or RTL change is
introduced. The compiler separately normalizes geometry and forwards safe
read-only native-layout views before constructing scheduling dependencies.

## Verified batch-copy measurements

The 2026-10-26 ELF-loaded gate passed all twelve byte-oracle/canary cases with
unchanged RTL. Retained evidence is under
`hw/rtl/cluster/tb/sim/test_local_copy_batch_20261026_elf/`: `results.xml`,
`sim.log`, `artifact-sha256.txt` and `copy_batch_cycles.csv`.

| Case | Geometry | Legacy cycles | Batch cycles |
|---|---|---:|---:|
| 0 | C16 pack, 137 runs | 4,962 | 1,378 |
| 1 | C16 unpack, 137 runs | 4,962 | 1,366 |
| 2 | C16 gaps on both sides | 4,962 | 1,374 |
| 3 | 32-byte blocks | 2,658 | 1,454 |
| 4 | Repeated-source 16-byte fill | 4,962 | 1,116 |
| 5 | Repeated-source 32-byte fill | 4,962 | 2,004 |
| 6 | 3D row gaps | 2,114 | 910 |
| 7 | Contiguous collapse | 6,638 | 3,950 |
| 8 | Unaligned 15-byte fallback | 2,114 | 2,296 |
| 9 | Singleton dimension | 1,002 | 424 |
| 10 | 4-byte blocks, 1,031-run VL tail | 37,146 | 2,358 |
| 11 | Adjacent strided planes | 2,114 | 758 |

This synthetic mix totals 78,596 -> 19,388 kernel cycles (4.05x); it is not a
model-weighted or end-to-end speedup. C16 cases improve by about 3.6x, while the
unaligned fallback costs 182 extra cycles (8.6%) from dispatch overhead.
Overall test PMU cycles (2,765,959) include initialization and scalar validation,
which the kernel timings exclude. Fresh model byte-exact/PMU gates remain
necessary before claiming YOLO or MobileNet improvement.

## Command

```sh
make -C sw/test/spatz_ops
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_operator_library
```
