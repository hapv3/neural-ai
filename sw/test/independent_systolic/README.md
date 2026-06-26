# Independent Systolic Test

## Target

Verify the systolic HAL path without graph scheduler dependencies. The test
compares full `M x 32` INT32 output tensors for boundary `M` values and one
two-block `K=64` accumulation fixture:

`1, 2, 31, 32, 33, 64, 128, 1024`.

## Scenario

1. Cocotb writes deterministic signed INT8 weights and IFM fixtures to L2.
2. Firmware stages fixed `32 x 32` weights and max `1024 x 32` IFM into TCDM.
3. Firmware calls `systolic_gemm32()` for each `M`.
4. Firmware runs `K=64` as one base GEMM32 block followed by one accumulated
   GEMM32 block through the psum path.
5. HAL handles safe tiling for large `M` where needed.
6. Firmware copies every INT32 OFM row back to L2.
7. Cocotb compares all returned words against Python golden output.

## Command

```sh
make -C sw/test/independent_systolic
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_systolic
```
