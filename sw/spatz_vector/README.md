# Spatz Vector Firmware Tests

Dedicated RVV firmware tests for proving that the Snitch frontend offloads vector
instructions to Spatz and that Spatz reaches shared TCDM through the VLSU ports.

## Build

```sh
make -C sw/spatz_vector
```

The suite uses the Spatz-local LLVM toolchain by default:

- Compiler: `hw/spatz/install/llvm/bin/clang`
- Objcopy/objdump: `hw/spatz/install/llvm/bin/llvm-objcopy` and `llvm-objdump`
- Hex conversion: `hw/spatz/install/riscv-gcc/bin/riscv32-unknown-elf-objcopy`

The firmware is assembled with `-march=rv32im_zve32x -mabi=ilp32`, so RVV
instructions are written as mnemonics instead of raw `.word` encodings.

## Current Test

`tests/basic_mem_arith.S` covers the baseline Spatz-vector execution path:

- Vector config: `vsetvli`
- Unit-stride memory: `vle32.v`, `vse32.v`
- Integer vector ALU: `vadd.vv`, `vsub.vv`
- Logical vector ALU: `vand.vv`, `vor.vv`, `vxor.vv`
- Immediate shifts: `vsll.vi`, `vsrl.vi`

The firmware verifies every vector store with scalar loads. It writes
`0xDEADBEEF` to `0x10008000` on success. On failure it writes
`0xBAD00000 | test_id` plus debug words for failing subtest, lane index, got, and
expected values. The cocotb test also reads the TCDM SRAM banks directly and
checks the full output vectors.

## RTL Regression

```sh
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_spatz_vector_basic
```

## Coverage Roadmap

The suite is intentionally split-ready: add one `.S` file per instruction group
and one cocotb wrapper or parameterized loader per binary.

- Config: `vsetivli`, `vsetvl`
- Memory widths: `vle8/16/32`, `vse8/16/32`
- Strided/indexed memory: `vlse/vsse`, `vluxei/vsuxei`, `vloxei/vsoxei`
- Arithmetic: `vmin/vminu`, `vmax/vmaxu`, `vmul`, `vmulh/u/su`
- Divide/remainder: `vdiv/u`, `vrem/u`
- Compare/mask: `vmseq`, `vmsne`, `vmslt/u`, `vmsle/u`, `vmsgt/u`
- Slide/move: `vslideup/down`, `vslide1up/down`, `vmv`
- Reductions and mask operations as supported by the integrated Spatz config
- FP vector instructions are negative/illegal tests in the current integer-only
  config (`N_FPU=0`, `RVF=0`, `RVD=0`)
