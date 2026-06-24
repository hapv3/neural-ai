# Independent Memory Test

## Target

Verify memory movement and TCDM address decoding independently from systolic and
Spatz compute blocks. This is the baseline suite for DMA/TCDM refactors.

## Scenario

1. Cocotb writes deterministic L2 fixtures after reset.
2. Cocotb loads firmware into I-TCM and releases `fetch_enable_i`.
3. `L2 -> TCDM` 1D transfer is verified inside firmware.
4. `TCDM -> L2` 1D transfer is verified by cocotb in external memory.
5. Firmware probes low/high representatives for all 16 TCDM banks.
6. `L2 -> TCDM` 2D and 3D transfers are verified inside firmware.
7. `TCDM -> L2` 2D and 3D transfers are verified by cocotb.

## Notes

- Source and destination strides intentionally differ in 2D/3D cases.
- Output-side copies are checked by cocotb to avoid firmware self-aliasing.
- Firmware reports completion through `NPU_IRQ_HOST_NOTIFY`; local phase/op words are firmware-private diagnostics.

## Command

```sh
make -C sw/test/independent_memory
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_independent_memory
```
