# Boot Smoke Test

## Target

Validate the minimum boot path needed before deeper subsystem tests run:

- Host loads `boot.bin` into I-TCM.
- Snitch fetches and executes from I-TCM.
- Firmware writes D-TCM status words visible to cocotb.
- Snitch accesses iDMA-compatible MMIO registers.
- TCDM-to-TCDM local copy works through the current iDMA runtime API.

## Scenario

1. Seed `0x10100000` with deterministic `0xCAFEBABE + i` words.
2. Clear `0x10100100` so stale memory cannot pass.
3. Write and read back `IDMA_LENGTH_LOW` as an MMIO smoke test.
4. Run `idma_L1ToL1()` for a 64-byte local copy.
5. Compare every copied word.
6. Publish `0xDEADBEEF` on success or `0xBADBADxx` with debug words on failure.

## Command

```sh
make -C sw/test/boot
env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp/ccache-tmp \
  make -C hw/rtl/cluster sim COCOTB_TEST_MODULES=test_snitch_boot
```
