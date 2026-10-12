"""Byte-exact automatic KGEN accumulation across the internal PSUM boundary."""

import cocotb
import numpy as np

from systolic_controller_case_utils import (
    REG_SYS_OFM_ROW_STRIDE, REG_SYS_OFM_TILE_COLS, REG_SYS_PSUM_ROW_STRIDE,
    configure_identity_requant, disable_optional_modes, mmio_write,
    program_linebuf, start_and_wait, start_clock_and_reset,
)


@cocotb.test()
async def external_psum_multik_boundary_and_yolo_geometry(dut):
    await start_clock_and_reset(dut)
    # Repeated jobs without resetting RTL also exercise first-tile pointer
    # selection when the preceding job left its K index on the last tile.
    for oh, ow, channels in ((3, 85, 16), (4, 64, 32), (1, 257, 16),
                             (7, 80, 16), (7, 80, 32)):
        rows = oh * ow
        ih, iw = (oh - 1) * 2 + 3, (ow - 1) * 2 + 3
        weights_base, input_base, output_base, psum_base = 0x1000, 0x10000, 0x30000, 0x50000
        output_stride, psum_stride = ow * 32 + 64, ow * 128 + 96
        x = np.fromfunction(lambda y, z, c: (3*y + 2*z + c) % 9 - 4,
                            (ih, iw, 32), dtype=int).astype(np.int8)
        x[:, :, channels:] = 99
        w = np.fromfunction(lambda t, c, n: (3*t + c + n) % 5 - 2,
                            (9, 32, 32), dtype=int).astype(np.int8)
        w[:, channels:, :] = 0
        image = bytearray([0xA5] * (512 * 1024))
        image[weights_base:weights_base+w.size] = w.tobytes()
        image[input_base:input_base+x.size] = x.tobytes()
        for address in range(0, len(image), 32):
            dut.tcdm_mem[address >> 5].value = int.from_bytes(image[address:address+32], "little")
        await disable_optional_modes(dut)
        await configure_identity_requant(dut, shift=4)
        await program_linebuf(
            dut, input_base=input_base, input_h=ih, input_w=iw, input_c=32,
            output_h=oh, output_w=ow, kernel_h=3, kernel_w=3,
            stride_h=2, stride_w=2, coalesce=True, kgen=True,
            c32_fast=channels == 32, generic_linear_k32=channels != 32,
            k_tiles=9, block_valid_bytes=channels, coalesce_k_bytes=9*channels,
            channel_offset=ih*iw*32 if channels == 32 else 0,
            weight_addr=weights_base, ofm_addr=output_base, psum_addr=psum_base)
        await mmio_write(dut, REG_SYS_OFM_ROW_STRIDE, output_stride)
        await mmio_write(dut, REG_SYS_OFM_TILE_COLS, ow)
        await mmio_write(dut, REG_SYS_PSUM_ROW_STRIDE, psum_stride)
        stats = await start_and_wait(dut, timeout_cycles=200000)
        assert stats["compute"] == rows * 9
        expected = bytearray(image)
        for y in range(oh):
            for z in range(ow):
                patches = x[y*2:y*2+3, z*2:z*2+3].reshape(9, 32).astype(np.int64)
                contributions = np.einsum("tk,tkn->tn", patches, w.astype(np.int64))
                acc = contributions.sum(axis=0)
                rounded = (np.abs(acc) + 8) >> 4
                quantized = np.clip(np.where(acc < 0, -rounded, rounded), -128, 127).astype(np.int8)
                dst = output_base + y * output_stride + z * 32
                expected[dst:dst+32] = quantized.tobytes()
                if rows > 256:
                    dst = psum_base + y * psum_stride + z * 128
                    expected[dst:dst+128] = contributions[:8].sum(axis=0).astype("<i4").tobytes()
        # Compare all memory: numerical output, accumulated INT32 PSUM, dirty
        # PSUM untouched for M<=256, stride gaps and IFM/weight guards.
        for address in range(0, len(expected), 32):
            actual = int(dut.tcdm_mem[address >> 5].value).to_bytes(32, "little")
            assert actual == expected[address:address+32], (
                f"M={rows} C={channels} mismatch at TCDM {address:#x}: "
                f"actual={actual.hex()} expected={expected[address:address+32].hex()}")
        dut._log.info("Byte-exact automatic PSUM PASS M=%d C=%d, full TCDM checked", rows, channels)
