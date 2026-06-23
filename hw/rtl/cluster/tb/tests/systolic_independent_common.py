import os

import cocotb
from cocotb.clock import Clock

from npu_test_utils import (
    firmware_path,
    hold_reset,
    load_firmware_tcm_backdoor,
    read_l2_bytes,
    release_reset,
    wait_for_status,
    write_dtcm_word,
    write_l2_bytes,
)


L2_WEIGHT = 0x80000000
L2_IFM = 0x80001000
L2_OUT = 0x80010000
SIG_START_WORD = 8
DIMS = [1, 2, 31, 32, 33, 64, 128, 1024]
MAX_M = 1024


def signal_to_int(signal):
    value = signal.value
    if not value.is_resolvable:
        return None
    if hasattr(value, "to_unsigned"):
        return value.to_unsigned()
    return int(value)


def to_u8(value):
    return value & 0xFF


def to_i8(value):
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def s32_to_bytes(value):
    return [(value >> shift) & 0xFF for shift in (0, 8, 16, 24)]


def weight_value(k, n):
    return ((k * 7 + n * 3) % 9) - 4


def ifm_value(m, k):
    return ((m * 5 + k * 11) % 11) - 5


def make_weight_bytes():
    return [to_u8(weight_value(k, n)) for k in range(32) for n in range(32)]


def make_ifm_bytes():
    return [to_u8(ifm_value(m, k)) for m in range(MAX_M) for k in range(32)]


def golden_for_dim(dim_m):
    out = []
    for m in range(dim_m):
        for n in range(32):
            acc = 0
            for k in range(32):
                acc += ifm_value(m, k) * weight_value(k, n)
            out.extend(s32_to_bytes(acc & 0xFFFFFFFF))
    return out


async def boot_and_run_independent_systolic(dut, test_file):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    await hold_reset(dut)
    fw_path = firmware_path(test_file, "sw/test/independent_systolic/independent_systolic.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/independent_systolic` first."
    load_firmware_tcm_backdoor(dut, fw_path)
    await release_reset(dut)

    await write_l2_bytes(dut, L2_WEIGHT, make_weight_bytes())
    await write_l2_bytes(dut, L2_IFM, make_ifm_bytes())
    write_dtcm_word(dut, SIG_START_WORD, 1)

    return await wait_for_status(dut, expected_pass_count=1 + len(DIMS), timeout_cycles=200000)


async def check_independent_systolic_output(dut):
    offset = 0
    for dim_m in DIMS:
        expected = golden_for_dim(dim_m)
        got = await read_l2_bytes(dut, L2_OUT + offset, len(expected))
        for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
            assert got_byte == exp_byte, (
                f"GEMM32 M={dim_m} byte {idx}: "
                f"got=0x{got_byte:02x} expected=0x{exp_byte:02x}"
            )
        offset += len(expected)
