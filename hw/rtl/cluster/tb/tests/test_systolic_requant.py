import cocotb

from npu_test_utils import read_l2_bytes, write_l2_bytes
from systolic_independent_common import (
    K64_M,
    ifm_value,
    ifm_k64_value,
    make_ifm_bytes,
    make_ifm_k64_bytes,
    make_weight_bytes,
    make_weight_k64_bytes,
    weight_value,
    weight_k64_value,
)

L2_WEIGHT = 0x80000000
L2_IFM = 0x80001000
L2_OUT = 0x80018000
L2_WEIGHT_K64 = 0x80040000
L2_IFM_K64 = 0x80041000
L2_OUT_K64 = 0x8001A000
DIMS = [1, 2, 31, 32, 33, 64]
MAX_M = 64


def qparams(channel):
    return {
        "bias": (channel - 16) * 3,
        "multiplier": (channel % 5) + 1,
        "shift": channel % 4,
        "zero_point": (channel % 7) - 3,
        "clamp_min": -50,
        "clamp_max": 60,
    }


def round_shift_right(value, shift):
    if shift == 0:
        return value
    offset = 1 << (shift - 1)
    if value >= 0:
        return (value + offset) >> shift
    return -(((-value) + offset) >> shift)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def requant(acc, channel):
    params = qparams(channel)
    biased = acc + params["bias"]
    scaled = biased * params["multiplier"]
    rounded = round_shift_right(scaled, params["shift"])
    with_zp = rounded + params["zero_point"]
    return clamp(with_zp, params["clamp_min"], params["clamp_max"]) & 0xFF


def golden_for_dim(dim_m):
    out = []
    for m in range(dim_m):
        for n in range(32):
            acc = 0
            for k in range(32):
                acc += ifm_value(m, k) * weight_value(k, n)
            out.append(requant(acc, n))
    return out


def golden_k64_requant():
    out = []
    for m in range(K64_M):
        for n in range(32):
            acc = 0
            for block in range(2):
                for k in range(32):
                    acc += ifm_k64_value(block, m, k) * weight_k64_value(block, k, n)
            out.append(requant(acc, n))
    return out


async def boot_and_run_systolic_requant(dut, test_file):
    from cocotb.clock import Clock
    from cocotbext.axi import AxiLiteBus, AxiLiteMaster
    from npu_test_utils import firmware_path, load_firmware_axi, release_fetch, reset_dut, wait_for_host_irq
    import os

    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())
    axi_master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)

    fw_path = firmware_path(test_file, "sw/test/systolic_requant/systolic_requant.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/test/systolic_requant` first."
    await reset_dut(dut)
    await load_firmware_axi(axi_master, fw_path)
    await write_l2_bytes(dut, L2_WEIGHT, make_weight_bytes())
    await write_l2_bytes(dut, L2_IFM, make_ifm_bytes()[: MAX_M * 32])
    await write_l2_bytes(dut, L2_WEIGHT_K64, make_weight_k64_bytes())
    await write_l2_bytes(dut, L2_IFM_K64, make_ifm_k64_bytes())
    await release_fetch(dut)
    await wait_for_host_irq(dut, timeout_cycles=80000)


@cocotb.test()
async def test_systolic_requant(dut):
    await boot_and_run_systolic_requant(dut, __file__)

    offset = 0
    for dim_m in DIMS:
        expected = golden_for_dim(dim_m)
        got = await read_l2_bytes(dut, L2_OUT + offset, len(expected))
        for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
            assert got_byte == exp_byte, (
                f"requant M={dim_m} byte {idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"
            )
        offset += len(expected)

    expected = golden_k64_requant()
    got = await read_l2_bytes(dut, L2_OUT_K64, len(expected))
    for idx, (got_byte, exp_byte) in enumerate(zip(got, expected)):
        assert got_byte == exp_byte, (
            f"accumulated requant K=64 byte {idx}: got=0x{got_byte:02x} expected=0x{exp_byte:02x}"
        )
