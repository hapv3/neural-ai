import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

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


async def monitor_ofm_fifo(dut, stats):
    ctrl = dut.u_npu_cluster.u_sys_ctrl
    fifo = ctrl.i_ofm_fifo

    while not stats["done"]:
        await RisingEdge(dut.clk_i)
        usage = signal_to_int(ctrl.ofm_fifo_usage)
        write_idx = signal_to_int(fifo.write_pointer_q)
        read_idx = signal_to_int(fifo.read_pointer_q)

        if usage is not None:
            stats["max_usage"] = max(stats["max_usage"], usage)
        if write_idx is not None:
            stats["max_write_idx"] = max(stats["max_write_idx"], write_idx)
        if read_idx is not None:
            stats["max_read_idx"] = max(stats["max_read_idx"], read_idx)


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


@cocotb.test()
async def test_independent_systolic(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    await hold_reset(dut)
    fw_path = firmware_path(__file__, "sw/independent_systolic_test/independent_systolic.bin")
    assert os.path.exists(fw_path), "Run `make -C sw/independent_systolic_test` first."
    load_firmware_tcm_backdoor(dut, fw_path)
    await release_reset(dut)

    await write_l2_bytes(dut, L2_WEIGHT, make_weight_bytes())
    await write_l2_bytes(dut, L2_IFM, make_ifm_bytes())
    write_dtcm_word(dut, SIG_START_WORD, 1)

    fifo_stats = {"done": False, "max_usage": 0, "max_write_idx": 0, "max_read_idx": 0}
    monitor_task = cocotb.start_soon(monitor_ofm_fifo(dut, fifo_stats))
    debug = await wait_for_status(dut, expected_pass_count=1 + len(DIMS), timeout_cycles=200000)
    fifo_stats["done"] = True
    await RisingEdge(dut.clk_i)
    monitor_task.cancel()
    dut._log.info(f"systolic suite passed: {debug}")
    dut._log.info(
        "ofm fifo high-water: "
        f"usage={fifo_stats['max_usage']} "
        f"write_idx={fifo_stats['max_write_idx']} "
        f"read_idx={fifo_stats['max_read_idx']}"
    )

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
