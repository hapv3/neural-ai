import cocotb
from cocotb.triggers import RisingEdge

from systolic_independent_common import (
    boot_and_run_independent_systolic,
    check_independent_systolic_output,
    signal_to_int,
)


async def monitor_small_ofm_fifo(dut, stats):
    ctrl = dut.u_npu_cluster.u_sys_ctrl

    while not stats["done"]:
        await RisingEdge(dut.clk_i)

        usage = signal_to_int(ctrl.ofm_fifo_usage)
        full = signal_to_int(ctrl.ofm_fifo_full)
        valid = signal_to_int(dut.u_npu_cluster.sys_ofm_valid)
        ready = signal_to_int(dut.u_npu_cluster.sys_ofm_ready)

        if usage is not None:
            stats["max_usage"] = max(stats["max_usage"], usage)
        if full == 1:
            stats["full_cycles"] += 1
        if valid == 1 and ready == 0:
            stats["backpressure_cycles"] += 1


@cocotb.test()
async def test_systolic_ofm_fifo_small_depth(dut):
    stats = {
        "done": False,
        "max_usage": 0,
        "full_cycles": 0,
        "backpressure_cycles": 0,
    }
    monitor_task = cocotb.start_soon(monitor_small_ofm_fifo(dut, stats))

    debug = await boot_and_run_independent_systolic(dut, __file__)
    stats["done"] = True
    await RisingEdge(dut.clk_i)
    monitor_task.cancel()

    dut._log.info(f"systolic small-depth suite passed: {debug}")
    dut._log.info(
        "small OFM FIFO stats: "
        f"max_usage={stats['max_usage']} "
        f"full_cycles={stats['full_cycles']} "
        f"backpressure_cycles={stats['backpressure_cycles']}"
    )
    assert stats["backpressure_cycles"] > 0, "Small OFM FIFO did not exercise array backpressure."
    await check_independent_systolic_output(dut)
