import cocotb
from cocotb.triggers import RisingEdge

from systolic_independent_common import (
    boot_and_run_independent_systolic,
    check_independent_systolic_output,
    signal_to_int,
)


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


@cocotb.test()
async def test_systolic_ofm_fifo_highwater(dut):
    fifo_stats = {"done": False, "max_usage": 0, "max_write_idx": 0, "max_read_idx": 0}
    monitor_task = cocotb.start_soon(monitor_ofm_fifo(dut, fifo_stats))

    debug = await boot_and_run_independent_systolic(dut, __file__)
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
    await check_independent_systolic_output(dut)
