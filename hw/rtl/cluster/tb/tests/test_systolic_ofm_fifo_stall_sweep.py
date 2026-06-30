import cocotb
from cocotb.triggers import RisingEdge

from systolic_independent_common import (
    boot_and_run_independent_systolic,
    check_independent_systolic_output,
    signal_to_int,
)


async def monitor_ofm_stall(dut, stats):
    ctrl = dut.u_npu_cluster.u_sys_ctrl
    fifo = ctrl.i_ofm_fifo
    cluster = dut.u_npu_cluster

    while not stats["done"]:
        await RisingEdge(dut.clk_i)

        usage = signal_to_int(ctrl.ofm_fifo_usage)
        write_idx = signal_to_int(fifo.write_pointer_q)
        read_idx = signal_to_int(fifo.read_pointer_q)
        full = signal_to_int(ctrl.ofm_fifo_full)
        ofm_valid = signal_to_int(cluster.sys_ofm_valid)
        ofm_ready = signal_to_int(cluster.sys_ofm_ready)
        otcdm_req = signal_to_int(cluster.sys_obi_o_req)
        otcdm_gnt = signal_to_int(cluster.sys_obi_o_gnt)
        stall_active = signal_to_int(cluster.sys_otcdm_stall_active)

        if usage is not None:
            stats["max_usage"] = max(stats["max_usage"], usage)
        if write_idx is not None:
            stats["max_write_idx"] = max(stats["max_write_idx"], write_idx)
        if read_idx is not None:
            stats["max_read_idx"] = max(stats["max_read_idx"], read_idx)
        if full == 1:
            stats["full_cycles"] += 1
        if ofm_valid == 1 and ofm_ready == 0:
            stats["array_backpressure_cycles"] += 1
        if stall_active == 1:
            stats["stall_active_cycles"] += 1
        if otcdm_req is not None and otcdm_gnt is not None:
            if otcdm_req != 0 and (otcdm_req & (~otcdm_gnt & 0xF)) != 0:
                stats["otcdm_req_stall_cycles"] += 1


@cocotb.test()
async def test_systolic_ofm_fifo_stall_sweep(dut):
    stats = {
        "done": False,
        "max_usage": 0,
        "max_write_idx": 0,
        "max_read_idx": 0,
        "full_cycles": 0,
        "array_backpressure_cycles": 0,
        "stall_active_cycles": 0,
        "otcdm_req_stall_cycles": 0,
    }
    monitor_task = cocotb.start_soon(monitor_ofm_stall(dut, stats))

    debug = await boot_and_run_independent_systolic(dut, __file__)
    stats["done"] = True
    await RisingEdge(dut.clk_i)
    monitor_task.cancel()

    dut._log.info(f"systolic stall sweep suite passed: {debug}")
    dut._log.info(
        "OFM_FIFO_SWEEP_RESULT "
        f"max_usage={stats['max_usage']} "
        f"full_cycles={stats['full_cycles']} "
        f"array_backpressure_cycles={stats['array_backpressure_cycles']} "
        f"stall_active_cycles={stats['stall_active_cycles']} "
        f"otcdm_req_stall_cycles={stats['otcdm_req_stall_cycles']} "
        f"write_idx={stats['max_write_idx']} "
        f"read_idx={stats['max_read_idx']}"
    )

    assert stats["stall_active_cycles"] > 0, "O-TCDM stall injection was not active."
    assert stats["otcdm_req_stall_cycles"] > 0, "No O-TCDM request observed under injected stall."
    await check_independent_systolic_output(dut)
