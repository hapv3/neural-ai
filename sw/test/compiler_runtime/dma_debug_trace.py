"""Opt-in, bounded L2-to-L1 DMA event trace; never drives DUT signals."""

import csv
from pathlib import Path

from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly
from cocotb.utils import get_sim_time


BUSY_FIELDS = (
    "raw_coupler", "eh_count", "eh_fsm", "write_legalizer",
    "read_legalizer", "write_datapath", "read_datapath", "buffer",
)


def decode_busy(value):
    return "|".join(name for bit, name in enumerate(BUSY_FIELDS) if value & (1 << bit)) or "idle"


class DMAHandshakeTrace:
    """Record pre-rising-edge events, not a waveform; counts are window-local.

    Frontend IDs and backend burst addresses are recorded separately. An ND
    transfer may issue many bursts, so they must not be conflated.
    """

    def __init__(self, dut, path, max_rows=20000):
        self.dut = dut
        self.path = Path(path)
        self.max_rows = max_rows
        self.rows = 0
        self.file = None
        self.handles = {}
        controller = dut.u_npu_cluster.u_idma_ctrl_mm
        wrapper = controller.i_l2_to_l1_backend
        backend = wrapper.i_idma_backend
        transport = backend.i_idma_transport_layer
        write = transport.i_idma_obi_write
        groups = (
            (controller, (
                "a2o_next_id", "a2o_done_id", "a2o_busy", "a2o_me_busy",
                "a2o_front_valid", "a2o_front_ready", "a2o_front_req",
                "a2o_be_req_valid", "a2o_be_req_ready", "a2o_be_rsp_valid",
                "a2o_be_rsp_ready", "a2o_rsp_last", "a2o_fe_rsp_valid",
                "axi_ar_valid_o", "axi_ar_ready_i", "axi_ar_addr_o", "axi_ar_len_o",
                "axi_r_valid_i", "axi_r_ready_o", "axi_r_last_i",
                "obi_write_req_o", "obi_write_gnt_i", "obi_write_rvalid_i",
                "obi_write_addr_o", "obi_write_be_o",
            )),
            (wrapper, ("req_src_addr_i", "req_dst_addr_i", "req_length_i", "req_last_i")),
            (backend, (
                "w_valid", "w_ready", "w_req", "w_last_burst", "w_super_last",
                "w_last_ready", "w_dp_rsp_valid", "w_dp_rsp_ready",
                "w_dp_req_out_valid", "w_dp_req_out_ready", "w_dp_req_out",
                "r_dp_req_out_valid", "aw_valid_dp", "aw_ready_dp",
            )),
            (transport, ("buffer_out_valid", "buffer_out_valid_shifted")),
            (write, ("mask_out", "ready_to_write")),
        )
        # Fail before firmware starts if a required signal is not exposed.
        for owner, names in groups:
            for name in names:
                self.handles[name] = getattr(owner, name)
        self.events = {
            "job_issue": ("a2o_front_valid", "a2o_front_ready"),
            "burst_issue": ("a2o_be_req_valid", "a2o_be_req_ready"),
            "burst_response": ("a2o_be_rsp_valid", "a2o_be_rsp_ready"),
            "job_retire": ("a2o_fe_rsp_valid",),
            "axi_ar": ("axi_ar_valid_o", "axi_ar_ready_i"),
            "axi_r": ("axi_r_valid_i", "axi_r_ready_o"),
            "obi_write": ("obi_write_req_o", "obi_write_gnt_i"),
            "obi_response": ("obi_write_rvalid_i",),
            "write_meta": ("w_valid", "w_ready"),
            "last_pop": ("w_dp_rsp_valid", "w_dp_rsp_ready"),
        }
        self.counts = dict.fromkeys(self.events, 0)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    async def run(self, start_cycle=0, cycles=1000000):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", newline="", buffering=1)
        fields = ["time_ps", "events", "busy_fields"]
        fields += [f"count_{event}" for event in self.events]
        fields += list(self.handles)
        writer = csv.DictWriter(self.file, fieldnames=fields)
        writer.writeheader()
        try:
            if start_cycle:
                await ClockCycles(self.dut.clk_i, start_cycle)
            last_state = None
            for cycle in range(cycles):
                # Interfaces are rising-edge synchronous. Read stable values
                # before that edge, not the FIFO state updated after it.
                await FallingEdge(self.dut.clk_i)
                await ReadOnly()
                cache = {}

                def value(name):
                    if name not in cache:
                        cache[name] = int(self.handles[name].value)
                    return cache[name]

                events = [name for name, signals in self.events.items()
                          if all(value(signal) for signal in signals)]
                for event in events:
                    self.counts[event] += 1
                state = tuple(value(name) for name in ("a2o_next_id", "a2o_done_id", "a2o_busy"))
                if events or state != last_state or cycle % 10000 == 0:
                    row = {name: value(name) for name in self.handles}
                    row.update({f"count_{name}": count for name, count in self.counts.items()})
                    row.update(time_ps=int(get_sim_time(unit="ps")),
                               events="|".join(events) or "snapshot",
                               busy_fields=decode_busy(state[2]))
                    writer.writerow(row)
                    self.rows += 1
                    if self.rows >= self.max_rows:
                        self.dut._log.warning("DMA trace row limit reached: %s", self.path)
                        break
                last_state = state
        finally:
            self.close()
