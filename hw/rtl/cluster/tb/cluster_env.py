import cocotb
from cocotb.triggers import Timer, RisingEdge
from cluster_driver import ClusterDriver
from cluster_monitor import ClusterMonitor
from cluster_scoreboard import ClusterScoreboard

class ClusterEnv:
    """
    Environment (Môi trường test):
    Chịu trách nhiệm khởi tạo các instance (Driver, Monitor, Scoreboard)
    và kết nối chúng lại với nhau cũng như với DUT.
    """
    def __init__(self, dut, clock, apb_clock):
        self.dut = dut
        self.clock = clock
        self.apb_clock = apb_clock
        self.driver = ClusterDriver(dut, clock, apb_clock)
        self.monitor = ClusterMonitor(dut)
        self.scoreboard = ClusterScoreboard(dut, self.monitor)

    async def reset(self):
        """
        Thực hiện reset hệ thống ban đầu
        """
        self.dut.rst_ni.value = 0
        self.dut.apb_rst_ni.value = 0
        self.dut.backdoor_we_i.value = 0
        await Timer(20, units="ns")
        self.dut.rst_ni.value = 1
        self.dut.apb_rst_ni.value = 1
        await RisingEdge(self.clock)
        await RisingEdge(self.apb_clock)
        self.dut._log.info("==================================================")
        self.dut._log.info("[ENV] Phase 2.5 NPU Cluster Booted")
        self.dut._log.info("==================================================")
