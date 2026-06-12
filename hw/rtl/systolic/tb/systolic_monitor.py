import cocotb
from cocotb.triggers import RisingEdge

class SystolicMonitor:
    """
    Monitor đóng vai trò người quan sát thụ động (Passive Observer).
    Nhiệm vụ: Liên tục theo dõi các tín hiệu đầu ra của DUT.
    Khi phát hiện tín hiệu valid_o, nó sẽ bắt (capture) dữ liệu ofm_data_o
    và gửi (push) sang Scoreboard.
    """
    def __init__(self, dut, scoreboard):
        self.dut = dut
        self.scoreboard = scoreboard

    async def start(self):
        """Khởi chạy vòng lặp vô tận chạy ngầm để bắt dữ liệu."""
        self.dut._log.info("[Monitor] Đang chạy ngầm, theo dõi bus Output...")
        while True:
            await RisingEdge(self.dut.clk_i)
            
            valid_val = self.dut.ofm_valid_o.value
            if valid_val.is_resolvable and int(valid_val) == 1:
                ofm_data = self.dut.ofm_data_o.value
                self.dut._log.info(f"[Monitor] Capture được tín hiệu OFM Valid! Gửi ngay sang Scoreboard...")
                
                # Push dữ liệu thực tế (Actual Data) cho Scoreboard xử lý
                self.scoreboard.compare(ofm_data)
                
                # Trong bài test đơn giản này, ta ngắt loop sau 1 lần nhận.
                # Thực tế Monitor sẽ chạy loop liên tục.
                break
