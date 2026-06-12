import cocotb
from cocotb.clock import Clock

# Import components
from systolic_item import SystolicTestItem
from systolic_driver import SystolicDriver
from systolic_monitor import SystolicMonitor
from systolic_scoreboard import SystolicScoreboard

@cocotb.test()
async def systolic_array_test(dut):
    """
    Kịch bản Test UVM-like Đầy đủ:
    Driver -> DUT -> Monitor -> Scoreboard
    """
    
    # 1. Khởi tạo xung nhịp 100MHz
    clock = Clock(dut.clk_i, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # 2. Khởi tạo các Components
    item = SystolicTestItem(array_dim=32)
    driver = SystolicDriver(dut, clock)
    scoreboard = SystolicScoreboard(dut, item)
    monitor = SystolicMonitor(dut, scoreboard)  # Link Monitor với Scoreboard
    
    dut._log.info("==================================================")
    dut._log.info("[TEST] Khởi chạy mô phỏng UVM-like hoàn chỉnh (có Monitor)")
    dut._log.info("==================================================")
    
    # 3. Chạy Monitor dưới dạng Background Task (Coroutine ngầm)
    cocotb.start_soon(monitor.start())
    
    # 4. Scoreboard tính toán Golden Model trước
    scoreboard.calculate_golden()
    
    # 5. Driver chủ động đẩy tín hiệu
    await driver.reset()
    await driver.load_weights(item)
    await driver.stream_ifms(item)
    
    # 6. Đợi một khoảng thời gian để luồng pipeline xả hết dữ liệu
    # Monitor sẽ tự động bắt được tín hiệu và gọi scoreboard.compare()
    dut._log.info("[TEST] Đợi Pipeline đẩy nốt dữ liệu cuối cùng...")
    for _ in range(50):
        await cocotb.triggers.RisingEdge(dut.clk_i)
        
    dut._log.info("==================================================")
    dut._log.info("[TEST] Hoàn thành luồng kiểm thử!")
    dut._log.info("==================================================")
