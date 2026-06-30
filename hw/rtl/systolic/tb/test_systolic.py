import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

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

    assert scoreboard.compare_count > 0, "Scoreboard did not capture any OFM output."
    assert scoreboard.last_error_count == 0, (
        f"Systolic output mismatch: {scoreboard.last_error_count} columns differ."
    )
        
    dut._log.info("==================================================")
    dut._log.info("[TEST] Hoàn thành luồng kiểm thử!")
    dut._log.info("==================================================")


@cocotb.test()
async def ofm_ready_backpressure_hold_test(dut):
    """
    Verify true output ready/valid behavior at array level.
    When ofm_valid_o is asserted and ofm_ready_i is deasserted, the array must
    hold output valid/data stable instead of dropping or advancing rows.
    """

    clock = Clock(dut.clk_i, 10, units="ns")
    cocotb.start_soon(clock.start())

    item = SystolicTestItem(array_dim=32)
    driver = SystolicDriver(dut, clock)

    await driver.reset()
    dut.ofm_ready_i.value = 1
    await driver.load_weights(item)
    await driver.stream_ifms(item)
    dut.ofm_ready_i.value = 0

    for _ in range(128):
        await RisingEdge(dut.clk_i)
        if dut.ofm_valid_o.value.is_resolvable and int(dut.ofm_valid_o.value) == 1:
            break
    else:
        raise AssertionError("OFM valid did not assert before timeout.")

    held_data = int(dut.ofm_data_o.value)

    for _ in range(5):
        await RisingEdge(dut.clk_i)
        assert int(dut.ofm_valid_o.value) == 1, "OFM valid dropped while ready was low."
        assert int(dut.ofm_data_o.value) == held_data, "OFM data changed while ready was low."

    dut.ofm_ready_i.value = 1
    await RisingEdge(dut.clk_i)
