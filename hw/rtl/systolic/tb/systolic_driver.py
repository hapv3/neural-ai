from cocotb.triggers import RisingEdge, Timer

class SystolicDriver:
    """
    Driver chịu trách nhiệm tạo ra các tín hiệu (Stimulus) điều khiển
    Device Under Test (DUT) theo trình tự mô phỏng (Reset, Load, Stream).
    """
    def __init__(self, dut, clock):
        self.dut = dut
        self.clock = clock
        self.array_dim = 32

    async def reset(self):
        self.dut._log.info("[Driver] Thực hiện quá trình Reset...")
        self.dut.rst_ni.value = 0
        self.dut.weight_load_en_i.value = 0
        self.dut.clear_acc_i.value = 0
        self.dut.compute_en_i.value = 0
        self.dut.weight_data_i.value = 0
        self.dut.ifm_data_i.value = 0
        self.dut.psum_data_i.value = 0
        
        await Timer(20, units="ns")
        self.dut.rst_ni.value = 1
        await RisingEdge(self.dut.clk_i)
        
    async def load_weights(self, item):
        self.dut._log.info("[Driver] Bắt đầu nạp Trọng số (Weight-Stationary)...")
        self.dut.weight_load_en_i.value = 1
        
        for i in range(self.array_dim):
            self.dut.weight_data_i.value = item.pack_weights_for_cycle(i)
            await RisingEdge(self.dut.clk_i)
            
        self.dut.weight_load_en_i.value = 0
        self.dut._log.info("[Driver] Đã nạp xong trọng số!")

    async def stream_ifms(self, item):
        self.dut._log.info("[Driver] Bắt đầu stream IFMs và cộng dồn MAC...")
        self.dut.clear_acc_i.value = 1
        self.dut.compute_en_i.value = 1
        
        for i in range(self.array_dim):
            self.dut.ifm_data_i.value = item.pack_ifms_for_cycle(i)
            await RisingEdge(self.dut.clk_i)
            self.dut.clear_acc_i.value = 0 # Chỉ clear ở chu kỳ đầu tiên
            
        self.dut.compute_en_i.value = 0
        self.dut._log.info("[Driver] Hoàn tất quá trình đẩy dữ liệu.")
