import cocotb
from cocotb.triggers import RisingEdge, Timer, ReadOnly

class APBMasterDriver:
    """
    Driver xử lý giao thức cấp thấp APB Bus
    """
    def __init__(self, dut, clock):
        self.dut = dut
        self.clock = clock
        self._init_signals()

    def _init_signals(self):
        self.dut.apb_paddr_i.value = 0
        self.dut.apb_psel_i.value = 0
        self.dut.apb_penable_i.value = 0
        self.dut.apb_pwrite_i.value = 0
        self.dut.apb_pwdata_i.value = 0

    async def write_reg(self, addr, data):
        await RisingEdge(self.clock)
        # Setup phase
        self.dut.apb_paddr_i.value = addr
        self.dut.apb_pwdata_i.value = data
        self.dut.apb_pwrite_i.value = 1
        self.dut.apb_psel_i.value = 1
        self.dut.apb_penable_i.value = 0
        
        await RisingEdge(self.clock)
        # Access phase
        self.dut.apb_penable_i.value = 1
        
        while True:
            # Wait for pready
            if self.dut.apb_pready_o.value == 1:
                break
            await RisingEdge(self.clock)
            
        await RisingEdge(self.clock)
        # Clear
        self.dut.apb_psel_i.value = 0
        self.dut.apb_penable_i.value = 0

    async def read_reg(self, addr):
        await RisingEdge(self.clock)
        # Setup phase
        self.dut.apb_paddr_i.value = addr
        self.dut.apb_pwrite_i.value = 0
        self.dut.apb_psel_i.value = 1
        self.dut.apb_penable_i.value = 0
        
        await RisingEdge(self.clock)
        # Access phase
        self.dut.apb_penable_i.value = 1
        
        while True:
            # Wait for pready
            if self.dut.apb_pready_o.value == 1:
                break
            await RisingEdge(self.clock)
            
        await ReadOnly() # Wait for combinational logic to settle
        data = int(self.dut.apb_prdata_o.value)
        await RisingEdge(self.clock)
        
        # Clear
        self.dut.apb_psel_i.value = 0
        self.dut.apb_penable_i.value = 0
        return data


class ClusterDriver:
    """
    Driver xử lý ở tầng cấp cao cho Cluster Testbench.
    Gói gọn thao tác điều khiển DMA và Memory Backdoor.
    """
    def __init__(self, dut, clock, apb_clock):
        self.dut = dut
        self.clock = clock
        self.apb_clock = apb_clock
        self.apb = APBMasterDriver(dut, apb_clock)
        
    async def backdoor_mem_write(self, start_addr, data_256bit):
        """
        Writes 256-bit (32 bytes) of data into axi_sim_mem via the SV backdoor pins.
        """
        for i in range(32):
            byte_val = (data_256bit >> (i * 8)) & 0xFF
            await RisingEdge(self.clock)
            self.dut.backdoor_we_i.value = 1
            self.dut.backdoor_addr_i.value = start_addr + i
            self.dut.backdoor_data_i.value = byte_val
        
        await RisingEdge(self.clock)
        self.dut.backdoor_we_i.value = 0

    async def load_payloads(self, item):
        """
        Loads the payloads from the item into external memory using the backdoor.
        """
        self.dut._log.info(f"[Driver] Preloading RAM at {hex(item.src_addr)}")
        addr = item.src_addr
        for p in item.payloads:
            await self.backdoor_mem_write(addr, p)
            addr += 32

    async def execute_dma(self, item):
        self.dut._log.info(f"[Driver] Configuring DMA: {hex(item.src_addr)} -> {hex(item.dst_addr)} ({item.length} Bytes)")
        await self.apb.write_reg(0x00, item.src_addr)
        await self.apb.write_reg(0x04, item.dst_addr)
        await self.apb.write_reg(0x08, item.length)
        await self.apb.write_reg(0x0C, 1) # Start

    async def wait_for_dma_done(self):
        self.dut._log.info("[Driver] Polling DMA Done Register...")
        timeout = 100
        while timeout > 0:
            done = await self.apb.read_reg(0x10)
            if done == 1:
                self.dut._log.info("[Driver] DMA Done Status Detected!")
                await self.apb.write_reg(0x10, 0)
                return
            await Timer(10, units="ns")
            timeout -= 1
        raise Exception("TIMEOUT: DMA did not complete!")
        self.dut._log.info("[Driver] DMA Transfer Complete!")
