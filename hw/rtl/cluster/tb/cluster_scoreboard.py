class ClusterScoreboard:
    """
    Scoreboard so sánh kết quả thực tế trên mô phỏng với dữ liệu kỳ vọng từ Test Item.
    """
    def __init__(self, dut, monitor):
        self.dut = dut
        self.monitor = monitor

    def check_dma_result(self, item, tcdm_word_addr):
        """
        Kiểm tra xem dữ liệu trong TCDM Banks có khớp với dữ liệu đã được đẩy vào DMA hay không.
        Mỗi payload 256-bit trong item.payloads tương ứng với 1 Bank.
        """
        if len(item.payloads) > 0:
            val0 = self.monitor.read_tcdm_bank(0, tcdm_word_addr)
            self.dut._log.info(f"[Scoreboard] Expected Bank 0: {hex(item.payloads[0])}")
            self.dut._log.info(f"[Scoreboard] Data in TCDM Bank 0: {hex(val0)}")
            assert val0 == item.payloads[0], f"Mismatch in Bank 0! Expected: {hex(item.payloads[0])}, Got: {hex(val0)}"

        if len(item.payloads) > 1:
            val1 = self.monitor.read_tcdm_bank(1, tcdm_word_addr)
            self.dut._log.info(f"[Scoreboard] Expected Bank 1: {hex(item.payloads[1])}")
            self.dut._log.info(f"[Scoreboard] Data in TCDM Bank 1: {hex(val1)}")
            assert val1 == item.payloads[1], f"Mismatch in Bank 1! Expected: {hex(item.payloads[1])}, Got: {hex(val1)}"

        self.dut._log.info("[Scoreboard] Item passed successfully!")
