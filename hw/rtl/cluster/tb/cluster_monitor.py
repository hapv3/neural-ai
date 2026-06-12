class ClusterMonitor:
    """
    Monitor nhiệm vụ quan sát kết quả xử lý của Cluster.
    Đọc dữ liệu từ TCDM Memory (Internal SRAM) để đưa vào Scoreboard kiểm tra.
    """
    def __init__(self, dut):
        self.dut = dut

    def read_tcdm_bank(self, bank_idx, word_addr):
        """
        Reads a 256-bit word from a specific TCDM bank at the given word address.
        """
        bank_mem = self.dut.u_npu_cluster.gen_sram_banks[bank_idx].u_sram_bank.mem
        return int(bank_mem[word_addr].value)
