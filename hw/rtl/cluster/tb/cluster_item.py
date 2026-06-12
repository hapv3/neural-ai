class ClusterTestItem:
    """
    Đại diện cho một transaction DMA truyền dữ liệu vào Cluster.
    Bao gồm thông tin cấu hình DMA và payload (dữ liệu kỳ vọng).
    """
    def __init__(self, src_addr, dst_addr, length, payloads):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.length = length
        self.payloads = payloads # Danh sách các số 256-bit sẽ được đẩy vào RAM
