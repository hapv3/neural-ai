import random

class SystolicTestItem:
    """
    Chứa dữ liệu (Transactions) phục vụ cho quá trình test.
    Đại diện cho cấu hình (Configuration), dữ liệu đầu vào (Input Data),
    và dữ liệu kỳ vọng (Expected Data).
    """
    def __init__(self, array_dim=32):
        self.array_dim = array_dim
        
        # Tạo ma trận ngẫu nhiên cho Trọng số (Weights) và Đầu vào (IFMs)
        self.weights = [[random.randint(0, 5) for _ in range(array_dim)] for _ in range(array_dim)]
        self.ifms = [[random.randint(0, 10) for _ in range(array_dim)] for _ in range(array_dim)]
        
        # Ma trận lưu dữ liệu đầu ra kỳ vọng (Expected OFMs)
        self.expected_ofms = [[0 for _ in range(array_dim)] for _ in range(array_dim)]
        
    def pack_weights_for_cycle(self, cycle):
        """
        Đóng gói 32 phần tử 8-bit của 1 hàng (hoặc cột) thành 1 số integer 256-bit
        để đẩy vào port weight_data_i của DUT.
        """
        val = 0
        for c in range(self.array_dim):
            w = self.weights[cycle][c] & 0xFF
            val |= (w << (c * 8))
        return val

    def pack_ifms_for_cycle(self, cycle):
        """
        Đóng gói 32 phần tử 8-bit thành 1 số integer 256-bit
        để đẩy vào port ifm_data_i của DUT.
        """
        val = 0
        for r in range(self.array_dim):
            i_val = self.ifms[cycle][r] & 0xFF
            val |= (i_val << (r * 8))
        return val
