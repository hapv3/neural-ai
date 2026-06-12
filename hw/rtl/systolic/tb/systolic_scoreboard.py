class SystolicScoreboard:
    """
    Scoreboard giờ đây chuyên biệt hoá chỉ làm nhiệm vụ Kiểm tra (Check).
    Nó sẽ độc lập 100% với việc capture dữ liệu từ bus.
    1. Lấy Transaction (Test Item) để tính Golden Model.
    2. Cung cấp hàm compare() để Monitor đẩy dữ liệu thực tế vào.
    3. Đánh giá đúng sai.
    """
    def __init__(self, dut, item):
        self.dut = dut
        self.item = item
        self.array_dim = 32

    def calculate_golden(self):
        self.dut._log.info("[Scoreboard] Bắt đầu tính toán Golden Model (Expected Results)...")
        
        # Khởi tạo mảng 1D chứa kết quả mong đợi cho chu kỳ Valid đầu tiên
        self.item.expected_ofms = [0] * self.array_dim
        
        # 1. Trích xuất Trọng số đã lưu trong phần cứng (W_hw)
        # Quá trình load 32 chu kỳ sẽ đẩy weight_data_i từ trên xuống.
        # Do đó, hàng r của phần cứng sẽ chứa data của chu kỳ (31 - r)
        W_hw = [[0 for _ in range(self.array_dim)] for _ in range(self.array_dim)]
        for r in range(self.array_dim):
            for c in range(self.array_dim):
                W_hw[r][c] = self.item.weights[31 - r][c]
                
        # 2. Tính toán PSum lan truyền cho chu kỳ Valid đầu tiên (t=32)
        # Tại t=32, OFM của cột c là tổng hợp của W_hw[r][c] * IFM_in(r, t_in)
        # Với t_in = r - c (độ trễ lan truyền chéo của IFM)
        for c in range(self.array_dim):
            acc = 0
            for r in range(self.array_dim):
                ifm_time = r - c
                # Nếu ifm_time nằm trong phạm vi 32 chu kỳ đã stream
                if 0 <= ifm_time < self.array_dim:
                    w_val = W_hw[r][c]
                    i_val = self.item.ifms[ifm_time][r]
                    acc += w_val * i_val
            self.item.expected_ofms[c] = acc
            
        self.dut._log.info("[Scoreboard] Tính toán Golden Model hoàn tất.")

    def compare(self, actual_ofm):
        """Hàm callback được gọi bởi Monitor mỗi khi capture được Output."""
        self.dut._log.info(f"[Scoreboard] Nhận dữ liệu thực tế từ Monitor: {hex(int(actual_ofm))}")
        
        # Giải nén chuỗi 1024-bit của phần cứng thành mảng 32 phần tử (32-bit signed int)
        actual_ofm_int = int(actual_ofm)
        actual_array = []
        for c in range(self.array_dim):
            val = (actual_ofm_int >> (c * 32)) & 0xFFFFFFFF
            # Ép kiểu signed 32-bit
            if val & 0x80000000:
                val -= 0x100000000
            actual_array.append(val)
            
        # Đối chiếu từng cột
        error_count = 0
        for c in range(self.array_dim):
            expected = self.item.expected_ofms[c]
            actual = actual_array[c]
            if expected != actual:
                self.dut._log.error(f"[Scoreboard] Lỗi tại cột {c}: Expected={expected}, Actual={actual}")
                error_count += 1
                
        if error_count == 0:
            self.dut._log.info("[Scoreboard] PASS! Toàn bộ 32 kết quả khớp 100% với Golden Model.")
        else:
            self.dut._log.error(f"[Scoreboard] FAIL! Có {error_count} lỗi sai lệch dữ liệu.")
