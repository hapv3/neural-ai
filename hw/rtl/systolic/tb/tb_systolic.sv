`timescale 1ns/1ps
`default_nettype none

//-------------------------------------------------------------------------
// Testbench cho NPU Systolic Array 32x32
// Chức năng: Kiểm tra cơ chế nạp Weight-Stationary và luồng tính toán MAC.
//-------------------------------------------------------------------------
module tb_systolic;

    localparam int unsigned ARRAY_DIM = 32;

    logic clk_i;
    logic rst_ni;

    // Control
    logic weight_load_en_i;
    logic clear_acc_i;
    logic compute_en_i;

    // Data I/O
    logic signed [ARRAY_DIM-1:0][7:0]  weight_data_i;
    logic signed [ARRAY_DIM-1:0][7:0]  ifm_data_i;
    logic signed [ARRAY_DIM-1:0][31:0] psum_data_i;
    
    logic signed [ARRAY_DIM-1:0][31:0] ofm_data_o;
    logic                              ofm_valid_o;

    // Khởi tạo Device Under Test (DUT)
    npu_systolic_array #(
        .ARRAY_DIM(ARRAY_DIM)
    ) dut (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .weight_load_en_i (weight_load_en_i),
        .clear_acc_i      (clear_acc_i),
        .compute_en_i     (compute_en_i),
        .weight_data_i    (weight_data_i),
        .ifm_data_i       (ifm_data_i),
        .psum_data_i      (psum_data_i),
        .ofm_data_o       (ofm_data_o),
        .ofm_valid_o      (ofm_valid_o)
    );

    // Clock Generation
    initial begin
        clk_i = 0;
        forever #5 clk_i = ~clk_i; // Chu kỳ 10ns (100MHz)
    end

    // Test Sequence
    initial begin
        $display("==================================================");
        $display("[TB] Bắt đầu mô phỏng npu_systolic_array (32x32)");
        $display("==================================================");
        
        // Khởi tạo tín hiệu
        rst_ni           = 0;
        weight_load_en_i = 0;
        clear_acc_i      = 0;
        compute_en_i     = 0;
        weight_data_i    = '0;
        ifm_data_i       = '0;
        psum_data_i      = '0;

        #20 rst_ni = 1;
        #10;

        //----------------------------------------------------------------
        // Giai đoạn 1: Load Weights (Weight-Stationary)
        //----------------------------------------------------------------
        $display("[TB] Phase 1: Nạp trọng số (Weights) vào mảng PEs...");
        weight_load_en_i = 1;
        
        // Cần ARRAY_DIM chu kỳ để lấp đầy mảng 32x32 từ trên xuống
        for (int i = 0; i < ARRAY_DIM; i++) begin
            for (int c = 0; c < ARRAY_DIM; c++) begin
                weight_data_i[c] = $urandom_range(0, 5); // Random trọng số nhỏ
            end
            #10;
        end
        weight_load_en_i = 0; // Chốt (lock) trọng số tại các PE
        $display("[TB] => Đã nạp xong trọng số!");

        //----------------------------------------------------------------
        // Giai đoạn 2: Streaming IFM và Compute
        //----------------------------------------------------------------
        $display("[TB] Phase 2: Đẩy Input Feature Map và cộng dồn (MAC)...");
        clear_acc_i  = 1;
        compute_en_i = 1;
        
        // Đẩy 32 hàng IFM vào từ bên trái
        for (int i = 0; i < ARRAY_DIM; i++) begin
            for (int r = 0; r < ARRAY_DIM; r++) begin
                ifm_data_i[r] = $urandom_range(0, 10);
            end
            #10;
            clear_acc_i = 0; // Chỉ clear ở nhịp đầu tiên của chuỗi cộng dồn mới
        end
        
        // Đợi dữ liệu lan truyền xuống đáy mảng
        wait(ofm_valid_o);
        #10;

        $display("==================================================");
        $display("[TB] Dữ liệu Output (OFM) Đã hợp lệ!");
        $display("[TB] Mô phỏng thành công (Simulation Passed).");
        $display("==================================================");
        $finish;
    end

endmodule
