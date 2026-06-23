`default_nettype none
`timescale 1ns/1ps

module tb_afu;

    //----------------------------------------------------------------------
    // Signals
    //----------------------------------------------------------------------
    logic clk_i;
    logic rst_ni;

    // OBI Slave Port (Configuration)
    logic        obi_s_req;
    logic        obi_s_gnt;
    logic [31:0] obi_s_addr;
    logic        obi_s_we;
    logic [3:0]  obi_s_be;
    logic [31:0] obi_s_wdata;
    logic        obi_s_rvalid;
    logic [31:0] obi_s_rdata;

    // OBI Master Port (Memory Access)
    logic        obi_m_req;
    logic        obi_m_gnt;
    logic [31:0] obi_m_addr;
    logic        obi_m_we;
    logic [3:0]  obi_m_be;
    logic [31:0] obi_m_wdata;
    logic        obi_m_rvalid;
    logic [31:0] obi_m_rdata;

    logic        done;

    //----------------------------------------------------------------------
    // Clock & Reset
    //----------------------------------------------------------------------
    initial begin
        clk_i = 0;
        forever #5 clk_i = ~clk_i; // 100MHz clock
    end

    initial begin
        rst_ni = 0;
        #20 rst_ni = 1;
    end

    //----------------------------------------------------------------------
    // Device Under Test (DUT)
    //----------------------------------------------------------------------
    afu #(
        .ADDR_WIDTH(32),
        .DATA_WIDTH(32)
    ) dut (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .obi_s_req_i    (obi_s_req),
        .obi_s_gnt_o    (obi_s_gnt),
        .obi_s_addr_i   (obi_s_addr),
        .obi_s_we_i     (obi_s_we),
        .obi_s_be_i     (obi_s_be),
        .obi_s_wdata_i  (obi_s_wdata),
        .obi_s_rvalid_o (obi_s_rvalid),
        .obi_s_rdata_o  (obi_s_rdata),
        .obi_m_req_o    (obi_m_req),
        .obi_m_gnt_i    (obi_m_gnt),
        .obi_m_addr_o   (obi_m_addr),
        .obi_m_we_o     (obi_m_we),
        .obi_m_be_o     (obi_m_be),
        .obi_m_wdata_o  (obi_m_wdata),
        .obi_m_rvalid_i (obi_m_rvalid),
        .obi_m_rdata_i  (obi_m_rdata),
        .done_o         (done)
    );

    //----------------------------------------------------------------------
    // Mock TCDM Memory (8 KB)
    //----------------------------------------------------------------------
    logic [31:0] tcdm_mem [0:2047];

    // OBI Master to TCDM Model
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            obi_m_gnt <= 0;
            obi_m_rvalid <= 0;
            obi_m_rdata <= 0;
        end else begin
            obi_m_gnt <= 0;
            obi_m_rvalid <= 0;
            
            if (obi_m_req) begin
                obi_m_gnt <= 1; // Always grant immediately for testing
                
                if (obi_m_we) begin
                    // Write
                    int word_idx = obi_m_addr[12:2];
                    if (obi_m_be[0]) tcdm_mem[word_idx][7:0]   <= obi_m_wdata[7:0];
                    if (obi_m_be[1]) tcdm_mem[word_idx][15:8]  <= obi_m_wdata[15:8];
                    if (obi_m_be[2]) tcdm_mem[word_idx][23:16] <= obi_m_wdata[23:16];
                    if (obi_m_be[3]) tcdm_mem[word_idx][31:24] <= obi_m_wdata[31:24];
                end else begin
                    // Read
                    obi_m_rvalid <= 1;
                    obi_m_rdata <= tcdm_mem[obi_m_addr[12:2]];
                end
            end
        end
    end

    //----------------------------------------------------------------------
    // Mock Snitch Driver Tasks
    //----------------------------------------------------------------------
    task automatic write_obi(input logic [31:0] addr, input logic [31:0] data);
        @(posedge clk_i);
        obi_s_req   = 1;
        obi_s_we    = 1;
        obi_s_addr  = addr;
        obi_s_wdata = data;
        obi_s_be    = 4'hF;
        wait(obi_s_gnt);
        @(posedge clk_i);
        obi_s_req = 0;
    endtask

    task automatic read_obi(input logic [31:0] addr, output logic [31:0] data);
        @(posedge clk_i);
        obi_s_req  = 1;
        obi_s_we   = 0;
        obi_s_addr = addr;
        obi_s_be   = 4'hF;
        wait(obi_s_gnt);
        @(posedge clk_i);
        obi_s_req = 0;
        wait(obi_s_rvalid);
        data = obi_s_rdata;
    endtask

    task automatic load_lut(input logic [31:0] lut_array [0:255]);
        for (int i = 0; i < 256; i++) begin
            write_obi(i * 4, lut_array[i]);
        end
    endtask

    task automatic start_afu(
        input logic [31:0] src_ptr,
        input logic [31:0] dst_ptr,
        input logic [31:0] length,
        input logic [1:0] mode
    );
        write_obi(32'h1004, src_ptr);
        write_obi(32'h1008, dst_ptr);
        write_obi(32'h100C, length);
        write_obi(32'h1010, {30'd0, mode});
        write_obi(32'h1000, 32'd1); // Start
    endtask

    task automatic wait_done();
        logic [31:0] status;
        do begin
            read_obi(32'h1000, status);
        end while (status[0] == 0);
    endtask

    // Helper to read/write bytes directly to memory array for verification
    function void write_tcdm_byte(int byte_addr, logic [7:0] val);
        int word_idx = byte_addr / 4;
        int byte_offset = byte_addr % 4;
        case (byte_offset)
            0: tcdm_mem[word_idx][7:0]   = val;
            1: tcdm_mem[word_idx][15:8]  = val;
            2: tcdm_mem[word_idx][23:16] = val;
            3: tcdm_mem[word_idx][31:24] = val;
        endcase
    endfunction

    function logic [7:0] read_tcdm_byte(int byte_addr);
        int word_idx = byte_addr / 4;
        int byte_offset = byte_addr % 4;
        case (byte_offset)
            0: return tcdm_mem[word_idx][7:0];
            1: return tcdm_mem[word_idx][15:8];
            2: return tcdm_mem[word_idx][23:16];
            3: return tcdm_mem[word_idx][31:24];
        endcase
    endfunction

    function logic [15:0] read_tcdm_half(int byte_addr);
        logic [7:0] b0 = read_tcdm_byte(byte_addr);
        logic [7:0] b1 = read_tcdm_byte(byte_addr+1);
        return {b1, b0};
    endfunction

    function logic [31:0] read_tcdm_word(int byte_addr);
        logic [15:0] h0 = read_tcdm_half(byte_addr);
        logic [15:0] h1 = read_tcdm_half(byte_addr+2);
        return {h1, h0};
    endfunction

    //----------------------------------------------------------------------
    // Test Sequences
    //----------------------------------------------------------------------
    int errors = 0;
    logic [31:0] lut_data [0:255];
    int test_len = 256;
    int src_base = 'h100;
    int dst_base = 'h400;

    // Mathematical approximations for tests
    real e = 2.718281828459;

    function real sigmoid(real x);
        return 1.0 / (1.0 + e**(-x));
    endfunction

    initial begin
        // Initialize
        obi_s_req = 0;
        obi_s_we = 0;
        obi_s_addr = 0;
        obi_s_wdata = 0;
        obi_s_be = 0;
        for (int i=0; i<2048; i++) tcdm_mem[i] = 0;

        wait(rst_ni);
        #100;
        $display("========================================");
        $display("[AFU TB] Starting Autonomous Tests...");

        //--------------------------------------------------
        // Test 1: SiLU (8-bit)
        //--------------------------------------------------
        $display("[AFU TB] Test 1: SiLU (8-bit mode)");
        for (int i=0; i<256; i++) begin
            real x = real'(i - 128) / 16.0; // Assume Q4.4 format
            real y = x * sigmoid(x);
            int y_quant = $rtoi(y * 16.0); // Back to Q4.4
            if (y_quant > 127) y_quant = 127;
            if (y_quant < -128) y_quant = -128;
            lut_data[i] = {24'd0, 8'(y_quant)}; // Only lowest byte matters for 8-bit mode
        end
        load_lut(lut_data);

        for (int i=0; i<test_len; i++) begin
            write_tcdm_byte(src_base + i, 8'(i));
        end

        start_afu(src_base, dst_base, test_len, 2'd0);
        wait_done();

        for (int i=0; i<test_len; i++) begin
            logic [7:0] expected = lut_data[i][7:0];
            logic [7:0] actual = read_tcdm_byte(dst_base + i);
            if (expected !== actual) begin
                $display("[FAIL] SiLU idx=%d, exp=%h, act=%h", i, expected, actual);
                errors++;
            end
        end
        if (errors==0) $display("[PASS] SiLU 8-bit");

        //--------------------------------------------------
        // Test 2: Sigmoid (8-bit)
        //--------------------------------------------------
        $display("[AFU TB] Test 2: Sigmoid (8-bit mode)");
        for (int i=0; i<256; i++) begin
            real x = real'(i - 128) / 16.0;
            real y = sigmoid(x);
            int y_quant = $rtoi(y * 255.0); // Assume Output is Q0.8 Unsigned
            if (y_quant > 255) y_quant = 255;
            if (y_quant < 0) y_quant = 0;
            lut_data[i] = {24'd0, 8'(y_quant)};
        end
        load_lut(lut_data);

        start_afu(src_base, dst_base, test_len, 2'd0);
        wait_done();

        for (int i=0; i<test_len; i++) begin
            logic [7:0] expected = lut_data[i][7:0];
            logic [7:0] actual = read_tcdm_byte(dst_base + i);
            if (expected !== actual) begin
                $display("[FAIL] Sigmoid idx=%d, exp=%h, act=%h", i, expected, actual);
                errors++;
            end
        end
        if (errors==0) $display("[PASS] Sigmoid 8-bit");

        //--------------------------------------------------
        // Test 3: Softmax Exp (16-bit)
        //--------------------------------------------------
        $display("[AFU TB] Test 3: Softmax Exp (16-bit mode)");
        for (int i=0; i<256; i++) begin
            real x = real'(i - 255) / 16.0; // Softmax usually subtracts max, so input is <= 0
            real y = e**(x);
            int y_quant = $rtoi(y * 32767.0); // Q1.15
            lut_data[i] = {16'd0, 16'(y_quant)};
        end
        load_lut(lut_data);

        start_afu(src_base, dst_base, test_len, 2'd1);
        wait_done();

        for (int i=0; i<test_len; i++) begin
            logic [15:0] expected = lut_data[i][15:0];
            logic [15:0] actual = read_tcdm_half(dst_base + i*2);
            if (expected !== actual) begin
                $display("[FAIL] Exp idx=%d, exp=%h, act=%h", i, expected, actual);
                errors++;
            end
        end
        if (errors==0) $display("[PASS] Softmax Exp 16-bit");

        //--------------------------------------------------
        // Test 4: Variance x^2 (32-bit)
        //--------------------------------------------------
        $display("[AFU TB] Test 4: Variance x^2 (32-bit mode)");
        for (int i=0; i<256; i++) begin
            int val = (i < 128) ? i : (i - 256);
            int y_val = val * val; // max is 128*128 = 16384, easily fits in 32-bit without overflow
            lut_data[i] = 32'(y_val);
        end
        load_lut(lut_data);

        start_afu(src_base, dst_base, test_len, 2'd2);
        wait_done();

        for (int i=0; i<test_len; i++) begin
            logic [31:0] expected = lut_data[i];
            logic [31:0] actual = read_tcdm_word(dst_base + i*4);
            if (expected !== actual) begin
                $display("[FAIL] Variance idx=%d, exp=%h, act=%h", i, expected, actual);
                errors++;
            end
        end
        if (errors==0) $display("[PASS] Variance x^2 32-bit");

        $display("========================================");
        if (errors == 0) $display("[AFU TB] ALL TESTS PASSED SUCCESSFULLY");
        else             $display("[AFU TB] COMPLETED WITH %d ERRORS", errors);
        $display("========================================");
        
        $finish;
    end

endmodule
