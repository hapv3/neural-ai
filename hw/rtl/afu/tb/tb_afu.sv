`default_nettype none
`timescale 1ns/1ps

module tb_afu;

    localparam int unsigned ADDR_WIDTH     = 32;
    localparam int unsigned CFG_DATA_WIDTH = 32;
    localparam int unsigned MEM_DATA_WIDTH = 256;
    localparam int unsigned MEM_BYTES      = MEM_DATA_WIDTH / 8;
    localparam int unsigned LUT_LANES      = 4;
    localparam int unsigned MEM_SIZE       = 16 * 1024;

    localparam logic [2:0] MODE_8BIT  = 3'd0;
    localparam logic [2:0] MODE_16BIT = 3'd1;
    localparam logic [2:0] MODE_32BIT = 3'd2;
    localparam logic [2:0] MODE_DFL4_ROW32_Q8 = 3'd5;
    localparam logic [2:0] MODE_CLASS_SIGMOID_ROW32_HIGH16 = 3'd6;
    localparam logic [31:0] AFU_CSR_BASE = 32'h400;
    localparam logic [31:0] AFU_DFL_EXP_LUT_BASE = 32'h800;
    localparam logic [31:0] AFU_DFL_RECIP_LUT_BASE = 32'hc00;

    logic clk_i;
    logic rst_ni;

    logic                          obi_s_req;
    logic                          obi_s_gnt;
    logic [ADDR_WIDTH-1:0]         obi_s_addr;
    logic                          obi_s_we;
    logic [(CFG_DATA_WIDTH/8)-1:0] obi_s_be;
    logic [CFG_DATA_WIDTH-1:0]     obi_s_wdata;
    logic                          obi_s_rvalid;
    logic [CFG_DATA_WIDTH-1:0]     obi_s_rdata;

    logic                          obi_m_req;
    logic                          obi_m_gnt;
    logic [ADDR_WIDTH-1:0]         obi_m_addr;
    logic                          obi_m_we;
    logic [(MEM_DATA_WIDTH/8)-1:0] obi_m_be;
    logic [MEM_DATA_WIDTH-1:0]     obi_m_wdata;
    logic                          obi_m_rvalid;
    logic [MEM_DATA_WIDTH-1:0]     obi_m_rdata;

    logic                          obi_rhs_req;
    logic                          obi_rhs_gnt;
    logic [ADDR_WIDTH-1:0]         obi_rhs_addr;
    logic                          obi_rhs_we;
    logic [(MEM_DATA_WIDTH/8)-1:0] obi_rhs_be;
    logic [MEM_DATA_WIDTH-1:0]     obi_rhs_wdata;
    logic                          obi_rhs_rvalid;
    logic [MEM_DATA_WIDTH-1:0]     obi_rhs_rdata;

    logic done;

    afu #(
        .ADDR_WIDTH     (ADDR_WIDTH),
        .CFG_DATA_WIDTH (CFG_DATA_WIDTH),
        .MEM_DATA_WIDTH (MEM_DATA_WIDTH),
        .LUT_LANES      (LUT_LANES)
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
        .obi_rhs_req_o  (obi_rhs_req),
        .obi_rhs_gnt_i  (obi_rhs_gnt),
        .obi_rhs_addr_o (obi_rhs_addr),
        .obi_rhs_we_o   (obi_rhs_we),
        .obi_rhs_be_o   (obi_rhs_be),
        .obi_rhs_wdata_o(obi_rhs_wdata),
        .obi_rhs_rvalid_i(obi_rhs_rvalid),
        .obi_rhs_rdata_i(obi_rhs_rdata),
        .done_o         (done)
    );

    assign obi_rhs_gnt = 1'b0;
    assign obi_rhs_rvalid = 1'b0;
    assign obi_rhs_rdata = '0;

    logic [7:0] tcdm_mem [0:MEM_SIZE-1];
    logic       read_pending_q;
    logic [31:0] read_addr_q;
    int unsigned mem_cycle_q;

    int errors;
    logic [31:0] lut_data [0:255];
    logic [31:0] dfl_exp_lut [0:255];
    logic [31:0] dfl_recip_lut [0:255];

    initial begin
        clk_i = 1'b0;
        forever #5 clk_i = ~clk_i;
    end

    initial begin
        rst_ni = 1'b0;
        #40;
        rst_ni = 1'b1;
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            obi_m_gnt      <= 1'b0;
            obi_m_rvalid   <= 1'b0;
            obi_m_rdata    <= '0;
            read_pending_q <= 1'b0;
            read_addr_q    <= '0;
            mem_cycle_q    <= '0;
        end else begin
            int unsigned base;

            mem_cycle_q  <= mem_cycle_q + 1;
            obi_m_gnt    <= 1'b0;
            obi_m_rvalid <= 1'b0;
            obi_m_rdata  <= '0;

            if (read_pending_q) begin
                base = read_addr_q % MEM_SIZE;
                for (int b = 0; b < MEM_BYTES; b++) begin
                    obi_m_rdata[b*8 +: 8] <= tcdm_mem[(base + b) % MEM_SIZE];
                end
                obi_m_rvalid   <= 1'b1;
                read_pending_q <= 1'b0;
            end

            if (obi_m_req && !obi_m_gnt && ((mem_cycle_q % 5) != 0)) begin
                base      = obi_m_addr % MEM_SIZE;
                obi_m_gnt <= 1'b1;
                if (obi_m_we) begin
                    for (int b = 0; b < MEM_BYTES; b++) begin
                        if (obi_m_be[b]) begin
                            tcdm_mem[(base + b) % MEM_SIZE] <= obi_m_wdata[b*8 +: 8];
                        end
                    end
                end else begin
                    read_pending_q <= 1'b1;
                    read_addr_q    <= obi_m_addr;
                end
            end
        end
    end

    task automatic write_obi(input logic [31:0] addr, input logic [31:0] data);
        @(posedge clk_i);
        obi_s_req   = 1'b1;
        obi_s_we    = 1'b1;
        obi_s_addr  = addr;
        obi_s_wdata = data;
        obi_s_be    = 4'hf;
        do begin
            @(posedge clk_i);
        end while (!obi_s_gnt);
        obi_s_req   = 1'b0;
        obi_s_we    = 1'b0;
        obi_s_addr  = '0;
        obi_s_wdata = '0;
        obi_s_be    = '0;
    endtask

    task automatic read_obi(input logic [31:0] addr, output logic [31:0] data);
        @(posedge clk_i);
        obi_s_req  = 1'b1;
        obi_s_we   = 1'b0;
        obi_s_addr = addr;
        obi_s_be   = 4'hf;
        do begin
            @(posedge clk_i);
        end while (!obi_s_gnt);
        obi_s_req  = 1'b0;
        obi_s_addr = '0;
        obi_s_be   = '0;
        while (!obi_s_rvalid) begin
            @(posedge clk_i);
        end
        data = obi_s_rdata;
    endtask

    task automatic load_lut();
        for (int i = 0; i < 256; i++) begin
            write_obi(i * 4, lut_data[i]);
        end
    endtask

    task automatic load_dfl_luts();
        for (int i = 0; i < 256; i++) begin
            write_obi(AFU_DFL_EXP_LUT_BASE + i * 4, dfl_exp_lut[i]);
            write_obi(AFU_DFL_RECIP_LUT_BASE + i * 4, dfl_recip_lut[i]);
        end
    endtask

    task automatic start_afu(
        input logic [31:0] src_ptr,
        input logic [31:0] dst_ptr,
        input logic [31:0] length,
        input logic [2:0]  mode
    );
        write_obi(AFU_CSR_BASE + 32'h04, src_ptr);
        write_obi(AFU_CSR_BASE + 32'h08, dst_ptr);
        write_obi(AFU_CSR_BASE + 32'h0c, length);
        write_obi(AFU_CSR_BASE + 32'h10, {29'd0, mode});
        write_obi(AFU_CSR_BASE + 32'h00, 32'd1);
    endtask

    task automatic wait_done(input string name);
        logic [31:0] status;
        for (int poll = 0; poll < 20000; poll++) begin
            read_obi(AFU_CSR_BASE + 32'h00, status);
            if (status[2]) begin
                $fatal(1, "[AFU TB] %s reported error status 0x%08x", name, status);
            end
            if (status[0]) begin
                return;
            end
        end
        $fatal(1, "[AFU TB] %s timed out waiting for done", name);
    endtask

    function automatic logic [7:0] input_pattern(input int index, input int pattern_id);
        unique case (pattern_id)
            0: input_pattern = 8'(index & 32'hff);
            1: input_pattern = 8'(((index * 37) + 11) & 32'hff);
            2: input_pattern = 8'(((index * 19) + 32'ha5) & 32'hff);
            default: input_pattern = 8'(((index * 53) + 7) & 32'hff);
        endcase
    endfunction

    function automatic logic [15:0] read_tcdm_half(input int byte_addr);
        read_tcdm_half = {tcdm_mem[(byte_addr + 1) % MEM_SIZE], tcdm_mem[byte_addr % MEM_SIZE]};
    endfunction

    function automatic logic [31:0] read_tcdm_word(input int byte_addr);
        read_tcdm_word = {
            tcdm_mem[(byte_addr + 3) % MEM_SIZE],
            tcdm_mem[(byte_addr + 2) % MEM_SIZE],
            tcdm_mem[(byte_addr + 1) % MEM_SIZE],
            tcdm_mem[byte_addr % MEM_SIZE]
        };
    endfunction

    function automatic logic signed [7:0] dfl_input_value(input int loc, input int channel);
        dfl_input_value = 8'(((loc * 13 + channel * 7 + 5) % 41) - 20);
    endfunction

    function automatic logic [15:0] dfl_exp_lut_value(input int index);
        int neg_delta;
        int shift;
        int frac;
        int base;
        int next_value;
        int value;
        begin
            if (index == 0) begin
                dfl_exp_lut_value = 16'd32768;
            end else begin
                neg_delta = 256 - index;
                shift = neg_delta >> 4;
                frac = neg_delta & 15;
                base = (shift >= 15) ? 1 : (32768 >> shift);
                next_value = (base > 1) ? (base >> 1) : 1;
                value = ((base * (16 - frac)) + (next_value * frac) + 8) >> 4;
                dfl_exp_lut_value = value[15:0];
            end
        end
    endfunction

    function automatic logic [31:0] dfl_recip_lut_value(input int index);
        longint unsigned midpoint_q9;
        longint unsigned numerator;
        begin
            midpoint_q9 = 512 + (index * 2) + 1;
            numerator = 64'(1) << 37;
            dfl_recip_lut_value = 32'((numerator + (midpoint_q9 >> 1)) / midpoint_q9);
        end
    endfunction

    function automatic int unsigned msb_pos18_tb(input int unsigned value);
        int unsigned pos;
        begin
            pos = 0;
            for (int i = 0; i < 18; i++) begin
                if (value[i]) pos = i;
            end
            msb_pos18_tb = pos;
        end
    endfunction

    function automatic logic [7:0] recip_index_from_sum_tb(input int unsigned sum);
        int unsigned shift;
        int unsigned norm_q8;
        begin
            shift = msb_pos18_tb(sum);
            norm_q8 = (sum << 8) >> shift;
            if (norm_q8 < 256) begin
                recip_index_from_sum_tb = 8'd0;
            end else if (norm_q8 > 511) begin
                recip_index_from_sum_tb = 8'd255;
            end else begin
                recip_index_from_sum_tb = norm_q8[7:0];
            end
        end
    endfunction

    function automatic logic [15:0] dfl_expected_approx(input int loc, input int side);
        logic signed [7:0] logits [0:3];
        logic signed [7:0] max_value;
        int unsigned exp_values [0:3];
        int unsigned sum;
        int unsigned weighted;
        int unsigned sum_shift;
        int unsigned recip_index;
        longint unsigned product;
        longint unsigned rounded;
        longint unsigned round_add;
        int unsigned total_shift;
        begin
            for (int bin = 0; bin < 4; bin++) begin
                logits[bin] = dfl_input_value(loc, side * 4 + bin);
            end
            max_value = logits[0];
            for (int bin = 1; bin < 4; bin++) begin
                if (logits[bin] > max_value) begin
                    max_value = logits[bin];
                end
            end
            for (int bin = 0; bin < 4; bin++) begin
                exp_values[bin] = dfl_exp_lut[8'(logits[bin] - max_value)][15:0];
            end
            sum = exp_values[0] + exp_values[1] + exp_values[2] + exp_values[3];
            weighted = exp_values[1] + (2 * exp_values[2]) + (3 * exp_values[3]);
            sum_shift = msb_pos18_tb(sum);
            recip_index = recip_index_from_sum_tb(sum);
            product = (longint'(weighted) << 8) * longint'(dfl_recip_lut[recip_index]);
            total_shift = 28 + sum_shift;
            round_add = 64'(1) << (total_shift - 1);
            rounded = (product + round_add) >> total_shift;
            dfl_expected_approx = rounded[15:0];
        end
    endfunction

    task automatic clear_tcdm(input logic [7:0] value);
        for (int i = 0; i < MEM_SIZE; i++) begin
            tcdm_mem[i] = value;
        end
    endtask

    task automatic fill_lut(input logic [2:0] mode, input int pattern_id);
        for (int i = 0; i < 256; i++) begin
            unique case (mode)
                MODE_8BIT: begin
                    lut_data[i] = ((i * 7) + pattern_id + 32'h5a) & 32'h0000_00ff;
                end
                MODE_16BIT: begin
                    lut_data[i] = ((i * 257) + 32'h1234 + pattern_id) & 32'h0000_ffff;
                end
                default: begin
                    lut_data[i] = (i * 32'h0101_0101) ^ (32'hdead_beef + pattern_id);
                end
            endcase
        end
    endtask

    task automatic check_case(
        input string       name,
        input logic [2:0]  mode,
        input int          src_base,
        input int          dst_base,
        input int          length,
        input int          pattern_id
    );
        int output_bytes;

        $display("[AFU TB] %s: mode=%0d src=0x%0h dst=0x%0h len=%0d",
                 name, mode, src_base, dst_base, length);
        $fflush();

        fill_lut(mode, pattern_id);
        load_lut();

        for (int i = 0; i < length; i++) begin
            tcdm_mem[(src_base + i) % MEM_SIZE] = input_pattern(i, pattern_id);
        end

        output_bytes = (mode == MODE_8BIT) ? 1 : ((mode == MODE_16BIT) ? 2 : 4);
        for (int i = 0; i < (length * output_bytes + 64); i++) begin
            tcdm_mem[(dst_base + i - 16 + MEM_SIZE) % MEM_SIZE] = 8'ha5;
        end

        start_afu(src_base, dst_base, length, mode);
        wait_done(name);

        for (int i = 0; i < length; i++) begin
            logic [7:0]  input_value;
            logic [31:0] expected;
            logic [31:0] actual;
            int          out_addr;

            input_value = input_pattern(i, pattern_id);
            expected    = lut_data[input_value];
            out_addr    = dst_base + i * output_bytes;

            unique case (mode)
                MODE_8BIT:  actual = {24'd0, tcdm_mem[out_addr % MEM_SIZE]};
                MODE_16BIT: actual = {16'd0, read_tcdm_half(out_addr)};
                default:    actual = read_tcdm_word(out_addr);
            endcase

            if (actual !== expected) begin
                $display("[FAIL] %s idx=%0d input=%0h exp=%08h act=%08h",
                         name, i, input_value, expected, actual);
                errors++;
            end
        end

        if (errors == 0) begin
            $display("[PASS] %s", name);
        end
    endtask

    task automatic check_dfl_case(
        input string name,
        input int    src_base,
        input int    dst_base,
        input int    locations
    );
        $display("[AFU TB] %s: fused DFL src=0x%0h dst=0x%0h locations=%0d",
                 name, src_base, dst_base, locations);
        $fflush();

        for (int i = 0; i < 256; i++) begin
            dfl_exp_lut[i] = {16'd0, dfl_exp_lut_value(i)};
            dfl_recip_lut[i] = dfl_recip_lut_value(i);
        end
        load_dfl_luts();

        for (int loc = 0; loc < locations; loc++) begin
            for (int ch = 0; ch < 32; ch++) begin
                if (ch < 16) begin
                    tcdm_mem[(src_base + loc * 32 + ch) % MEM_SIZE] = dfl_input_value(loc, ch);
                end else begin
                    tcdm_mem[(src_base + loc * 32 + ch) % MEM_SIZE] = 8'(ch + loc);
                end
            end
        end
        for (int i = 0; i < locations * 8 + 64; i++) begin
            tcdm_mem[(dst_base + i) % MEM_SIZE] = 8'ha5;
        end

        start_afu(src_base, dst_base, locations * 32, MODE_DFL4_ROW32_Q8);
        wait_done(name);

        for (int loc = 0; loc < locations; loc++) begin
            for (int side = 0; side < 4; side++) begin
                logic [15:0] expected;
                logic [15:0] actual;
                int out_addr;
                out_addr = dst_base + loc * 8 + side * 2;
                expected = dfl_expected_approx(loc, side);
                actual = read_tcdm_half(out_addr);
                if (actual !== expected) begin
                    $display("[FAIL] %s loc=%0d side=%0d exp=%0d act=%0d",
                             name, loc, side, expected, actual);
                    errors++;
                end
            end
        end

        if (errors == 0) begin
            $display("[PASS] %s", name);
        end
    endtask

    task automatic check_class_sigmoid_case(
        input string name,
        input int    src_base,
        input int    dst_base,
        input int    locations
    );
        $display("[AFU TB] %s: class sigmoid src=0x%0h dst=0x%0h locations=%0d",
                 name, src_base, dst_base, locations);
        $fflush();

        fill_lut(MODE_8BIT, 9);
        load_lut();

        for (int loc = 0; loc < locations; loc++) begin
            for (int ch = 0; ch < 32; ch++) begin
                tcdm_mem[(src_base + loc * 32 + ch) % MEM_SIZE] =
                    8'((loc * 17 + ch * 5 + 3) & 8'hff);
            end
        end
        for (int i = 0; i < locations * 16 + 64; i++) begin
            tcdm_mem[(dst_base + i) % MEM_SIZE] = 8'ha5;
        end

        start_afu(src_base, dst_base, locations * 32, MODE_CLASS_SIGMOID_ROW32_HIGH16);
        wait_done(name);

        for (int loc = 0; loc < locations; loc++) begin
            for (int cls = 0; cls < 16; cls++) begin
                logic [7:0] input_value;
                logic [7:0] expected;
                logic [7:0] actual;
                int out_addr;

                input_value = tcdm_mem[(src_base + loc * 32 + 16 + cls) % MEM_SIZE];
                expected = lut_data[input_value][7:0];
                out_addr = dst_base + loc * 16 + cls;
                actual = tcdm_mem[out_addr % MEM_SIZE];
                if (actual !== expected) begin
                    $display("[FAIL] %s loc=%0d cls=%0d input=%0h exp=%0h act=%0h",
                             name, loc, cls, input_value, expected, actual);
                    errors++;
                end
            end
        end

        if (errors == 0) begin
            $display("[PASS] %s", name);
        end
    endtask

    initial begin
        errors      = 0;
        obi_s_req   = 1'b0;
        obi_s_we    = 1'b0;
        obi_s_addr  = '0;
        obi_s_wdata = '0;
        obi_s_be    = '0;
        clear_tcdm(8'h00);

        wait (rst_ni);
        repeat (5) @(posedge clk_i);

        $display("========================================");
        $display("[AFU TB] Starting 256-bit beat-engine tests");
        $fflush();

        check_case("zero_length", MODE_8BIT,  'h100, 'h400, 0,   0);
        check_case("mode8_aligned_257",  MODE_8BIT,  'h100, 'h400, 257, 1);
        check_case("mode16_aligned_129", MODE_16BIT, 'h100, 'h500, 129, 2);
        check_case("mode32_aligned_67",  MODE_32BIT, 'h200, 'h600, 67,  3);
        check_case("mode8_unaligned_65", MODE_8BIT,  'h123, 'h477, 65,  0);
        check_case("mode16_unaligned_33", MODE_16BIT, 'h13f, 'h584, 33, 1);
        check_case("mode32_unaligned_17", MODE_32BIT, 'h255, 'h684, 17, 2);
        check_dfl_case("dfl_row32_aligned_17", 'h1000, 'h2000, 17);
        check_class_sigmoid_case("class_sigmoid_row32_high16_17", 'h1200, 'h2800, 17);
        check_case("mode8_after_dfl_pingpong", MODE_8BIT, 'h300, 'h900, 37, 3);

        $display("========================================");
        if (errors == 0) begin
            $display("[AFU TB] ALL TESTS PASSED SUCCESSFULLY");
        end else begin
            $display("[AFU TB] COMPLETED WITH %0d ERRORS", errors);
            $fatal(1, "[AFU TB] failures detected");
        end
        $display("========================================");

        $finish;
    end

endmodule
