`default_nettype none
`timescale 1ns/1ps

module tb_afu;

    localparam int unsigned ADDR_WIDTH     = 32;
    localparam int unsigned CFG_DATA_WIDTH = 32;
    localparam int unsigned MEM_DATA_WIDTH = 256;
    localparam int unsigned MEM_BYTES      = MEM_DATA_WIDTH / 8;
    localparam int unsigned LUT_LANES      = 4;
    localparam int unsigned MEM_SIZE       = 16 * 1024;

    localparam logic [1:0] MODE_8BIT  = 2'd0;
    localparam logic [1:0] MODE_16BIT = 2'd1;
    localparam logic [1:0] MODE_32BIT = 2'd2;

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
        .done_o         (done)
    );

    logic [7:0] tcdm_mem [0:MEM_SIZE-1];
    logic       read_pending_q;
    logic [31:0] read_addr_q;
    int unsigned mem_cycle_q;

    int errors;
    logic [31:0] lut_data [0:255];

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

    task automatic start_afu(
        input logic [31:0] src_ptr,
        input logic [31:0] dst_ptr,
        input logic [31:0] length,
        input logic [1:0]  mode
    );
        write_obi(32'h1004, src_ptr);
        write_obi(32'h1008, dst_ptr);
        write_obi(32'h100c, length);
        write_obi(32'h1010, {30'd0, mode});
        write_obi(32'h1000, 32'd1);
    endtask

    task automatic wait_done(input string name);
        logic [31:0] status;
        for (int poll = 0; poll < 20000; poll++) begin
            read_obi(32'h1000, status);
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

    task automatic clear_tcdm(input logic [7:0] value);
        for (int i = 0; i < MEM_SIZE; i++) begin
            tcdm_mem[i] = value;
        end
    endtask

    task automatic fill_lut(input logic [1:0] mode, input int pattern_id);
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
        input logic [1:0]  mode,
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
