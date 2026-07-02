`timescale 1ns/1ps
`default_nettype none

module tb_systolic_controller;
    localparam int unsigned ADDR_WIDTH = 32;
    localparam int unsigned DATA_WIDTH = 256;
    localparam int unsigned CFG_DATA_WIDTH = 32;
    localparam int unsigned TCDM_WORDS = 4096;

    logic clk_i;
    logic rst_ni;

    logic                          ctrl_req_i;
    logic                          ctrl_gnt_o;
    logic [ADDR_WIDTH-1:0]         ctrl_addr_i;
    logic                          ctrl_we_i;
    logic [(CFG_DATA_WIDTH/8)-1:0] ctrl_be_i;
    logic [CFG_DATA_WIDTH-1:0]     ctrl_wdata_i;
    logic                          ctrl_rvalid_o;
    logic [CFG_DATA_WIDTH-1:0]     ctrl_rdata_o;
    logic                          cfg_sys_done_o;

    logic                          obi_i_req_o;
    logic                          obi_i_gnt_i;
    logic [ADDR_WIDTH-1:0]         obi_i_addr_o;
    logic                          obi_i_we_o;
    logic [(DATA_WIDTH/8)-1:0]     obi_i_be_o;
    logic [DATA_WIDTH-1:0]         obi_i_wdata_o;
    logic                          obi_i_rvalid_i;
    logic [DATA_WIDTH-1:0]         obi_i_rdata_i;

    logic [3:0]                    obi_o_req_o;
    logic [3:0]                    obi_o_gnt_i;
    logic [3:0][ADDR_WIDTH-1:0]    obi_o_addr_o;
    logic [3:0]                    obi_o_we_o;
    logic [3:0][(DATA_WIDTH/8)-1:0] obi_o_be_o;
    logic [3:0][DATA_WIDTH-1:0]    obi_o_wdata_o;
    logic [3:0]                    obi_o_rvalid_i;
    logic [3:0][DATA_WIDTH-1:0]    obi_o_rdata_i;

    logic                          perf_weight_load_en_o;
    logic                          perf_compute_en_o;
    logic                          perf_ofm_valid_o;
    logic                          perf_ofm_ready_o;
    logic [2:0]                    debug_state_o;
    logic [1:0]                    debug_drain_state_o;
    logic [4:0]                    debug_linebuf_state_o;

    logic [DATA_WIDTH-1:0]         tcdm_mem [TCDM_WORDS];

    systolic_controller #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .CFG_DATA_WIDTH   (CFG_DATA_WIDTH),
        .ARRAY_DIM        (32),
        .INPUT_ELEM_WIDTH (8),
        .OFM_ELEM_WIDTH   (32),
        .INPUT_FIFO_DEPTH (4),
        .OFM_FIFO_DEPTH   (8)
    ) dut (
        .clk_i                 (clk_i),
        .rst_ni                (rst_ni),
        .ctrl_req_i            (ctrl_req_i),
        .ctrl_gnt_o            (ctrl_gnt_o),
        .ctrl_addr_i           (ctrl_addr_i),
        .ctrl_we_i             (ctrl_we_i),
        .ctrl_be_i             (ctrl_be_i),
        .ctrl_wdata_i          (ctrl_wdata_i),
        .ctrl_rvalid_o         (ctrl_rvalid_o),
        .ctrl_rdata_o          (ctrl_rdata_o),
        .cfg_sys_done_o        (cfg_sys_done_o),
        .obi_i_req_o           (obi_i_req_o),
        .obi_i_gnt_i           (obi_i_gnt_i),
        .obi_i_addr_o          (obi_i_addr_o),
        .obi_i_we_o            (obi_i_we_o),
        .obi_i_be_o            (obi_i_be_o),
        .obi_i_wdata_o         (obi_i_wdata_o),
        .obi_i_rvalid_i        (obi_i_rvalid_i),
        .obi_i_rdata_i         (obi_i_rdata_i),
        .obi_o_req_o           (obi_o_req_o),
        .obi_o_gnt_i           (obi_o_gnt_i),
        .obi_o_addr_o          (obi_o_addr_o),
        .obi_o_we_o            (obi_o_we_o),
        .obi_o_be_o            (obi_o_be_o),
        .obi_o_wdata_o         (obi_o_wdata_o),
        .obi_o_rvalid_i        (obi_o_rvalid_i),
        .obi_o_rdata_i         (obi_o_rdata_i),
        .perf_weight_load_en_o (perf_weight_load_en_o),
        .perf_compute_en_o     (perf_compute_en_o),
        .perf_ofm_valid_o      (perf_ofm_valid_o),
        .perf_ofm_ready_o      (perf_ofm_ready_o),
        .debug_state_o         (debug_state_o),
        .debug_drain_state_o   (debug_drain_state_o),
        .debug_linebuf_state_o (debug_linebuf_state_o)
    );

    initial begin
        clk_i = 1'b0;
        forever #5 clk_i = ~clk_i;
    end

    assign obi_i_gnt_i = obi_i_req_o;
    assign obi_o_gnt_i = obi_o_req_o;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            obi_i_rvalid_i <= 1'b0;
            obi_i_rdata_i <= '0;
        end else begin
            obi_i_rvalid_i <= obi_i_req_o && obi_i_gnt_i && !obi_i_we_o;
            obi_i_rdata_i <= tcdm_mem[obi_i_addr_o[16:5]];
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            obi_o_rvalid_i <= '0;
            obi_o_rdata_i <= '0;
        end else begin
            obi_o_rvalid_i <= '0;
            obi_o_rdata_i <= '0;
            for (int unsigned port = 0; port < 4; port++) begin
                if (obi_o_req_o[port] && obi_o_gnt_i[port] && !obi_o_we_o[port]) begin
                    obi_o_rvalid_i[port] <= 1'b1;
                    obi_o_rdata_i[port] <= tcdm_mem[obi_o_addr_o[port][16:5]];
                end
            end
        end
    end

    always_ff @(posedge clk_i) begin
        for (int unsigned port = 0; port < 4; port++) begin
            if (obi_o_req_o[port] && obi_o_gnt_i[port] && obi_o_we_o[port]) begin
                tcdm_mem[obi_o_addr_o[port][16:5]] <= obi_o_wdata_o[port];
            end
        end
    end
endmodule
