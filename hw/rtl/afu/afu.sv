// Copyright (c) 2026
// AFU Top Module

module afu #(
    parameter int unsigned ADDR_WIDTH     = 32,
    parameter int unsigned CFG_DATA_WIDTH = 32,
    parameter int unsigned MEM_DATA_WIDTH = 256,
    parameter int unsigned LUT_LANES      = 4
)(
    input  logic                          clk_i,
    input  logic                          rst_ni,

    // OBI target interface (configuration & LUT programming)
    input  logic                          obi_s_req_i,
    output logic                          obi_s_gnt_o,
    input  logic [ADDR_WIDTH-1:0]         obi_s_addr_i,
    input  logic                          obi_s_we_i,
    input  logic [(CFG_DATA_WIDTH/8)-1:0] obi_s_be_i,
    input  logic [CFG_DATA_WIDTH-1:0]     obi_s_wdata_i,
    output logic                          obi_s_rvalid_o,
    output logic [CFG_DATA_WIDTH-1:0]     obi_s_rdata_o,

    // OBI initiator interface (memory access)
    output logic                          obi_m_req_o,
    input  logic                          obi_m_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_m_addr_o,
    output logic                          obi_m_we_o,
    output logic [(MEM_DATA_WIDTH/8)-1:0] obi_m_be_o,
    output logic [MEM_DATA_WIDTH-1:0]     obi_m_wdata_o,
    input  logic                          obi_m_rvalid_i,
    input  logic [MEM_DATA_WIDTH-1:0]     obi_m_rdata_i,
    
    // Interrupt / Status
    output logic                          done_o
);

    // CSRs
    logic [31:0] cfg_src_ptr;
    logic [31:0] cfg_dst_ptr;
    logic [31:0] cfg_length;
    logic [1:0]  cfg_mode;
    logic        cfg_start;
    
    // LUT write interface
    logic        lut_we;
    logic [7:0]  lut_addr;
    logic [31:0] lut_wdata;
    logic [3:0]  lut_be;
    
    // Read FIFO interface
    logic rfifo_full, rfifo_almost_full, rfifo_empty;
    logic rfifo_push, rfifo_pop;
    logic [255:0] rfifo_wdata, rfifo_rdata;
    
    // Write FIFO interface
    logic wfifo_full, wfifo_almost_full, wfifo_empty, wfifo_all_empty;
    logic wfifo_push, wfifo_pop;
    logic [287:0] wfifo_wdata, wfifo_rdata;

    logic core_done;
    logic core_busy;
    logic backend_idle;
    logic afu_error;

    assign afu_error = 1'b0;
    
    afu_frontend #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .DATA_WIDTH (CFG_DATA_WIDTH)
    ) i_frontend (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .obi_s_req_i    (obi_s_req_i),
        .obi_s_gnt_o    (obi_s_gnt_o),
        .obi_s_addr_i   (obi_s_addr_i),
        .obi_s_we_i     (obi_s_we_i),
        .obi_s_be_i     (obi_s_be_i),
        .obi_s_wdata_i  (obi_s_wdata_i),
        .obi_s_rvalid_o (obi_s_rvalid_o),
        .obi_s_rdata_o  (obi_s_rdata_o),
        .cfg_src_ptr_o  (cfg_src_ptr),
        .cfg_dst_ptr_o  (cfg_dst_ptr),
        .cfg_length_o   (cfg_length),
        .cfg_mode_o     (cfg_mode),
        .cfg_start_o    (cfg_start),
        .lut_we_o       (lut_we),
        .lut_addr_o     (lut_addr),
        .lut_wdata_o    (lut_wdata),
        .lut_be_o       (lut_be),
        .afu_done_i     (done_o),
        .afu_busy_i     (core_busy || !backend_idle),
        .afu_error_i    (afu_error)
    );
    
    afu_backend #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .DATA_WIDTH (MEM_DATA_WIDTH),
        .BE_WIDTH   (MEM_DATA_WIDTH/8)
    ) i_backend (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .cfg_src_ptr_i  (cfg_src_ptr),
        .cfg_dst_ptr_i  (cfg_dst_ptr),
        .cfg_length_i   (cfg_length),
        .cfg_mode_i     (cfg_mode),
        .cfg_start_i    (cfg_start),
        .obi_m_req_o    (obi_m_req_o),
        .obi_m_gnt_i    (obi_m_gnt_i),
        .obi_m_addr_o   (obi_m_addr_o),
        .obi_m_we_o     (obi_m_we_o),
        .obi_m_be_o     (obi_m_be_o),
        .obi_m_wdata_o  (obi_m_wdata_o),
        .obi_m_rvalid_i (obi_m_rvalid_i),
        .obi_m_rdata_i  (obi_m_rdata_i),
        .rfifo_almost_full_i (rfifo_almost_full),
        .rfifo_push_o   (rfifo_push),
        .rfifo_data_o   (rfifo_wdata),
        .wfifo_empty_i  (wfifo_empty),
        .wfifo_pop_o    (wfifo_pop),
        .wfifo_data_i   (wfifo_rdata),
        .idle_o         (backend_idle)
    );

    assign done_o = core_done && wfifo_all_empty && backend_idle;
    
    afu_core #(
        .LUT_LANES (LUT_LANES)
    ) i_core (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .cfg_src_ptr_i  (cfg_src_ptr),
        .cfg_dst_ptr_i  (cfg_dst_ptr),
        .cfg_length_i   (cfg_length),
        .cfg_mode_i     (cfg_mode),
        .cfg_start_i    (cfg_start),
        .lut_we_i       (lut_we),
        .lut_addr_i     (lut_addr),
        .lut_wdata_i    (lut_wdata),
        .lut_be_i       (lut_be),
        .rfifo_empty_i  (rfifo_empty),
        .rfifo_pop_o    (rfifo_pop),
        .rfifo_data_i   (rfifo_rdata),
        .wfifo_full_i   (wfifo_full),
        .wfifo_push_o   (wfifo_push),
        .wfifo_data_o   (wfifo_wdata),
        .done_o         (core_done),
        .busy_o         (core_busy)
    );
    
    afu_fifo_ff #(
        .NAME("RFIFO"),
        .DATA_WIDTH(256),
        .DEPTH(2)
    ) i_rfifo (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .flush_i        (cfg_start),
        .full_o         (rfifo_full),
        .almost_full_o  (rfifo_almost_full),
        .empty_o        (rfifo_empty),
        .all_empty_o    (),
        .data_i         (rfifo_wdata),
        .push_i         (rfifo_push),
        .data_o         (rfifo_rdata),
        .pop_i          (rfifo_pop)
    );
    
    afu_fifo_ff #(
        .NAME("WFIFO"),
        .DATA_WIDTH(288),
        .DEPTH(2)
    ) i_wfifo (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .flush_i        (cfg_start),
        .full_o         (wfifo_full),
        .almost_full_o  (wfifo_almost_full),
        .empty_o        (wfifo_empty),
        .all_empty_o    (wfifo_all_empty),
        .data_i         (wfifo_wdata),
        .push_i         (wfifo_push),
        .data_o         (wfifo_rdata),
        .pop_i          (wfifo_pop)
    );

endmodule
