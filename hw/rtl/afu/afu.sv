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

    // OBI initiator interface (RHS read-only memory access for binary modes)
    output logic                          obi_rhs_req_o,
    input  logic                          obi_rhs_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_rhs_addr_o,
    output logic                          obi_rhs_we_o,
    output logic [(MEM_DATA_WIDTH/8)-1:0] obi_rhs_be_o,
    output logic [MEM_DATA_WIDTH-1:0]     obi_rhs_wdata_o,
    input  logic                          obi_rhs_rvalid_i,
    input  logic [MEM_DATA_WIDTH-1:0]     obi_rhs_rdata_i,
    
    // Interrupt / Status
    output logic                          done_o,

    output logic                          perf_start_o,
    output logic                          perf_active_o,
    output logic [4:0]                    perf_state_o,
    output logic                          perf_lhs_consume_o,
    output logic                          perf_rhs_consume_o,
    output logic                          perf_result_produce_o,
    output logic                          perf_input_wait_o,
    output logic                          perf_rhs_wait_o,
    output logic                          perf_output_stall_o
);

    // CSRs
    logic [31:0] cfg_src_ptr;
    logic [31:0] cfg_src2_ptr;
    logic [31:0] cfg_dst_ptr;
    logic [31:0] cfg_length;
    logic [3:0]  cfg_mode;
    logic signed [31:0] cfg_add_bias;
    logic [1:0] cfg_binary_mode;
    logic signed [31:0] cfg_binary_lhs_multiplier;
    logic [6:0] cfg_binary_lhs_shift;
    logic signed [31:0] cfg_binary_rhs_multiplier;
    logic [6:0] cfg_binary_rhs_shift;
    logic signed [31:0] cfg_binary_output_multiplier;
    logic [6:0] cfg_binary_output_shift;
    logic signed [31:0] cfg_binary_lhs_zero_point;
    logic signed [31:0] cfg_binary_rhs_zero_point;
    logic signed [31:0] cfg_binary_output_zero_point;
    logic signed [31:0] cfg_binary_clamp_min;
    logic signed [31:0] cfg_binary_clamp_max;
    logic [5:0] cfg_binary_double_round_shift;
    logic        cfg_start;

    localparam logic [3:0] MODE_BINARY_QUANT = 4'd8;
    localparam logic [3:0] MODE_LUT_BINARY_QUANT = 4'd9;
    localparam logic [3:0] MODE_GLOBAL_AVGPOOL_REQUANT = 4'd10;
    
    // LUT write interface
    logic        lut_we;
    logic [7:0]  lut_addr;
    logic [31:0] lut_wdata;
    logic [3:0]  lut_be;
    logic        lut_fixed_bank;
    logic        lut_bank;
    
    // Read FIFO interface
    logic rfifo_full, rfifo_almost_full, rfifo_empty;
    logic rfifo_push, rfifo_pop;
    logic [255:0] rfifo_wdata, rfifo_rdata;

    // RHS read FIFO interface
    logic rhs_rfifo_full, rhs_rfifo_almost_full, rhs_rfifo_empty;
    logic rhs_rfifo_push, rhs_rfifo_pop;
    logic [255:0] rhs_rfifo_wdata, rhs_rfifo_rdata;
    
    // Write FIFO interface
    logic wfifo_full, wfifo_almost_full, wfifo_empty, wfifo_all_empty;
    logic wfifo_push, wfifo_pop;
    logic [287:0] wfifo_wdata, wfifo_rdata;

    logic core_done;
    logic core_busy;
    logic core_start;
    logic [2:0] core_mode;
    logic core_rfifo_pop;
    logic core_rhs_rfifo_pop;
    logic core_wfifo_full;
    logic core_wfifo_push;
    logic [287:0] core_wfifo_wdata;
    logic [4:0] core_perf_state;
    logic core_perf_input_wait;
    logic core_perf_rhs_wait;
    logic core_perf_output_stall;
    logic binary_done;
    logic binary_busy;
    logic binary_error;
    logic binary_lhs_pop;
    logic binary_rhs_pop;
    logic binary_chain_ready;
    logic binary_out_valid;
    logic [287:0] binary_out_data;
    logic binary_input_wait;
    logic binary_rhs_wait;
    logic binary_output_stall;
    logic binary_standalone;
    logic binary_chain;
    logic binary_active;
    logic global_avgpool_requant;
    logic operation_done;
    logic backend_idle;
    logic afu_error;

    assign binary_standalone = cfg_mode == MODE_BINARY_QUANT;
    assign binary_chain = cfg_mode == MODE_LUT_BINARY_QUANT;
    assign binary_active = binary_standalone || binary_chain;
    assign global_avgpool_requant = cfg_mode == MODE_GLOBAL_AVGPOOL_REQUANT;
    assign core_start = cfg_start && !binary_standalone;
    assign core_mode = binary_chain ? 3'd0 :
                       (global_avgpool_requant ? 3'd7 : cfg_mode[2:0]);
    assign operation_done = binary_active ? binary_done : core_done;
    assign afu_error = binary_active && binary_error;
    assign perf_start_o = cfg_start;
    assign perf_active_o = core_busy || binary_busy || !backend_idle;
    assign perf_state_o = binary_active ? {3'd6, binary_busy, binary_chain} : core_perf_state;
    assign perf_lhs_consume_o = rfifo_pop;
    assign perf_rhs_consume_o = rhs_rfifo_pop;
    assign perf_result_produce_o = wfifo_push;
    assign perf_input_wait_o = binary_active ? binary_input_wait : core_perf_input_wait;
    assign perf_rhs_wait_o = binary_active ? binary_rhs_wait : core_perf_rhs_wait;
    assign perf_output_stall_o = binary_active ? binary_output_stall : core_perf_output_stall;

    assign rfifo_pop = binary_standalone ? binary_lhs_pop : core_rfifo_pop;
    assign rhs_rfifo_pop = binary_active ? binary_rhs_pop : core_rhs_rfifo_pop;
    assign core_wfifo_full = binary_chain ? !binary_chain_ready : wfifo_full;
    assign wfifo_push = binary_active ? (binary_out_valid && !wfifo_full) : core_wfifo_push;
    assign wfifo_wdata = binary_active ? binary_out_data : core_wfifo_wdata;
    
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
        .cfg_src2_ptr_o (cfg_src2_ptr),
        .cfg_dst_ptr_o  (cfg_dst_ptr),
        .cfg_length_o   (cfg_length),
        .cfg_mode_o     (cfg_mode),
        .cfg_add_bias_o (cfg_add_bias),
        .cfg_binary_mode_o(cfg_binary_mode),
        .cfg_binary_lhs_multiplier_o(cfg_binary_lhs_multiplier),
        .cfg_binary_lhs_shift_o(cfg_binary_lhs_shift),
        .cfg_binary_rhs_multiplier_o(cfg_binary_rhs_multiplier),
        .cfg_binary_rhs_shift_o(cfg_binary_rhs_shift),
        .cfg_binary_output_multiplier_o(cfg_binary_output_multiplier),
        .cfg_binary_output_shift_o(cfg_binary_output_shift),
        .cfg_binary_lhs_zero_point_o(cfg_binary_lhs_zero_point),
        .cfg_binary_rhs_zero_point_o(cfg_binary_rhs_zero_point),
        .cfg_binary_output_zero_point_o(cfg_binary_output_zero_point),
        .cfg_binary_clamp_min_o(cfg_binary_clamp_min),
        .cfg_binary_clamp_max_o(cfg_binary_clamp_max),
        .cfg_binary_double_round_shift_o(cfg_binary_double_round_shift),
        .cfg_start_o    (cfg_start),
        .lut_we_o       (lut_we),
        .lut_addr_o     (lut_addr),
        .lut_wdata_o    (lut_wdata),
        .lut_be_o       (lut_be),
        .lut_fixed_bank_o(lut_fixed_bank),
        .lut_bank_o      (lut_bank),
        .afu_done_i     (done_o),
        .afu_busy_i     (core_busy || binary_busy || !backend_idle),
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
        .cfg_src2_ptr_i (cfg_src2_ptr),
        .cfg_dst_ptr_i  (cfg_dst_ptr),
        .cfg_length_i   (cfg_length),
        .cfg_mode_i     (cfg_mode),
        .cfg_start_i    (cfg_start),
        .read_stop_i    (operation_done),
        .obi_m_req_o    (obi_m_req_o),
        .obi_m_gnt_i    (obi_m_gnt_i),
        .obi_m_addr_o   (obi_m_addr_o),
        .obi_m_we_o     (obi_m_we_o),
        .obi_m_be_o     (obi_m_be_o),
        .obi_m_wdata_o  (obi_m_wdata_o),
        .obi_m_rvalid_i (obi_m_rvalid_i),
        .obi_m_rdata_i  (obi_m_rdata_i),
        .obi_rhs_req_o  (obi_rhs_req_o),
        .obi_rhs_gnt_i  (obi_rhs_gnt_i),
        .obi_rhs_addr_o (obi_rhs_addr_o),
        .obi_rhs_we_o   (obi_rhs_we_o),
        .obi_rhs_be_o   (obi_rhs_be_o),
        .obi_rhs_wdata_o(obi_rhs_wdata_o),
        .obi_rhs_rvalid_i(obi_rhs_rvalid_i),
        .obi_rhs_rdata_i(obi_rhs_rdata_i),
        .rfifo_almost_full_i (rfifo_almost_full),
        .rfifo_push_o   (rfifo_push),
        .rfifo_data_o   (rfifo_wdata),
        .rhs_rfifo_almost_full_i(rhs_rfifo_almost_full),
        .rhs_rfifo_push_o(rhs_rfifo_push),
        .rhs_rfifo_data_o(rhs_rfifo_wdata),
        .wfifo_empty_i  (wfifo_empty),
        .wfifo_pop_o    (wfifo_pop),
        .wfifo_data_i   (wfifo_rdata),
        .idle_o         (backend_idle)
    );

    assign done_o = operation_done && wfifo_all_empty && backend_idle;
    
    afu_core #(
        .LUT_LANES (LUT_LANES)
    ) i_core (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .cfg_src_ptr_i  (cfg_src_ptr),
        .cfg_src2_ptr_i (cfg_src2_ptr),
        .cfg_dst_ptr_i  (cfg_dst_ptr),
        .cfg_length_i   (cfg_length),
        .cfg_mode_i     (core_mode),
        .cfg_add_bias_i (cfg_add_bias),
        .cfg_gap_requant_i(global_avgpool_requant),
        .cfg_output_multiplier_i(cfg_binary_output_multiplier),
        .cfg_output_shift_i(cfg_binary_output_shift),
        .cfg_output_zero_point_i(cfg_binary_output_zero_point),
        .cfg_clamp_min_i(cfg_binary_clamp_min),
        .cfg_clamp_max_i(cfg_binary_clamp_max),
        .cfg_double_round_shift_i(cfg_binary_double_round_shift),
        .cfg_start_i    (core_start),
        .lut_we_i       (lut_we),
        .lut_addr_i     (lut_addr),
        .lut_wdata_i    (lut_wdata),
        .lut_be_i       (lut_be),
        .lut_fixed_bank_i(lut_fixed_bank),
        .lut_bank_i      (lut_bank),
        .rfifo_empty_i  (rfifo_empty),
        .rfifo_pop_o    (core_rfifo_pop),
        .rfifo_data_i   (rfifo_rdata),
        .rhs_rfifo_empty_i(rhs_rfifo_empty),
        .rhs_rfifo_pop_o(core_rhs_rfifo_pop),
        .rhs_rfifo_data_i(rhs_rfifo_rdata),
        .wfifo_full_i   (core_wfifo_full),
        .wfifo_push_o   (core_wfifo_push),
        .wfifo_data_o   (core_wfifo_wdata),
        .done_o         (core_done),
        .busy_o         (core_busy),
        .perf_state_o   (core_perf_state),
        .perf_input_wait_o(core_perf_input_wait),
        .perf_rhs_wait_o(core_perf_rhs_wait),
        .perf_output_stall_o(core_perf_output_stall)
    );

    afu_binary_requant_engine #(
        .LANES(32)
    ) i_binary_requant_engine (
        .clk_i,
        .rst_ni,
        .start_i                  (cfg_start && binary_active),
        .chain_i                  (binary_chain),
        .length_i                 (cfg_length),
        .mode_i                   (cfg_binary_mode),
        .lhs_multiplier_i         (cfg_binary_lhs_multiplier),
        .lhs_shift_i              (cfg_binary_lhs_shift),
        .rhs_multiplier_i         (cfg_binary_rhs_multiplier),
        .rhs_shift_i              (cfg_binary_rhs_shift),
        .output_multiplier_i      (cfg_binary_output_multiplier),
        .output_shift_i           (cfg_binary_output_shift),
        .lhs_zero_point_i         (cfg_binary_lhs_zero_point),
        .rhs_zero_point_i         (cfg_binary_rhs_zero_point),
        .output_zero_point_i      (cfg_binary_output_zero_point),
        .clamp_min_i              (cfg_binary_clamp_min),
        .clamp_max_i              (cfg_binary_clamp_max),
        .double_round_shift_i     (cfg_binary_double_round_shift),
        .lhs_empty_i              (rfifo_empty),
        .lhs_pop_o                (binary_lhs_pop),
        .lhs_data_i               (rfifo_rdata),
        .chain_valid_i            (binary_chain && core_wfifo_push),
        .chain_ready_o            (binary_chain_ready),
        .chain_data_i             (core_wfifo_wdata),
        .rhs_empty_i              (rhs_rfifo_empty),
        .rhs_pop_o                (binary_rhs_pop),
        .rhs_data_i               (rhs_rfifo_rdata),
        .out_valid_o              (binary_out_valid),
        .out_ready_i              (!wfifo_full),
        .out_data_o               (binary_out_data),
        .done_o                   (binary_done),
        .busy_o                   (binary_busy),
        .error_o                  (binary_error),
        .input_wait_o             (binary_input_wait),
        .rhs_wait_o               (binary_rhs_wait),
        .output_stall_o           (binary_output_stall)
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
        .NAME("RHS_RFIFO"),
        .DATA_WIDTH(256),
        .DEPTH(2)
    ) i_rhs_rfifo (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .flush_i        (cfg_start),
        .full_o         (rhs_rfifo_full),
        .almost_full_o  (rhs_rfifo_almost_full),
        .empty_o        (rhs_rfifo_empty),
        .all_empty_o    (),
        .data_i         (rhs_rfifo_wdata),
        .push_i         (rhs_rfifo_push),
        .data_o         (rhs_rfifo_rdata),
        .pop_i          (rhs_rfifo_pop)
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
