`default_nettype none

module systolic_controller #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned CFG_DATA_WIDTH = 32,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned OFM_ELEM_WIDTH = 32,
    parameter int unsigned INPUT_FIFO_DEPTH = 4,
    parameter int unsigned OFM_FIFO_DEPTH = 128
)(
    input  logic clk_i,
    input  logic rst_ni,

    // MMIO slave for integrated systolic/requant register block.
    input  logic                          ctrl_req_i,
    output logic                          ctrl_gnt_o,
    input  logic [ADDR_WIDTH-1:0]         ctrl_addr_i,
    input  logic                          ctrl_we_i,
    input  logic [(CFG_DATA_WIDTH/8)-1:0] ctrl_be_i,
    input  logic [CFG_DATA_WIDTH-1:0]     ctrl_wdata_i,
    output logic                          ctrl_rvalid_o,
    output logic [CFG_DATA_WIDTH-1:0]     ctrl_rdata_o,

    // Completion pulse to cluster interrupt controller.
    output logic                      cfg_sys_done_o,

    // OBI Master for I-TCDM IFM/linebuffer reads.
    output logic                      obi_i_req_o,
    input  logic                      obi_i_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_i_addr_o,
    output logic                      obi_i_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_i_be_o,
    output logic [DATA_WIDTH-1:0]     obi_i_wdata_o,
    input  logic                      obi_i_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_i_rdata_i,

    // OBI Master for I-TCDM weight reads. This lets linebuffer prefetch run
    // concurrently with weight preload; TCDM bank conflicts are still handled
    // by the shared interconnect.
    output logic                      obi_w_req_o,
    input  logic                      obi_w_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_w_addr_o,
    output logic                      obi_w_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_w_be_o,
    output logic [DATA_WIDTH-1:0]     obi_w_wdata_o,
    input  logic                      obi_w_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_w_rdata_i,

    // OBI master for the independently quantized binary post-op operand.
    output logic                      obi_b_req_o,
    input  logic                      obi_b_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_b_addr_o,
    output logic                      obi_b_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_b_be_o,
    output logic [DATA_WIDTH-1:0]     obi_b_wdata_o,
    input  logic                      obi_b_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_b_rdata_i,


    // 4x OBI Masters for O-TCDM (Write OFM)
    output logic [3:0]                obi_o_req_o,
    input  logic [3:0]                obi_o_gnt_i,
    output logic [3:0][ADDR_WIDTH-1:0]obi_o_addr_o,
    output logic [3:0]                obi_o_we_o,
    output logic [3:0][(DATA_WIDTH/8)-1:0] obi_o_be_o,
    output logic [3:0][DATA_WIDTH-1:0]obi_o_wdata_o,
    input  logic [3:0]                obi_o_rvalid_i,
    input  logic [3:0][DATA_WIDTH-1:0]obi_o_rdata_i,

    // Performance/debug pulses exported to PMU. The array is local.
    output logic                      perf_weight_load_en_o,
    output logic                      perf_compute_en_o,
    output logic                      perf_ofm_valid_o,
    output logic                      perf_ofm_ready_o,
    output logic                      perf_start_o,
    output logic                      perf_linebuf_busy_o,
    output logic                      perf_linebuf_prefetch_busy_o,
    output logic                      perf_binary_busy_o,
    output logic [2:0]                debug_state_o,
    output logic [1:0]                debug_drain_state_o,
    output logic [4:0]                debug_linebuf_state_o,
    output logic [1:0]                debug_linebuf_fetch_main_state_o,
    output logic [2:0]                debug_linebuf_fetch_background_state_o,
    output logic [2:0]                debug_linebuf_bypass_state_o
);

    logic [2:0] state_q;

    logic [31:0] drain_cnt_q;

    localparam int unsigned PSUM_BUF_M = 256;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0]   ofm_row_t;

    logic          ofm_fifo_empty;
    logic          psum_fifo_empty;
    logic          array_pipe_ready;

    logic          fifo_flush;
    logic          input_feed_start;
    logic          input_feed_done;
    logic          input_side_ready;
    logic          weight_load_en;
    logic          weight_load_done;
    logic          weight_preload_done;
    logic          weight_preload_consume;
    logic          weight_load_service;
    logic          weight_preload_allow;
    logic          weight_depthwise_group_start;
    logic [31:0]   weight_depthwise_group_ptr;
    logic          clear_acc;
    logic          compute_en;
    input_row_t    weight_data;
    input_row_t    ifm_data;
    ofm_row_t      psum_data;
    ofm_row_t      ofm_data;
    logic          ofm_valid;
    logic          ofm_ready;
    logic          requant_out_valid;
    logic          requant_out_ready;
    logic          requant_config_invalid;
    logic          binary_config_invalid;
    logic          binary_operand_busy;
    logic          quantized_out_valid;
    logic          cfg_sys_start_i;
    logic [31:0]   cfg_sys_weight_ptr_i;
    logic [31:0]   cfg_sys_ifm_ptr_i;
    logic [31:0]   cfg_sys_ofm_ptr_i;
    logic [31:0]   cfg_sys_psum_ptr_i;
    logic [31:0]   cfg_sys_dim_m_i;
    logic          cfg_sys_accum_en_i;
    logic [31:0]   cfg_sys_ofm_row_stride_bytes_i;
    logic [31:0]   cfg_sys_ofm_tile_cols_i;
    logic [31:0]   cfg_sys_psum_row_stride_bytes_i;
    logic          cfg_requant_en_i;
    logic [ARRAY_DIM-1:0][31:0] cfg_requant_bias_i;
    logic [ARRAY_DIM-1:0][31:0] cfg_requant_multiplier_i;
    logic [ARRAY_DIM-1:0][7:0] cfg_requant_shift_i;
    logic [ARRAY_DIM-1:0][31:0] cfg_requant_zero_point_i;
    logic [31:0]   cfg_requant_clamp_min_i;
    logic [31:0]   cfg_requant_clamp_max_i;
    logic          cfg_binary_en_i;
    logic [1:0]    cfg_binary_mode_i;
    logic [31:0]   cfg_binary_rhs_ptr_i;
    logic [31:0]   cfg_binary_rhs_row_stride_bytes_i;
    logic [31:0]   cfg_binary_rhs_tile_cols_i;
    logic [31:0]   cfg_binary_lhs_multiplier_i;
    logic [6:0]    cfg_binary_lhs_shift_i;
    logic [31:0]   cfg_binary_rhs_multiplier_i;
    logic [6:0]    cfg_binary_rhs_shift_i;
    logic [31:0]   cfg_binary_output_multiplier_i;
    logic [6:0]    cfg_binary_output_shift_i;
    logic signed [31:0] cfg_binary_lhs_zero_point_i;
    logic signed [31:0] cfg_binary_rhs_zero_point_i;
    logic signed [31:0] cfg_binary_output_zero_point_i;
    logic signed [31:0] cfg_binary_clamp_min_i;
    logic signed [31:0] cfg_binary_clamp_max_i;
    logic [5:0]    cfg_binary_double_round_shift_i;
    logic          cfg_linebuf_en_i;
    logic          cfg_linebuf_coalesce_i;
    logic          cfg_linebuf_kgen_i;
    logic          cfg_linebuf_pool_i;
    logic          cfg_linebuf_c32_fast_i;
    logic          cfg_linebuf_depthwise_i;
    logic          cfg_linebuf_c32_group_stationary_i;
    logic          cfg_linebuf_generic_linear_k32_i;
    logic [31:0]   cfg_linebuf_input_base_i;
    logic [15:0]   cfg_linebuf_input_h_i;
    logic [15:0]   cfg_linebuf_input_w_i;
    logic [15:0]   cfg_linebuf_input_c_i;
    logic [15:0]   cfg_linebuf_output_w_i;
    logic [15:0]   cfg_linebuf_stride_h_i;
    logic [15:0]   cfg_linebuf_stride_w_i;
    logic [15:0]   cfg_linebuf_pad_h_i;
    logic [15:0]   cfg_linebuf_pad_w_i;
    logic [31:0]   cfg_linebuf_row_stride_bytes_i;
    logic [31:0]   cfg_linebuf_pixel_stride_bytes_i;
    logic [31:0]   cfg_linebuf_ow_step_bytes_i;
    logic [31:0]   cfg_linebuf_oh_step_bytes_i;
    logic [15:0]   cfg_linebuf_kernel_h_i;
    logic [15:0]   cfg_linebuf_kernel_w_i;
    logic [15:0]   cfg_linebuf_c_base_i;
    logic [5:0]    cfg_linebuf_lane_base_i;
    logic [31:0]   cfg_linebuf_k_tiles_i;
    logic [15:0]   cfg_linebuf_k_seed_ic_i;
    logic [7:0]    cfg_linebuf_k_seed_kw_i;
    logic [7:0]    cfg_linebuf_k_seed_kh_i;
    logic [31:0]   cfg_linebuf_spatial_m_i;
    logic [5:0]    cfg_linebuf_block_valid_bytes_i;
    logic [31:0]   cfg_linebuf_channel_addr_offset_i;
    logic [31:0]   cfg_linebuf_coalesce_k_bytes_i;
    logic [31:0]   cfg_linebuf_channel_addr_offset_eff;
    logic [31:0]   linebuf_spatial_m;
    logic [15:0]   linebuf_c_base_eff;
    logic [15:0]   linebuf_seed_ic_eff;
    logic [7:0]    linebuf_seed_kw_eff;
    logic [7:0]    linebuf_seed_kh_eff;
    logic          linebuf_kgen_multi;
    logic          linebuf_c32_group_stationary;
    logic          linebuf_pool_mode;
    logic          linebuf_depthwise_mode;
    logic          accum_active;
    logic          requant_active;
    logic          drain_enabled;
    logic          drain_tile_advance;
    logic          drain_tile_advance_overlap;
    logic          drain_tile_start;
    logic          drain_tile_start_add_rows;
    logic          drain_depthwise_group_start;
    logic [31:0]   drain_depthwise_group_output_ptr;
    logic          linebuf_use_next_cfg;
    logic          compute_service;
    logic          drain_service;
    logic          input_preload_hold;
    logic [31:0]   k_tile_idx_q;
    logic [15:0]   k_seed_ic_q;
    logic [7:0]    k_seed_kw_q;
    logic [7:0]    k_seed_kh_q;
    logic [15:0]   k_seed_ic_next;
    logic [7:0]    k_seed_kw_next;
    logic [7:0]    k_seed_kh_next;
    logic [31:0]   k_channel_offset_q;
    logic [31:0]   k_channel_offset_next;
    logic          k_tile_advance;
    logic          linebuf_has_next_k_tile;
    logic          psum_buf_active;
    logic          psum_buf_needs_external;
    logic          psum_buf_final_tile;
    logic          psum_buf_overlap_active;
    logic          psum_buf_drain_entry;

    logic          linebuf_start;
    logic          linebuf_next_tile;
    input_row_t    linebuf_row_data;
    logic          linebuf_row_valid;
    logic          linebuf_busy;
    logic          linebuf_prefetch_busy;
    logic [4:0]    linebuf_debug_state;
    input_row_t    pool_out_data;
    logic          pool_in_ready;
    logic          pool_out_valid;
    logic          pool_out_ready;
    logic [31:0]   pool_kernel_vectors;
    localparam int unsigned DW_MAX_TAPS = 25;
    localparam int unsigned DW_TAP_COUNT_W = $clog2(DW_MAX_TAPS + 1);
    input_row_t    dw_weight;
    logic [DW_TAP_COUNT_W-1:0] dw_tap_count_q;
    logic [31:0]   dw_group_idx_q;
    logic [31:0]   dw_group_input_offset_q;
    logic [5:0]    dw_group_valid_bytes;
    logic          dw_tap_is_last;
    logic          dw_engine_in_valid;
    logic          dw_engine_in_ready;
    logic          dw_engine_out_valid;
    logic          dw_engine_out_ready;
    ofm_row_t      dw_engine_out_acc;
    logic [5:0]    cfg_linebuf_block_valid_bytes_eff;

    assign array_pipe_ready = !ofm_valid || ofm_ready;
    assign psum_data = '0;
    assign clear_acc = 1'b0;
    assign perf_weight_load_en_o = weight_load_en;
    assign perf_compute_en_o = compute_en;
    assign perf_ofm_valid_o = ofm_valid;
    assign perf_ofm_ready_o = ofm_ready;
    assign perf_start_o = cfg_sys_start_i;
    assign perf_linebuf_busy_o = linebuf_busy;
    assign perf_linebuf_prefetch_busy_o = linebuf_prefetch_busy;
    assign perf_binary_busy_o = binary_operand_busy;
    assign debug_state_o = state_q;
    assign debug_linebuf_state_o = linebuf_debug_state;
    assign linebuf_spatial_m = (cfg_linebuf_spatial_m_i != 32'd0) ? cfg_linebuf_spatial_m_i : cfg_sys_dim_m_i;
    assign linebuf_kgen_multi = cfg_linebuf_en_i && cfg_linebuf_coalesce_i && cfg_linebuf_kgen_i &&
                                (cfg_linebuf_c32_group_stationary_i ||
                                 cfg_linebuf_generic_linear_k32_i) &&
                                (cfg_linebuf_k_tiles_i > 32'd1);
    assign linebuf_c32_group_stationary = linebuf_kgen_multi &&
                                          cfg_linebuf_c32_group_stationary_i;
    assign linebuf_pool_mode = cfg_linebuf_en_i && cfg_linebuf_pool_i;
    assign linebuf_depthwise_mode = cfg_linebuf_en_i && cfg_linebuf_depthwise_i;
    assign accum_active = cfg_sys_accum_en_i || (linebuf_kgen_multi && (k_tile_idx_q != 32'd0));
    assign requant_active = cfg_requant_en_i && (!linebuf_kgen_multi || !linebuf_has_next_k_tile);
    assign psum_buf_active = linebuf_kgen_multi && (cfg_sys_dim_m_i <= 32'(PSUM_BUF_M));
    assign psum_buf_needs_external = psum_buf_active && cfg_sys_accum_en_i && (k_tile_idx_q == 32'd0);
    assign psum_buf_final_tile = psum_buf_active && !linebuf_has_next_k_tile;
    assign psum_buf_overlap_active = psum_buf_active && linebuf_kgen_multi && linebuf_has_next_k_tile;
    assign linebuf_c_base_eff = linebuf_depthwise_mode ? (cfg_linebuf_c_base_i + 16'(dw_group_idx_q << 5)) :
                                cfg_linebuf_kgen_i ? {linebuf_seed_ic_eff[15:5], 5'b0} : cfg_linebuf_c_base_i;
    assign linebuf_seed_ic_eff = cfg_linebuf_kgen_i ? (linebuf_use_next_cfg ? k_seed_ic_next : k_seed_ic_q) :
                                                       cfg_linebuf_k_seed_ic_i;
    assign linebuf_seed_kw_eff = cfg_linebuf_kgen_i ? (linebuf_use_next_cfg ? k_seed_kw_next : k_seed_kw_q) :
                                                       cfg_linebuf_k_seed_kw_i;
    assign linebuf_seed_kh_eff = cfg_linebuf_kgen_i ? (linebuf_use_next_cfg ? k_seed_kh_next : k_seed_kh_q) :
                                                       cfg_linebuf_k_seed_kh_i;
    function automatic logic [31:0] kernel_tap_count(
        input logic [15:0] kernel_h,
        input logic [15:0] kernel_w
    );
        logic [31:0] kw;
        begin
            kw = {16'd0, kernel_w};
            unique case (kernel_h[2:0])
                3'd0: kernel_tap_count = 32'd0;
                3'd1: kernel_tap_count = kw;
                3'd2: kernel_tap_count = kw << 1;
                3'd3: kernel_tap_count = (kw << 1) + kw;
                3'd4: kernel_tap_count = kw << 2;
                3'd5: kernel_tap_count = (kw << 2) + kw;
                default: kernel_tap_count = 32'd25;
            endcase
        end
    endfunction

    assign pool_kernel_vectors = kernel_tap_count(cfg_linebuf_kernel_h_i, cfg_linebuf_kernel_w_i);
    assign cfg_linebuf_block_valid_bytes_eff = linebuf_depthwise_mode ?
                                               dw_group_valid_bytes :
                                               cfg_linebuf_block_valid_bytes_i;
    assign cfg_linebuf_channel_addr_offset_eff = linebuf_depthwise_mode ?
                                                 (cfg_linebuf_channel_addr_offset_i + dw_group_input_offset_q) :
                                                 linebuf_c32_group_stationary ?
                                                 (linebuf_use_next_cfg ?
                                                  k_channel_offset_next :
                                                  k_channel_offset_q) :
                                                 cfg_linebuf_channel_addr_offset_i;

    systolic_job_sequencer #(
        .ARRAY_DIM        (ARRAY_DIM),
        .DW_MAX_TAPS      (DW_MAX_TAPS),
        .DW_TAP_COUNT_W   (DW_TAP_COUNT_W)
    ) i_job_sequencer (
        .clk_i,
        .rst_ni,
        .start_i                         (cfg_sys_start_i),
        .linebuf_enable_i                (cfg_linebuf_en_i),
        .pool_mode_i                     (linebuf_pool_mode),
        .depthwise_mode_i                (linebuf_depthwise_mode),
        .kgen_multi_i                    (linebuf_kgen_multi),
        .tile_index_i                    (k_tile_idx_q),
        .has_next_tile_i                 (linebuf_has_next_k_tile),
        .psum_overlap_active_i           (psum_buf_overlap_active),
        .requant_enable_i                (cfg_requant_en_i),
        .requant_config_invalid_i        (requant_config_invalid),
        .binary_config_invalid_i         (binary_config_invalid),
        .binary_enable_i                 (cfg_binary_en_i),
        .weight_load_done_i              (weight_load_done),
        .weight_preload_done_i           (weight_preload_done),
        .input_feed_done_i               (input_feed_done),
        .array_pipe_ready_i              (array_pipe_ready),
        .linebuf_row_valid_i             (linebuf_row_valid),
        .linebuf_busy_i                  (linebuf_busy),
        .linebuf_prefetch_busy_i         (linebuf_prefetch_busy),
        .drain_remaining_i               (drain_cnt_q),
        .ofm_empty_i                     (ofm_fifo_empty),
        .binary_operand_busy_i           (binary_operand_busy),
        .depthwise_input_ready_i         (dw_engine_in_ready),
        .depthwise_output_valid_i        (dw_engine_out_valid),
        .quantized_output_valid_i        (quantized_out_valid),
        .pool_input_ready_i              (pool_in_ready),
        .pool_output_valid_i             (pool_out_valid),
        .weight_base_ptr_i               (cfg_sys_weight_ptr_i),
        .output_base_ptr_i               (cfg_sys_ofm_ptr_i),
        .input_c_i                       (cfg_linebuf_input_c_i),
        .input_h_i                       (cfg_linebuf_input_h_i),
        .input_row_stride_bytes_i        (cfg_linebuf_row_stride_bytes_i),
        .spatial_row_count_i             (linebuf_spatial_m),
        .kernel_vectors_i                (pool_kernel_vectors),
        .job_start_o                     (fifo_flush),
        .done_o                          (cfg_sys_done_o),
        .state_o                         (state_q),
        .load_service_o                  (weight_load_service),
        .compute_service_o               (compute_service),
        .drain_service_o                 (drain_service),
        .drain_active_o                  (drain_enabled),
        .weight_preload_allow_o          (weight_preload_allow),
        .input_preload_hold_o            (input_preload_hold),
        .use_next_tile_config_o          (linebuf_use_next_cfg),
        .input_feed_start_o              (input_feed_start),
        .input_side_ready_o              (input_side_ready),
        .weight_preload_consume_o        (weight_preload_consume),
        .weight_depthwise_group_start_o  (weight_depthwise_group_start),
        .weight_depthwise_group_ptr_o    (weight_depthwise_group_ptr),
        .drain_tile_advance_o            (drain_tile_advance),
        .drain_tile_advance_overlap_o    (drain_tile_advance_overlap),
        .drain_tile_start_o              (drain_tile_start),
        .drain_tile_start_add_rows_o     (drain_tile_start_add_rows),
        .drain_depthwise_group_start_o   (drain_depthwise_group_start),
        .drain_depthwise_group_output_ptr_o(drain_depthwise_group_output_ptr),
        .linebuf_start_o                 (linebuf_start),
        .linebuf_next_tile_o             (linebuf_next_tile),
        .k_tile_advance_o                (k_tile_advance),
        .depthwise_input_valid_o         (dw_engine_in_valid),
        .depthwise_tap_index_o           (dw_tap_count_q),
        .depthwise_tap_is_last_o         (dw_tap_is_last),
        .depthwise_group_index_o         (dw_group_idx_q),
        .depthwise_group_input_offset_o  (dw_group_input_offset_q),
        .depthwise_group_valid_bytes_o   (dw_group_valid_bytes)
    );

    systolic_k_tile_scheduler #(
        .ARRAY_DIM (ARRAY_DIM)
    ) i_k_tile_scheduler (
        .clk_i,
        .rst_ni,
        .job_start_i          (fifo_flush),
        .advance_i            (k_tile_advance),
        .kgen_multi_i         (linebuf_kgen_multi),
        .c32_group_stationary_i(linebuf_c32_group_stationary),
        .generic_linear_k32_i (cfg_linebuf_generic_linear_k32_i),
        .k_tiles_i            (cfg_linebuf_k_tiles_i),
        .input_c_i            (cfg_linebuf_input_c_i),
        .kernel_h_i           (cfg_linebuf_kernel_h_i),
        .kernel_w_i           (cfg_linebuf_kernel_w_i),
        .initial_seed_ic_i    (cfg_linebuf_k_seed_ic_i),
        .initial_seed_kw_i    (cfg_linebuf_k_seed_kw_i),
        .initial_seed_kh_i    (cfg_linebuf_k_seed_kh_i),
        .channel_addr_offset_i(cfg_linebuf_channel_addr_offset_i),
        .has_next_o           (linebuf_has_next_k_tile),
        .tile_index_o         (k_tile_idx_q),
        .seed_ic_o            (k_seed_ic_q),
        .seed_kw_o            (k_seed_kw_q),
        .seed_kh_o            (k_seed_kh_q),
        .channel_offset_o     (k_channel_offset_q),
        .next_seed_ic_o       (k_seed_ic_next),
        .next_seed_kw_o       (k_seed_kw_next),
        .next_seed_kh_o       (k_seed_kh_next),
        .next_channel_offset_o(k_channel_offset_next)
    );

    systolic_maxpool_engine #(
        .LANES           (ARRAY_DIM),
        .ELEM_WIDTH      (INPUT_ELEM_WIDTH),
        .TAP_COUNT_WIDTH (8)
    ) i_maxpool_engine (
        .clk_i,
        .rst_ni,
        .flush_i          (fifo_flush),
        .kernel_vectors_i (pool_kernel_vectors),
        .in_data_i        (linebuf_row_data),
        .in_valid_i       (linebuf_pool_mode && compute_service && linebuf_row_valid),
        .in_ready_o       (pool_in_ready),
        .out_data_o       (pool_out_data),
        .out_valid_o      (pool_out_valid),
        .out_ready_i      (pool_out_ready)
    );

    depthwise_mac_engine #(
        .ARRAY_DIM       (ARRAY_DIM),
        .INPUT_ELEM_WIDTH(INPUT_ELEM_WIDTH),
        .ACC_WIDTH       (OFM_ELEM_WIDTH)
    ) i_depthwise_mac_engine (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .flush_i      (fifo_flush),
        .in_valid_i   (dw_engine_in_valid),
        .in_ready_o   (dw_engine_in_ready),
        .ifm_i        (linebuf_row_data),
        .weight_i     (dw_weight),
        .clear_i      (dw_tap_count_q == '0),
        .last_i       (dw_tap_is_last),
        .valid_lanes_i(dw_group_valid_bytes),
        .out_valid_o  (dw_engine_out_valid),
        .out_ready_i  (dw_engine_out_ready),
        .acc_o        (dw_engine_out_acc)
    );

    systolic_output_drain #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .OFM_ELEM_WIDTH   (OFM_ELEM_WIDTH),
        .INPUT_FIFO_DEPTH (INPUT_FIFO_DEPTH),
        .OFM_FIFO_DEPTH   (OFM_FIFO_DEPTH)
    ) i_output_drain (
        .clk_i,
        .rst_ni,
        .job_start_i                     (fifo_flush),
        .tile_advance_i                  (drain_tile_advance),
        .tile_advance_overlap_i          (drain_tile_advance_overlap),
        .tile_start_i                    (drain_tile_start),
        .tile_start_add_rows_i           (drain_tile_start_add_rows),
        .depthwise_group_start_i         (drain_depthwise_group_start),
        .depthwise_group_output_ptr_i    (drain_depthwise_group_output_ptr),
        .drain_active_i                  (drain_enabled),
        .compute_phase_i                 (compute_service),
        .pool_mode_i                     (linebuf_pool_mode),
        .depthwise_mode_i                (linebuf_depthwise_mode),
        .external_accum_enable_i         (cfg_sys_accum_en_i),
        .accum_active_i                  (accum_active),
        .requant_active_i                (requant_active),
        .psum_buf_active_i               (psum_buf_active),
        .psum_buf_needs_external_i       (psum_buf_needs_external),
        .psum_buf_final_tile_i           (psum_buf_final_tile),
        .k_tile_idx_i                    (k_tile_idx_q),
        .ofm_base_ptr_i                  (cfg_sys_ofm_ptr_i),
        .psum_base_ptr_i                 (cfg_sys_psum_ptr_i),
        .row_count_i                     (cfg_sys_dim_m_i),
        .spatial_row_count_i             (linebuf_spatial_m),
        .ofm_row_stride_bytes_i          (cfg_sys_ofm_row_stride_bytes_i),
        .ofm_tile_cols_i                 (cfg_sys_ofm_tile_cols_i),
        .psum_row_stride_bytes_i         (cfg_sys_psum_row_stride_bytes_i),
        .result_i                        (ofm_data),
        .result_valid_i                  (ofm_valid),
        .result_ready_o                  (ofm_ready),
        .depthwise_result_i              (dw_engine_out_acc),
        .depthwise_result_valid_i        (dw_engine_out_valid),
        .depthwise_result_ready_o        (dw_engine_out_ready),
        .pool_result_i                   (pool_out_data),
        .pool_result_valid_i             (pool_out_valid),
        .pool_result_ready_o             (pool_out_ready),
        .requant_enable_i                (cfg_requant_en_i),
        .requant_bias_i                  (cfg_requant_bias_i),
        .requant_multiplier_i            (cfg_requant_multiplier_i),
        .requant_shift_i                 (cfg_requant_shift_i),
        .requant_zero_point_i            (cfg_requant_zero_point_i),
        .requant_clamp_min_i             (cfg_requant_clamp_min_i),
        .requant_clamp_max_i             (cfg_requant_clamp_max_i),
        .binary_enable_i                 (cfg_binary_en_i),
        .binary_mode_i                   (cfg_binary_mode_i),
        .binary_rhs_ptr_i                (cfg_binary_rhs_ptr_i),
        .binary_rhs_row_stride_bytes_i   (cfg_binary_rhs_row_stride_bytes_i),
        .binary_rhs_tile_cols_i          (cfg_binary_rhs_tile_cols_i),
        .binary_lhs_multiplier_i         (cfg_binary_lhs_multiplier_i),
        .binary_lhs_shift_i              (cfg_binary_lhs_shift_i),
        .binary_rhs_multiplier_i         (cfg_binary_rhs_multiplier_i),
        .binary_rhs_shift_i              (cfg_binary_rhs_shift_i),
        .binary_output_multiplier_i      (cfg_binary_output_multiplier_i),
        .binary_output_shift_i           (cfg_binary_output_shift_i),
        .binary_lhs_zero_point_i         (cfg_binary_lhs_zero_point_i),
        .binary_rhs_zero_point_i         (cfg_binary_rhs_zero_point_i),
        .binary_output_zero_point_i      (cfg_binary_output_zero_point_i),
        .binary_clamp_min_i              (cfg_binary_clamp_min_i),
        .binary_clamp_max_i              (cfg_binary_clamp_max_i),
        .binary_double_round_shift_i     (cfg_binary_double_round_shift_i),
        .obi_b_req_o,
        .obi_b_gnt_i,
        .obi_b_addr_o,
        .obi_b_we_o,
        .obi_b_be_o,
        .obi_b_wdata_o,
        .obi_b_rvalid_i,
        .obi_b_rdata_i,
        .obi_o_req_o,
        .obi_o_gnt_i,
        .obi_o_addr_o,
        .obi_o_we_o,
        .obi_o_be_o,
        .obi_o_wdata_o,
        .obi_o_rvalid_i,
        .obi_o_rdata_i,
        .remaining_rows_o               (drain_cnt_q),
        .ofm_empty_o                     (ofm_fifo_empty),
        .psum_empty_o                    (psum_fifo_empty),
        .psum_buf_drain_entry_o          (psum_buf_drain_entry),
        .binary_busy_o                   (binary_operand_busy),
        .postprocess_out_valid_o         (quantized_out_valid),
        .requant_config_invalid_o        (requant_config_invalid),
        .binary_config_invalid_o         (binary_config_invalid),
        .debug_requant_out_valid_o       (requant_out_valid),
        .debug_requant_out_ready_o       (requant_out_ready),
        .debug_state_o                   (debug_drain_state_o)
    );


    npu_systolic_array #(
        .ARRAY_DIM(ARRAY_DIM)
    ) i_systolic_array (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .weight_load_en_i (weight_load_en),
        .clear_acc_i      (clear_acc),
        .compute_en_i     (compute_en),
        .ofm_ready_i      (ofm_ready),
        .weight_data_i    (weight_data),
        .ifm_data_i       (ifm_data),
        .psum_data_i      (psum_data),
        .ofm_data_o       (ofm_data),
        .ofm_valid_o      (ofm_valid)
    );

    systolic_weight_engine #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH),
        .FIFO_DEPTH       (INPUT_FIFO_DEPTH),
        .DW_MAX_TAPS      (DW_MAX_TAPS)
    ) i_weight_engine (
        .clk_i,
        .rst_ni,
        .job_start_i                 (fifo_flush),
        .load_service_i              (weight_load_service),
        .preload_service_i           (drain_service),
        .preload_allow_i             (weight_preload_allow),
        .preload_consume_i           (weight_preload_consume),
        .depthwise_group_start_i     (weight_depthwise_group_start),
        .depthwise_mode_i            (linebuf_depthwise_mode),
        .array_pipe_ready_i          (array_pipe_ready),
        .weight_base_ptr_i           (cfg_sys_weight_ptr_i),
        .depthwise_group_weight_ptr_i(weight_depthwise_group_ptr),
        .depthwise_tap_count_i       (pool_kernel_vectors),
        .depthwise_tap_index_i       (32'(dw_tap_count_q)),
        .next_tile_index_i           (k_tile_idx_q + 32'd1),
        .obi_req_o                   (obi_w_req_o),
        .obi_gnt_i                   (obi_w_gnt_i),
        .obi_addr_o                  (obi_w_addr_o),
        .obi_we_o                    (obi_w_we_o),
        .obi_be_o                    (obi_w_be_o),
        .obi_wdata_o                 (obi_w_wdata_o),
        .obi_rvalid_i                (obi_w_rvalid_i),
        .obi_rdata_i                 (obi_w_rdata_i),
        .weight_load_en_o            (weight_load_en),
        .weight_data_o               (weight_data),
        .depthwise_weight_o          (dw_weight),
        .load_done_o                 (weight_load_done),
        .preload_done_o              (weight_preload_done)
    );

    systolic_ctrl_regs #(
        .ADDR_WIDTH(ADDR_WIDTH)
    ) i_systolic_ctrl_regs (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .req_i              (ctrl_req_i),
        .gnt_o              (ctrl_gnt_o),
        .addr_i             (ctrl_addr_i),
        .we_i               (ctrl_we_i),
        .be_i               (ctrl_be_i),
        .wdata_i            (ctrl_wdata_i),
        .rvalid_o           (ctrl_rvalid_o),
        .rdata_o            (ctrl_rdata_o),
        .cfg_sys_start_o    (cfg_sys_start_i),
        .cfg_sys_weight_ptr_o(cfg_sys_weight_ptr_i),
        .cfg_sys_ifm_ptr_o  (cfg_sys_ifm_ptr_i),
        .cfg_sys_ofm_ptr_o  (cfg_sys_ofm_ptr_i),
        .cfg_sys_psum_ptr_o (cfg_sys_psum_ptr_i),
        .cfg_sys_dim_m_o    (cfg_sys_dim_m_i),
        .cfg_sys_accum_en_o (cfg_sys_accum_en_i),
        .cfg_sys_ofm_row_stride_bytes_o(cfg_sys_ofm_row_stride_bytes_i),
        .cfg_sys_ofm_tile_cols_o(cfg_sys_ofm_tile_cols_i),
        .cfg_sys_psum_row_stride_bytes_o(cfg_sys_psum_row_stride_bytes_i),
        .cfg_requant_en_o   (cfg_requant_en_i),
        .cfg_requant_bias_o (cfg_requant_bias_i),
        .cfg_requant_multiplier_o(cfg_requant_multiplier_i),
        .cfg_requant_shift_o(cfg_requant_shift_i),
        .cfg_requant_zero_point_o(cfg_requant_zero_point_i),
        .cfg_requant_clamp_min_o(cfg_requant_clamp_min_i),
        .cfg_requant_clamp_max_o(cfg_requant_clamp_max_i),
        .cfg_binary_en_o    (cfg_binary_en_i),
        .cfg_binary_mode_o  (cfg_binary_mode_i),
        .cfg_binary_rhs_ptr_o(cfg_binary_rhs_ptr_i),
        .cfg_binary_rhs_row_stride_bytes_o(cfg_binary_rhs_row_stride_bytes_i),
        .cfg_binary_rhs_tile_cols_o(cfg_binary_rhs_tile_cols_i),
        .cfg_binary_lhs_multiplier_o(cfg_binary_lhs_multiplier_i),
        .cfg_binary_lhs_shift_o(cfg_binary_lhs_shift_i),
        .cfg_binary_rhs_multiplier_o(cfg_binary_rhs_multiplier_i),
        .cfg_binary_rhs_shift_o(cfg_binary_rhs_shift_i),
        .cfg_binary_output_multiplier_o(cfg_binary_output_multiplier_i),
        .cfg_binary_output_shift_o(cfg_binary_output_shift_i),
        .cfg_binary_lhs_zero_point_o(cfg_binary_lhs_zero_point_i),
        .cfg_binary_rhs_zero_point_o(cfg_binary_rhs_zero_point_i),
        .cfg_binary_output_zero_point_o(cfg_binary_output_zero_point_i),
        .cfg_binary_clamp_min_o(cfg_binary_clamp_min_i),
        .cfg_binary_clamp_max_o(cfg_binary_clamp_max_i),
        .cfg_binary_double_round_shift_o(cfg_binary_double_round_shift_i),
        .cfg_linebuf_en_o   (cfg_linebuf_en_i),
        .cfg_linebuf_coalesce_o(cfg_linebuf_coalesce_i),
        .cfg_linebuf_pool_o (cfg_linebuf_pool_i),
        .cfg_linebuf_c32_fast_o(cfg_linebuf_c32_fast_i),
        .cfg_linebuf_depthwise_o(cfg_linebuf_depthwise_i),
        .cfg_linebuf_c32_group_stationary_o(cfg_linebuf_c32_group_stationary_i),
        .cfg_linebuf_generic_linear_k32_o(cfg_linebuf_generic_linear_k32_i),
        .cfg_linebuf_kgen_o (cfg_linebuf_kgen_i),
        .cfg_linebuf_input_base_o(cfg_linebuf_input_base_i),
        .cfg_linebuf_input_h_o(cfg_linebuf_input_h_i),
        .cfg_linebuf_input_w_o(cfg_linebuf_input_w_i),
        .cfg_linebuf_input_c_o(cfg_linebuf_input_c_i),
        .cfg_linebuf_output_w_o(cfg_linebuf_output_w_i),
        .cfg_linebuf_stride_h_o(cfg_linebuf_stride_h_i),
        .cfg_linebuf_stride_w_o(cfg_linebuf_stride_w_i),
        .cfg_linebuf_pad_h_o(cfg_linebuf_pad_h_i),
        .cfg_linebuf_pad_w_o(cfg_linebuf_pad_w_i),
        .cfg_linebuf_row_stride_bytes_o(cfg_linebuf_row_stride_bytes_i),
        .cfg_linebuf_pixel_stride_bytes_o(cfg_linebuf_pixel_stride_bytes_i),
        .cfg_linebuf_ow_step_bytes_o(cfg_linebuf_ow_step_bytes_i),
        .cfg_linebuf_oh_step_bytes_o(cfg_linebuf_oh_step_bytes_i),
        .cfg_linebuf_kernel_h_o(cfg_linebuf_kernel_h_i),
        .cfg_linebuf_kernel_w_o(cfg_linebuf_kernel_w_i),
        .cfg_linebuf_c_base_o(cfg_linebuf_c_base_i),
        .cfg_linebuf_lane_base_o(cfg_linebuf_lane_base_i),
        .cfg_linebuf_k_tiles_o(cfg_linebuf_k_tiles_i),
        .cfg_linebuf_k_seed_ic_o(cfg_linebuf_k_seed_ic_i),
        .cfg_linebuf_k_seed_kw_o(cfg_linebuf_k_seed_kw_i),
        .cfg_linebuf_k_seed_kh_o(cfg_linebuf_k_seed_kh_i),
        .cfg_linebuf_spatial_m_o(cfg_linebuf_spatial_m_i),
        .cfg_linebuf_block_valid_bytes_o(cfg_linebuf_block_valid_bytes_i),
        .cfg_linebuf_channel_addr_offset_o(cfg_linebuf_channel_addr_offset_i),
        .cfg_linebuf_coalesce_k_bytes_o(cfg_linebuf_coalesce_k_bytes_i),
        .cfg_sys_done_i     (cfg_sys_done_o)
    );

    systolic_input_engine #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH),
        .FIFO_DEPTH       (INPUT_FIFO_DEPTH),
        .MAX_INPUT_W      (640)
    ) i_input_engine (
        .clk_i,
        .rst_ni,
        .job_start_i             (fifo_flush),
        .feed_start_i            (input_feed_start),
        .feed_service_i          (compute_service),
        .drain_service_i         (drain_service),
        .linebuf_start_i         (linebuf_start),
        .linebuf_next_tile_i     (linebuf_next_tile),
        .preload_service_i       (drain_service),
        .preload_has_next_i      (linebuf_has_next_k_tile),
        .preload_hold_i          (input_preload_hold),
        .linebuf_enable_i        (cfg_linebuf_en_i),
        .side_stream_mode_i      (linebuf_pool_mode || linebuf_depthwise_mode),
        .array_pipe_ready_i      (array_pipe_ready),
        .side_ready_i            (input_side_ready),
        .ifm_base_ptr_i          (cfg_sys_ifm_ptr_i),
        .row_count_i             (cfg_sys_dim_m_i),
        .cfg_spatial_m_i         (linebuf_spatial_m),
        .cfg_k_tiles_i           (cfg_linebuf_k_tiles_i),
        .cfg_origin_base_i       (cfg_linebuf_input_base_i),
        .cfg_row_stride_bytes_i  (cfg_linebuf_row_stride_bytes_i),
        .cfg_pixel_stride_bytes_i(cfg_linebuf_pixel_stride_bytes_i),
        .cfg_ow_step_bytes_i     (cfg_linebuf_ow_step_bytes_i),
        .cfg_oh_step_bytes_i     (cfg_linebuf_oh_step_bytes_i),
        .cfg_input_h_i           (cfg_linebuf_input_h_i),
        .cfg_input_w_i           (cfg_linebuf_input_w_i),
        .cfg_input_c_i           (cfg_linebuf_input_c_i),
        .cfg_output_w_i          (cfg_linebuf_output_w_i),
        .cfg_kernel_h_i          (cfg_linebuf_kernel_h_i),
        .cfg_kernel_w_i          (cfg_linebuf_kernel_w_i),
        .cfg_stride_h_i          (cfg_linebuf_stride_h_i),
        .cfg_stride_w_i          (cfg_linebuf_stride_w_i),
        .cfg_pad_h_i             (cfg_linebuf_pad_h_i),
        .cfg_pad_w_i             (cfg_linebuf_pad_w_i),
        .cfg_c_base_i            (linebuf_c_base_eff),
        .cfg_lane_base_i         (cfg_linebuf_lane_base_i),
        .cfg_coalesce_i          (cfg_linebuf_coalesce_i),
        .cfg_kgen_i              (cfg_linebuf_kgen_i),
        .cfg_pool_i              (cfg_linebuf_pool_i),
        .cfg_c32_fast_i          (cfg_linebuf_c32_fast_i),
        .cfg_depthwise_i         (cfg_linebuf_depthwise_i),
        .cfg_block_valid_bytes_i (cfg_linebuf_block_valid_bytes_eff),
        .cfg_channel_addr_offset_i(cfg_linebuf_channel_addr_offset_eff),
        .cfg_coalesce_k_bytes_i  (cfg_linebuf_coalesce_k_bytes_i),
        .cfg_k_seed_kh_i         (linebuf_seed_kh_eff),
        .cfg_k_seed_kw_i         (linebuf_seed_kw_eff),
        .cfg_k_seed_ic_i         (linebuf_seed_ic_eff),
        .obi_req_o               (obi_i_req_o),
        .obi_gnt_i               (obi_i_gnt_i),
        .obi_addr_o              (obi_i_addr_o),
        .obi_we_o                (obi_i_we_o),
        .obi_be_o                (obi_i_be_o),
        .obi_wdata_o             (obi_i_wdata_o),
        .obi_rvalid_i            (obi_i_rvalid_i),
        .obi_rdata_i             (obi_i_rdata_i),
        .compute_en_o            (compute_en),
        .compute_data_o          (ifm_data),
        .feed_done_o             (input_feed_done),
        .side_data_o             (linebuf_row_data),
        .side_valid_o            (linebuf_row_valid),
        .linebuf_row_ready_o     (),
        .linebuf_busy_o          (linebuf_busy),
        .linebuf_done_o          (),
        .prefetch_busy_o         (linebuf_prefetch_busy),
        .request_count_o         (),
        .response_count_o        (),
        .emitted_vectors_o       (),
        .fetch_beats_o           (),
        .bypass_vectors_o        (),
        .debug_state_o           (linebuf_debug_state),
        .debug_fetch_main_state_o(debug_linebuf_fetch_main_state_o),
        .debug_fetch_background_state_o(debug_linebuf_fetch_background_state_o),
        .debug_bypass_state_o    (debug_linebuf_bypass_state_o)
    );

endmodule
