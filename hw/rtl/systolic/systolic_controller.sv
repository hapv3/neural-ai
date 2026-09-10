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
    output logic [2:0]                debug_state_o,
    output logic [1:0]                debug_drain_state_o,
    output logic [4:0]                debug_linebuf_state_o
);

    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        COMPUTE,
        WAIT_DRAIN,
        DONE
    } state_e;

    state_e state_q;
    state_e state_d;

    logic [31:0] i_ptr_q, i_ptr_d;
    logic [31:0] req_cnt_q, req_cnt_d; // Counter for requests
    logic [31:0] rsp_cnt_q, rsp_cnt_d; // Counter for responses
    logic [31:0] drain_cnt_q;

    localparam int unsigned ARRAY_FLUSH_CYCLES = (2 * ARRAY_DIM) - 1;
    localparam int unsigned ARRAY_FLUSH_COUNT_W = $clog2(ARRAY_FLUSH_CYCLES + 1);
    localparam int unsigned PSUM_BUF_M = 256;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0]   ofm_row_t;

    input_row_t    ifm_fifo_data;
    input_row_t    ifm_fifo_out;
    logic          ifm_fifo_push;
    logic          ifm_fifo_pop;
    logic          ifm_fifo_full;
    logic          ifm_fifo_empty;

    logic          ofm_fifo_empty;
    logic          psum_fifo_empty;
    logic          array_pipe_ready;

    logic          fifo_flush;
    logic          weight_load_en;
    logic          weight_load_done;
    logic          weight_preload_done;
    logic          weight_preload_consume;
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
    logic [31:0]   k_tile_idx_q, k_tile_idx_d;
    logic [15:0]   k_seed_ic_q, k_seed_ic_d;
    logic [7:0]    k_seed_kw_q, k_seed_kw_d;
    logic [7:0]    k_seed_kh_q, k_seed_kh_d;
    logic [15:0]   k_seed_ic_next;
    logic [7:0]    k_seed_kw_next;
    logic [7:0]    k_seed_kh_next;
    logic [31:0]   k_channel_offset_q, k_channel_offset_d;
    logic [31:0]   k_channel_offset_next;
    logic [ARRAY_FLUSH_COUNT_W-1:0] array_flush_cnt_q, array_flush_cnt_d;
    logic          linebuf_has_next_k_tile;
    logic          psum_buf_active;
    logic          psum_buf_needs_external;
    logic          psum_buf_final_tile;
    logic          psum_buf_overlap_active;
    logic          psum_buf_overlap_next_ready;
    logic          psum_buf_overlap_next_safe;
    logic          psum_buf_drain_entry;

    logic          linebuf_start;
    logic          linebuf_next_tile;
    logic          linebuf_prefetch;
    logic          linebuf_prefetch_req_q, linebuf_prefetch_req_d;
    logic          linebuf_obi_req;
    logic [ADDR_WIDTH-1:0] linebuf_obi_addr;
    input_row_t    linebuf_row_data;
    logic          linebuf_row_valid;
    logic          linebuf_row_ready;
    logic          linebuf_done;
    logic          linebuf_busy;
    logic          linebuf_prefetch_busy;
    logic [31:0]   linebuf_emitted_vectors;
    logic [31:0]   linebuf_fetch_beats;
    logic [31:0]   linebuf_bypass_vectors;
    logic [4:0]    linebuf_debug_state;
    input_row_t    pool_out_data;
    logic          pool_in_ready;
    logic          pool_out_valid;
    logic          pool_out_ready;
    logic [31:0]   pool_kernel_vectors;
    localparam int unsigned DW_MAX_TAPS = 25;
    localparam int unsigned DW_TAP_COUNT_W = $clog2(DW_MAX_TAPS + 1);
    input_row_t    dw_weight;
    logic [DW_TAP_COUNT_W-1:0] dw_tap_count_q, dw_tap_count_d;
    logic [31:0]   dw_group_idx_q, dw_group_idx_d;
    logic [31:0]   dw_group_count;
    logic [31:0]   dw_group_span_bytes;
    logic [31:0]   dw_group_output_bytes;
    logic [31:0]   dw_weight_group_bytes;
    logic [31:0]   dw_group_input_offset_q, dw_group_input_offset_d;
    logic [31:0]   dw_group_output_offset_q, dw_group_output_offset_d;
    logic [31:0]   dw_group_weight_offset_q, dw_group_weight_offset_d;
    logic          dw_last_group;
    logic [5:0]    dw_group_valid_bytes;
    logic          dw_tap_is_last;
    logic          dw_engine_in_valid;
    logic          dw_engine_in_ready;
    logic          dw_engine_out_valid;
    logic          dw_engine_out_ready;
    ofm_row_t      dw_engine_out_acc;
    logic [5:0]    cfg_linebuf_block_valid_bytes_eff;

    function automatic logic [5:0] depthwise_group_valid_bytes(
        input logic [15:0] input_c,
        input logic [31:0] group_idx
    );
        logic [31:0] group_base;
        logic [31:0] remaining;
        begin
            group_base = group_idx << 5;
            if (group_base >= {16'd0, input_c}) begin
                depthwise_group_valid_bytes = 6'd0;
            end else begin
                remaining = {16'd0, input_c} - group_base;
                depthwise_group_valid_bytes = (remaining >= 32'd32) ? 6'd32 : remaining[5:0];
            end
        end
    endfunction

    assign fifo_flush = (state_q == IDLE) && cfg_sys_start_i;

    assign ifm_fifo_data    = obi_i_rdata_i;
    assign array_pipe_ready = !ofm_valid || ofm_ready;
    assign psum_data = '0;
    assign perf_weight_load_en_o = weight_load_en;
    assign perf_compute_en_o = compute_en;
    assign perf_ofm_valid_o = ofm_valid;
    assign perf_ofm_ready_o = ofm_ready;
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
    assign dw_tap_is_last = (({27'd0, dw_tap_count_q} + 32'd1) == pool_kernel_vectors);
    assign linebuf_has_next_k_tile = linebuf_kgen_multi && ((k_tile_idx_q + 32'd1) < cfg_linebuf_k_tiles_i);
    assign accum_active = cfg_sys_accum_en_i || (linebuf_kgen_multi && (k_tile_idx_q != 32'd0));
    assign requant_active = cfg_requant_en_i && (!linebuf_kgen_multi || !linebuf_has_next_k_tile);
    assign psum_buf_active = linebuf_kgen_multi && (cfg_sys_dim_m_i <= 32'(PSUM_BUF_M));
    assign psum_buf_needs_external = psum_buf_active && cfg_sys_accum_en_i && (k_tile_idx_q == 32'd0);
    assign psum_buf_final_tile = psum_buf_active && !linebuf_has_next_k_tile;
    assign psum_buf_overlap_active = psum_buf_active && linebuf_kgen_multi && linebuf_has_next_k_tile;
    assign psum_buf_overlap_next_ready = psum_buf_overlap_active && linebuf_has_next_k_tile &&
                                         weight_preload_done && !linebuf_prefetch_busy &&
                                         (array_flush_cnt_q == '0);
    // Direct WAIT_DRAIN->COMPUTE overlap preserves the current drain/prefetch
    // state. OFM FIFO ordering keeps an external-psum first tile ahead of later
    // on-chip psum-buffer tiles, so the next tile cannot consume a row before
    // the external accumulation wrote that row into the psum buffer.
    assign psum_buf_overlap_next_safe = psum_buf_overlap_next_ready;
    assign drain_enabled = !linebuf_pool_mode && !linebuf_depthwise_mode &&
                           ((state_q == LOAD_WEIGHTS) || (state_q == COMPUTE) || (state_q == WAIT_DRAIN));
    assign linebuf_use_next_cfg = (state_q == WAIT_DRAIN) && linebuf_has_next_k_tile;
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
    assign dw_group_count = ({16'd0, cfg_linebuf_input_c_i} + 32'd31) >> 5;
    assign dw_group_span_bytes = {16'd0, cfg_linebuf_input_h_i} * cfg_linebuf_row_stride_bytes_i;
    assign dw_group_output_bytes = linebuf_spatial_m << 5;
    assign dw_weight_group_bytes = pool_kernel_vectors << 5;
    assign dw_last_group = (dw_group_idx_q + 32'd1) >= dw_group_count;
    assign dw_group_valid_bytes = depthwise_group_valid_bytes(cfg_linebuf_input_c_i, dw_group_idx_q);
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

    function automatic void advance_k_seed32_c32_group_stationary(
        input  logic [7:0]  kh_i,
        input  logic [7:0]  kw_i,
        input  logic [15:0] ic_i,
        input  logic [15:0] input_c_i,
        input  logic [15:0] kernel_h_i,
        input  logic [15:0] kernel_w_i,
        output logic [7:0]  kh_o,
        output logic [7:0]  kw_o,
        output logic [15:0] ic_o
    );
        logic [7:0]  kh;
        logic [7:0]  kw;
        logic [15:0] ic;
        begin
            kh = kh_i;
            kw = kw_i;
            ic = {ic_i[15:5], 5'b0};
            if ((kw + 8'd1) == kernel_w_i[7:0]) begin
                kw = '0;
                if ((kh + 8'd1) == kernel_h_i[7:0]) begin
                    kh = '0;
                    if ((ic + 16'(ARRAY_DIM)) >= input_c_i) begin
                        ic = '0;
                    end else begin
                        ic = ic + 16'(ARRAY_DIM);
                    end
                end else begin
                    kh = kh + 8'd1;
                end
            end else begin
                kw = kw + 8'd1;
            end
            kh_o = kh;
            kw_o = kw;
            ic_o = ic;
        end
    endfunction

    function automatic void advance_k_seed32_generic_linear_k32(
        input  logic [7:0]  kh_i,
        input  logic [7:0]  kw_i,
        input  logic [15:0] ic_i,
        input  logic [15:0] kernel_h_i,
        input  logic [15:0] kernel_w_i,
        output logic [7:0]  kh_o,
        output logic [7:0]  kw_o,
        output logic [15:0] ic_o
    );
        begin
            kh_o = kh_i;
            kw_o = kw_i;
            ic_o = ic_i;
            if ((kw_i + 8'd1) == kernel_w_i[7:0]) begin
                kw_o = '0;
                if ((kh_i + 8'd1) == kernel_h_i[7:0]) begin
                    kh_o = '0;
                end else begin
                    kh_o = kh_i + 8'd1;
                end
            end else begin
                kw_o = kw_i + 8'd1;
            end
        end
    endfunction

    always_comb begin
        if (linebuf_c32_group_stationary) begin
            advance_k_seed32_c32_group_stationary(k_seed_kh_q,
                                                  k_seed_kw_q,
                                                  k_seed_ic_q,
                                                  cfg_linebuf_input_c_i,
                                                  cfg_linebuf_kernel_h_i,
                                                  cfg_linebuf_kernel_w_i,
                                                  k_seed_kh_next,
                                                  k_seed_kw_next,
                                                  k_seed_ic_next);
        end else if (linebuf_kgen_multi && cfg_linebuf_generic_linear_k32_i) begin
            advance_k_seed32_generic_linear_k32(k_seed_kh_q,
                                                k_seed_kw_q,
                                                k_seed_ic_q,
                                                cfg_linebuf_kernel_h_i,
                                                cfg_linebuf_kernel_w_i,
                                                k_seed_kh_next,
                                                k_seed_kw_next,
                                                k_seed_ic_next);
        end else begin
            k_seed_kh_next = cfg_linebuf_k_seed_kh_i;
            k_seed_kw_next = cfg_linebuf_k_seed_kw_i;
            k_seed_ic_next = cfg_linebuf_k_seed_ic_i;
        end

        k_channel_offset_next = k_channel_offset_q;
        if (linebuf_c32_group_stationary && (k_seed_ic_next[15:5] != k_seed_ic_q[15:5])) begin
            if (k_seed_ic_next[15:5] == 11'd0) begin
                k_channel_offset_next = 32'd0;
            end else begin
                k_channel_offset_next = k_channel_offset_q + cfg_linebuf_channel_addr_offset_i;
            end
        end
    end

    task automatic advance_to_next_k_tile(input logic launch_direct_compute);
        begin
            k_seed_kh_d = k_seed_kh_next;
            k_seed_kw_d = k_seed_kw_next;
            k_seed_ic_d = k_seed_ic_next;
            k_channel_offset_d = k_channel_offset_next;
            k_tile_idx_d = k_tile_idx_q + 32'd1;
            i_ptr_d = cfg_sys_ifm_ptr_i;
            drain_tile_advance = 1'b1;
            drain_tile_advance_overlap = launch_direct_compute;
            if (launch_direct_compute) begin
                req_cnt_d = cfg_sys_dim_m_i;
                rsp_cnt_d = cfg_sys_dim_m_i;
                weight_preload_consume = 1'b1;
                linebuf_next_tile = 1'b1;
                state_d = COMPUTE;
            end else begin
                state_d = LOAD_WEIGHTS;
            end
        end
    endtask

    task automatic service_linebuf_prefetch_engine();
        begin
            if (linebuf_has_next_k_tile) begin
                linebuf_prefetch_req_d = linebuf_prefetch_busy ||
                                         (array_flush_cnt_q != '0) ||
                                         (drain_cnt_q != 0) ||
                                         !ofm_fifo_empty;
            end else begin
                linebuf_prefetch_req_d = 1'b0;
            end

            if (linebuf_prefetch_req_q) begin
                obi_i_req_o = linebuf_obi_req;
                obi_i_addr_o = linebuf_obi_addr;
            end
        end
    endtask

    task automatic launch_linebuf_compute_engine();
        begin
            obi_i_req_o = linebuf_obi_req;
            obi_i_addr_o = linebuf_obi_addr;
            if (linebuf_row_valid && array_pipe_ready) begin
                compute_en = 1'b1;
                clear_acc = 1'b0;
                ifm_data = linebuf_row_data;
                linebuf_row_ready = 1'b1;
                req_cnt_d = req_cnt_q - 1;
            end
            if (req_cnt_q == 1 && linebuf_row_valid && array_pipe_ready) begin
                rsp_cnt_d = '0;
                array_flush_cnt_d = ARRAY_FLUSH_COUNT_W'(ARRAY_FLUSH_CYCLES);
                state_d = WAIT_DRAIN;
            end
        end
    endtask

    task automatic launch_fifo_compute_engine();
        begin
            if (req_cnt_q > 0) begin
                obi_i_req_o = !ifm_fifo_full && array_pipe_ready;
                obi_i_addr_o = i_ptr_q;
                if (obi_i_req_o && obi_i_gnt_i) begin
                    i_ptr_d = i_ptr_q + 32;
                    req_cnt_d = req_cnt_q - 1;
                end
            end
            ifm_fifo_push = obi_i_rvalid_i && !ifm_fifo_full;
            if (!ifm_fifo_empty && array_pipe_ready) begin
                compute_en = 1'b1;
                clear_acc = 1'b0;
                ifm_fifo_pop = 1'b1;
                ifm_data = ifm_fifo_out;
                rsp_cnt_d = rsp_cnt_q - 1;
            end
            if (req_cnt_q == 0 && rsp_cnt_q == 1 && ifm_fifo_pop) begin
                array_flush_cnt_d = ARRAY_FLUSH_COUNT_W'(ARRAY_FLUSH_CYCLES);
                state_d = WAIT_DRAIN;
            end
        end
    endtask

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
        .in_valid_i       (linebuf_pool_mode && (state_q == COMPUTE) && linebuf_row_valid),
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
        .compute_phase_i                 (state_q == COMPUTE),
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
        .load_service_i              (state_q == LOAD_WEIGHTS),
        .preload_service_i           (state_q == WAIT_DRAIN),
        .preload_allow_i             (linebuf_has_next_k_tile &&
                                      (array_flush_cnt_q == '0) &&
                                      !linebuf_prefetch_busy),
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

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (INPUT_FIFO_DEPTH),
        .dtype        (input_row_t)
    ) i_ifm_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (fifo_flush),
        .testmode_i (1'b0),
        .full_o     (ifm_fifo_full),
        .empty_o    (ifm_fifo_empty),
        .usage_o    (),
        .data_i     (ifm_fifo_data),
        .push_i     (ifm_fifo_push),
        .data_o     (ifm_fifo_out),
        .pop_i      (ifm_fifo_pop)
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

    conv_linebuf_stream_packer #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH),
        .MAX_INPUT_W      (640)
    ) i_conv_channel_linebuf_packer (
        .clk_i                   (clk_i),
        .rst_ni                  (rst_ni),
        .start_i                 (linebuf_start),
        .next_tile_i             (linebuf_next_tile),
        .prefetch_i              (linebuf_prefetch),
        .dim_m_i                 (linebuf_spatial_m),
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
        .obi_req_o               (linebuf_obi_req),
        .obi_gnt_i               (obi_i_gnt_i),
        .obi_addr_o              (linebuf_obi_addr),
        .obi_rvalid_i            (obi_i_rvalid_i),
        .obi_rdata_i             (obi_i_rdata_i),
        .row_data_o              (linebuf_row_data),
        .row_valid_o             (linebuf_row_valid),
        .row_ready_i             (linebuf_row_ready),
        .busy_o                  (linebuf_busy),
        .done_o                  (linebuf_done),
        .prefetch_busy_o         (linebuf_prefetch_busy),
        .emitted_vectors_o       (linebuf_emitted_vectors),
        .fetch_beats_o           (linebuf_fetch_beats),
        .bypass_vectors_o        (linebuf_bypass_vectors),
        .debug_state_o           (linebuf_debug_state)
    );

    assign obi_i_we_o = 1'b0;
    assign obi_i_be_o = '1;
    assign obi_i_wdata_o = '0;

    // FSM
    // The engine helper tasks below are side-effecting but are only invoked from
    // this single next-state block. Verilator reports their writes as separate
    // procedural writers, so keep this suppression scoped to the controller
    // next-state logic.
    /* verilator lint_off MULTIDRIVEN */
    always_comb begin
        state_d = state_q;
        i_ptr_d = i_ptr_q;
        req_cnt_d = req_cnt_q;
        rsp_cnt_d = rsp_cnt_q;
        k_tile_idx_d = k_tile_idx_q;
        k_seed_ic_d = k_seed_ic_q;
        k_seed_kw_d = k_seed_kw_q;
        k_seed_kh_d = k_seed_kh_q;
        k_channel_offset_d = k_channel_offset_q;
        array_flush_cnt_d = array_flush_cnt_q;
        dw_tap_count_d = dw_tap_count_q;
        dw_group_idx_d = dw_group_idx_q;
        dw_group_input_offset_d = dw_group_input_offset_q;
        dw_group_output_offset_d = dw_group_output_offset_q;
        dw_group_weight_offset_d = dw_group_weight_offset_q;
        drain_tile_advance = 1'b0;
        drain_tile_advance_overlap = 1'b0;
        drain_tile_start = 1'b0;
        drain_tile_start_add_rows = 1'b0;
        drain_depthwise_group_start = 1'b0;
        drain_depthwise_group_output_ptr = cfg_sys_ofm_ptr_i;
        linebuf_start = 1'b0;
        linebuf_next_tile = 1'b0;
        linebuf_prefetch_req_d = (state_q == WAIT_DRAIN) ? linebuf_prefetch_req_q : 1'b0;
        linebuf_prefetch = linebuf_prefetch_req_q && (state_q == WAIT_DRAIN);
        linebuf_row_ready = 1'b0;
        weight_preload_consume = 1'b0;
        weight_depthwise_group_start = 1'b0;
        weight_depthwise_group_ptr = cfg_sys_weight_ptr_i;

        cfg_sys_done_o = 1'b0;

        obi_i_req_o = 1'b0;
        obi_i_addr_o = '0;

        compute_en = 1'b0;
        clear_acc = 1'b0;
        ifm_data = '0;
        ifm_fifo_push = 1'b0;
        ifm_fifo_pop = 1'b0;
        dw_engine_in_valid = 1'b0;

        case (state_q)
            IDLE: begin
                if (cfg_sys_start_i) begin
                    i_ptr_d = cfg_sys_ifm_ptr_i;
                    req_cnt_d = '0;
                    rsp_cnt_d = '0;
                    array_flush_cnt_d = '0;
                    k_tile_idx_d = '0;
                    k_seed_ic_d = cfg_linebuf_k_seed_ic_i;
                    k_seed_kw_d = cfg_linebuf_k_seed_kw_i;
                    k_seed_kh_d = cfg_linebuf_k_seed_kh_i;
                    k_channel_offset_d = '0;
                    dw_tap_count_d = '0;
                    dw_group_idx_d = '0;
                    dw_group_input_offset_d = '0;
                    dw_group_output_offset_d = '0;
                    dw_group_weight_offset_d = '0;
                    state_d = LOAD_WEIGHTS;

                    if (linebuf_pool_mode) begin
                        req_cnt_d = '0;
                        rsp_cnt_d = '0;
                        linebuf_start = 1'b1;
                        state_d = COMPUTE;
                    end

                    if (linebuf_depthwise_mode) begin
                        state_d = LOAD_WEIGHTS;
                    end

                    if ((cfg_requant_en_i && requant_config_invalid) ||
                        binary_config_invalid) begin
                        req_cnt_d = '0;
                        rsp_cnt_d = '0;
                        state_d = DONE;
                    end
                end
            end

            LOAD_WEIGHTS: begin
                if (linebuf_depthwise_mode) begin
                    if (weight_load_done) begin
                        linebuf_start = 1'b1;
                        dw_tap_count_d = '0;
                        state_d = COMPUTE;
                    end
                end else if (weight_preload_done) begin
                    req_cnt_d = cfg_sys_dim_m_i;
                    rsp_cnt_d = cfg_sys_dim_m_i;
                    drain_tile_start = 1'b1;
                    drain_tile_start_add_rows = psum_buf_overlap_active &&
                                                (k_tile_idx_q != 32'd0);
                    array_flush_cnt_d = '0;
                    weight_preload_consume = 1'b1;
                    if (cfg_linebuf_en_i) begin
                        linebuf_next_tile = 1'b1;
                    end
                    state_d = COMPUTE;
                end else if (weight_load_done) begin
                    req_cnt_d = cfg_sys_dim_m_i;
                    rsp_cnt_d = cfg_sys_dim_m_i;
                    drain_tile_start = 1'b1;
                    drain_tile_start_add_rows = psum_buf_overlap_active &&
                                                (k_tile_idx_q != 32'd0);
                    if (cfg_linebuf_en_i) begin
                        if (linebuf_kgen_multi && (k_tile_idx_q != 32'd0)) begin
                            linebuf_next_tile = 1'b1;
                        end else begin
                            linebuf_start = 1'b1;
                        end
                    end
                    state_d = COMPUTE;
                end
            end

            COMPUTE: begin
                if (linebuf_depthwise_mode) begin
                    obi_i_req_o = linebuf_obi_req;
                    obi_i_addr_o = linebuf_obi_addr;

                    if (linebuf_row_valid && !requant_config_invalid && dw_engine_in_ready) begin
                        dw_engine_in_valid = 1'b1;
                        linebuf_row_ready = 1'b1;
                        if (dw_tap_is_last) begin
                            dw_tap_count_d = '0;
                        end else begin
                            dw_tap_count_d = dw_tap_count_q + 1'b1;
                        end
                    end

                    if ((drain_cnt_q == 32'd0) && !linebuf_busy &&
                        !dw_engine_out_valid && !quantized_out_valid) begin
                        if (!dw_last_group) begin
                            dw_group_idx_d = dw_group_idx_q + 32'd1;
                            dw_group_input_offset_d = dw_group_input_offset_q + dw_group_span_bytes;
                            dw_group_output_offset_d = dw_group_output_offset_q + dw_group_output_bytes;
                            dw_group_weight_offset_d = dw_group_weight_offset_q + dw_weight_group_bytes;
                            weight_depthwise_group_start = 1'b1;
                            weight_depthwise_group_ptr = cfg_sys_weight_ptr_i +
                                dw_group_weight_offset_q + dw_weight_group_bytes;
                            drain_depthwise_group_start = 1'b1;
                            drain_depthwise_group_output_ptr = cfg_sys_ofm_ptr_i +
                                dw_group_output_offset_q + dw_group_output_bytes;
                            dw_tap_count_d = '0;
                            state_d = LOAD_WEIGHTS;
                        end else begin
                            state_d = DONE;
                        end
                    end
                end else if (linebuf_pool_mode) begin
                    obi_i_req_o = linebuf_obi_req;
                    obi_i_addr_o = linebuf_obi_addr;

                    linebuf_row_ready = linebuf_row_valid && pool_in_ready;

                    if ((drain_cnt_q == 32'd0) && !pool_out_valid && !linebuf_busy) begin
                        state_d = DONE;
                    end
                end else if (cfg_linebuf_en_i) begin
                    launch_linebuf_compute_engine();
                end else begin
                    launch_fifo_compute_engine();
                end
            end

            WAIT_DRAIN: begin
                if ((array_flush_cnt_q != '0) && array_pipe_ready) begin
                    array_flush_cnt_d = array_flush_cnt_q - 1'b1;
                end

                // The line-buffer formatter may still hold its final row after
                // the last compute input was accepted.  Keep the output
                // handshake open until that pipeline is empty; otherwise a
                // following one-cycle START can arrive while the line-buffer
                // is still in CH_STREAM_DONE and be lost.
                if (cfg_linebuf_en_i) begin
                    linebuf_row_ready = 1'b1;
                end

                service_linebuf_prefetch_engine();

                if (accum_active) begin
                    if (psum_buf_overlap_next_safe) begin
                        advance_to_next_k_tile(1'b1);
                    end else if (drain_cnt_q == 0 && ofm_fifo_empty &&
                                 (!cfg_binary_en_i || !binary_operand_busy)) begin
                        if (linebuf_has_next_k_tile && weight_preload_done && !linebuf_prefetch_busy) begin
                            advance_to_next_k_tile(1'b0);
                        end else if (linebuf_has_next_k_tile) begin
                            state_d = WAIT_DRAIN;
                        end else if (!cfg_linebuf_en_i || !linebuf_busy) begin
                            state_d = DONE;
                        end
                    end
                end else begin
                    if (psum_buf_overlap_next_safe) begin
                        advance_to_next_k_tile(1'b1);
                    end else if (drain_cnt_q == 0 && ofm_fifo_empty &&
                                 (!cfg_binary_en_i || !binary_operand_busy)) begin
                        if (linebuf_has_next_k_tile && weight_preload_done && !linebuf_prefetch_busy) begin
                            advance_to_next_k_tile(1'b0);
                        end else if (linebuf_has_next_k_tile) begin
                            state_d = WAIT_DRAIN;
                        end else if (!cfg_linebuf_en_i || !linebuf_busy) begin
                            state_d = DONE;
                        end
                    end
                end
            end

            DONE: begin
                cfg_sys_done_o = 1'b1;
                linebuf_prefetch_req_d = 1'b0;
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end
    /* verilator lint_on MULTIDRIVEN */

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q         <= IDLE;
            i_ptr_q         <= '0;
            req_cnt_q       <= '0;
            rsp_cnt_q       <= '0;
            k_tile_idx_q    <= '0;
            k_seed_ic_q     <= '0;
            k_seed_kw_q     <= '0;
            k_seed_kh_q     <= '0;
            k_channel_offset_q <= '0;
            linebuf_prefetch_req_q <= 1'b0;
            array_flush_cnt_q <= '0;
            dw_tap_count_q <= '0;
            dw_group_idx_q <= '0;
            dw_group_input_offset_q <= '0;
            dw_group_output_offset_q <= '0;
            dw_group_weight_offset_q <= '0;
        end else begin
            state_q     <= state_d;
            i_ptr_q     <= i_ptr_d;
            req_cnt_q   <= req_cnt_d;
            rsp_cnt_q   <= rsp_cnt_d;
            k_tile_idx_q <= k_tile_idx_d;
            k_seed_ic_q <= k_seed_ic_d;
            k_seed_kw_q <= k_seed_kw_d;
            k_seed_kh_q <= k_seed_kh_d;
            k_channel_offset_q <= k_channel_offset_d;
            linebuf_prefetch_req_q <= linebuf_prefetch_req_d;
            array_flush_cnt_q <= array_flush_cnt_d;
            dw_tap_count_q <= dw_tap_count_d;
            dw_group_idx_q <= dw_group_idx_d;
            dw_group_input_offset_q <= dw_group_input_offset_d;
            dw_group_output_offset_q <= dw_group_output_offset_d;
            dw_group_weight_offset_q <= dw_group_weight_offset_d;
        end
    end

endmodule
