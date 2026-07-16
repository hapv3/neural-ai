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

    typedef enum logic [1:0] {
        DRAIN_IDLE,
        DRAIN_ACCUM_READ,
        DRAIN_ACCUM_WRITE,
        DRAIN_ACCUM_REQUANT
    } drain_state_e;

    drain_state_e drain_state_q;
    drain_state_e drain_state_d;

    logic [31:0] w_ptr_q, w_ptr_d;
    logic [31:0] i_ptr_q, i_ptr_d;
    logic [31:0] o_ptr_q, o_ptr_d;
    logic [31:0] a_ptr_q, a_ptr_d;
    logic [31:0] o_col_q, o_col_d;
    logic [31:0] a_col_q, a_col_d;
    logic [31:0] req_cnt_q, req_cnt_d; // Counter for requests
    logic [31:0] rsp_cnt_q, rsp_cnt_d; // Counter for responses
    logic [31:0] drain_cnt_q, drain_cnt_d; // Counter for valid outputs

    localparam int unsigned OFM_BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned OFM_ROW_BYTES = (ARRAY_DIM * OFM_ELEM_WIDTH) / 8;
    localparam int unsigned REQUANT_ROW_BYTES = ARRAY_DIM;
    localparam int unsigned OFM_ELEMS_PER_OBI = DATA_WIDTH / OFM_ELEM_WIDTH;
    localparam int unsigned OFM_ROW_BEATS = OFM_ROW_BYTES / OFM_BEAT_BYTES;
    localparam int unsigned OFM_ROW_BEAT_COUNT_W = (OFM_ROW_BEATS > 1) ? $clog2(OFM_ROW_BEATS + 1) : 1;
    localparam int unsigned ARRAY_FLUSH_CYCLES = (2 * ARRAY_DIM) - 1;
    localparam int unsigned ARRAY_FLUSH_COUNT_W = $clog2(ARRAY_FLUSH_CYCLES + 1);
    localparam int unsigned PSUM_BUF_M = 256;
    localparam int unsigned PSUM_BUF_ADDR_WIDTH = $clog2(PSUM_BUF_M);

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0]   ofm_row_t;

    localparam int unsigned OFM_FIFO_ADDR_DEPTH = (OFM_FIFO_DEPTH > 1) ? $clog2(OFM_FIFO_DEPTH) : 1;

    function automatic logic [31:0] next_strided_ptr(
        input logic [31:0] ptr,
        input logic [31:0] col,
        input logic [31:0] row_bytes,
        input logic [31:0] row_stride_bytes,
        input logic [31:0] tile_cols
    );
        logic [31:0] row_gap;
        begin
            row_gap = row_stride_bytes - ((tile_cols - 32'd1) * row_bytes);
            if ((row_stride_bytes != 32'd0) && (tile_cols != 32'd0) &&
                ((col + 32'd1) == tile_cols)) begin
                next_strided_ptr = ptr + row_gap;
            end else begin
                next_strided_ptr = ptr + row_bytes;
            end
        end
    endfunction

    function automatic logic [31:0] next_strided_col(
        input logic [31:0] col,
        input logic [31:0] tile_cols
    );
        begin
            if ((tile_cols != 32'd0) && ((col + 32'd1) == tile_cols)) begin
                next_strided_col = 32'd0;
            end else begin
                next_strided_col = col + 32'd1;
            end
        end
    endfunction

    input_row_t    weight_fifo_data;
    input_row_t    weight_fifo_out;
    logic          weight_fifo_push;
    logic          weight_fifo_pop;
    logic          weight_fifo_full;
    logic          weight_fifo_empty;

    input_row_t    ifm_fifo_data;
    input_row_t    ifm_fifo_out;
    logic          ifm_fifo_push;
    logic          ifm_fifo_pop;
    logic          ifm_fifo_full;
    logic          ifm_fifo_empty;

    typedef struct packed {
        ofm_row_t    row;
        logic [31:0] row_idx;
        logic [31:0] k_tile_idx;
        logic        psum_buf_active;
        logic        needs_external_psum;
        logic        final_tile;
    } ofm_fifo_entry_t;

    ofm_fifo_entry_t ofm_fifo_data;
    ofm_fifo_entry_t ofm_fifo_out;
    ofm_row_t      accum_row_q, accum_row_d;
    ofm_row_t      accum_sum;
    ofm_row_t      psum_buf_old;
    ofm_row_t      psum_buf_sum;
    ofm_row_t      requant_acc;
    logic          ofm_fifo_push;
    logic          ofm_fifo_pop;
    logic          ofm_fifo_full;
    logic          ofm_fifo_empty;
    logic [OFM_FIFO_ADDR_DEPTH-1:0] ofm_fifo_usage;
    ofm_row_t      psum_fifo_data;
    ofm_row_t      psum_fifo_out;
    logic          psum_fifo_push;
    logic          psum_fifo_pop;
    logic          psum_fifo_full;
    logic          psum_fifo_empty;
    logic          array_pipe_ready;

    logic          fifo_flush;
    logic          weight_load_en;
    logic          clear_acc;
    logic          compute_en;
    input_row_t    weight_data;
    input_row_t    ifm_data;
    ofm_row_t      psum_data;
    ofm_row_t      ofm_data;
    logic          ofm_valid;
    logic          ofm_ready;
    logic          requant_in_valid;
    logic          requant_in_ready;
    logic          requant_out_valid;
    logic          requant_out_ready;
    logic [255:0]  requant_packed_data;
    logic          requant_invalid;
    logic          requant_config_invalid;
    logic [OFM_ROW_BEAT_COUNT_W-1:0] accum_beat_q, accum_beat_d;
    logic [OFM_ROW_BEAT_COUNT_W-1:0] accum_req_beat_q, accum_req_beat_d;
    logic          accum_requant_sent_q, accum_requant_sent_d;
    ofm_row_t      psum_read_row_q, psum_read_row_d;
    logic          psum_read_active_q, psum_read_active_d;
    logic [3:0]    psum_read_resp_mask_q, psum_read_resp_mask_d;
    logic [31:0]   psum_prefetch_rows_q, psum_prefetch_rows_d;
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
    logic          cfg_linebuf_en_i;
    logic          cfg_linebuf_coalesce_i;
    logic          cfg_linebuf_kgen_i;
    logic          cfg_linebuf_pool_i;
    logic          cfg_linebuf_c32_fast_i;
    logic          cfg_linebuf_depthwise_i;
    logic          cfg_linebuf_c32_group_stationary_i;
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
    logic          weight_preload_active_q, weight_preload_active_d;
    logic          weight_preload_done_q, weight_preload_done_d;
    logic [31:0]   weight_preload_req_cnt_q, weight_preload_req_cnt_d;
    logic [31:0]   weight_preload_rsp_cnt_q, weight_preload_rsp_cnt_d;
    logic [31:0]   weight_preload_obi_rsp_cnt_q, weight_preload_obi_rsp_cnt_d;
    logic [31:0]   weight_preload_ptr_q, weight_preload_ptr_d;
    logic          linebuf_has_next_k_tile;
    logic          weight_preload_fetch_done;
    logic          psum_buf_active;
    logic          psum_buf_needs_external;
    logic          accum_uses_tcdm_psum;
    logic          psum_buf_final_tile;
    logic          psum_buf_overlap_active;
    logic          psum_buf_overlap_next_ready;
    logic          psum_buf_overlap_next_safe;
    logic          psum_buf_drain_entry;
    logic          psum_buf_sel_q, psum_buf_sel_d;
    logic          psum_buf_we;
    logic [PSUM_BUF_ADDR_WIDTH-1:0] psum_buf_addr;
    ofm_row_t      psum_buf_wdata;
    ofm_row_t      psum_buf_rdata;
    ofm_row_t      psum_buf_q [2][PSUM_BUF_M];
    logic [31:0]   ofm_push_row_idx_q, ofm_push_row_idx_d;

    logic          linebuf_start;
    logic          linebuf_next_tile;
    logic          linebuf_prefetch;
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
    input_row_t    pool_acc_q, pool_acc_d;
    input_row_t    pool_out_q, pool_out_d;
    input_row_t    pool_next_acc;
    logic          pool_out_valid_q, pool_out_valid_d;
    logic [7:0]    pool_tap_count_q, pool_tap_count_d;
    logic [31:0]   pool_kernel_vectors;
    localparam int unsigned DW_MAX_TAPS = 25;
    localparam int unsigned DW_TAP_COUNT_W = $clog2(DW_MAX_TAPS + 1);
    input_row_t    dw_weight_q [DW_MAX_TAPS];
    input_row_t    dw_weight_d [DW_MAX_TAPS];
    ofm_row_t      dw_acc_q, dw_acc_d;
    ofm_row_t      dw_next_acc;
    ofm_row_t      dw_requant_acc;
    logic [DW_TAP_COUNT_W-1:0] dw_tap_count_q, dw_tap_count_d;
    logic [DW_TAP_COUNT_W-1:0] dw_weight_rsp_idx;
    logic [31:0]   dw_group_idx_q, dw_group_idx_d;
    logic [31:0]   dw_group_count;
    logic [31:0]   dw_group_span_bytes;
    logic [31:0]   dw_group_output_bytes;
    logic [31:0]   dw_weight_group_bytes;
    logic [31:0]   dw_group_input_offset;
    logic [31:0]   dw_group_output_offset;
    logic [31:0]   dw_group_weight_offset;
    logic          dw_last_group;
    logic [5:0]    dw_group_valid_bytes;
    logic [5:0]    cfg_linebuf_block_valid_bytes_eff;
    logic [255:0]  requant_packed_write_data;

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

    function automatic logic [255:0] mask_packed_lanes(
        input logic [255:0] data,
        input logic [5:0] valid_bytes
    );
        logic [255:0] result;
        begin
            result = '0;
            for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
                if (ch < valid_bytes) begin
                    result[ch * 8 +: 8] = data[ch * 8 +: 8];
                end
            end
            mask_packed_lanes = result;
        end
    endfunction

    assign fifo_flush = (state_q == IDLE) && cfg_sys_start_i;

    assign weight_fifo_data = obi_w_rdata_i;
    assign ifm_fifo_data    = obi_i_rdata_i;
    assign ofm_ready = !ofm_fifo_full;
    assign array_pipe_ready = !ofm_valid || ofm_ready;
    assign psum_data = '0;
    assign perf_weight_load_en_o = weight_load_en;
    assign perf_compute_en_o = compute_en;
    assign perf_ofm_valid_o = ofm_valid;
    assign perf_ofm_ready_o = ofm_ready;
    assign debug_state_o = state_q;
    assign debug_drain_state_o = drain_state_q;
    assign debug_linebuf_state_o = linebuf_debug_state;
    assign linebuf_spatial_m = (cfg_linebuf_spatial_m_i != 32'd0) ? cfg_linebuf_spatial_m_i : cfg_sys_dim_m_i;
    assign linebuf_kgen_multi = cfg_linebuf_en_i && cfg_linebuf_coalesce_i && cfg_linebuf_kgen_i &&
                                cfg_linebuf_c32_fast_i &&
                                (cfg_linebuf_lane_base_i == 6'd0) &&
                                (cfg_linebuf_block_valid_bytes_eff == 6'(ARRAY_DIM)) &&
                                (cfg_linebuf_input_c_i >= 16'(ARRAY_DIM)) &&
                                (cfg_linebuf_input_c_i[4:0] == 5'd0) &&
                                (cfg_linebuf_k_tiles_i > 32'd1);
    assign linebuf_c32_group_stationary = linebuf_kgen_multi;
    assign linebuf_pool_mode = cfg_linebuf_en_i && cfg_linebuf_pool_i;
    assign linebuf_depthwise_mode = cfg_linebuf_en_i && cfg_linebuf_depthwise_i;
    assign linebuf_has_next_k_tile = linebuf_kgen_multi && ((k_tile_idx_q + 32'd1) < cfg_linebuf_k_tiles_i);
    assign accum_active = cfg_sys_accum_en_i || (linebuf_kgen_multi && (k_tile_idx_q != 32'd0));
    assign requant_active = cfg_requant_en_i && (!linebuf_kgen_multi || !linebuf_has_next_k_tile);
    assign psum_buf_active = linebuf_kgen_multi && (cfg_sys_dim_m_i <= 32'(PSUM_BUF_M));
    assign psum_buf_needs_external = psum_buf_active && cfg_sys_accum_en_i && (k_tile_idx_q == 32'd0);
    assign accum_uses_tcdm_psum = accum_active && (!psum_buf_active || psum_buf_needs_external);
    assign psum_buf_final_tile = psum_buf_active && !linebuf_has_next_k_tile;
    assign psum_buf_overlap_active = psum_buf_active && linebuf_kgen_multi && linebuf_has_next_k_tile;
    assign psum_buf_overlap_next_ready = psum_buf_overlap_active && linebuf_has_next_k_tile &&
                                         weight_preload_done_q && !linebuf_prefetch_busy &&
                                         (array_flush_cnt_q == '0);
    // Direct WAIT_DRAIN->COMPUTE overlap preserves the current drain/prefetch
    // state. OFM FIFO ordering keeps an external-psum first tile ahead of later
    // on-chip psum-buffer tiles, so the next tile cannot consume a row before
    // the external accumulation wrote that row into the psum buffer.
    assign psum_buf_overlap_next_safe = psum_buf_overlap_next_ready;
    assign psum_buf_drain_entry = !ofm_fifo_empty && ofm_fifo_out.psum_buf_active;
    assign drain_enabled = !linebuf_pool_mode && !linebuf_depthwise_mode &&
                           ((state_q == LOAD_WEIGHTS) || (state_q == COMPUTE) || (state_q == WAIT_DRAIN));
    assign weight_preload_fetch_done = weight_preload_done_q ||
                                       (weight_preload_active_q &&
                                        (weight_preload_req_cnt_q == 32'd0) &&
                                        (weight_preload_obi_rsp_cnt_q == 32'd0));
    assign linebuf_use_next_cfg = (state_q == WAIT_DRAIN) && linebuf_has_next_k_tile;
    assign linebuf_c_base_eff = linebuf_depthwise_mode ? (cfg_linebuf_c_base_i + 16'(dw_group_idx_q << 5)) :
                                cfg_linebuf_kgen_i ? {linebuf_seed_ic_eff[15:5], 5'b0} : cfg_linebuf_c_base_i;
    assign linebuf_seed_ic_eff = cfg_linebuf_kgen_i ? (linebuf_use_next_cfg ? k_seed_ic_next : k_seed_ic_q) :
                                                       cfg_linebuf_k_seed_ic_i;
    assign linebuf_seed_kw_eff = cfg_linebuf_kgen_i ? (linebuf_use_next_cfg ? k_seed_kw_next : k_seed_kw_q) :
                                                       cfg_linebuf_k_seed_kw_i;
    assign linebuf_seed_kh_eff = cfg_linebuf_kgen_i ? (linebuf_use_next_cfg ? k_seed_kh_next : k_seed_kh_q) :
                                                       cfg_linebuf_k_seed_kh_i;
    assign pool_kernel_vectors = 32'(cfg_linebuf_kernel_h_i) * 32'(cfg_linebuf_kernel_w_i);
    assign dw_weight_rsp_idx = DW_TAP_COUNT_W'(pool_kernel_vectors - rsp_cnt_q);
    assign dw_group_count = ({16'd0, cfg_linebuf_input_c_i} + 32'd31) >> 5;
    assign dw_group_span_bytes = {16'd0, cfg_linebuf_input_h_i} * cfg_linebuf_row_stride_bytes_i;
    assign dw_group_output_bytes = linebuf_spatial_m * 32'd32;
    assign dw_weight_group_bytes = pool_kernel_vectors * 32'd32;
    assign dw_group_input_offset = dw_group_idx_q * dw_group_span_bytes;
    assign dw_group_output_offset = dw_group_idx_q * dw_group_output_bytes;
    assign dw_group_weight_offset = dw_group_idx_q * dw_weight_group_bytes;
    assign dw_last_group = (dw_group_idx_q + 32'd1) >= dw_group_count;
    assign dw_group_valid_bytes = depthwise_group_valid_bytes(cfg_linebuf_input_c_i, dw_group_idx_q);
    assign cfg_linebuf_block_valid_bytes_eff = linebuf_depthwise_mode ?
                                               dw_group_valid_bytes :
                                               cfg_linebuf_block_valid_bytes_i;
    assign cfg_linebuf_channel_addr_offset_eff = linebuf_depthwise_mode ?
                                                 (cfg_linebuf_channel_addr_offset_i + dw_group_input_offset) :
                                                 linebuf_c32_group_stationary ?
                                                 (linebuf_use_next_cfg ?
                                                  k_channel_offset_next :
                                                  k_channel_offset_q) :
                                                 cfg_linebuf_channel_addr_offset_i;
    assign requant_packed_write_data = linebuf_depthwise_mode ?
                                       mask_packed_lanes(requant_packed_data, dw_group_valid_bytes) :
                                       requant_packed_data;

    function automatic input_row_t max_i8_row(
        input input_row_t lhs,
        input input_row_t rhs
    );
        input_row_t result;
        begin
            for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
                result[ch] = ($signed(lhs[ch]) >= $signed(rhs[ch])) ? lhs[ch] : rhs[ch];
            end
            max_i8_row = result;
        end
    endfunction

    function automatic ofm_row_t depthwise_mac_row(
        input ofm_row_t acc,
        input input_row_t ifm,
        input input_row_t weight,
        input logic clear,
        input logic [5:0] valid_bytes
    );
        ofm_row_t result;
        logic signed [15:0] product;
        logic signed [31:0] product_ext;
        begin
            for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
                if (ch < valid_bytes) begin
                    product = $signed(ifm[ch]) * $signed(weight[ch]);
                    product_ext = {{16{product[15]}}, product};
                    result[ch] = clear ? product_ext : (acc[ch] + product_ext);
                end else begin
                    result[ch] = '0;
                end
            end
            depthwise_mac_row = result;
        end
    endfunction

    always_comb begin
        for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
            accum_sum[ch] = ofm_fifo_out.row[ch] + psum_fifo_out[ch];
            psum_buf_sum[ch] = ofm_fifo_out.row[ch] + psum_buf_old[ch];
        end
    end

    assign requant_acc = linebuf_depthwise_mode ? dw_requant_acc :
                         (psum_buf_drain_entry && requant_active) ? psum_buf_sum :
                         ((accum_active && requant_active) ? accum_sum : ofm_fifo_out.row);

    function automatic void advance_k_seed32(
        input  logic [7:0]  kh_i,
        input  logic [7:0]  kw_i,
        input  logic [15:0] ic_i,
        input  logic [15:0] input_c_i,
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
            ic = ic_i;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                if ((ic + 16'd1) == input_c_i) begin
                    ic = '0;
                    if ((kw + 8'd1) == kernel_w_i[7:0]) begin
                        kw = '0;
                        kh = kh + 8'd1;
                    end else begin
                        kw = kw + 8'd1;
                    end
                end else begin
                    ic = ic + 16'd1;
                end
            end
            kh_o = kh;
            kw_o = kw;
            ic_o = ic;
        end
    endfunction

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

    function automatic void advance_linebuf_k_seed(
        input  logic        c32_group_stationary_i,
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
        begin
            if (c32_group_stationary_i) begin
                advance_k_seed32_c32_group_stationary(kh_i,
                                                      kw_i,
                                                      ic_i,
                                                      input_c_i,
                                                      kernel_h_i,
                                                      kernel_w_i,
                                                      kh_o,
                                                      kw_o,
                                                      ic_o);
            end else begin
                advance_k_seed32(kh_i,
                                 kw_i,
                                 ic_i,
                                 input_c_i,
                                 kernel_w_i,
                                 kh_o,
                                 kw_o,
                                 ic_o);
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

    always_comb begin
        requant_config_invalid = ($signed(cfg_requant_clamp_min_i) > $signed(cfg_requant_clamp_max_i));
        for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
            if (cfg_requant_shift_i[ch] > 8'd31) begin
                requant_config_invalid = 1'b1;
            end
        end
    end

    requant_pipeline #(
        .ARRAY_DIM(ARRAY_DIM)
    ) i_requant_pipeline (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .in_valid_i    (requant_in_valid),
        .in_ready_o    (requant_in_ready),
        .acc_i         (requant_acc),
        .bias_i        (cfg_requant_bias_i),
        .multiplier_i  (cfg_requant_multiplier_i),
        .shift_i       (cfg_requant_shift_i),
        .zero_point_i  (cfg_requant_zero_point_i),
        .clamp_min_i   (cfg_requant_clamp_min_i),
        .clamp_max_i   (cfg_requant_clamp_max_i),
        .out_valid_o   (requant_out_valid),
        .out_ready_i   (requant_out_ready),
        .packed_o      (requant_packed_data),
        .invalid_o     (requant_invalid)
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

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (INPUT_FIFO_DEPTH),
        .dtype        (input_row_t)
    ) i_weight_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (fifo_flush),
        .testmode_i (1'b0),
        .full_o     (weight_fifo_full),
        .empty_o    (weight_fifo_empty),
        .usage_o    (),
        .data_i     (weight_fifo_data),
        .push_i     (weight_fifo_push),
        .data_o     (weight_fifo_out),
        .pop_i      (weight_fifo_pop)
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

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (OFM_FIFO_DEPTH),
        .dtype        (ofm_fifo_entry_t)
    ) i_ofm_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (fifo_flush),
        .testmode_i (1'b0),
        .full_o     (ofm_fifo_full),
        .empty_o    (ofm_fifo_empty),
        .usage_o    (ofm_fifo_usage),
        .data_i     (ofm_fifo_data),
        .push_i     (ofm_fifo_push),
        .data_o     (ofm_fifo_out),
        .pop_i      (ofm_fifo_pop)
    );

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (OFM_FIFO_DEPTH),
        .dtype        (ofm_row_t)
    ) i_psum_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (fifo_flush),
        .testmode_i (1'b0),
        .full_o     (psum_fifo_full),
        .empty_o    (psum_fifo_empty),
        .usage_o    (),
        .data_i     (psum_fifo_data),
        .push_i     (psum_fifo_push),
        .data_o     (psum_fifo_out),
        .pop_i      (psum_fifo_pop)
    );

    systolic_ctrl_regs #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(CFG_DATA_WIDTH)
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
        .cfg_linebuf_en_o   (cfg_linebuf_en_i),
        .cfg_linebuf_coalesce_o(cfg_linebuf_coalesce_i),
        .cfg_linebuf_pool_o (cfg_linebuf_pool_i),
        .cfg_linebuf_c32_fast_o(cfg_linebuf_c32_fast_i),
        .cfg_linebuf_depthwise_o(cfg_linebuf_depthwise_i),
        .cfg_linebuf_c32_group_stationary_o(cfg_linebuf_c32_group_stationary_i),
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
        .K_MAX            (5),
        .MAX_INPUT_W      (640),
        .STRIDE_MAX       (2)
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
    assign obi_w_we_o = 1'b0;
    assign obi_w_be_o = '1;
    assign obi_w_wdata_o = '0;

    // FSM
    always_comb begin
        state_d = state_q;
        drain_state_d = drain_state_q;
        w_ptr_d = w_ptr_q;
        i_ptr_d = i_ptr_q;
        o_ptr_d = o_ptr_q;
        a_ptr_d = a_ptr_q;
        o_col_d = o_col_q;
        a_col_d = a_col_q;
        req_cnt_d = req_cnt_q;
        rsp_cnt_d = rsp_cnt_q;
        drain_cnt_d = drain_cnt_q;
        accum_row_d = accum_row_q;
        accum_beat_d = accum_beat_q;
        accum_req_beat_d = accum_req_beat_q;
        accum_requant_sent_d = accum_requant_sent_q;
        psum_read_row_d = psum_read_row_q;
        psum_read_active_d = psum_read_active_q;
        psum_read_resp_mask_d = psum_read_resp_mask_q;
        psum_prefetch_rows_d = psum_prefetch_rows_q;
        k_tile_idx_d = k_tile_idx_q;
        k_seed_ic_d = k_seed_ic_q;
        k_seed_kw_d = k_seed_kw_q;
        k_seed_kh_d = k_seed_kh_q;
        k_channel_offset_d = k_channel_offset_q;
        array_flush_cnt_d = array_flush_cnt_q;
        weight_preload_active_d = weight_preload_active_q;
        weight_preload_done_d = weight_preload_done_q;
        weight_preload_req_cnt_d = weight_preload_req_cnt_q;
        weight_preload_rsp_cnt_d = weight_preload_rsp_cnt_q;
        weight_preload_obi_rsp_cnt_d = weight_preload_obi_rsp_cnt_q;
        weight_preload_ptr_d = weight_preload_ptr_q;
        ofm_push_row_idx_d = ofm_push_row_idx_q;
        psum_buf_sel_d = psum_buf_sel_q;
        pool_acc_d = pool_acc_q;
        pool_out_d = pool_out_q;
        pool_out_valid_d = pool_out_valid_q;
        pool_tap_count_d = pool_tap_count_q;
        pool_next_acc = pool_acc_q;
        dw_acc_d = dw_acc_q;
        dw_next_acc = dw_acc_q;
        dw_requant_acc = dw_acc_q;
        dw_tap_count_d = dw_tap_count_q;
        dw_group_idx_d = dw_group_idx_q;
        for (int unsigned tap = 0; tap < DW_MAX_TAPS; tap++) begin
            dw_weight_d[tap] = dw_weight_q[tap];
        end
        psum_buf_we = 1'b0;
        psum_buf_addr = '0;
        psum_buf_wdata = '0;
        psum_buf_old = '0;
        if (psum_buf_drain_entry) begin
            psum_buf_addr = PSUM_BUF_ADDR_WIDTH'(ofm_fifo_out.row_idx);
        end else if ((drain_cnt_q != 32'd0) && (drain_cnt_q <= cfg_sys_dim_m_i)) begin
            psum_buf_addr = PSUM_BUF_ADDR_WIDTH'(cfg_sys_dim_m_i - drain_cnt_q);
        end
        psum_buf_rdata = psum_buf_q[psum_buf_sel_q][psum_buf_addr];
        if (psum_buf_drain_entry && ofm_fifo_out.needs_external_psum) begin
            psum_buf_old = psum_fifo_out;
        end else if (psum_buf_drain_entry &&
                     ((ofm_fifo_out.k_tile_idx != 32'd0) || cfg_sys_accum_en_i)) begin
            psum_buf_old = psum_buf_rdata;
        end
        linebuf_start = 1'b0;
        linebuf_next_tile = 1'b0;
        linebuf_prefetch = 1'b0;
        linebuf_row_ready = 1'b0;

        cfg_sys_done_o = 1'b0;

        obi_i_req_o = 1'b0;
        obi_i_addr_o = '0;
        obi_w_req_o = 1'b0;
        obi_w_addr_o = '0;

        weight_load_en = 1'b0;
        compute_en = 1'b0;
        clear_acc = 1'b0;
        weight_data = '0;
        ifm_data = '0;
        weight_fifo_push = 1'b0;
        weight_fifo_pop = 1'b0;
        ifm_fifo_push = 1'b0;
        ifm_fifo_pop = 1'b0;
        ofm_fifo_data = '0;
        ofm_fifo_data.row = ofm_data;
        ofm_fifo_data.row_idx = ofm_push_row_idx_q;
        ofm_fifo_data.k_tile_idx = k_tile_idx_q;
        ofm_fifo_data.psum_buf_active = psum_buf_active;
        ofm_fifo_data.needs_external_psum = psum_buf_needs_external;
        ofm_fifo_data.final_tile = psum_buf_final_tile;
        ofm_fifo_pop = 1'b0;
        psum_fifo_push = 1'b0;
        psum_fifo_pop = 1'b0;
        psum_fifo_data = psum_read_row_q;
        requant_in_valid = 1'b0;
        requant_out_ready = 1'b0;
        ofm_fifo_push = drain_enabled && ofm_valid && ofm_ready;
        if (ofm_fifo_push) begin
            ofm_push_row_idx_d = ofm_push_row_idx_q + 32'd1;
        end

        obi_o_req_o = '0;
        obi_o_we_o = '1;
        obi_o_be_o = '1;
        obi_o_addr_o[0] = o_ptr_q + 0;
        obi_o_addr_o[1] = o_ptr_q + OFM_BEAT_BYTES;
        obi_o_addr_o[2] = o_ptr_q + (2 * OFM_BEAT_BYTES);
        obi_o_addr_o[3] = o_ptr_q + (3 * OFM_BEAT_BYTES);
        obi_o_wdata_o[0] = ofm_fifo_out.row[OFM_ELEMS_PER_OBI-1:0];
        obi_o_wdata_o[1] = ofm_fifo_out.row[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
        obi_o_wdata_o[2] = ofm_fifo_out.row[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
        obi_o_wdata_o[3] = ofm_fifo_out.row[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];

        if (drain_enabled) begin
            if (psum_buf_active || psum_buf_drain_entry) begin
                drain_state_d = DRAIN_IDLE;

                if (psum_read_active_q) begin
                    drain_state_d = DRAIN_ACCUM_READ;
                    for (int unsigned port = 0; port < 4; port++) begin
                        if (obi_o_rvalid_i[port]) begin
                            for (int unsigned elem = 0; elem < OFM_ELEMS_PER_OBI; elem++) begin
                                psum_read_row_d[(port * OFM_ELEMS_PER_OBI) + elem] =
                                    obi_o_rdata_i[port][elem * OFM_ELEM_WIDTH +: OFM_ELEM_WIDTH];
                            end
                            psum_read_resp_mask_d[port] = 1'b1;
                        end
                    end
                    if ((psum_read_resp_mask_q | obi_o_rvalid_i) == 4'b1111) begin
                        psum_fifo_data = psum_read_row_d;
                        psum_fifo_push = 1'b1;
                        psum_read_active_d = 1'b0;
                        psum_read_resp_mask_d = '0;
                    end
                end

                if (accum_requant_sent_q && requant_out_valid) begin
                    drain_state_d = DRAIN_ACCUM_REQUANT;
                    obi_o_req_o[0] = 1'b1;
                    obi_o_wdata_o[0] = requant_packed_data;
                    if (obi_o_gnt_i[0] || requant_invalid) begin
                        requant_out_ready = 1'b1;
                        o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                                   cfg_sys_ofm_row_stride_bytes_i,
                                                   cfg_sys_ofm_tile_cols_i);
                        o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                        drain_cnt_d = drain_cnt_q - 1;
                        accum_requant_sent_d = 1'b0;
                    end
                    if (requant_invalid) begin
                        obi_o_req_o[0] = 1'b0;
                    end
                end

                if (!ofm_fifo_empty &&
                    (!ofm_fifo_out.needs_external_psum || !psum_fifo_empty)) begin
                    drain_state_d = DRAIN_ACCUM_WRITE;
                    if (ofm_fifo_out.final_tile && requant_active) begin
                        drain_state_d = DRAIN_ACCUM_REQUANT;
                        if (!accum_requant_sent_q && requant_in_ready && !requant_config_invalid) begin
                            requant_in_valid = 1'b1;
                            ofm_fifo_pop = 1'b1;
                            psum_fifo_pop = ofm_fifo_out.needs_external_psum;
                            accum_requant_sent_d = 1'b1;
                        end
                        if (requant_out_valid) begin
                            obi_o_req_o[0] = 1'b1;
                            obi_o_wdata_o[0] = requant_packed_data;
                            if (obi_o_gnt_i[0] || requant_invalid) begin
                                requant_out_ready = 1'b1;
                                o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                                           cfg_sys_ofm_row_stride_bytes_i,
                                                           cfg_sys_ofm_tile_cols_i);
                                o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                                drain_cnt_d = drain_cnt_q - 1;
                                accum_requant_sent_d = 1'b0;
                            end
                            if (requant_invalid) begin
                                obi_o_req_o[0] = 1'b0;
                            end
                        end
                    end else if (ofm_fifo_out.final_tile) begin
                        obi_o_we_o = '1;
                        obi_o_wdata_o[0] = psum_buf_sum[OFM_ELEMS_PER_OBI-1:0];
                        obi_o_wdata_o[1] = psum_buf_sum[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
                        obi_o_wdata_o[2] = psum_buf_sum[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
                        obi_o_wdata_o[3] = psum_buf_sum[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];
                        obi_o_req_o = 4'b1111;
                        if (obi_o_gnt_i == 4'b1111) begin
                            ofm_fifo_pop = 1'b1;
                            psum_fifo_pop = ofm_fifo_out.needs_external_psum;
                            o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(OFM_ROW_BYTES),
                                                       cfg_sys_ofm_row_stride_bytes_i,
                                                       cfg_sys_ofm_tile_cols_i);
                            o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                            drain_cnt_d = drain_cnt_q - 1;
                        end
                    end else begin
                        psum_buf_we = 1'b1;
                        psum_buf_wdata = psum_buf_sum;
                        ofm_fifo_pop = 1'b1;
                        psum_fifo_pop = ofm_fifo_out.needs_external_psum;
                        drain_cnt_d = drain_cnt_q - 1;
                    end
                end

                if (psum_buf_needs_external &&
                    (psum_prefetch_rows_q != 0) && !psum_read_active_q && !psum_fifo_full &&
                    !(|obi_o_req_o)) begin
                    drain_state_d = DRAIN_ACCUM_READ;
                    obi_o_req_o = 4'b1111;
                    obi_o_we_o = '0;
                    obi_o_addr_o[0] = a_ptr_q + 0;
                    obi_o_addr_o[1] = a_ptr_q + OFM_BEAT_BYTES;
                    obi_o_addr_o[2] = a_ptr_q + (2 * OFM_BEAT_BYTES);
                    obi_o_addr_o[3] = a_ptr_q + (3 * OFM_BEAT_BYTES);
                    if (obi_o_gnt_i == 4'b1111) begin
                        a_ptr_d = next_strided_ptr(a_ptr_q, a_col_q, 32'(OFM_ROW_BYTES),
                                                   cfg_sys_psum_row_stride_bytes_i,
                                                   cfg_sys_ofm_tile_cols_i);
                        a_col_d = next_strided_col(a_col_q, cfg_sys_ofm_tile_cols_i);
                        psum_prefetch_rows_d = psum_prefetch_rows_q - 1;
                        psum_read_active_d = 1'b1;
                        psum_read_resp_mask_d = '0;
                        psum_read_row_d = '0;
                    end
                end
            end else if (!accum_active && requant_active) begin
                if (requant_out_valid) begin
                    obi_o_req_o[0] = 1'b1;
                    obi_o_wdata_o[0] = requant_packed_data;
                    if (obi_o_gnt_i[0] || requant_invalid) begin
                        requant_out_ready = 1'b1;
                        o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                                   cfg_sys_ofm_row_stride_bytes_i,
                                                   cfg_sys_ofm_tile_cols_i);
                        o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                        drain_cnt_d = drain_cnt_q - 1;
                    end
                    if (requant_invalid) begin
                        obi_o_req_o[0] = 1'b0;
                    end
                end
                if (!ofm_fifo_empty && requant_in_ready && !requant_config_invalid) begin
                    requant_in_valid = 1'b1;
                    ofm_fifo_pop = 1'b1;
                end
            end else if (!accum_active && !ofm_fifo_empty) begin
                obi_o_req_o = 4'b1111;
                if (obi_o_gnt_i == 4'b1111) begin
                    ofm_fifo_pop = 1'b1;
                    o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(OFM_ROW_BYTES),
                                               cfg_sys_ofm_row_stride_bytes_i,
                                               cfg_sys_ofm_tile_cols_i);
                    o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                    drain_cnt_d = drain_cnt_q - 1;
                end
            end else if (accum_active) begin
                drain_state_d = DRAIN_IDLE;

                if (psum_read_active_q) begin
                    drain_state_d = DRAIN_ACCUM_READ;
                    for (int unsigned port = 0; port < 4; port++) begin
                        if (obi_o_rvalid_i[port]) begin
                            for (int unsigned elem = 0; elem < OFM_ELEMS_PER_OBI; elem++) begin
                                psum_read_row_d[(port * OFM_ELEMS_PER_OBI) + elem] =
                                    obi_o_rdata_i[port][elem * OFM_ELEM_WIDTH +: OFM_ELEM_WIDTH];
                            end
                            psum_read_resp_mask_d[port] = 1'b1;
                        end
                    end
                    if ((psum_read_resp_mask_q | obi_o_rvalid_i) == 4'b1111) begin
                        psum_fifo_data = psum_read_row_d;
                        psum_fifo_push = 1'b1;
                        psum_read_active_d = 1'b0;
                        psum_read_resp_mask_d = '0;
                    end
                end

                if (!requant_active && !ofm_fifo_empty && !psum_fifo_empty) begin
                    drain_state_d = DRAIN_ACCUM_WRITE;
                    obi_o_we_o = '1;
                    obi_o_wdata_o[0] = accum_sum[OFM_ELEMS_PER_OBI-1:0];
                    obi_o_wdata_o[1] = accum_sum[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
                    obi_o_wdata_o[2] = accum_sum[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
                    obi_o_wdata_o[3] = accum_sum[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];
                    obi_o_req_o = 4'b1111;
                    if (obi_o_gnt_i == 4'b1111) begin
                        ofm_fifo_pop = 1'b1;
                        psum_fifo_pop = 1'b1;
                        o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(OFM_ROW_BYTES),
                                                   cfg_sys_ofm_row_stride_bytes_i,
                                                   cfg_sys_ofm_tile_cols_i);
                        o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                        drain_cnt_d = drain_cnt_q - 1;
                    end
                end else if (requant_active) begin
                    if (!accum_requant_sent_q && !ofm_fifo_empty && !psum_fifo_empty &&
                        requant_in_ready && !requant_config_invalid) begin
                        drain_state_d = DRAIN_ACCUM_REQUANT;
                        requant_in_valid = 1'b1;
                        ofm_fifo_pop = 1'b1;
                        psum_fifo_pop = 1'b1;
                        accum_requant_sent_d = 1'b1;
                    end

                    if (requant_out_valid) begin
                        drain_state_d = DRAIN_ACCUM_REQUANT;
                        obi_o_req_o[0] = 1'b1;
                        obi_o_wdata_o[0] = requant_packed_write_data;
                        if (obi_o_gnt_i[0] || requant_invalid) begin
                            requant_out_ready = 1'b1;
                            o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                                       cfg_sys_ofm_row_stride_bytes_i,
                                                       cfg_sys_ofm_tile_cols_i);
                            o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                            drain_cnt_d = drain_cnt_q - 1;
                            accum_requant_sent_d = 1'b0;
                        end
                        if (requant_invalid) begin
                            obi_o_req_o[0] = 1'b0;
                        end
                    end
                end

                if ((psum_prefetch_rows_q != 0) && !psum_read_active_q && !psum_fifo_full &&
                    !(|obi_o_req_o)) begin
                    drain_state_d = DRAIN_ACCUM_READ;
                    obi_o_req_o = 4'b1111;
                    obi_o_we_o = '0;
                    obi_o_addr_o[0] = a_ptr_q + 0;
                    obi_o_addr_o[1] = a_ptr_q + OFM_BEAT_BYTES;
                    obi_o_addr_o[2] = a_ptr_q + (2 * OFM_BEAT_BYTES);
                    obi_o_addr_o[3] = a_ptr_q + (3 * OFM_BEAT_BYTES);
                    if (obi_o_gnt_i == 4'b1111) begin
                        a_ptr_d = next_strided_ptr(a_ptr_q, a_col_q, 32'(OFM_ROW_BYTES),
                                                   cfg_sys_psum_row_stride_bytes_i,
                                                   cfg_sys_ofm_tile_cols_i);
                        a_col_d = next_strided_col(a_col_q, cfg_sys_ofm_tile_cols_i);
                        psum_prefetch_rows_d = psum_prefetch_rows_q - 1;
                        psum_read_active_d = 1'b1;
                        psum_read_resp_mask_d = '0;
                        psum_read_row_d = '0;
                    end
                end
            end
        end

        case (state_q)
            IDLE: begin
                if (cfg_sys_start_i) begin
                    w_ptr_d = cfg_sys_weight_ptr_i + ((ARRAY_DIM - 1) * 32);
                    i_ptr_d = cfg_sys_ifm_ptr_i;
                    o_ptr_d = cfg_sys_ofm_ptr_i;
                    a_ptr_d = cfg_sys_psum_ptr_i;
                    o_col_d = '0;
                    a_col_d = '0;
                    req_cnt_d = ARRAY_DIM;
                    rsp_cnt_d = ARRAY_DIM;
                    drain_cnt_d = cfg_sys_dim_m_i;
                    drain_state_d = DRAIN_IDLE;
                    accum_row_d = '0;
                    accum_beat_d = '0;
                    accum_req_beat_d = '0;
                    accum_requant_sent_d = 1'b0;
                    psum_read_row_d = '0;
                    psum_read_active_d = 1'b0;
                    psum_read_resp_mask_d = '0;
                    psum_prefetch_rows_d = cfg_sys_accum_en_i ? cfg_sys_dim_m_i : '0;
                    array_flush_cnt_d = '0;
                    weight_preload_active_d = 1'b0;
                    weight_preload_done_d = 1'b0;
                    weight_preload_req_cnt_d = '0;
                    weight_preload_rsp_cnt_d = '0;
                    weight_preload_obi_rsp_cnt_d = '0;
                    weight_preload_ptr_d = '0;
                    k_tile_idx_d = '0;
                    k_seed_ic_d = cfg_linebuf_k_seed_ic_i;
                    k_seed_kw_d = cfg_linebuf_k_seed_kw_i;
                    k_seed_kh_d = cfg_linebuf_k_seed_kh_i;
                    k_channel_offset_d = '0;
                    ofm_push_row_idx_d = '0;
                    pool_acc_d = '0;
                    pool_out_d = '0;
                    pool_out_valid_d = 1'b0;
                    pool_tap_count_d = '0;
                    dw_acc_d = '0;
                    dw_tap_count_d = '0;
                    dw_group_idx_d = '0;
                    if (psum_buf_active) begin
                        psum_buf_sel_d = ~psum_buf_sel_q;
                    end
                    state_d = LOAD_WEIGHTS;

                    if (linebuf_pool_mode) begin
                        req_cnt_d = '0;
                        rsp_cnt_d = '0;
                        drain_cnt_d = linebuf_spatial_m;
                        psum_prefetch_rows_d = '0;
                        linebuf_start = 1'b1;
                        state_d = COMPUTE;
                    end

                    if (linebuf_depthwise_mode) begin
                        w_ptr_d = cfg_sys_weight_ptr_i;
                        o_ptr_d = cfg_sys_ofm_ptr_i;
                        req_cnt_d = pool_kernel_vectors;
                        rsp_cnt_d = pool_kernel_vectors;
                        drain_cnt_d = linebuf_spatial_m;
                        psum_prefetch_rows_d = '0;
                        state_d = LOAD_WEIGHTS;
                    end

                    if (cfg_requant_en_i && requant_config_invalid) begin
                        w_ptr_d = cfg_sys_weight_ptr_i;
                        req_cnt_d = '0;
                        rsp_cnt_d = '0;
                        drain_cnt_d = '0;
                        psum_prefetch_rows_d = '0;
                        state_d = DONE;
                    end
                end
            end

            LOAD_WEIGHTS: begin
                if (linebuf_depthwise_mode) begin
                    if (req_cnt_q > 0) begin
                        obi_w_req_o = 1'b1;
                        obi_w_addr_o = w_ptr_q;
                        if (obi_w_req_o && obi_w_gnt_i) begin
                            w_ptr_d = w_ptr_q + 32;
                            req_cnt_d = req_cnt_q - 1;
                        end
                    end
                    if (obi_w_rvalid_i) begin
                        if (dw_weight_rsp_idx < DW_TAP_COUNT_W'(DW_MAX_TAPS)) begin
                            dw_weight_d[dw_weight_rsp_idx] = obi_w_rdata_i;
                        end
                        rsp_cnt_d = rsp_cnt_q - 1;
                    end
                    if ((req_cnt_q == 0) && (rsp_cnt_q == 1) && obi_w_rvalid_i) begin
                        linebuf_start = 1'b1;
                        dw_acc_d = '0;
                        dw_tap_count_d = '0;
                        state_d = COMPUTE;
                    end
                end else if (weight_preload_done_q) begin
                    req_cnt_d = cfg_sys_dim_m_i;
                    rsp_cnt_d = cfg_sys_dim_m_i;
                    drain_cnt_d = cfg_sys_dim_m_i;
                    drain_state_d = DRAIN_IDLE;
                    accum_row_d = '0;
                    accum_beat_d = '0;
                    accum_req_beat_d = '0;
                    accum_requant_sent_d = 1'b0;
                    psum_read_row_d = '0;
                    psum_read_active_d = 1'b0;
                    psum_read_resp_mask_d = '0;
                    psum_prefetch_rows_d = accum_uses_tcdm_psum ? cfg_sys_dim_m_i : '0;
                    array_flush_cnt_d = '0;
                    drain_cnt_d = (psum_buf_overlap_active && (k_tile_idx_q != 32'd0)) ?
                                  (drain_cnt_q + cfg_sys_dim_m_i) : cfg_sys_dim_m_i;
                    ofm_push_row_idx_d = '0;
                    weight_preload_done_d = 1'b0;
                    if (cfg_linebuf_en_i) begin
                        linebuf_next_tile = 1'b1;
                    end
                    state_d = COMPUTE;
                end else if (req_cnt_q > 0) begin
                    obi_w_req_o = !weight_fifo_full;
                    obi_w_addr_o = w_ptr_q;
                    if (obi_w_req_o && obi_w_gnt_i) begin
                        w_ptr_d = w_ptr_q - 32;
                        req_cnt_d   = req_cnt_q - 1;
                    end
                end
                if (!linebuf_depthwise_mode) begin
                    weight_fifo_push = obi_w_rvalid_i && !weight_fifo_full;
                    if (!weight_fifo_empty) begin
                        weight_load_en = 1'b1;
                        weight_fifo_pop  = 1'b1;
                        weight_data    = weight_fifo_out;
                        rsp_cnt_d = rsp_cnt_q - 1;
                    end
                    if (req_cnt_q == 0 && rsp_cnt_q == 1 && weight_fifo_pop) begin
                        req_cnt_d = cfg_sys_dim_m_i;
                        rsp_cnt_d = cfg_sys_dim_m_i;
                        drain_cnt_d = cfg_sys_dim_m_i;
                        drain_state_d = DRAIN_IDLE;
                        accum_row_d = '0;
                        accum_beat_d = '0;
                        accum_req_beat_d = '0;
                        accum_requant_sent_d = 1'b0;
                        psum_read_row_d = '0;
                        psum_read_active_d = 1'b0;
                        psum_read_resp_mask_d = '0;
                        psum_prefetch_rows_d = accum_uses_tcdm_psum ? cfg_sys_dim_m_i : '0;
                        drain_cnt_d = (psum_buf_overlap_active && (k_tile_idx_q != 32'd0)) ?
                                      (drain_cnt_q + cfg_sys_dim_m_i) : cfg_sys_dim_m_i;
                        ofm_push_row_idx_d = '0;
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
            end

            COMPUTE: begin
                if (linebuf_depthwise_mode) begin
                    obi_i_req_o = linebuf_obi_req;
                    obi_i_addr_o = linebuf_obi_addr;

                    if (requant_out_valid) begin
                        obi_o_req_o[0] = 1'b1;
                        obi_o_wdata_o[0] = requant_packed_data;
                        if (obi_o_gnt_i[0] || requant_invalid) begin
                            requant_out_ready = 1'b1;
                            o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                                       cfg_sys_ofm_row_stride_bytes_i,
                                                       cfg_sys_ofm_tile_cols_i);
                            o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                            drain_cnt_d = drain_cnt_q - 1'b1;
                        end
                        if (requant_invalid) begin
                            obi_o_req_o[0] = 1'b0;
                        end
                    end

                    if (linebuf_row_valid && !requant_config_invalid &&
                        (({27'd0, dw_tap_count_q} + 32'd1) != pool_kernel_vectors || requant_in_ready)) begin
                        dw_next_acc = depthwise_mac_row(dw_acc_q,
                                                        linebuf_row_data,
                                                        dw_weight_q[dw_tap_count_q],
                                                        (dw_tap_count_q == '0),
                                                        dw_group_valid_bytes);
                        linebuf_row_ready = 1'b1;
                        if ({27'd0, dw_tap_count_q} + 32'd1 == pool_kernel_vectors) begin
                            dw_requant_acc = dw_next_acc;
                            requant_in_valid = cfg_requant_en_i;
                            dw_acc_d = '0;
                            dw_tap_count_d = '0;
                        end else begin
                            dw_acc_d = dw_next_acc;
                            dw_tap_count_d = dw_tap_count_q + 1'b1;
                        end
                    end

                    if ((drain_cnt_q == 32'd0) && !linebuf_busy && !requant_out_valid) begin
                        if (!dw_last_group) begin
                            dw_group_idx_d = dw_group_idx_q + 32'd1;
                            w_ptr_d = cfg_sys_weight_ptr_i + dw_group_weight_offset + dw_weight_group_bytes;
                            o_ptr_d = cfg_sys_ofm_ptr_i + dw_group_output_offset + dw_group_output_bytes;
                            o_col_d = '0;
                            req_cnt_d = pool_kernel_vectors;
                            rsp_cnt_d = pool_kernel_vectors;
                            drain_cnt_d = linebuf_spatial_m;
                            dw_acc_d = '0;
                            dw_tap_count_d = '0;
                            state_d = LOAD_WEIGHTS;
                        end else begin
                            state_d = DONE;
                        end
                    end
                end else if (linebuf_pool_mode) begin
                    obi_i_req_o = linebuf_obi_req;
                    obi_i_addr_o = linebuf_obi_addr;

                    if (pool_out_valid_q) begin
                        obi_o_req_o[0] = 1'b1;
                        obi_o_we_o[0] = 1'b1;
                        obi_o_be_o[0] = '1;
                        obi_o_addr_o[0] = o_ptr_q;
                        obi_o_wdata_o[0] = pool_out_q;
                        if (obi_o_gnt_i[0]) begin
                            pool_out_valid_d = 1'b0;
                            o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                                       cfg_sys_ofm_row_stride_bytes_i,
                                                       cfg_sys_ofm_tile_cols_i);
                            o_col_d = next_strided_col(o_col_q, cfg_sys_ofm_tile_cols_i);
                            drain_cnt_d = drain_cnt_q - 1'b1;
                        end
                    end

                    if (linebuf_row_valid && !pool_out_valid_q) begin
                        linebuf_row_ready = 1'b1;
                        if (pool_tap_count_q == 8'd0) begin
                            pool_next_acc = linebuf_row_data;
                        end else begin
                            pool_next_acc = max_i8_row(pool_acc_q, linebuf_row_data);
                        end
                        pool_acc_d = pool_next_acc;

                        if ({24'd0, pool_tap_count_q} + 32'd1 == pool_kernel_vectors) begin
                            pool_out_d = pool_next_acc;
                            pool_out_valid_d = 1'b1;
                            pool_tap_count_d = '0;
                        end else begin
                            pool_tap_count_d = pool_tap_count_q + 8'd1;
                        end
                    end

                    if ((drain_cnt_q == 32'd0) && !pool_out_valid_q && !linebuf_busy) begin
                        state_d = DONE;
                    end
                end else if (cfg_linebuf_en_i) begin
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
                end else begin
                    if (req_cnt_q > 0) begin
                        obi_i_req_o = !ifm_fifo_full && array_pipe_ready;
                        obi_i_addr_o = i_ptr_q;
                        if (obi_i_req_o && obi_i_gnt_i) begin
                            i_ptr_d = i_ptr_q + 32;
                            req_cnt_d   = req_cnt_q - 1;
                        end
                    end
                    ifm_fifo_push = obi_i_rvalid_i && !ifm_fifo_full;
                    if (!ifm_fifo_empty && array_pipe_ready) begin
                        compute_en = 1'b1;
                        clear_acc  = 1'b0;
                        ifm_fifo_pop = 1'b1;
                        ifm_data   = ifm_fifo_out;
                        rsp_cnt_d = rsp_cnt_q - 1;
                    end
                    if (req_cnt_q == 0 && rsp_cnt_q == 1 && ifm_fifo_pop) begin
                        array_flush_cnt_d = ARRAY_FLUSH_COUNT_W'(ARRAY_FLUSH_CYCLES);
                        state_d = WAIT_DRAIN;
                    end
                end
            end

            WAIT_DRAIN: begin
                if ((array_flush_cnt_q != '0) && array_pipe_ready) begin
                    array_flush_cnt_d = array_flush_cnt_q - 1'b1;
                end

                if (linebuf_has_next_k_tile && !weight_preload_active_q &&
                    !weight_preload_done_q && (array_flush_cnt_q == '0) &&
                    !linebuf_prefetch_busy) begin
                    weight_preload_active_d = 1'b1;
                    weight_preload_req_cnt_d = ARRAY_DIM;
                    weight_preload_rsp_cnt_d = ARRAY_DIM;
                    weight_preload_obi_rsp_cnt_d = ARRAY_DIM;
                    weight_preload_ptr_d = cfg_sys_weight_ptr_i +
                                            (((k_tile_idx_q + 32'd1) * ARRAY_DIM * 32) +
                                             ((ARRAY_DIM - 1) * 32));
                end

                if (weight_preload_active_q) begin
                    if (weight_preload_req_cnt_q > 0) begin
                        obi_w_req_o = !weight_fifo_full;
                        obi_w_addr_o = weight_preload_ptr_q;
                        if (obi_w_req_o && obi_w_gnt_i) begin
                            weight_preload_ptr_d = weight_preload_ptr_q - 32;
                            weight_preload_req_cnt_d = weight_preload_req_cnt_q - 1;
                        end
                    end
                    weight_fifo_push = (weight_preload_obi_rsp_cnt_q != 32'd0) &&
                                       obi_w_rvalid_i && !weight_fifo_full;
                    if (weight_fifo_push) begin
                        weight_preload_obi_rsp_cnt_d = weight_preload_obi_rsp_cnt_q - 1'b1;
                    end
                    if (!weight_fifo_empty && array_pipe_ready) begin
                        weight_load_en = 1'b1;
                        weight_fifo_pop = 1'b1;
                        weight_data = weight_fifo_out;
                        weight_preload_rsp_cnt_d = weight_preload_rsp_cnt_q - 1;
                    end
                    if (weight_preload_req_cnt_q == 0 &&
                        weight_preload_rsp_cnt_q == 1 &&
                        weight_fifo_pop) begin
                        weight_preload_active_d = 1'b0;
                        weight_preload_done_d = 1'b1;
                    end
                end

                linebuf_prefetch = linebuf_has_next_k_tile &&
                                   (linebuf_prefetch_busy ||
                                    (array_flush_cnt_q != '0) ||
                                    (drain_cnt_q != 0) ||
                                    !ofm_fifo_empty);
                if (linebuf_prefetch) begin
                    obi_i_req_o = linebuf_obi_req;
                    obi_i_addr_o = linebuf_obi_addr;
                end

                if (accum_active) begin
                    if (psum_buf_overlap_next_safe) begin
                        k_seed_kh_d = k_seed_kh_next;
                        k_seed_kw_d = k_seed_kw_next;
                        k_seed_ic_d = k_seed_ic_next;
                        k_channel_offset_d = k_channel_offset_next;
                        k_tile_idx_d = k_tile_idx_q + 32'd1;
                        i_ptr_d = cfg_sys_ifm_ptr_i;
                        o_ptr_d = cfg_sys_ofm_ptr_i;
                        a_ptr_d = cfg_sys_psum_ptr_i;
                        o_col_d = '0;
                        a_col_d = '0;
                        req_cnt_d = cfg_sys_dim_m_i;
                        rsp_cnt_d = cfg_sys_dim_m_i;
                        drain_cnt_d = drain_cnt_q + cfg_sys_dim_m_i;
                        ofm_push_row_idx_d = '0;
                        weight_preload_done_d = 1'b0;
                        linebuf_next_tile = 1'b1;
                        state_d = COMPUTE;
                    end else if (drain_cnt_q == 0 && ofm_fifo_empty) begin
                        if (linebuf_has_next_k_tile && weight_preload_done_q && !linebuf_prefetch_busy) begin
                            k_seed_kh_d = k_seed_kh_next;
                            k_seed_kw_d = k_seed_kw_next;
                            k_seed_ic_d = k_seed_ic_next;
                            k_channel_offset_d = k_channel_offset_next;
                            k_tile_idx_d = k_tile_idx_q + 32'd1;
                            i_ptr_d = cfg_sys_ifm_ptr_i;
                            o_ptr_d = cfg_sys_ofm_ptr_i;
                            a_ptr_d = cfg_sys_psum_ptr_i;
                            o_col_d = '0;
                            a_col_d = '0;
                            state_d = LOAD_WEIGHTS;
                        end else if (linebuf_has_next_k_tile) begin
                            state_d = WAIT_DRAIN;
                        end else begin
                            state_d = DONE;
                        end
                    end
                end else begin
                    if (psum_buf_overlap_next_safe) begin
                        k_seed_kh_d = k_seed_kh_next;
                        k_seed_kw_d = k_seed_kw_next;
                        k_seed_ic_d = k_seed_ic_next;
                        k_channel_offset_d = k_channel_offset_next;
                        k_tile_idx_d = k_tile_idx_q + 32'd1;
                        i_ptr_d = cfg_sys_ifm_ptr_i;
                        o_ptr_d = cfg_sys_ofm_ptr_i;
                        a_ptr_d = cfg_sys_psum_ptr_i;
                        o_col_d = '0;
                        a_col_d = '0;
                        req_cnt_d = cfg_sys_dim_m_i;
                        rsp_cnt_d = cfg_sys_dim_m_i;
                        drain_cnt_d = drain_cnt_q + cfg_sys_dim_m_i;
                        ofm_push_row_idx_d = '0;
                        weight_preload_done_d = 1'b0;
                        linebuf_next_tile = 1'b1;
                        state_d = COMPUTE;
                    end else if (drain_cnt_q == 0 && ofm_fifo_empty) begin
                        if (linebuf_has_next_k_tile && weight_preload_done_q && !linebuf_prefetch_busy) begin
                            k_seed_kh_d = k_seed_kh_next;
                            k_seed_kw_d = k_seed_kw_next;
                            k_seed_ic_d = k_seed_ic_next;
                            k_channel_offset_d = k_channel_offset_next;
                            k_tile_idx_d = k_tile_idx_q + 32'd1;
                            i_ptr_d = cfg_sys_ifm_ptr_i;
                            o_ptr_d = cfg_sys_ofm_ptr_i;
                            a_ptr_d = cfg_sys_psum_ptr_i;
                            o_col_d = '0;
                            a_col_d = '0;
                            state_d = LOAD_WEIGHTS;
                        end else if (linebuf_has_next_k_tile) begin
                            state_d = WAIT_DRAIN;
                        end else begin
                            state_d = DONE;
                        end
                    end
                end
            end

            DONE: begin
                cfg_sys_done_o = 1'b1;
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q         <= IDLE;
            drain_state_q   <= DRAIN_IDLE;
            w_ptr_q         <= '0;
            i_ptr_q         <= '0;
            o_ptr_q         <= '0;
            a_ptr_q         <= '0;
            o_col_q         <= '0;
            a_col_q         <= '0;
            req_cnt_q       <= '0;
            rsp_cnt_q       <= '0;
            drain_cnt_q     <= '0;
            accum_row_q     <= '0;
            accum_beat_q    <= '0;
            accum_req_beat_q <= '0;
            accum_requant_sent_q <= 1'b0;
            psum_read_row_q <= '0;
            psum_read_active_q <= 1'b0;
            psum_read_resp_mask_q <= '0;
            psum_prefetch_rows_q <= '0;
            k_tile_idx_q    <= '0;
            k_seed_ic_q     <= '0;
            k_seed_kw_q     <= '0;
            k_seed_kh_q     <= '0;
            k_channel_offset_q <= '0;
            array_flush_cnt_q <= '0;
            weight_preload_active_q <= 1'b0;
            weight_preload_done_q <= 1'b0;
            weight_preload_req_cnt_q <= '0;
            weight_preload_rsp_cnt_q <= '0;
            weight_preload_obi_rsp_cnt_q <= '0;
            weight_preload_ptr_q <= '0;
            ofm_push_row_idx_q <= '0;
            psum_buf_sel_q <= 1'b0;
            pool_acc_q <= '0;
            pool_out_q <= '0;
            pool_out_valid_q <= 1'b0;
            pool_tap_count_q <= '0;
            dw_acc_q <= '0;
            dw_tap_count_q <= '0;
            dw_group_idx_q <= '0;
            for (int unsigned tap = 0; tap < DW_MAX_TAPS; tap++) begin
                dw_weight_q[tap] <= '0;
            end
        end else begin
            state_q     <= state_d;
            drain_state_q <= drain_state_d;
            w_ptr_q     <= w_ptr_d;
            i_ptr_q     <= i_ptr_d;
            o_ptr_q     <= o_ptr_d;
            a_ptr_q     <= a_ptr_d;
            o_col_q     <= o_col_d;
            a_col_q     <= a_col_d;
            req_cnt_q   <= req_cnt_d;
            rsp_cnt_q   <= rsp_cnt_d;
            drain_cnt_q <= drain_cnt_d;
            accum_row_q <= accum_row_d;
            accum_beat_q <= accum_beat_d;
            accum_req_beat_q <= accum_req_beat_d;
            accum_requant_sent_q <= accum_requant_sent_d;
            psum_read_row_q <= psum_read_row_d;
            psum_read_active_q <= psum_read_active_d;
            psum_read_resp_mask_q <= psum_read_resp_mask_d;
            psum_prefetch_rows_q <= psum_prefetch_rows_d;
            k_tile_idx_q <= k_tile_idx_d;
            k_seed_ic_q <= k_seed_ic_d;
            k_seed_kw_q <= k_seed_kw_d;
            k_seed_kh_q <= k_seed_kh_d;
            k_channel_offset_q <= k_channel_offset_d;
            array_flush_cnt_q <= array_flush_cnt_d;
            weight_preload_active_q <= weight_preload_active_d;
            weight_preload_done_q <= weight_preload_done_d;
            weight_preload_req_cnt_q <= weight_preload_req_cnt_d;
            weight_preload_rsp_cnt_q <= weight_preload_rsp_cnt_d;
            weight_preload_obi_rsp_cnt_q <= weight_preload_obi_rsp_cnt_d;
            weight_preload_ptr_q <= weight_preload_ptr_d;
            ofm_push_row_idx_q <= ofm_push_row_idx_d;
            psum_buf_sel_q <= psum_buf_sel_d;
            pool_acc_q <= pool_acc_d;
            pool_out_q <= pool_out_d;
            pool_out_valid_q <= pool_out_valid_d;
            pool_tap_count_q <= pool_tap_count_d;
            dw_acc_q <= dw_acc_d;
            dw_tap_count_q <= dw_tap_count_d;
            dw_group_idx_q <= dw_group_idx_d;
            for (int unsigned tap = 0; tap < DW_MAX_TAPS; tap++) begin
                dw_weight_q[tap] <= dw_weight_d[tap];
            end
            if (psum_buf_we) begin
                psum_buf_q[psum_buf_sel_q][psum_buf_addr] <= psum_buf_wdata;
            end
        end
    end

endmodule
