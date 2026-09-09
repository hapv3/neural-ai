`default_nettype none

module conv_linebuf_stream_packer #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned MAX_INPUT_W = 640
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      start_i,
    input  logic                      next_tile_i,
    input  logic                      prefetch_i,
    input  logic [31:0]               dim_m_i,
    input  logic [31:0]               cfg_k_tiles_i,

    input  logic [31:0]               cfg_origin_base_i,
    input  logic [31:0]               cfg_row_stride_bytes_i,
    input  logic [31:0]               cfg_pixel_stride_bytes_i,
    input  logic [31:0]               cfg_ow_step_bytes_i,
    input  logic [31:0]               cfg_oh_step_bytes_i,
    input  logic [15:0]               cfg_input_h_i,
    input  logic [15:0]               cfg_input_w_i,
    input  logic [15:0]               cfg_input_c_i,
    input  logic [15:0]               cfg_output_w_i,
    input  logic [15:0]               cfg_kernel_h_i,
    input  logic [15:0]               cfg_kernel_w_i,
    input  logic [15:0]               cfg_stride_h_i,
    input  logic [15:0]               cfg_stride_w_i,
    input  logic [15:0]               cfg_pad_h_i,
    input  logic [15:0]               cfg_pad_w_i,
    input  logic [15:0]               cfg_c_base_i,
    input  logic [5:0]                cfg_lane_base_i,
    input  logic                      cfg_coalesce_i,
    input  logic                      cfg_kgen_i,
    input  logic                      cfg_pool_i,
    input  logic                      cfg_c32_fast_i,
    input  logic                      cfg_depthwise_i,
    input  logic [5:0]                cfg_block_valid_bytes_i,
    input  logic [31:0]               cfg_channel_addr_offset_i,
    input  logic [31:0]               cfg_coalesce_k_bytes_i,
    input  logic [7:0]                cfg_k_seed_kh_i,
    input  logic [7:0]                cfg_k_seed_kw_i,
    input  logic [15:0]               cfg_k_seed_ic_i,

    output logic                      obi_req_o,
    input  logic                      obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_addr_o,
    input  logic                      obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_rdata_i,

    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] row_data_o,
    output logic                      row_valid_o,
    input  logic                      row_ready_i,
    output logic                      busy_o,
    output logic                      done_o,
    output logic                      prefetch_busy_o,
    output logic [31:0]               emitted_vectors_o,
    output logic [31:0]               fetch_beats_o,
    output logic [31:0]               bypass_vectors_o,
    output logic [4:0]                debug_state_o
);

    localparam int unsigned K_MAX = 5;
    localparam int unsigned STRIDE_MAX = 2;
    localparam int unsigned ROW_SLOTS = K_MAX + STRIDE_MAX;
    localparam int unsigned BANKS = ROW_SLOTS * STRIDE_MAX;
    localparam int unsigned BANK_DEPTH = (MAX_INPUT_W + STRIDE_MAX - 1) / STRIDE_MAX;
    localparam int unsigned BANK_ADDR_WIDTH = $clog2(BANK_DEPTH);
    localparam int unsigned ROW_PENDING_WIDTH = $clog2((2 * MAX_INPUT_W) + 1);

    typedef enum logic [4:0] {
        CH_IDLE,
        CH_ENSURE,
        CH_FILL_REQ0,
        CH_FILL_REQ1,
        CH_FILL_DRAIN,
        CH_WINDOW_REQ,
        CH_WINDOW_WAIT,
        CH_STREAM_PRIME,
        CH_STREAM_EMIT,
        CH_BYPASS_PREP,
        CH_BYPASS_REQ0,
        CH_BYPASS_WAIT0,
        CH_BYPASS_REQ1,
        CH_BYPASS_WAIT1,
        CH_STREAM_DONE
    } state_e;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_t;

    state_e state_q;

    logic [BANKS-1:0] bank_w_req;
    logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_w_addr;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_w_data;
    logic [BANKS-1:0] bank_r_req;
    logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_r_addr;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_rdata;

    window_t window_data;

    logic [31:0] output_row_base_addr_q;
    logic [31:0] output_spatial_addr_q;
    logic signed [31:0] output_base_ih_q;
    logic signed [31:0] output_base_iw_q;
    logic [15:0] ow_q;
    logic [15:0] kh_q;
    logic [15:0] kw_q;
    logic [31:0] spatial_rows_q;
    logic [31:0] emitted_vectors_q;
    logic [31:0] fetch_engine_beats;
    logic [31:0] k_tile_idx_q;
    logic row_cache_full;
    logic [15:0] cached_c_base;

    logic [15:0] fill_kh_q;

    logic [15:0] window_kw_q;
    logic [15:0] window_req_kw;
    logic window_clear;
    logic window_load_request;
    logic window_load_capture;
    logic window_slide_commit;

    logic bg_started_for_row_q;

    input_row_t row_data_q;
    logic row_valid_out_q;
    logic done_q;
    logic prefetch_active_q;
    logic prefetch_ready_q;
    logic [15:0] prefetched_c_base_q;

    logic [5:0] block_valid_bytes;
    logic [31:0] coalesce_k_bytes;
    logic [ARRAY_DIM-1:0][7:0] lane_kh;
    logic [ARRAY_DIM-1:0][7:0] lane_kw;
    logic [ARRAY_DIM-1:0][15:0] lane_ic;

    logic signed [31:0] fill_ih;
    logic fill_row_in_bounds;
    logic bypass_active;
    logic config_rejected;
    logic emit_fire;
    logic stg1_fire;
    logic last_kernel_vector;
    logic vector_last_for_spatial;
    logic last_spatial;
    logic has_next_same_row;
    logic has_next2_same_row;
    logic more_k_tiles;
    logic row_cache_full_mode;
    logic row_cache_reuse;
    logic row_ring_mode;
    logic c32_blocked_mode;
    logic c32_kgen_fast;
    logic [15:0] effective_c_base;
    logic [31:0] channel_addr_offset;
    logic [15:0] fill_row_slot;
    logic fill_row_cached;
    logic fill_row_pending;
    logic fill_row_ready;
    logic [15:0] fill_done_rows;
    logic bg_can_start;
    logic fetch_main_start;
    logic fetch_main_request_accepted;
    logic [1:0] fetch_main_next_phase;
    logic fetch_main_done;
    logic fetch_background_start;
    logic fetch_background_idle;
    logic fetch_background_reset;
    logic [$clog2(ROW_SLOTS)-1:0] fetch_background_query_slot;
    logic [15:0] fetch_background_query_ih;
    logic [$clog2(ROW_SLOTS)-1:0] fetch_main_alloc_slot;
    logic [15:0] fetch_main_alloc_ih;
    logic [$clog2(ROW_SLOTS)-1:0] fetch_background_alloc_slot;
    logic [15:0] fetch_background_alloc_ih;
    logic fetch_obi_req;
    logic fetch_obi_gnt;
    logic [ADDR_WIDTH-1:0] fetch_obi_addr;
    logic bypass_start;
    logic bypass_obi_req;
    logic [ADDR_WIDTH-1:0] bypass_obi_addr;
    input_row_t bypass_row;
    logic bypass_row_valid;
    logic [31:0] bypass_fetch_beats;
    logic [31:0] bypass_emitted_vectors;
    logic [4:0] bypass_debug_state;
    logic beat_push;
    logic [$clog2(ROW_SLOTS)-1:0] beat_push_slot;
    logic beat_push_last_for_row;
    logic beat_pop;
    logic [$clog2(ROW_SLOTS)-1:0] beat_pop_slot;
    logic row_store_job_start;
    logic row_store_invalidate;
    logic row_store_cached_c_base_set;
    logic row_store_main_alloc;
    logic row_store_background_alloc;
    logic row_store_main_cached;
    logic row_store_main_pending;
    logic row_store_background_cached;
    logic row_store_background_pending;
    logic tile_advance_event;
    logic prefetch_start_event;
    logic [15:0] next_kh;
    logic [15:0] next_kw;
    logic signed [31:0] slide_from_iw;
    logic slide_req_active;
    logic [DATA_WIDTH-1:0] pad_vector;
    logic [31:0] pad_row_offset;
    
    // Stage 1 Pipeline Registers (Coordinates)
    logic [ARRAY_DIM-1:0][7:0] stg1_lane_kh_q;
    logic [ARRAY_DIM-1:0][7:0] stg1_lane_kw_q;
    logic [ARRAY_DIM-1:0][15:0] stg1_lane_ic_q;
    logic [15:0] stg1_tap_kh_q;
    logic [15:0] stg1_tap_kw_q;
    logic        stg1_valid_q;

    logic stg2_ready;
    logic formatter_pipe_ready;
    input_row_t formatter_row;
    logic formatter_valid;
    logic formatter_empty;
    logic formatter_stream_drained;

    assign formatter_pipe_ready = row_ready_i || !row_valid_out_q;
    assign stg2_ready = formatter_pipe_ready;
    assign formatter_stream_drained = !stg1_valid_q &&
                                      formatter_empty &&
                                      (!row_valid_out_q || row_ready_i);

    assign row_data_o = bypass_active ? bypass_row : row_data_q;
    assign row_valid_o = bypass_active ? bypass_row_valid : row_valid_out_q;
    assign busy_o = state_q != CH_IDLE;
    assign done_o = done_q;
    assign prefetch_busy_o = prefetch_active_q;
    assign emitted_vectors_o = emitted_vectors_q;
    assign fetch_beats_o = fetch_engine_beats + bypass_fetch_beats;
    assign bypass_vectors_o = bypass_emitted_vectors;
    assign debug_state_o = (state_q == CH_BYPASS_PREP) ? bypass_debug_state : state_q;

    conv_linebuf_config_decoder #(
        .DATA_WIDTH (DATA_WIDTH),
        .ARRAY_DIM  (ARRAY_DIM)
    ) i_config_decoder (
        .cfg_k_tiles_i,
        .cfg_row_stride_bytes_i,
        .cfg_input_h_i,
        .cfg_input_c_i,
        .cfg_kernel_h_i,
        .cfg_kernel_w_i,
        .cfg_stride_h_i,
        .cfg_stride_w_i,
        .cfg_pad_h_i,
        .cfg_c_base_i,
        .cfg_lane_base_i,
        .cfg_coalesce_i,
        .cfg_kgen_i,
        .cfg_pool_i,
        .cfg_c32_fast_i,
        .cfg_depthwise_i,
        .cfg_block_valid_bytes_i,
        .cfg_channel_addr_offset_i,
        .cfg_coalesce_k_bytes_i,
        .cfg_k_seed_kh_i,
        .cfg_k_seed_kw_i,
        .cfg_k_seed_ic_i,
        .row_cache_full_i       (row_cache_full),
        .cached_c_base_i        (cached_c_base),
        .block_valid_bytes_o    (block_valid_bytes),
        .coalesce_k_bytes_o     (coalesce_k_bytes),
        .lane_kh_o              (lane_kh),
        .lane_kw_o              (lane_kw),
        .lane_ic_o              (lane_ic),
        .effective_c_base_o     (effective_c_base),
        .channel_addr_offset_o  (channel_addr_offset),
        .c32_blocked_mode_o     (c32_blocked_mode),
        .c32_kgen_fast_o        (c32_kgen_fast),
        .row_cache_full_mode_o  (row_cache_full_mode),
        .row_cache_reuse_o      (row_cache_reuse),
        .row_ring_mode_o        (row_ring_mode),
        .fill_done_rows_o       (fill_done_rows),
        .pad_vector_o           (pad_vector),
        .pad_row_offset_o       (pad_row_offset)
    );

    conv_linebuf_fetch_engine #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .ROW_SLOTS        (ROW_SLOTS),
        .BANKS            (BANKS),
        .BANK_ADDR_WIDTH  (BANK_ADDR_WIDTH)
    ) i_fetch_engine (
        .clk_i,
        .rst_ni,
        .clear_count_i                  (row_store_job_start),
        .reset_background_i             (fetch_background_reset),
        .row_ring_mode_i                (row_ring_mode),
        .c32_blocked_mode_i             (c32_blocked_mode),
        .input_h_i                      (cfg_input_h_i),
        .input_w_i                      (cfg_input_w_i),
        .kernel_h_i                     (cfg_kernel_h_i),
        .pixel_stride_bytes_i           (cfg_pixel_stride_bytes_i),
        .row_stride_bytes_i             (cfg_row_stride_bytes_i),
        .channel_addr_offset_i          (channel_addr_offset),
        .main_start_i                   (fetch_main_start),
        .main_base_addr_i               (
            row_cache_full ?
            row_tap_addr(cfg_origin_base_i + pad_row_offset, fill_kh_q) :
            row_tap_addr(output_row_base_addr_q, fill_kh_q)
        ),
        .main_row_slot_i                (row_ring_mode ? fill_row_slot : fill_kh_q),
        .main_row_ih_i                  (fill_ih[15:0]),
        .main_valid_bytes_i             (block_valid_bytes),
        .main_row_ready_i               (fill_row_ready),
        .main_request_accepted_o        (fetch_main_request_accepted),
        .main_next_phase_o              (fetch_main_next_phase),
        .main_done_o                    (fetch_main_done),
        .main_alloc_valid_o             (row_store_main_alloc),
        .main_alloc_slot_o              (fetch_main_alloc_slot),
        .main_alloc_ih_o                (fetch_main_alloc_ih),
        .background_start_i             (fetch_background_start),
        .background_base_ih_i           (
            output_base_ih_q + $signed({16'd0, cfg_stride_h_i})
        ),
        .background_row_base_addr_i     (
            output_row_base_addr_q + cfg_oh_step_bytes_i
        ),
        .background_idle_o              (fetch_background_idle),
        .background_query_slot_o        (fetch_background_query_slot),
        .background_query_ih_o          (fetch_background_query_ih),
        .background_row_cached_i        (row_store_background_cached),
        .background_row_pending_i       (row_store_background_pending),
        .background_alloc_valid_o       (row_store_background_alloc),
        .background_alloc_slot_o        (fetch_background_alloc_slot),
        .background_alloc_ih_o          (fetch_background_alloc_ih),
        .obi_req_o                      (fetch_obi_req),
        .obi_gnt_i                      (fetch_obi_gnt),
        .obi_addr_o                     (fetch_obi_addr),
        .obi_rvalid_i,
        .obi_rdata_i,
        .fetch_beats_o                  (fetch_engine_beats),
        .beat_push_o                    (beat_push),
        .beat_push_slot_o               (beat_push_slot),
        .beat_push_last_for_row_o       (beat_push_last_for_row),
        .beat_pop_o                     (beat_pop),
        .beat_pop_slot_o                (beat_pop_slot),
        .bank_write_req_o               (bank_w_req),
        .bank_write_addr_o              (bank_w_addr),
        .bank_write_data_o              (bank_w_data)
    );

    conv_linebuf_bypass_engine #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH)
    ) i_bypass_engine (
        .clk_i,
        .rst_ni,
        .start_i                 (bypass_start),
        .last_i                  (last_spatial),
        .spatial_addr_i          (output_spatial_addr_q),
        .base_ih_i               (output_base_ih_q),
        .base_iw_i               (output_base_iw_q),
        .input_h_i               (cfg_input_h_i),
        .input_w_i               (cfg_input_w_i),
        .channel_addr_offset_i   (channel_addr_offset),
        .valid_bytes_i           (block_valid_bytes),
        .lane_base_i             (cfg_lane_base_i),
        .c32_blocked_mode_i      (c32_blocked_mode),
        .obi_req_o               (bypass_obi_req),
        .obi_gnt_i,
        .obi_addr_o              (bypass_obi_addr),
        .obi_rvalid_i,
        .obi_rdata_i,
        .row_o                   (bypass_row),
        .row_valid_o             (bypass_row_valid),
        .row_ready_i,
        .fetch_beats_o           (bypass_fetch_beats),
        .emitted_vectors_o       (bypass_emitted_vectors),
        .debug_state_o           (bypass_debug_state)
    );

    conv_linebuf_row_store #(
        .DATA_WIDTH        (DATA_WIDTH),
        .ROW_SLOTS        (ROW_SLOTS),
        .BANKS            (BANKS),
        .BANK_DEPTH       (BANK_DEPTH),
        .BANK_ADDR_WIDTH  (BANK_ADDR_WIDTH),
        .ROW_PENDING_WIDTH(ROW_PENDING_WIDTH)
    ) i_row_store (
        .clk_i,
        .rst_ni,
        .job_start_i                 (row_store_job_start),
        .job_full_mode_i             (row_cache_full_mode),
        .job_c_base_i                (effective_c_base),
        .invalidate_i                (row_store_invalidate),
        .cached_c_base_set_i         (row_store_cached_c_base_set),
        .cached_c_base_i             (effective_c_base),
        .alloc_main_valid_i          (row_store_main_alloc),
        .alloc_main_slot_i           (fetch_main_alloc_slot),
        .alloc_main_ih_i             (fetch_main_alloc_ih),
        .alloc_background_valid_i    (row_store_background_alloc),
        .alloc_background_slot_i     (fetch_background_alloc_slot),
        .alloc_background_ih_i       (fetch_background_alloc_ih),
        .beat_push_i                 (beat_push),
        .beat_push_slot_i            (beat_push_slot),
        .beat_push_last_for_row_i    (beat_push_last_for_row),
        .beat_pop_i                  (beat_pop),
        .beat_pop_slot_i             (beat_pop_slot),
        .bank_write_req_i            (bank_w_req),
        .bank_write_addr_i           (bank_w_addr),
        .bank_write_data_i           (bank_w_data),
        .bank_read_req_i             (bank_r_req),
        .bank_read_addr_i            (bank_r_addr),
        .bank_read_data_o            (bank_rdata),
        .query_main_slot_i           (fill_row_slot[$clog2(ROW_SLOTS)-1:0]),
        .query_main_ih_i             (fill_ih[15:0]),
        .query_main_cached_o         (row_store_main_cached),
        .query_main_pending_o        (row_store_main_pending),
        .query_background_slot_i     (fetch_background_query_slot),
        .query_background_ih_i       (fetch_background_query_ih),
        .query_background_cached_o   (row_store_background_cached),
        .query_background_pending_o  (row_store_background_pending),
        .row_cache_full_o            (row_cache_full),
        .cached_c_base_o             (cached_c_base)
    );

    conv_linebuf_window_engine #(
        .DATA_WIDTH       (DATA_WIDTH),
        .K_MAX            (K_MAX),
        .BANKS            (BANKS),
        .BANK_ADDR_WIDTH  (BANK_ADDR_WIDTH)
    ) i_window_engine (
        .clk_i,
        .rst_ni,
        .clear_i             (window_clear),
        .load_request_i      (window_load_request),
        .load_capture_i      (window_load_capture),
        .load_request_kw_i   (window_req_kw),
        .load_capture_kw_i   (window_kw_q),
        .slide_request_i     (slide_req_active),
        .slide_from_iw_i     (slide_from_iw),
        .slide_commit_i      (window_slide_commit),
        .input_h_i           (cfg_input_h_i),
        .input_w_i           (cfg_input_w_i),
        .kernel_h_i          (cfg_kernel_h_i),
        .kernel_w_i          (cfg_kernel_w_i),
        .stride_w_i          (cfg_stride_w_i),
        .base_ih_i           (output_base_ih_q),
        .base_iw_i           (output_base_iw_q),
        .row_ring_mode_i     (row_ring_mode),
        .row_cache_full_i    (row_cache_full),
        .pad_vector_i        (pad_vector),
        .bank_read_req_o     (bank_r_req),
        .bank_read_addr_o    (bank_r_addr),
        .bank_read_data_i    (bank_rdata),
        .window_o            (window_data)
    );

    conv_linebuf_formatter_pipeline #(
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH),
        .K_MAX            (K_MAX)
    ) i_formatter_pipeline (
        .clk_i,
        .rst_ni,
        .flush_i             (state_q == CH_IDLE),
        .advance_i           (formatter_pipe_ready && !bypass_active),
        .window_i            (window_data),
        .lane_kh_i           (stg1_lane_kh_q),
        .lane_kw_i           (stg1_lane_kw_q),
        .lane_ic_i           (stg1_lane_ic_q),
        .tap_kh_i            (stg1_tap_kh_q),
        .tap_kw_i            (stg1_tap_kw_q),
        .kernel_h_i          (cfg_kernel_h_i),
        .kernel_w_i          (cfg_kernel_w_i),
        .c_base_i            (cfg_c_base_i),
        .input_c_i           (cfg_input_c_i),
        .lane_base_i         (cfg_lane_base_i),
        .block_valid_bytes_i (block_valid_bytes),
        .coalesce_i          (cfg_coalesce_i),
        .kgen_i              (cfg_kgen_i),
        .c32_kgen_fast_i     (c32_kgen_fast),
        .valid_i             (stg1_valid_q),
        .row_o               (formatter_row),
        .valid_o             (formatter_valid),
        .empty_o             (formatter_empty)
    );

    function automatic logic [2:0] mod7_u16(input logic [15:0] value);
        logic [5:0] sum0;
        logic [5:0] rem0;
        logic [5:0] rem1;
        logic [5:0] rem2;
        begin
            // ROW_SLOTS is fixed at 7. Since 8 mod 7 == 1, n mod 7 is the
            // modulo-7 sum of its 3-bit chunks. This keeps the slot mapper
            // shallow and avoids a synthesized divider/subtractor chain.
            sum0 = {3'd0, value[2:0]} +
                   {3'd0, value[5:3]} +
                   {3'd0, value[8:6]} +
                   {3'd0, value[11:9]} +
                   {3'd0, value[14:12]} +
                   {5'd0, value[15]};
            rem0 = (sum0 >= 6'd28) ? (sum0 - 6'd28) : sum0;
            rem1 = (rem0 >= 6'd14) ? (rem0 - 6'd14) : rem0;
            rem2 = (rem1 >= 6'd7)  ? (rem1 - 6'd7)  : rem1;
            mod7_u16 = rem2[2:0];
        end
    endfunction

    function automatic logic [15:0] cache_row_slot(input logic [15:0] ih);
        cache_row_slot = {13'd0, mod7_u16(ih)};
    endfunction

    function automatic logic [31:0] row_stride_offset(input logic [15:0] kh);
        logic [31:0] stride_x2;
        logic [31:0] stride_x4;
        begin
            stride_x2 = cfg_row_stride_bytes_i << 1;
            stride_x4 = cfg_row_stride_bytes_i << 2;
            unique case (kh[2:0])
                3'd0: row_stride_offset = 32'd0;
                3'd1: row_stride_offset = cfg_row_stride_bytes_i;
                3'd2: row_stride_offset = stride_x2;
                3'd3: row_stride_offset = stride_x2 + cfg_row_stride_bytes_i;
                3'd4: row_stride_offset = stride_x4;
                default: row_stride_offset = 32'd0;
            endcase
        end
    endfunction

    function automatic logic [31:0] row_tap_addr(
        input logic [31:0] row_base,
        input logic [15:0] kh
    );
        row_tap_addr = row_base + row_stride_offset(kh) + channel_addr_offset;
    endfunction

    task automatic derive_fill_coordinates;
        begin
            fill_ih = row_cache_full ?
                      $signed({16'd0, fill_kh_q}) :
                      (output_base_ih_q + $signed({16'd0, fill_kh_q}));
            fill_row_in_bounds = row_cache_full ?
                                 (fill_kh_q < cfg_input_h_i) :
                                 ((fill_ih >= 32'sd0) &&
                                  (fill_ih < $signed({16'd0, cfg_input_h_i})));
            fill_row_slot = fill_row_in_bounds ? cache_row_slot(fill_ih[15:0]) : 16'd0;
        end
    endtask

    task automatic derive_fill_cache_status;
        begin
            fill_row_cached = row_ring_mode && fill_row_in_bounds &&
                              row_store_main_cached;
            fill_row_pending = row_ring_mode && fill_row_in_bounds &&
                               row_store_main_pending;
            fill_row_ready = fill_row_cached;
        end
    endtask

    task automatic derive_background_start;
        begin
            bg_can_start = row_ring_mode &&
                           fetch_background_idle &&
                           !bg_started_for_row_q &&
                           ((state_q == CH_WINDOW_REQ) ||
                            (state_q == CH_WINDOW_WAIT) ||
                            (state_q == CH_STREAM_PRIME) ||
                            (state_q == CH_STREAM_EMIT)) &&
                           (block_valid_bytes != 6'd0) &&
                           ((spatial_rows_q + 32'(cfg_output_w_i)) < dim_m_i);
            fetch_background_start = bg_can_start;
        end
    endtask

    task automatic derive_bypass_status;
        begin
            bypass_active = (cfg_kernel_h_i == 16'd1) &&
                            (cfg_kernel_w_i == 16'd1) &&
                            (cfg_pad_h_i == 16'd0) &&
                            (cfg_pad_w_i == 16'd0);
            config_rejected = (dim_m_i == 32'd0) ||
                              (cfg_output_w_i == 16'd0) ||
                              (cfg_kernel_h_i == 16'd0) ||
                              (cfg_kernel_w_i == 16'd0) ||
                              (cfg_kernel_h_i > K_MAX[15:0]) ||
                              (cfg_kernel_w_i > K_MAX[15:0]) ||
                              (cfg_pad_h_i > K_MAX[15:0]) ||
                              (cfg_input_w_i > MAX_INPUT_W[15:0]) ||
                              (cfg_stride_h_i == 16'd0) ||
                              (cfg_stride_w_i == 16'd0) ||
                              (cfg_stride_h_i > STRIDE_MAX[15:0]) ||
                              (cfg_stride_w_i > STRIDE_MAX[15:0]) ||
                              (cfg_coalesce_i && !cfg_kgen_i &&
                               (coalesce_k_bytes > 32'(ARRAY_DIM)));
            bypass_start = (state_q == CH_IDLE) && start_i &&
                           bypass_active && !config_rejected;
        end
    endtask

    task automatic derive_stream_status;
        begin
            last_kernel_vector = ((kh_q + 16'd1) == cfg_kernel_h_i) &&
                                 ((kw_q + 16'd1) == cfg_kernel_w_i);
            stg1_fire = stg1_valid_q && stg2_ready;
            emit_fire = bypass_active ?
                        (bypass_row_valid && row_ready_i) : stg1_fire;

            vector_last_for_spatial = cfg_coalesce_i || last_kernel_vector;
            last_spatial = (spatial_rows_q + 32'd1) == dim_m_i;
            more_k_tiles = cfg_coalesce_i && cfg_kgen_i && (cfg_k_tiles_i > 32'd1) &&
                           ((k_tile_idx_q + 32'd1) < cfg_k_tiles_i);
            has_next_same_row = !last_spatial && ((ow_q + 16'd1) != cfg_output_w_i);
            has_next2_same_row = ((spatial_rows_q + 32'd2) < dim_m_i) &&
                                 ((ow_q + 16'd2) < cfg_output_w_i);

            if ((kw_q + 16'd1) != cfg_kernel_w_i) begin
                next_kh = kh_q;
                next_kw = kw_q + 16'd1;
            end else begin
                next_kh = kh_q + 16'd1;
                next_kw = '0;
            end

            slide_req_active = (state_q == CH_STREAM_PRIME) ||
                               ((state_q == CH_STREAM_EMIT) && emit_fire &&
                                vector_last_for_spatial && has_next2_same_row);
            slide_from_iw = output_base_iw_q;
            if ((state_q == CH_STREAM_EMIT) && emit_fire && vector_last_for_spatial) begin
                slide_from_iw = output_base_iw_q + $signed({16'd0, cfg_stride_w_i});
            end

            window_req_kw = window_kw_q;
            if ((state_q == CH_WINDOW_WAIT) &&
                ((window_kw_q + 16'd1) != cfg_kernel_w_i)) begin
                window_req_kw = window_kw_q + 16'd1;
            end
        end
    endtask

    task automatic derive_row_store_events;
        begin
            row_store_job_start = (state_q == CH_IDLE) && start_i;
            fetch_main_start = (state_q == CH_ENSURE) &&
                               (fill_kh_q != fill_done_rows) &&
                               fill_row_in_bounds &&
                               !fill_row_ready &&
                               !fill_row_pending &&
                               (block_valid_bytes != 6'd0);

            tile_advance_event = (state_q == CH_STREAM_DONE) &&
                                 formatter_stream_drained &&
                                 more_k_tiles && next_tile_i;
            prefetch_start_event = (state_q == CH_STREAM_DONE) &&
                                   formatter_stream_drained &&
                                   more_k_tiles && !next_tile_i &&
                                   prefetch_i && !prefetch_active_q &&
                                   !prefetch_ready_q && !row_cache_reuse;
            row_store_invalidate = (tile_advance_event || prefetch_start_event) &&
                                   !(row_ring_mode &&
                                     (effective_c_base == cached_c_base));
            row_store_cached_c_base_set =
                (tile_advance_event &&
                 !(row_cache_reuse ||
                   (prefetch_ready_q &&
                    (prefetched_c_base_q == effective_c_base)))) ||
                prefetch_start_event;
            fetch_background_reset = row_store_job_start ||
                                     tile_advance_event ||
                                     prefetch_start_event;
        end
    endtask

    task automatic derive_window_control;
        begin
            window_clear = ((state_q == CH_IDLE) && start_i) ||
                           (state_q == CH_ENSURE) ||
                           tile_advance_event ||
                           prefetch_start_event ||
                           ((state_q == CH_STREAM_EMIT) && emit_fire &&
                            vector_last_for_spatial && !last_spatial &&
                            !bypass_active &&
                            ((ow_q + 16'd1) == cfg_output_w_i));
            window_load_request = (state_q == CH_WINDOW_REQ) ||
                                  ((state_q == CH_WINDOW_WAIT) &&
                                   ((window_kw_q + 16'd1) != cfg_kernel_w_i));
            window_load_capture = state_q == CH_WINDOW_WAIT;
            window_slide_commit = (state_q == CH_STREAM_EMIT) && emit_fire &&
                                  vector_last_for_spatial && !bypass_active &&
                                  has_next_same_row;
        end
    endtask

    task automatic drive_obi_request_mux;
        begin
            fetch_obi_gnt = !bypass_obi_req && obi_gnt_i;
            obi_req_o = bypass_obi_req || fetch_obi_req;
            obi_addr_o = bypass_obi_req ? bypass_obi_addr : fetch_obi_addr;
        end
    endtask

    always_comb derive_fill_coordinates();
    always_comb derive_fill_cache_status();
    always_comb derive_background_start();
    always_comb derive_bypass_status();
    always_comb derive_stream_status();
    always_comb derive_row_store_events();
    always_comb derive_window_control();
    always_comb drive_obi_request_mux();

    task automatic reset_sequential_state;
        begin
            state_q <= CH_IDLE;
            output_row_base_addr_q <= '0;
            output_spatial_addr_q <= '0;
            output_base_ih_q <= '0;
            output_base_iw_q <= '0;
            ow_q <= '0;
            kh_q <= '0;
            kw_q <= '0;
            spatial_rows_q <= '0;
            emitted_vectors_q <= '0;
            k_tile_idx_q <= '0;
            fill_kh_q <= '0;
            bg_started_for_row_q <= 1'b0;
            window_kw_q <= '0;
            row_data_q <= '0;
            row_valid_out_q <= 1'b0;
            done_q <= 1'b0;
            prefetch_active_q <= 1'b0;
            prefetch_ready_q <= 1'b0;
            prefetched_c_base_q <= '0;
            stg1_valid_q <= 1'b0;
            stg1_lane_kh_q <= '0;
            stg1_lane_kw_q <= '0;
            stg1_lane_ic_q <= '0;
            stg1_tap_kh_q <= '0;
            stg1_tap_kw_q <= '0;
        end
    endtask

    task automatic tick_lane_pipeline;
        begin
            stg1_lane_kh_q <= lane_kh;
            stg1_lane_kw_q <= lane_kw;
            stg1_lane_ic_q <= lane_ic;
        end
    endtask

    task automatic reset_spatial_walk;
        begin
            output_row_base_addr_q <= cfg_origin_base_i;
            output_spatial_addr_q <= cfg_origin_base_i;
            output_base_ih_q <= -$signed({16'd0, cfg_pad_h_i});
            output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
            ow_q <= '0;
            kh_q <= '0;
            kw_q <= '0;
            spatial_rows_q <= '0;
            fill_kh_q <= '0;
            window_kw_q <= '0;
        end
    endtask

    task automatic tick_idle_state;
        begin
            stg1_valid_q <= 1'b0;
            row_valid_out_q <= 1'b0;
            if (start_i) begin
                reset_spatial_walk();
                emitted_vectors_q <= '0;
                k_tile_idx_q <= '0;
                bg_started_for_row_q <= 1'b0;
                prefetch_active_q <= 1'b0;
                prefetch_ready_q <= 1'b0;
                prefetched_c_base_q <= '0;
                if (config_rejected) begin
                    done_q <= 1'b1;
                    state_q <= CH_IDLE;
                end else if ((cfg_kernel_h_i == 16'd1) &&
                             (cfg_kernel_w_i == 16'd1) &&
                             (cfg_pad_h_i == 16'd0) &&
                             (cfg_pad_w_i == 16'd0)) begin
                    state_q <= CH_BYPASS_PREP;
                end else begin
                    state_q <= CH_ENSURE;
                end
            end
        end
    endtask

    task automatic tick_bypass_active;
        begin
            if (emit_fire) begin
                emitted_vectors_q <= emitted_vectors_q + 32'd1;
                spatial_rows_q <= spatial_rows_q + 32'd1;
                if (last_spatial) begin
                    state_q <= CH_STREAM_DONE;
                end else if ((ow_q + 16'd1) == cfg_output_w_i) begin
                    advance_to_next_output_row();
                end else begin
                    advance_to_next_output_col();
                end
            end
        end
    endtask

    task automatic tick_fill_states;
        begin
            unique case (state_q)
                CH_ENSURE: begin
                    if (fill_kh_q == fill_done_rows) begin
                        if (prefetch_active_q) begin
                            prefetch_active_q <= 1'b0;
                            prefetch_ready_q <= 1'b1;
                            prefetched_c_base_q <= effective_c_base;
                            state_q <= CH_STREAM_DONE;
                        end else begin
                            window_kw_q <= '0;
                            state_q <= CH_WINDOW_REQ;
                        end
                    end else if (!fill_row_in_bounds || fill_row_ready || (block_valid_bytes == 6'd0)) begin
                        fill_kh_q <= fill_kh_q + 16'd1;
                    end else if (fill_row_pending) begin
                        state_q <= CH_ENSURE;
                    end else begin
                        state_q <= CH_FILL_REQ0;
                    end
                end

                CH_FILL_REQ0: begin
                    if (fetch_main_request_accepted) begin
                        unique case (fetch_main_next_phase)
                            2'd1: state_q <= CH_FILL_REQ1;
                            2'd2: state_q <= CH_FILL_DRAIN;
                            default: state_q <= CH_FILL_REQ0;
                        endcase
                    end
                end

                CH_FILL_REQ1: begin
                    if (fetch_main_request_accepted) begin
                        state_q <= (fetch_main_next_phase == 2'd2) ?
                                   CH_FILL_DRAIN : CH_FILL_REQ0;
                    end
                end

                CH_FILL_DRAIN: begin
                    if (fetch_main_done) begin
                        fill_kh_q <= fill_kh_q + 16'd1;
                        state_q <= CH_ENSURE;
                    end
                end

                default: begin
                end
            endcase
        end
    endtask

    task automatic tick_window_states;
        begin
            unique case (state_q)
                CH_WINDOW_REQ: begin
                    state_q <= CH_WINDOW_WAIT;
                end

                CH_WINDOW_WAIT: begin
                    if ((window_kw_q + 16'd1) == cfg_kernel_w_i) begin
                        kh_q <= '0;
                        kw_q <= '0;
                        state_q <= CH_STREAM_PRIME;
                    end else begin
                        window_kw_q <= window_kw_q + 16'd1;
                        state_q <= CH_WINDOW_WAIT;
                    end
                end

                default: begin
                end
            endcase
        end
    endtask

    task automatic advance_to_next_output_row;
        begin
            ow_q <= '0;
            output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
            output_base_ih_q <= output_base_ih_q + $signed({16'd0, cfg_stride_h_i});
            output_row_base_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
            output_spatial_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
            bg_started_for_row_q <= 1'b0;
        end
    endtask

    task automatic advance_to_next_output_col;
        begin
            ow_q <= ow_q + 16'd1;
            output_base_iw_q <= output_base_iw_q + $signed({16'd0, cfg_stride_w_i});
            output_spatial_addr_q <= output_spatial_addr_q + cfg_ow_step_bytes_i;
        end
    endtask

    task automatic tick_stream_states;
        begin
            unique case (state_q)
                CH_STREAM_PRIME: begin
                    stg1_tap_kh_q <= kh_q;
                    stg1_tap_kw_q <= kw_q;
                    stg1_valid_q <= 1'b1;
                    state_q <= CH_STREAM_EMIT;
                end

                CH_STREAM_EMIT: begin
                    if (emit_fire) begin
                        emitted_vectors_q <= emitted_vectors_q + 32'd1;

                        if (!vector_last_for_spatial) begin
                            kh_q <= next_kh;
                            kw_q <= next_kw;
                            stg1_tap_kh_q <= next_kh;
                            stg1_tap_kw_q <= next_kw;
                        end else begin
                            spatial_rows_q <= spatial_rows_q + 32'd1;
                            kh_q <= '0;
                            kw_q <= '0;
                            if (last_spatial) begin
                                stg1_valid_q <= 1'b0;
                                state_q <= CH_STREAM_DONE;
                            end else if ((ow_q + 16'd1) == cfg_output_w_i) begin
                                stg1_valid_q <= 1'b0;
                                advance_to_next_output_row();
                                fill_kh_q <= '0;
                                window_kw_q <= '0;
                                state_q <= row_cache_full ? CH_WINDOW_REQ : CH_ENSURE;
                            end else if (has_next_same_row) begin
                                stg1_tap_kh_q <= '0;
                                stg1_tap_kw_q <= '0;
                                advance_to_next_output_col();
                            end
                        end
                    end
                end

                CH_STREAM_DONE: begin
                    stg1_valid_q <= 1'b0;
                    if (formatter_stream_drained) begin
                        if (more_k_tiles) begin
                            if (next_tile_i) begin
                                k_tile_idx_q <= k_tile_idx_q + 32'd1;
                                reset_spatial_walk();
                                bg_started_for_row_q <= 1'b0;
                                prefetch_active_q <= 1'b0;
                                prefetch_ready_q <= 1'b0;
                                if (row_cache_reuse ||
                                    (prefetch_ready_q && (prefetched_c_base_q == effective_c_base))) begin
                                    state_q <= CH_WINDOW_REQ;
                                end else begin
                                    state_q <= CH_ENSURE;
                                end
                            end else if (prefetch_i && !prefetch_active_q && !prefetch_ready_q &&
                                         !row_cache_reuse) begin
                                reset_spatial_walk();
                                bg_started_for_row_q <= 1'b0;
                                prefetch_active_q <= 1'b1;
                                state_q <= CH_ENSURE;
                            end
                        end else begin
                            row_valid_out_q <= 1'b0;
                            done_q <= 1'b1;
                            state_q <= CH_IDLE;
                        end
                    end
                end

                default: begin
                end
            endcase
        end
    endtask

    task automatic tick_main_fsm;
        begin
            unique case (state_q)
                CH_IDLE: tick_idle_state();
                CH_BYPASS_PREP: tick_bypass_active();
                CH_ENSURE,
                CH_FILL_REQ0,
                CH_FILL_REQ1,
                CH_FILL_DRAIN: tick_fill_states();
                CH_WINDOW_REQ,
                CH_WINDOW_WAIT: tick_window_states();
                CH_STREAM_PRIME,
                CH_STREAM_EMIT,
                CH_STREAM_DONE: tick_stream_states();
                default: begin
                    state_q <= CH_IDLE;
                end
            endcase
        end
    endtask

    task automatic tick_output_stage;
        begin
            if (formatter_pipe_ready && !bypass_active) begin
                row_data_q <= formatter_row;
                row_valid_out_q <= formatter_valid;
            end
        end
    endtask

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            reset_sequential_state();
        end else begin
            done_q <= 1'b0;
            tick_lane_pipeline();
            tick_main_fsm();
            tick_output_stage();
            if (fetch_background_start) begin
                bg_started_for_row_q <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
