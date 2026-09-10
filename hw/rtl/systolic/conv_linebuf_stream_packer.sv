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

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_t;

    logic [BANKS-1:0] bank_w_req;
    logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_w_addr;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_w_data;
    logic [BANKS-1:0] bank_r_req;
    logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_r_addr;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_rdata;

    window_t window_data;

    logic [31:0] fetch_engine_beats;
    logic row_cache_full;
    logic [15:0] cached_c_base;

    logic [5:0] block_valid_bytes;
    logic [31:0] coalesce_k_bytes;
    logic [ARRAY_DIM-1:0][7:0] lane_kh;
    logic [ARRAY_DIM-1:0][7:0] lane_kw;
    logic [ARRAY_DIM-1:0][15:0] lane_ic;

    logic last_spatial;
    logic row_cache_full_mode;
    logic row_cache_reuse;
    logic row_ring_mode;
    logic c32_blocked_mode;
    logic c32_kgen_fast;
    logic [15:0] effective_c_base;
    logic [31:0] channel_addr_offset;
    logic fill_row_ready;
    logic [15:0] fill_done_rows;
    logic fetch_main_start;
    logic [ADDR_WIDTH-1:0] fetch_main_base_addr;
    logic [15:0] fetch_main_row_slot;
    logic [15:0] fetch_main_row_ih;
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
    logic [$clog2(ROW_SLOTS)-1:0] row_store_query_slot;
    logic [15:0] row_store_query_ih;
    logic signed [31:0] output_base_ih;
    logic signed [31:0] output_base_iw;
    logic [ADDR_WIDTH-1:0] output_spatial_addr;
    logic signed [31:0] fetch_background_base_ih;
    logic [ADDR_WIDTH-1:0] fetch_background_row_base_addr;
    logic window_clear;
    logic window_load_request;
    logic window_load_capture;
    logic [15:0] window_req_kw;
    logic [15:0] window_capture_kw;
    logic signed [31:0] slide_from_iw;
    logic slide_req_active;
    logic window_slide_commit;
    logic [DATA_WIDTH-1:0] pad_vector;
    logic [31:0] pad_row_offset;
    logic formatter_flush;
    logic formatter_advance;
    logic [ARRAY_DIM-1:0][7:0] formatter_lane_kh;
    logic [ARRAY_DIM-1:0][7:0] formatter_lane_kw;
    logic [ARRAY_DIM-1:0][15:0] formatter_lane_ic;
    logic [15:0] formatter_tap_kh;
    logic [15:0] formatter_tap_kw;
    logic formatter_input_valid;
    input_row_t formatter_row;
    logic formatter_valid;
    logic formatter_empty;

    assign fetch_beats_o = fetch_engine_beats + bypass_fetch_beats;
    assign bypass_vectors_o = bypass_emitted_vectors;

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

    conv_linebuf_spatial_scheduler #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH),
        .K_MAX            (K_MAX),
        .STRIDE_MAX       (STRIDE_MAX),
        .ROW_SLOTS        (ROW_SLOTS),
        .MAX_INPUT_W      (MAX_INPUT_W)
    ) i_spatial_scheduler (
        .clk_i,
        .rst_ni,
        .start_i,
        .next_tile_i,
        .prefetch_i,
        .dim_m_i,
        .cfg_k_tiles_i,
        .cfg_origin_base_i,
        .cfg_row_stride_bytes_i,
        .cfg_ow_step_bytes_i,
        .cfg_oh_step_bytes_i,
        .cfg_input_h_i,
        .cfg_input_w_i,
        .cfg_output_w_i,
        .cfg_kernel_h_i,
        .cfg_kernel_w_i,
        .cfg_stride_h_i,
        .cfg_stride_w_i,
        .cfg_pad_h_i,
        .cfg_pad_w_i,
        .cfg_coalesce_i,
        .cfg_kgen_i,
        .block_valid_bytes_i            (block_valid_bytes),
        .coalesce_k_bytes_i             (coalesce_k_bytes),
        .lane_kh_i                      (lane_kh),
        .lane_kw_i                      (lane_kw),
        .lane_ic_i                      (lane_ic),
        .effective_c_base_i             (effective_c_base),
        .channel_addr_offset_i          (channel_addr_offset),
        .row_cache_reuse_i              (row_cache_reuse),
        .row_ring_mode_i                (row_ring_mode),
        .fill_done_rows_i               (fill_done_rows),
        .pad_row_offset_i               (pad_row_offset),
        .row_cache_full_i               (row_cache_full),
        .cached_c_base_i                (cached_c_base),
        .row_store_main_cached_i        (row_store_main_cached),
        .row_store_main_pending_i       (row_store_main_pending),
        .fetch_main_request_accepted_i  (fetch_main_request_accepted),
        .fetch_main_next_phase_i        (fetch_main_next_phase),
        .fetch_main_done_i              (fetch_main_done),
        .fetch_background_idle_i        (fetch_background_idle),
        .formatter_row_i                (formatter_row),
        .formatter_valid_i              (formatter_valid),
        .formatter_empty_i              (formatter_empty),
        .bypass_row_i                   (bypass_row),
        .bypass_row_valid_i             (bypass_row_valid),
        .bypass_debug_state_i           (bypass_debug_state),
        .row_ready_i,
        .fetch_main_start_o             (fetch_main_start),
        .fetch_main_base_addr_o         (fetch_main_base_addr),
        .fetch_main_row_slot_o          (fetch_main_row_slot),
        .fetch_main_row_ih_o            (fetch_main_row_ih),
        .fetch_main_row_ready_o         (fill_row_ready),
        .fetch_background_start_o       (fetch_background_start),
        .fetch_background_reset_o       (fetch_background_reset),
        .fetch_background_base_ih_o     (fetch_background_base_ih),
        .fetch_background_row_base_addr_o(fetch_background_row_base_addr),
        .row_store_job_start_o          (row_store_job_start),
        .row_store_invalidate_o         (row_store_invalidate),
        .row_store_cached_c_base_set_o  (row_store_cached_c_base_set),
        .row_store_query_slot_o         (row_store_query_slot),
        .row_store_query_ih_o           (row_store_query_ih),
        .bypass_start_o                 (bypass_start),
        .bypass_last_o                  (last_spatial),
        .bypass_spatial_addr_o          (output_spatial_addr),
        .bypass_base_ih_o               (output_base_ih),
        .bypass_base_iw_o               (output_base_iw),
        .window_clear_o                 (window_clear),
        .window_load_request_o          (window_load_request),
        .window_load_capture_o          (window_load_capture),
        .window_load_request_kw_o       (window_req_kw),
        .window_load_capture_kw_o       (window_capture_kw),
        .window_slide_request_o         (slide_req_active),
        .window_slide_from_iw_o         (slide_from_iw),
        .window_slide_commit_o          (window_slide_commit),
        .formatter_flush_o              (formatter_flush),
        .formatter_advance_o            (formatter_advance),
        .formatter_lane_kh_o            (formatter_lane_kh),
        .formatter_lane_kw_o            (formatter_lane_kw),
        .formatter_lane_ic_o            (formatter_lane_ic),
        .formatter_tap_kh_o             (formatter_tap_kh),
        .formatter_tap_kw_o             (formatter_tap_kw),
        .formatter_valid_o              (formatter_input_valid),
        .row_data_o,
        .row_valid_o,
        .busy_o,
        .done_o,
        .prefetch_busy_o,
        .emitted_vectors_o,
        .debug_state_o
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
        .main_base_addr_i               (fetch_main_base_addr),
        .main_row_slot_i                (fetch_main_row_slot),
        .main_row_ih_i                  (fetch_main_row_ih),
        .main_valid_bytes_i             (block_valid_bytes),
        .main_row_ready_i               (fill_row_ready),
        .main_request_accepted_o        (fetch_main_request_accepted),
        .main_next_phase_o              (fetch_main_next_phase),
        .main_done_o                    (fetch_main_done),
        .main_alloc_valid_o             (row_store_main_alloc),
        .main_alloc_slot_o              (fetch_main_alloc_slot),
        .main_alloc_ih_o                (fetch_main_alloc_ih),
        .background_start_i             (fetch_background_start),
        .background_base_ih_i           (fetch_background_base_ih),
        .background_row_base_addr_i     (fetch_background_row_base_addr),
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
        .spatial_addr_i          (output_spatial_addr),
        .base_ih_i               (output_base_ih),
        .base_iw_i               (output_base_iw),
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
        .query_main_slot_i           (row_store_query_slot),
        .query_main_ih_i             (row_store_query_ih),
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
        .load_capture_kw_i   (window_capture_kw),
        .slide_request_i     (slide_req_active),
        .slide_from_iw_i     (slide_from_iw),
        .slide_commit_i      (window_slide_commit),
        .input_h_i           (cfg_input_h_i),
        .input_w_i           (cfg_input_w_i),
        .kernel_h_i          (cfg_kernel_h_i),
        .kernel_w_i          (cfg_kernel_w_i),
        .stride_w_i          (cfg_stride_w_i),
        .base_ih_i           (output_base_ih),
        .base_iw_i           (output_base_iw),
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
        .flush_i             (formatter_flush),
        .advance_i           (formatter_advance),
        .window_i            (window_data),
        .lane_kh_i           (formatter_lane_kh),
        .lane_kw_i           (formatter_lane_kw),
        .lane_ic_i           (formatter_lane_ic),
        .tap_kh_i            (formatter_tap_kh),
        .tap_kw_i            (formatter_tap_kw),
        .kernel_h_i          (cfg_kernel_h_i),
        .kernel_w_i          (cfg_kernel_w_i),
        .c_base_i            (cfg_c_base_i),
        .input_c_i           (cfg_input_c_i),
        .lane_base_i         (cfg_lane_base_i),
        .block_valid_bytes_i (block_valid_bytes),
        .coalesce_i          (cfg_coalesce_i),
        .kgen_i              (cfg_kgen_i),
        .c32_kgen_fast_i     (c32_kgen_fast),
        .valid_i             (formatter_input_valid),
        .row_o               (formatter_row),
        .valid_o             (formatter_valid),
        .empty_o             (formatter_empty)
    );

    always_comb begin
        fetch_obi_gnt = !bypass_obi_req && obi_gnt_i;
        obi_req_o = bypass_obi_req || fetch_obi_req;
        obi_addr_o = bypass_obi_req ? bypass_obi_addr : fetch_obi_addr;
    end

endmodule

`default_nettype wire
