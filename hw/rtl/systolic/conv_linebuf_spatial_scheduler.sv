`default_nettype none

module conv_linebuf_spatial_scheduler #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned K_MAX = 5,
    parameter int unsigned STRIDE_MAX = 2,
    parameter int unsigned ROW_SLOTS = K_MAX + STRIDE_MAX,
    parameter int unsigned MAX_INPUT_W = 640
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic start_i,
    input  logic next_tile_i,
    input  logic prefetch_i,
    input  logic [31:0] dim_m_i,
    input  logic [31:0] cfg_k_tiles_i,
    input  logic [31:0] cfg_origin_base_i,
    input  logic [31:0] cfg_row_stride_bytes_i,
    input  logic [31:0] cfg_ow_step_bytes_i,
    input  logic [31:0] cfg_oh_step_bytes_i,
    input  logic [15:0] cfg_input_h_i,
    input  logic [15:0] cfg_input_w_i,
    input  logic [15:0] cfg_output_w_i,
    input  logic [15:0] cfg_kernel_h_i,
    input  logic [15:0] cfg_kernel_w_i,
    input  logic [15:0] cfg_stride_h_i,
    input  logic [15:0] cfg_stride_w_i,
    input  logic [15:0] cfg_pad_h_i,
    input  logic [15:0] cfg_pad_w_i,
    input  logic cfg_coalesce_i,
    input  logic cfg_kgen_i,

    input  logic [5:0] block_valid_bytes_i,
    input  logic [31:0] coalesce_k_bytes_i,
    input  logic [ARRAY_DIM-1:0][7:0] lane_kh_i,
    input  logic [ARRAY_DIM-1:0][7:0] lane_kw_i,
    input  logic [ARRAY_DIM-1:0][15:0] lane_ic_i,
    input  logic [15:0] effective_c_base_i,
    input  logic [31:0] channel_addr_offset_i,
    input  logic row_cache_reuse_i,
    input  logic row_ring_mode_i,
    input  logic [15:0] fill_done_rows_i,
    input  logic [31:0] pad_row_offset_i,

    input  logic row_cache_full_i,
    input  logic [15:0] cached_c_base_i,
    input  logic row_store_main_cached_i,
    input  logic row_store_main_pending_i,

    input  logic fetch_main_request_accepted_i,
    input  logic [1:0] fetch_main_next_phase_i,
    input  logic fetch_main_done_i,
    input  logic fetch_background_idle_i,

    input  logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] formatter_row_i,
    input  logic formatter_valid_i,
    input  logic formatter_empty_i,

    input  logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] bypass_row_i,
    input  logic bypass_row_valid_i,
    input  logic [4:0] bypass_debug_state_i,

    input  logic row_ready_i,

    output logic fetch_main_start_o,
    output logic [ADDR_WIDTH-1:0] fetch_main_base_addr_o,
    output logic [15:0] fetch_main_row_slot_o,
    output logic [15:0] fetch_main_row_ih_o,
    output logic fetch_main_row_ready_o,
    output logic fetch_background_start_o,
    output logic fetch_background_reset_o,
    output logic signed [31:0] fetch_background_base_ih_o,
    output logic [ADDR_WIDTH-1:0] fetch_background_row_base_addr_o,

    output logic row_store_job_start_o,
    output logic row_store_invalidate_o,
    output logic row_store_cached_c_base_set_o,
    output logic [$clog2(ROW_SLOTS)-1:0] row_store_query_slot_o,
    output logic [15:0] row_store_query_ih_o,

    output logic bypass_start_o,
    output logic bypass_last_o,
    output logic [ADDR_WIDTH-1:0] bypass_spatial_addr_o,
    output logic signed [31:0] bypass_base_ih_o,
    output logic signed [31:0] bypass_base_iw_o,

    output logic window_clear_o,
    output logic window_load_request_o,
    output logic window_load_capture_o,
    output logic [15:0] window_load_request_kw_o,
    output logic [15:0] window_load_capture_kw_o,
    output logic window_slide_request_o,
    output logic signed [31:0] window_slide_from_iw_o,
    output logic window_slide_commit_o,

    output logic formatter_flush_o,
    output logic formatter_advance_o,
    output logic [ARRAY_DIM-1:0][7:0] formatter_lane_kh_o,
    output logic [ARRAY_DIM-1:0][7:0] formatter_lane_kw_o,
    output logic [ARRAY_DIM-1:0][15:0] formatter_lane_ic_o,
    output logic [15:0] formatter_tap_kh_o,
    output logic [15:0] formatter_tap_kw_o,
    output logic formatter_valid_o,

    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] row_data_o,
    output logic row_valid_o,
    output logic busy_o,
    output logic done_o,
    output logic prefetch_busy_o,
    output logic [31:0] emitted_vectors_o,
    output logic [4:0] debug_state_o
);

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

    state_e state_q;
    logic [31:0] output_row_base_addr_q;
    logic [31:0] output_spatial_addr_q;
    logic signed [31:0] output_base_ih_q;
    logic signed [31:0] output_base_iw_q;
    logic [15:0] ow_q;
    logic [15:0] kh_q;
    logic [15:0] kw_q;
    logic [31:0] spatial_rows_q;
    logic [31:0] emitted_vectors_q;
    logic [31:0] k_tile_idx_q;
    logic [15:0] fill_kh_q;
    logic [15:0] window_kw_q;
    logic bg_started_for_row_q;
    input_row_t row_data_q;
    logic row_valid_q;
    logic done_q;
    logic prefetch_active_q;
    logic prefetch_ready_q;
    logic [15:0] prefetched_c_base_q;
    logic [ARRAY_DIM-1:0][7:0] stg1_lane_kh_q;
    logic [ARRAY_DIM-1:0][7:0] stg1_lane_kw_q;
    logic [ARRAY_DIM-1:0][15:0] stg1_lane_ic_q;
    logic [15:0] stg1_tap_kh_q;
    logic [15:0] stg1_tap_kw_q;
    logic stg1_valid_q;

    logic signed [31:0] fill_ih;
    logic fill_row_in_bounds;
    logic [15:0] fill_row_slot;
    logic fill_row_cached;
    logic fill_row_pending;
    logic fill_row_ready;
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
    logic bg_can_start;
    logic formatter_pipe_ready;
    logic formatter_stream_drained;
    logic tile_advance_event;
    logic prefetch_start_event;
    logic [15:0] next_kh;
    logic [15:0] next_kw;

    assign row_data_o = bypass_active ? bypass_row_i : row_data_q;
    assign row_valid_o = bypass_active ? bypass_row_valid_i : row_valid_q;
    assign busy_o = state_q != CH_IDLE;
    assign done_o = done_q;
    assign prefetch_busy_o = prefetch_active_q;
    assign emitted_vectors_o = emitted_vectors_q;
    assign debug_state_o = (state_q == CH_BYPASS_PREP) ? bypass_debug_state_i : state_q;

    assign bypass_last_o = last_spatial;
    assign bypass_spatial_addr_o = output_spatial_addr_q;
    assign bypass_base_ih_o = output_base_ih_q;
    assign bypass_base_iw_o = output_base_iw_q;
    assign fetch_main_row_slot_o = row_ring_mode_i ? fill_row_slot : fill_kh_q;
    assign fetch_main_row_ih_o = fill_ih[15:0];
    assign fetch_main_row_ready_o = fill_row_ready;
    assign fetch_background_base_ih_o =
        output_base_ih_q + $signed({16'd0, cfg_stride_h_i});
    assign fetch_background_row_base_addr_o =
        output_row_base_addr_q + cfg_oh_step_bytes_i;

    assign row_store_query_slot_o = fill_row_slot[$clog2(ROW_SLOTS)-1:0];
    assign row_store_query_ih_o = fill_ih[15:0];

    assign formatter_flush_o = state_q == CH_IDLE;
    assign formatter_advance_o = formatter_pipe_ready && !bypass_active;
    assign formatter_lane_kh_o = stg1_lane_kh_q;
    assign formatter_lane_kw_o = stg1_lane_kw_q;
    assign formatter_lane_ic_o = stg1_lane_ic_q;
    assign formatter_tap_kh_o = stg1_tap_kh_q;
    assign formatter_tap_kw_o = stg1_tap_kw_q;
    assign formatter_valid_o = stg1_valid_q;

    function automatic logic [2:0] mod7_u16(input logic [15:0] value);
        logic [5:0] sum0;
        logic [5:0] rem0;
        logic [5:0] rem1;
        logic [5:0] rem2;
        begin
            sum0 = {3'd0, value[2:0]} +
                   {3'd0, value[5:3]} +
                   {3'd0, value[8:6]} +
                   {3'd0, value[11:9]} +
                   {3'd0, value[14:12]} +
                   {5'd0, value[15]};
            rem0 = (sum0 >= 6'd28) ? (sum0 - 6'd28) : sum0;
            rem1 = (rem0 >= 6'd14) ? (rem0 - 6'd14) : rem0;
            rem2 = (rem1 >= 6'd7) ? (rem1 - 6'd7) : rem1;
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
        row_tap_addr = row_base + row_stride_offset(kh) + channel_addr_offset_i;
    endfunction

    always_comb begin
        fill_ih = row_cache_full_i ?
                  $signed({16'd0, fill_kh_q}) :
                  (output_base_ih_q + $signed({16'd0, fill_kh_q}));
        fill_row_in_bounds = row_cache_full_i ?
                             (fill_kh_q < cfg_input_h_i) :
                             ((fill_ih >= 32'sd0) &&
                              (fill_ih < $signed({16'd0, cfg_input_h_i})));
        fill_row_slot = fill_row_in_bounds ? cache_row_slot(fill_ih[15:0]) : 16'd0;

        fill_row_cached = row_ring_mode_i && fill_row_in_bounds &&
                          row_store_main_cached_i;
        fill_row_pending = row_ring_mode_i && fill_row_in_bounds &&
                           row_store_main_pending_i;
        fill_row_ready = fill_row_cached;
    end

    always_comb begin
        bg_can_start = row_ring_mode_i &&
                       fetch_background_idle_i &&
                       !bg_started_for_row_q &&
                       ((state_q == CH_WINDOW_REQ) ||
                        (state_q == CH_WINDOW_WAIT) ||
                        (state_q == CH_STREAM_PRIME) ||
                        (state_q == CH_STREAM_EMIT)) &&
                       (block_valid_bytes_i != 6'd0) &&
                       ((spatial_rows_q + 32'(cfg_output_w_i)) < dim_m_i);
        fetch_background_start_o = bg_can_start;
    end

    always_comb begin
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
                           (coalesce_k_bytes_i > 32'(ARRAY_DIM)));
        bypass_start_o = (state_q == CH_IDLE) && start_i &&
                         bypass_active && !config_rejected;
    end

    always_comb begin
        last_kernel_vector = ((kh_q + 16'd1) == cfg_kernel_h_i) &&
                             ((kw_q + 16'd1) == cfg_kernel_w_i);
        formatter_pipe_ready = row_ready_i || !row_valid_q;
        stg1_fire = stg1_valid_q && formatter_pipe_ready;
        emit_fire = bypass_active ?
                    (bypass_row_valid_i && row_ready_i) : stg1_fire;

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

        window_slide_request_o = (state_q == CH_STREAM_PRIME) ||
                                 ((state_q == CH_STREAM_EMIT) && emit_fire &&
                                  vector_last_for_spatial && has_next2_same_row);
        window_slide_from_iw_o = output_base_iw_q;
        if ((state_q == CH_STREAM_EMIT) && emit_fire && vector_last_for_spatial) begin
            window_slide_from_iw_o =
                output_base_iw_q + $signed({16'd0, cfg_stride_w_i});
        end

        window_load_request_kw_o = window_kw_q;
        if ((state_q == CH_WINDOW_WAIT) &&
            ((window_kw_q + 16'd1) != cfg_kernel_w_i)) begin
            window_load_request_kw_o = window_kw_q + 16'd1;
        end
        window_load_capture_kw_o = window_kw_q;
    end

    always_comb begin
        formatter_stream_drained = !stg1_valid_q &&
                                   formatter_empty_i &&
                                   (!row_valid_q || row_ready_i);

        row_store_job_start_o = (state_q == CH_IDLE) && start_i;
        fetch_main_start_o = (state_q == CH_ENSURE) &&
                             (fill_kh_q != fill_done_rows_i) &&
                             fill_row_in_bounds &&
                             !fill_row_ready &&
                             !fill_row_pending &&
                             (block_valid_bytes_i != 6'd0);

        tile_advance_event = (state_q == CH_STREAM_DONE) &&
                             formatter_stream_drained &&
                             more_k_tiles && next_tile_i;
        prefetch_start_event = (state_q == CH_STREAM_DONE) &&
                               formatter_stream_drained &&
                               more_k_tiles && !next_tile_i &&
                               prefetch_i && !prefetch_active_q &&
                               !prefetch_ready_q && !row_cache_reuse_i;
        row_store_invalidate_o = (tile_advance_event || prefetch_start_event) &&
                                 !(row_ring_mode_i &&
                                   (effective_c_base_i == cached_c_base_i));
        row_store_cached_c_base_set_o =
            (tile_advance_event &&
             !(row_cache_reuse_i ||
               (prefetch_ready_q &&
                (prefetched_c_base_q == effective_c_base_i)))) ||
            prefetch_start_event;
        fetch_background_reset_o = row_store_job_start_o ||
                                   tile_advance_event ||
                                   prefetch_start_event;
    end

    always_comb begin
        window_clear_o = ((state_q == CH_IDLE) && start_i) ||
                         (state_q == CH_ENSURE) ||
                         tile_advance_event ||
                         prefetch_start_event ||
                         ((state_q == CH_STREAM_EMIT) && emit_fire &&
                          vector_last_for_spatial && !last_spatial &&
                          !bypass_active &&
                          ((ow_q + 16'd1) == cfg_output_w_i));
        window_load_request_o = (state_q == CH_WINDOW_REQ) ||
                                ((state_q == CH_WINDOW_WAIT) &&
                                 ((window_kw_q + 16'd1) != cfg_kernel_w_i));
        window_load_capture_o = state_q == CH_WINDOW_WAIT;
        window_slide_commit_o = (state_q == CH_STREAM_EMIT) && emit_fire &&
                                vector_last_for_spatial && !bypass_active &&
                                has_next_same_row;
    end

    always_comb begin
        fetch_main_base_addr_o = row_cache_full_i ?
            row_tap_addr(cfg_origin_base_i + pad_row_offset_i, fill_kh_q) :
            row_tap_addr(output_row_base_addr_q, fill_kh_q);
    end

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
            row_valid_q <= 1'b0;
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

    task automatic tick_idle_state;
        begin
            stg1_valid_q <= 1'b0;
            row_valid_q <= 1'b0;
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
                end else if (bypass_active) begin
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
                    if (fill_kh_q == fill_done_rows_i) begin
                        if (prefetch_active_q) begin
                            prefetch_active_q <= 1'b0;
                            prefetch_ready_q <= 1'b1;
                            prefetched_c_base_q <= effective_c_base_i;
                            state_q <= CH_STREAM_DONE;
                        end else begin
                            window_kw_q <= '0;
                            state_q <= CH_WINDOW_REQ;
                        end
                    end else if (!fill_row_in_bounds || fill_row_ready ||
                                 (block_valid_bytes_i == 6'd0)) begin
                        fill_kh_q <= fill_kh_q + 16'd1;
                    end else if (fill_row_pending) begin
                        state_q <= CH_ENSURE;
                    end else begin
                        state_q <= CH_FILL_REQ0;
                    end
                end

                CH_FILL_REQ0: begin
                    if (fetch_main_request_accepted_i) begin
                        unique case (fetch_main_next_phase_i)
                            2'd1: state_q <= CH_FILL_REQ1;
                            2'd2: state_q <= CH_FILL_DRAIN;
                            default: state_q <= CH_FILL_REQ0;
                        endcase
                    end
                end

                CH_FILL_REQ1: begin
                    if (fetch_main_request_accepted_i) begin
                        state_q <= (fetch_main_next_phase_i == 2'd2) ?
                                   CH_FILL_DRAIN : CH_FILL_REQ0;
                    end
                end

                CH_FILL_DRAIN: begin
                    if (fetch_main_done_i) begin
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
                CH_WINDOW_REQ: state_q <= CH_WINDOW_WAIT;

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
                                state_q <= row_cache_full_i ? CH_WINDOW_REQ : CH_ENSURE;
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
                                if (row_cache_reuse_i ||
                                    (prefetch_ready_q &&
                                     (prefetched_c_base_q == effective_c_base_i))) begin
                                    state_q <= CH_WINDOW_REQ;
                                end else begin
                                    state_q <= CH_ENSURE;
                                end
                            end else if (prefetch_i && !prefetch_active_q &&
                                         !prefetch_ready_q && !row_cache_reuse_i) begin
                                reset_spatial_walk();
                                bg_started_for_row_q <= 1'b0;
                                prefetch_active_q <= 1'b1;
                                state_q <= CH_ENSURE;
                            end
                        end else begin
                            row_valid_q <= 1'b0;
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

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            reset_sequential_state();
        end else begin
            done_q <= 1'b0;
            stg1_lane_kh_q <= lane_kh_i;
            stg1_lane_kw_q <= lane_kw_i;
            stg1_lane_ic_q <= lane_ic_i;

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
                default: state_q <= CH_IDLE;
            endcase

            if (formatter_pipe_ready && !bypass_active) begin
                row_data_q <= formatter_row_i;
                row_valid_q <= formatter_valid_i;
            end
            if (fetch_background_start_o) begin
                bg_started_for_row_q <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
