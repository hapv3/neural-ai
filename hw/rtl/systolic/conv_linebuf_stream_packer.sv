`default_nettype none

module conv_linebuf_stream_packer #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned K_MAX = 5,
    parameter int unsigned MAX_INPUT_W = 640,
    parameter int unsigned STRIDE_MAX = 2
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

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BYTE_SEL_BITS = $clog2(BEAT_BYTES);
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

    typedef enum logic [2:0] {
        BG_IDLE,
        BG_SCAN,
        BG_REQ0,
        BG_REQ1,
        BG_DRAIN
    } bg_state_e;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_t;

    state_e state_q;
    bg_state_e bg_state_q;

    // --- Pipelined OBI fetch: beat-level tracking FIFO ---
    localparam int unsigned BEAT_FIFO_DEPTH = 4;
    typedef struct packed {
        logic [BYTE_SEL_BITS-1:0] addr_lsb;
        logic [5:0]  valid_bytes;
        logic [15:0] kh;
        logic [15:0] x;
        logic        is_beat0_of_cross; // first beat of crossing pixel
        logic        is_solo;           // non-crossing pixel (single beat)
    } beat_meta_t;

    beat_meta_t [BEAT_FIFO_DEPTH-1:0] beat_fifo_q;
    logic [$clog2(BEAT_FIFO_DEPTH)-1:0] bf_wptr_q, bf_rptr_q;
    logic [$clog2(BEAT_FIFO_DEPTH):0] bf_count_q;
    logic bf_full, bf_empty;
    assign bf_full  = bf_count_q >= ($clog2(BEAT_FIFO_DEPTH)+1)'(BEAT_FIFO_DEPTH - 1);
    assign bf_empty = bf_count_q == '0;

    // Response engine state
    logic [DATA_WIDTH-1:0] resp_beat0_q; // saved beat0 for crossing pixels
    logic resp_write_bank;  // combinational: response engine wants to write bank
    logic [$clog2(BANKS)-1:0] resp_bank_idx;
    logic [BANK_ADDR_WIDTH-1:0] resp_bank_addr;
    logic [DATA_WIDTH-1:0] resp_bank_wdata;
    beat_meta_t resp_meta;
    assign resp_meta = beat_fifo_q[bf_rptr_q];

    logic [BANKS-1:0] bank_w_req;
    logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_w_addr;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_w_data;
    logic [BANKS-1:0] bank_r_req;
    logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_r_addr;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_rdata;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_w_rdata_unused;
    logic [DATA_WIDTH/8-1:0] bank_be;

    window_t window_q;
    window_t slide_window;

    logic [31:0] output_row_base_addr_q;
    logic [31:0] output_spatial_addr_q;
    logic signed [31:0] output_base_ih_q;
    logic signed [31:0] output_base_iw_q;
    logic [15:0] ow_q;
    logic [15:0] kh_q;
    logic [15:0] kw_q;
    logic [31:0] spatial_rows_q;
    logic [31:0] emitted_vectors_q;
    logic [31:0] fetch_beats_q;
    logic [31:0] bypass_vectors_q;
    logic [31:0] k_tile_idx_q;
    logic row_cache_full_q;
    logic [15:0] cached_c_base_q;
    logic [ROW_SLOTS-1:0] row_slot_valid_q;
    logic [ROW_SLOTS-1:0] row_fetch_active_q;
    logic [ROW_SLOTS-1:0] row_fetch_done_q;
    logic [ROW_SLOTS-1:0][15:0] row_slot_ih_q;
    logic [ROW_SLOTS-1:0][ROW_PENDING_WIDTH-1:0] row_pending_q;

    logic [15:0] fill_kh_q;
    logic [15:0] fill_x_q;
    logic [31:0] fill_addr_q;
    logic [31:0] pending_beat_addr_q;
    logic [5:0] fill_valid_bytes_q;

    logic [15:0] window_kw_q;
    logic [15:0] window_req_kw;

    logic signed [31:0] bg_base_ih_q;
    logic [31:0] bg_row_base_addr_q;
    logic [15:0] bg_kh_q;
    logic [15:0] bg_x_q;
    logic [31:0] bg_addr_q;
    logic [31:0] bg_pending_beat_addr_q;
    logic [5:0] bg_valid_bytes_q;
    logic bg_started_for_row_q;

    logic [31:0] bypass_addr_q;
    logic [31:0] bypass_candidate_addr;
    logic [DATA_WIDTH-1:0] bypass_beat0_q;
    logic [5:0] bypass_valid_bytes_q;
    logic bypass_crosses_beat_q;

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
    logic fill_crosses_current;
    logic bypass_in_bounds;
    logic bypass_crosses_current;
    logic bypass_active;
    logic emit_fire;
    logic stg1_fire;
    logic output_fire;
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
    logic [31:0] channel_addr_offset;
    logic [15:0] fill_row_slot;
    logic fill_row_cached;
    logic fill_row_pending;
    logic fill_row_ready;
    logic [15:0] fill_done_rows;
    logic signed [31:0] bg_ih;
    logic bg_row_in_bounds;
    logic [15:0] bg_row_slot;
    logic bg_row_cached;
    logic bg_row_pending;
    logic bg_row_ready;
    logic bg_crosses_current;
    logic bg_can_start;
    logic bg_obi_req;
    logic bg_obi_gnt;
    logic main_obi_req;
    logic main_fill_beat_push;
    logic bg_fill_beat_push;
    logic beat_push;
    logic [$clog2(ROW_SLOTS)-1:0] beat_push_slot;
    logic beat_push_last_for_row;
    logic beat_pop;
    logic [$clog2(ROW_SLOTS)-1:0] beat_pop_slot;
    logic [ROW_SLOTS-1:0][ROW_PENDING_WIDTH-1:0] row_pending_next;
    logic [ROW_SLOTS-1:0] row_ready_next;
    logic [15:0] next_kh;
    logic [15:0] next_kw;
    logic signed [31:0] slide_from_iw;
    logic slide_req_active;
    logic [DATA_WIDTH-1:0] pad_vector;
    
    // Stage 1 Pipeline Registers (Coordinates)
    logic [ARRAY_DIM-1:0][7:0] stg1_lane_kh_q;
    logic [ARRAY_DIM-1:0][7:0] stg1_lane_kw_q;
    logic [ARRAY_DIM-1:0][15:0] stg1_lane_ic_q;
    logic [15:0] stg1_tap_kh_q;
    logic [15:0] stg1_tap_kw_q;
    logic        stg1_valid_q;

    // Stage 2 Pipeline Registers (Output)
    // row_data_q and row_valid_out_q already exist and serve as Stage 2 registers

    logic       stg2_ready;

    assign stg2_ready = row_ready_i || !row_valid_out_q;

    assign row_data_o = row_data_q;
    assign row_valid_o = row_valid_out_q;
    assign busy_o = state_q != CH_IDLE;
    assign done_o = done_q;
    assign prefetch_busy_o = prefetch_active_q;
    assign emitted_vectors_o = emitted_vectors_q;
    assign fetch_beats_o = fetch_beats_q;
    assign bypass_vectors_o = bypass_vectors_q;
    assign debug_state_o = state_q;
    assign bank_be = '1;
    assign pad_vector = cfg_pool_i ? {BEAT_BYTES{8'h80}} : DATA_WIDTH'(0);

    for (genvar bank = 0; bank < BANKS; bank++) begin : gen_line_banks
        tc_sram #(
            .NumWords    (BANK_DEPTH),
            .DataWidth   (DATA_WIDTH),
            .ByteWidth   (8),
            .NumPorts    (2),
            .Latency     (1),
            .SimInit     ("none"),
            .PrintSimCfg (1'b0)
        ) i_bank_sram (
            .clk_i   (clk_i),
            .rst_ni  (rst_ni),
            .req_i   ({bank_r_req[bank],  bank_w_req[bank]}),
            .we_i    ({1'b0,              1'b1}),
            .addr_i  ({bank_r_addr[bank], bank_w_addr[bank]}),
            .wdata_i ({DATA_WIDTH'(0),    bank_w_data[bank]}),
            .be_i    ({bank_be,           bank_be}),
            .rdata_o ({bank_rdata[bank],  bank_w_rdata_unused[bank]})
        );
    end

    function automatic logic [31:0] beat_base(input logic [31:0] addr);
        beat_base = {addr[31:BYTE_SEL_BITS], {BYTE_SEL_BITS{1'b0}}};
    endfunction

    function automatic logic [5:0] valid_c_bytes(
        input logic [15:0] input_c,
        input logic [15:0] c_base,
        input logic [5:0]  lane_base
    );
        logic [15:0] rem;
        logic [6:0] lane_room;
        begin
            lane_room = (lane_base >= 6'(ARRAY_DIM)) ? 7'd0 : (7'(ARRAY_DIM) - {1'b0, lane_base});
            rem = input_c - c_base;
            if ((c_base >= input_c) || (lane_room == 7'd0)) begin
                valid_c_bytes = 6'd0;
            end else if (rem >= {9'd0, lane_room}) begin
                valid_c_bytes = lane_room[5:0];
            end else begin
                valid_c_bytes = {1'b0, rem[4:0]};
            end
        end
    endfunction

    function automatic logic [DATA_WIDTH-1:0] merge_beats(
        input logic [DATA_WIDTH-1:0] beat0,
        input logic [DATA_WIDTH-1:0] beat1,
        input logic [BYTE_SEL_BITS-1:0] addr_lsb,
        input logic [5:0] valid_bytes
    );
        logic [DATA_WIDTH-1:0] merged;
        logic [BYTE_SEL_BITS:0] byte_sel;
        begin
            merged = '0;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                byte_sel = {1'b0, addr_lsb} + (BYTE_SEL_BITS+1)'(lane);
                if (lane < valid_bytes) begin
                    if (byte_sel < (BYTE_SEL_BITS+1)'(BEAT_BYTES)) begin
                        merged[lane * 8 +: 8] = beat0[byte_sel[BYTE_SEL_BITS-1:0] * 8 +: 8];
                    end else begin
                        merged[lane * 8 +: 8] = beat1[byte_sel[BYTE_SEL_BITS-1:0] * 8 +: 8];
                    end
                end
            end
            merge_beats = merged;
        end
    endfunction

    function automatic input_row_t unpack_row(
        input logic [DATA_WIDTH-1:0] data,
        input logic [5:0] valid_bytes,
        input logic [5:0] lane_base
    );
        input_row_t row;
        logic [6:0] dst_lane;
        begin
            row = '0;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                dst_lane = {1'b0, lane_base} + 7'(lane);
                if ((lane < valid_bytes) && (dst_lane < 7'(ARRAY_DIM))) begin
                    row[dst_lane[4:0]] = data[lane * 8 +: 8];
                end
            end
            unpack_row = row;
        end
    endfunction

    function automatic logic [$clog2(BANKS)-1:0] bank_index(
        input logic [15:0] row_slot,
        input logic [15:0] x
    );
        bank_index = ($clog2(BANKS))'((32'(row_slot) * STRIDE_MAX) + (32'(x) % STRIDE_MAX));
    endfunction

    function automatic logic [BANK_ADDR_WIDTH-1:0] bank_word_addr(input logic [15:0] x);
        bank_word_addr = BANK_ADDR_WIDTH'(32'(x) / STRIDE_MAX);
    endfunction

    function automatic logic [15:0] cache_row_slot(input logic [15:0] ih);
        cache_row_slot = 16'(32'(ih) % ROW_SLOTS);
    endfunction

    function automatic logic [31:0] row_tap_addr(
        input logic [31:0] row_base,
        input logic [15:0] kh
    );
        row_tap_addr = row_base +
                       (32'(kh) * cfg_row_stride_bytes_i) +
                       channel_addr_offset;
    endfunction

    function automatic input_row_t build_emit_row(
        input window_t win,
        input logic [15:0] tap_kh,
        input logic [15:0] tap_kw,
        input logic [ARRAY_DIM-1:0][7:0] l_kh,
        input logic [ARRAY_DIM-1:0][7:0] l_kw,
        input logic [ARRAY_DIM-1:0][15:0] l_ic
    );
        input_row_t row;
        logic [6:0] dst_lane;
        logic [15:0] src_lane;
        logic [15:0] dst_count;
        begin
            row = '0;
            if (c32_kgen_fast) begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    row[lane] = win[l_kh[lane][2:0]][l_kw[lane][2:0]][lane * 8 +: 8];
                end
            end else if (cfg_coalesce_i && cfg_kgen_i) begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    if (({8'd0, l_kh[lane]} < cfg_kernel_h_i) &&
                        ({8'd0, l_kw[lane]} < cfg_kernel_w_i) &&
                        (l_ic[lane] >= cfg_c_base_i) &&
                        (l_ic[lane] < (cfg_c_base_i + 16'(ARRAY_DIM))) &&
                        (l_ic[lane] < cfg_input_c_i)) begin
                        src_lane = l_ic[lane] - cfg_c_base_i;
                        row[lane] = win[l_kh[lane][2:0]][l_kw[lane][2:0]][src_lane[4:0] * 8 +: 8];
                    end
                end
            end else if (cfg_coalesce_i) begin
                dst_count = {10'd0, cfg_lane_base_i};
                for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                    for (int unsigned kw = 0; kw < K_MAX; kw++) begin
                        for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                            if ((kh < cfg_kernel_h_i) &&
                                (kw < cfg_kernel_w_i) &&
                                (lane < block_valid_bytes) &&
                                (dst_count < 16'(ARRAY_DIM))) begin
                                row[dst_count[4:0]] = win[kh][kw][lane * 8 +: 8];
                                dst_count = dst_count + 16'd1;
                            end
                        end
                    end
                end
            end else begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    dst_lane = {1'b0, cfg_lane_base_i} + 7'(lane);
                    if ((lane < block_valid_bytes) && (dst_lane < 7'(ARRAY_DIM))) begin
                        row[dst_lane[4:0]] = win[tap_kh[2:0]][tap_kw[2:0]][lane * 8 +: 8];
                    end
                end
            end
            build_emit_row = row;
        end
    endfunction

    task automatic derive_format_config;
        logic [7:0] gen_kh;
        logic [7:0] gen_kw;
        logic [15:0] gen_ic;
        begin
            block_valid_bytes = (cfg_block_valid_bytes_i != 6'd0) ?
                                cfg_block_valid_bytes_i :
                                valid_c_bytes(cfg_input_c_i, cfg_c_base_i, cfg_lane_base_i);
            coalesce_k_bytes = (cfg_coalesce_k_bytes_i != 32'd0) ?
                               cfg_coalesce_k_bytes_i :
                               (32'(cfg_kernel_h_i) * 32'(cfg_kernel_w_i) * 32'(block_valid_bytes));

            channel_addr_offset = (cfg_channel_addr_offset_i != 32'd0) ?
                                  cfg_channel_addr_offset_i :
                                  {16'd0, cfg_c_base_i};
            c32_blocked_mode = cfg_c32_fast_i &&
                               (cfg_block_valid_bytes_i == 6'(BEAT_BYTES)) &&
                               (cfg_channel_addr_offset_i[BYTE_SEL_BITS-1:0] == '0);
            c32_kgen_fast = c32_blocked_mode && cfg_coalesce_i && cfg_kgen_i &&
                             (cfg_lane_base_i == 6'd0) &&
                             (cfg_c_base_i[4:0] == 5'd0);

            gen_kh = cfg_k_seed_kh_i;
            gen_kw = cfg_k_seed_kw_i;
            gen_ic = cfg_k_seed_ic_i;
            if (c32_kgen_fast) begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    lane_kh[lane] = cfg_k_seed_kh_i;
                    lane_kw[lane] = cfg_k_seed_kw_i;
                    lane_ic[lane] = cfg_c_base_i + 16'(lane);
                end
            end else begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    lane_kh[lane] = gen_kh;
                    lane_kw[lane] = gen_kw;
                    lane_ic[lane] = gen_ic;
                    if ((gen_ic + 16'd1) == cfg_input_c_i) begin
                        gen_ic = '0;
                        if ((gen_kw + 8'd1) == cfg_kernel_w_i[7:0]) begin
                            gen_kw = '0;
                            gen_kh = gen_kh + 8'd1;
                        end else begin
                            gen_kw = gen_kw + 8'd1;
                        end
                    end else begin
                        gen_ic = gen_ic + 16'd1;
                    end
                end
            end
        end
    endtask

    task automatic derive_cache_mode;
        begin
            row_cache_full_mode = cfg_coalesce_i && cfg_kgen_i &&
                                  (cfg_k_tiles_i > 32'd1) &&
                                  (cfg_input_h_i <= K_MAX[15:0]);
            row_cache_reuse = row_cache_full_q && (cfg_c_base_i == cached_c_base_q);
            row_ring_mode = (cfg_depthwise_i || (cfg_coalesce_i && cfg_kgen_i)) &&
                            !row_cache_full_q &&
                            (cfg_kernel_h_i <= K_MAX[15:0]) &&
                            (cfg_kernel_w_i <= K_MAX[15:0]) &&
                            (cfg_stride_h_i != 16'd0) &&
                            (cfg_stride_h_i <= STRIDE_MAX[15:0]) &&
                            (cfg_stride_w_i != 16'd0) &&
                            (cfg_stride_w_i <= STRIDE_MAX[15:0]);
            fill_done_rows = row_cache_full_q ? cfg_input_h_i : cfg_kernel_h_i;
        end
    endtask

    task automatic derive_fill_status;
        begin
            fill_ih = row_cache_full_q ?
                      $signed({16'd0, fill_kh_q}) :
                      (output_base_ih_q + $signed({16'd0, fill_kh_q}));
            fill_row_in_bounds = row_cache_full_q ?
                                 (fill_kh_q < cfg_input_h_i) :
                                 ((fill_ih >= 32'sd0) &&
                                  (fill_ih < $signed({16'd0, cfg_input_h_i})));
            fill_row_slot = fill_row_in_bounds ? cache_row_slot(fill_ih[15:0]) : 16'd0;
            fill_row_cached = row_ring_mode && fill_row_in_bounds &&
                              row_slot_valid_q[fill_row_slot[$clog2(ROW_SLOTS)-1:0]] &&
                              (row_slot_ih_q[fill_row_slot[$clog2(ROW_SLOTS)-1:0]] == fill_ih[15:0]);
            fill_row_pending = row_ring_mode && fill_row_in_bounds &&
                               row_fetch_active_q[fill_row_slot[$clog2(ROW_SLOTS)-1:0]] &&
                               (row_slot_ih_q[fill_row_slot[$clog2(ROW_SLOTS)-1:0]] == fill_ih[15:0]);
            fill_row_ready = fill_row_cached;
            fill_crosses_current = ({2'b00, fill_addr_q[BYTE_SEL_BITS-1:0]} +
                                    {1'b0, fill_valid_bytes_q}) >
                                   (BYTE_SEL_BITS+2)'(BEAT_BYTES);
        end
    endtask

    task automatic derive_background_status;
        begin
            bg_ih = bg_base_ih_q + $signed({16'd0, bg_kh_q});
            bg_row_in_bounds = (bg_kh_q < cfg_kernel_h_i) &&
                               (bg_ih >= 32'sd0) &&
                               (bg_ih < $signed({16'd0, cfg_input_h_i}));
            bg_row_slot = bg_row_in_bounds ? cache_row_slot(bg_ih[15:0]) : 16'd0;
            bg_row_cached = row_ring_mode && bg_row_in_bounds &&
                            row_slot_valid_q[bg_row_slot[$clog2(ROW_SLOTS)-1:0]] &&
                            (row_slot_ih_q[bg_row_slot[$clog2(ROW_SLOTS)-1:0]] == bg_ih[15:0]);
            bg_row_pending = row_ring_mode && bg_row_in_bounds &&
                             row_fetch_active_q[bg_row_slot[$clog2(ROW_SLOTS)-1:0]] &&
                             (row_slot_ih_q[bg_row_slot[$clog2(ROW_SLOTS)-1:0]] == bg_ih[15:0]);
            bg_row_ready = bg_row_cached;
            bg_crosses_current = ({2'b00, bg_addr_q[BYTE_SEL_BITS-1:0]} +
                                  {1'b0, bg_valid_bytes_q}) >
                                 (BYTE_SEL_BITS+2)'(BEAT_BYTES);
            bg_can_start = row_ring_mode &&
                           (bg_state_q == BG_IDLE) &&
                           !bg_started_for_row_q &&
                           ((state_q == CH_WINDOW_REQ) ||
                            (state_q == CH_WINDOW_WAIT) ||
                            (state_q == CH_STREAM_PRIME) ||
                            (state_q == CH_STREAM_EMIT)) &&
                           (block_valid_bytes != 6'd0) &&
                           ((spatial_rows_q + 32'(cfg_output_w_i)) < dim_m_i);
        end
    endtask

    task automatic derive_bypass_status;
        begin
            bypass_candidate_addr = output_spatial_addr_q + channel_addr_offset;
            bypass_in_bounds = (output_base_ih_q >= 32'sd0) &&
                               (output_base_iw_q >= 32'sd0) &&
                               (output_base_ih_q < $signed({16'd0, cfg_input_h_i})) &&
                               (output_base_iw_q < $signed({16'd0, cfg_input_w_i})) &&
                               (block_valid_bytes != 6'd0);
            bypass_crosses_current = ({2'b00, bypass_candidate_addr[BYTE_SEL_BITS-1:0]} +
                                      {1'b0, block_valid_bytes}) >
                                     (BYTE_SEL_BITS+2)'(BEAT_BYTES);
            bypass_active = (cfg_kernel_h_i == 16'd1) &&
                            (cfg_kernel_w_i == 16'd1) &&
                            (cfg_pad_h_i == 16'd0) &&
                            (cfg_pad_w_i == 16'd0);
        end
    endtask

    task automatic derive_stream_status;
        begin
            last_kernel_vector = ((kh_q + 16'd1) == cfg_kernel_h_i) &&
                                 ((kw_q + 16'd1) == cfg_kernel_w_i);
            output_fire = row_valid_out_q && row_ready_i;
            stg1_fire = stg1_valid_q && stg2_ready;
            emit_fire = bypass_active ? output_fire : stg1_fire;

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

    task automatic derive_beat_accounting;
        begin
            main_fill_beat_push = ((state_q == CH_FILL_REQ0) || (state_q == CH_FILL_REQ1)) &&
                                  obi_gnt_i && !bf_full;
            bg_fill_beat_push = ((bg_state_q == BG_REQ0) || (bg_state_q == BG_REQ1)) &&
                                obi_gnt_i && !bf_full &&
                                !((state_q == CH_FILL_REQ0) || (state_q == CH_FILL_REQ1) ||
                                  (state_q == CH_BYPASS_REQ0) || (state_q == CH_BYPASS_REQ1));
            beat_push = row_ring_mode && (main_fill_beat_push || bg_fill_beat_push);
            beat_push_slot = main_fill_beat_push ?
                             fill_row_slot[$clog2(ROW_SLOTS)-1:0] :
                             bg_row_slot[$clog2(ROW_SLOTS)-1:0];
            beat_push_last_for_row = main_fill_beat_push ?
                                     (((state_q == CH_FILL_REQ0) && !fill_crosses_current &&
                                       ((fill_x_q + 16'd1) == cfg_input_w_i)) ||
                                      ((state_q == CH_FILL_REQ1) &&
                                       ((fill_x_q + 16'd1) == cfg_input_w_i))) :
                                     (((bg_state_q == BG_REQ0) && !bg_crosses_current &&
                                       ((bg_x_q + 16'd1) == cfg_input_w_i)) ||
                                      ((bg_state_q == BG_REQ1) &&
                                       ((bg_x_q + 16'd1) == cfg_input_w_i)));
            beat_pop = obi_rvalid_i && !bf_empty;
            beat_pop_slot = resp_meta.kh[$clog2(ROW_SLOTS)-1:0];

            row_pending_next = row_pending_q;
            if (beat_push) begin
                row_pending_next[beat_push_slot] =
                    row_pending_next[beat_push_slot] + ROW_PENDING_WIDTH'(1);
            end
            if (beat_pop && row_fetch_active_q[beat_pop_slot]) begin
                row_pending_next[beat_pop_slot] =
                    row_pending_next[beat_pop_slot] - ROW_PENDING_WIDTH'(1);
            end

            row_ready_next = '0;
            for (int unsigned slot = 0; slot < ROW_SLOTS; slot++) begin
                row_ready_next[slot] = row_fetch_active_q[slot] &&
                                       row_fetch_done_q[slot] &&
                                       (row_pending_next[slot] == '0);
            end
        end
    endtask

    task automatic drive_response_writeback;
        begin
            bank_w_req = '0;
            bank_w_addr = '0;
            bank_w_data = '0;
            resp_write_bank = 1'b0;
            resp_bank_idx = '0;
            resp_bank_addr = '0;
            resp_bank_wdata = '0;

            if (obi_rvalid_i && !bf_empty) begin
                if (resp_meta.is_beat0_of_cross) begin
                    resp_write_bank = 1'b0;
                end else if (resp_meta.is_solo) begin
                    resp_write_bank = 1'b1;
                    resp_bank_idx = bank_index(resp_meta.kh, resp_meta.x);
                    resp_bank_addr = bank_word_addr(resp_meta.x);
                    resp_bank_wdata = (c32_blocked_mode &&
                                       (resp_meta.addr_lsb == '0) &&
                                       (resp_meta.valid_bytes == 6'(BEAT_BYTES))) ?
                                      obi_rdata_i :
                                      merge_beats(obi_rdata_i, '0,
                                                  resp_meta.addr_lsb,
                                                  resp_meta.valid_bytes);
                end else begin
                    resp_write_bank = 1'b1;
                    resp_bank_idx = bank_index(resp_meta.kh, resp_meta.x);
                    resp_bank_addr = bank_word_addr(resp_meta.x);
                    resp_bank_wdata = merge_beats(resp_beat0_q, obi_rdata_i,
                                                  resp_meta.addr_lsb,
                                                  resp_meta.valid_bytes);
                end
            end

            if (resp_write_bank) begin
                bank_w_req[resp_bank_idx] = 1'b1;
                bank_w_addr[resp_bank_idx] = resp_bank_addr;
                bank_w_data[resp_bank_idx] = resp_bank_wdata;
            end
        end
    endtask

    task automatic drive_window_read_requests;
        logic signed [31:0] cell_ih;
        logic signed [31:0] cell_iw;
        logic signed [31:0] slide_iw;
        logic [15:0] slide_new_cols;
        logic [15:0] slide_target_kw;
        logic [$clog2(BANKS)-1:0] idx;
        begin
            bank_r_req = '0;
            bank_r_addr = '0;

            if ((state_q == CH_WINDOW_REQ) ||
                ((state_q == CH_WINDOW_WAIT) &&
                 ((window_kw_q + 16'd1) != cfg_kernel_w_i))) begin
                for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                    cell_ih = output_base_ih_q + $signed(32'(kh));
                    cell_iw = output_base_iw_q + $signed({16'd0, window_req_kw});
                    if ((kh < cfg_kernel_h_i) &&
                        (window_req_kw < cfg_kernel_w_i) &&
                        (cell_ih >= 32'sd0) &&
                        (cell_iw >= 32'sd0) &&
                        (cell_ih < $signed({16'd0, cfg_input_h_i})) &&
                        (cell_iw < $signed({16'd0, cfg_input_w_i}))) begin
                        idx = bank_index(row_ring_mode ? cache_row_slot(cell_ih[15:0]) :
                                         (row_cache_full_q ? cell_ih[15:0] : 16'(kh)),
                                         cell_iw[15:0]);
                        bank_r_req[idx] = 1'b1;
                        bank_r_addr[idx] = bank_word_addr(cell_iw[15:0]);
                    end
                end
            end else if (slide_req_active) begin
                slide_new_cols = (cfg_stride_w_i >= cfg_kernel_w_i) ? cfg_kernel_w_i : cfg_stride_w_i;
                for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                    for (int unsigned col = 0; col < STRIDE_MAX; col++) begin
                        if ((kh < cfg_kernel_h_i) && (col < slide_new_cols)) begin
                            slide_target_kw = (cfg_stride_w_i >= cfg_kernel_w_i) ?
                                              16'(col) :
                                              (cfg_kernel_w_i - cfg_stride_w_i + 16'(col));
                            cell_ih = output_base_ih_q + $signed(32'(kh));
                            slide_iw = slide_from_iw + $signed({16'd0, cfg_stride_w_i}) +
                                       $signed({16'd0, slide_target_kw});
                            if ((cell_ih >= 32'sd0) &&
                                (slide_iw >= 32'sd0) &&
                                (cell_ih < $signed({16'd0, cfg_input_h_i})) &&
                                (slide_iw < $signed({16'd0, cfg_input_w_i}))) begin
                                idx = bank_index(row_ring_mode ? cache_row_slot(cell_ih[15:0]) :
                                                 (row_cache_full_q ? cell_ih[15:0] : 16'(kh)),
                                                 slide_iw[15:0]);
                                bank_r_req[idx] = 1'b1;
                                bank_r_addr[idx] = bank_word_addr(slide_iw[15:0]);
                            end
                        end
                    end
                end
            end
        end
    endtask

    task automatic drive_obi_request_mux;
        begin
            main_obi_req = 1'b0;
            if ((state_q == CH_FILL_REQ0 || state_q == CH_FILL_REQ1) && !bf_full) begin
                main_obi_req = 1'b1;
            end else if (state_q == CH_BYPASS_REQ0 || state_q == CH_BYPASS_REQ1) begin
                main_obi_req = 1'b1;
            end

            bg_obi_req = ((bg_state_q == BG_REQ0) || (bg_state_q == BG_REQ1)) &&
                         !bf_full && !main_obi_req;
            bg_obi_gnt = bg_obi_req && obi_gnt_i;

            obi_req_o = main_obi_req || bg_obi_req;
            obi_addr_o = main_obi_req ? pending_beat_addr_q : bg_pending_beat_addr_q;
        end
    endtask

    task automatic build_slide_window_next;
        logic signed [31:0] cell_ih;
        logic signed [31:0] slide_iw;
        logic [15:0] slide_new_cols;
        logic [15:0] slide_target_kw;
        logic [$clog2(BANKS)-1:0] idx;
        begin
            slide_window = window_q;
            if (cfg_stride_w_i >= cfg_kernel_w_i) begin
                slide_window = '0;
            end else begin
                for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                    for (int unsigned kw = 0; kw < K_MAX; kw++) begin
                        if ((kh < cfg_kernel_h_i) &&
                            ((32'(kw) + 32'(cfg_stride_w_i)) < 32'(cfg_kernel_w_i))) begin
                            slide_window[kh][kw] = window_q[kh][kw + 32'(cfg_stride_w_i)];
                        end else begin
                            slide_window[kh][kw] = pad_vector;
                        end
                    end
                end
            end

            slide_new_cols = (cfg_stride_w_i >= cfg_kernel_w_i) ? cfg_kernel_w_i : cfg_stride_w_i;
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                for (int unsigned col = 0; col < STRIDE_MAX; col++) begin
                    if ((kh < cfg_kernel_h_i) && (col < slide_new_cols)) begin
                        slide_target_kw = (cfg_stride_w_i >= cfg_kernel_w_i) ?
                                          16'(col) :
                                          (cfg_kernel_w_i - cfg_stride_w_i + 16'(col));
                        cell_ih = output_base_ih_q + $signed(32'(kh));
                        slide_iw = output_base_iw_q + $signed({16'd0, cfg_stride_w_i}) +
                                   $signed({16'd0, slide_target_kw});
                        if ((cell_ih >= 32'sd0) &&
                            (slide_iw >= 32'sd0) &&
                            (cell_ih < $signed({16'd0, cfg_input_h_i})) &&
                            (slide_iw < $signed({16'd0, cfg_input_w_i}))) begin
                            idx = bank_index(row_ring_mode ? cache_row_slot(cell_ih[15:0]) :
                                             (row_cache_full_q ? cell_ih[15:0] : 16'(kh)),
                                             slide_iw[15:0]);
                            slide_window[kh][slide_target_kw[2:0]] = bank_rdata[idx];
                        end else begin
                            slide_window[kh][slide_target_kw[2:0]] = pad_vector;
                        end
                    end
                end
            end
        end
    endtask

    always_comb begin
        derive_format_config();
        derive_cache_mode();
        derive_fill_status();
        derive_background_status();
        derive_bypass_status();
        derive_stream_status();
        derive_beat_accounting();
        drive_response_writeback();
        drive_window_read_requests();
        drive_obi_request_mux();
        build_slide_window_next();
    end

    task automatic reset_sequential_state;
        begin
            state_q <= CH_IDLE;
            bg_state_q <= BG_IDLE;
            window_q <= '0;
            output_row_base_addr_q <= '0;
            output_spatial_addr_q <= '0;
            output_base_ih_q <= '0;
            output_base_iw_q <= '0;
            ow_q <= '0;
            kh_q <= '0;
            kw_q <= '0;
            spatial_rows_q <= '0;
            emitted_vectors_q <= '0;
            fetch_beats_q <= '0;
            bypass_vectors_q <= '0;
            k_tile_idx_q <= '0;
            row_cache_full_q <= 1'b0;
            cached_c_base_q <= '0;
            row_slot_valid_q <= '0;
            row_fetch_active_q <= '0;
            row_fetch_done_q <= '0;
            row_slot_ih_q <= '0;
            row_pending_q <= '0;
            fill_kh_q <= '0;
            fill_x_q <= '0;
            fill_addr_q <= '0;
            pending_beat_addr_q <= '0;
            fill_valid_bytes_q <= '0;
            bg_base_ih_q <= '0;
            bg_row_base_addr_q <= '0;
            bg_kh_q <= '0;
            bg_x_q <= '0;
            bg_addr_q <= '0;
            bg_pending_beat_addr_q <= '0;
            bg_valid_bytes_q <= '0;
            bg_started_for_row_q <= 1'b0;
            bf_wptr_q <= '0;
            bf_rptr_q <= '0;
            bf_count_q <= '0;
            resp_beat0_q <= '0;
            window_kw_q <= '0;
            bypass_addr_q <= '0;
            bypass_beat0_q <= '0;
            bypass_valid_bytes_q <= '0;
            bypass_crosses_beat_q <= 1'b0;
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
        end
    endtask

    task automatic tick_response_engine;
        begin
            if (obi_rvalid_i && !bf_empty) begin
                fetch_beats_q <= fetch_beats_q + 32'd1;
                if (resp_meta.is_beat0_of_cross) begin
                    resp_beat0_q <= obi_rdata_i;
                end
                bf_rptr_q <= bf_rptr_q + 1;
                bf_count_q <= bf_count_q - 1;
            end
        end
    endtask

    task automatic tick_lane_pipeline;
        begin
            stg1_lane_kh_q <= lane_kh;
            stg1_lane_kw_q <= lane_kw;
            stg1_lane_ic_q <= lane_ic;
        end
    endtask

    task automatic tick_row_fetch_accounting;
        begin
            row_pending_q <= row_pending_next;
            if (beat_push && beat_push_last_for_row) begin
                row_fetch_done_q[beat_push_slot] <= 1'b1;
            end
            for (int unsigned slot = 0; slot < ROW_SLOTS; slot++) begin
                if (row_ready_next[slot]) begin
                    row_slot_valid_q[slot] <= 1'b1;
                    row_fetch_active_q[slot] <= 1'b0;
                    row_fetch_done_q[slot] <= 1'b0;
                end
            end
        end
    endtask

    task automatic reset_spatial_walk;
        begin
            window_q <= '0;
            output_row_base_addr_q <= cfg_origin_base_i;
            output_spatial_addr_q <= cfg_origin_base_i;
            output_base_ih_q <= -$signed({16'd0, cfg_pad_h_i});
            output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
            ow_q <= '0;
            kh_q <= '0;
            kw_q <= '0;
            spatial_rows_q <= '0;
            fill_kh_q <= '0;
            fill_x_q <= '0;
            window_kw_q <= '0;
        end
    endtask

    task automatic invalidate_row_tracking_if_needed;
        begin
            if (!(row_ring_mode && (cfg_c_base_i == cached_c_base_q))) begin
                row_slot_valid_q <= '0;
                row_fetch_active_q <= '0;
                row_fetch_done_q <= '0;
                row_pending_q <= '0;
            end
        end
    endtask

    task automatic push_beat_metadata(
        input logic [BYTE_SEL_BITS-1:0] addr_lsb,
        input logic [5:0] valid_bytes,
        input logic [15:0] kh,
        input logic [15:0] x,
        input logic is_beat0_of_cross,
        input logic is_solo
    );
        begin
            beat_fifo_q[bf_wptr_q].addr_lsb <= addr_lsb;
            beat_fifo_q[bf_wptr_q].valid_bytes <= valid_bytes;
            beat_fifo_q[bf_wptr_q].kh <= kh;
            beat_fifo_q[bf_wptr_q].x <= x;
            beat_fifo_q[bf_wptr_q].is_beat0_of_cross <= is_beat0_of_cross;
            beat_fifo_q[bf_wptr_q].is_solo <= is_solo;
            bf_wptr_q <= bf_wptr_q + 1;
            if (obi_rvalid_i && !bf_empty) begin
                bf_count_q <= bf_count_q;
            end else begin
                bf_count_q <= bf_count_q + 1;
            end
        end
    endtask

    task automatic start_row_fetch(
        input logic [15:0] row_slot,
        input logic [15:0] row_ih
    );
        begin
            row_slot_valid_q[row_slot[$clog2(ROW_SLOTS)-1:0]] <= 1'b0;
            row_fetch_active_q[row_slot[$clog2(ROW_SLOTS)-1:0]] <= 1'b1;
            row_fetch_done_q[row_slot[$clog2(ROW_SLOTS)-1:0]] <= 1'b0;
            row_pending_q[row_slot[$clog2(ROW_SLOTS)-1:0]] <= '0;
            row_slot_ih_q[row_slot[$clog2(ROW_SLOTS)-1:0]] <= row_ih;
        end
    endtask

    task automatic tick_idle_state;
        begin
            stg1_valid_q <= 1'b0;
            row_valid_out_q <= 1'b0;
            if (start_i) begin
                reset_spatial_walk();
                emitted_vectors_q <= '0;
                fetch_beats_q <= '0;
                bypass_vectors_q <= '0;
                k_tile_idx_q <= '0;
                row_cache_full_q <= row_cache_full_mode;
                cached_c_base_q <= cfg_c_base_i;
                row_slot_valid_q <= '0;
                row_fetch_active_q <= '0;
                row_fetch_done_q <= '0;
                row_pending_q <= '0;
                bg_state_q <= BG_IDLE;
                bg_started_for_row_q <= 1'b0;
                prefetch_active_q <= 1'b0;
                prefetch_ready_q <= 1'b0;
                prefetched_c_base_q <= '0;
                if ((dim_m_i == 32'd0) ||
                    (cfg_output_w_i == 16'd0) ||
                    (cfg_kernel_h_i == 16'd0) ||
                    (cfg_kernel_w_i == 16'd0) ||
                    (cfg_kernel_h_i > K_MAX[15:0]) ||
                    (cfg_kernel_w_i > K_MAX[15:0]) ||
                    (cfg_input_w_i > MAX_INPUT_W[15:0]) ||
                    (cfg_stride_h_i == 16'd0) ||
                    (cfg_stride_w_i == 16'd0) ||
                    (cfg_stride_h_i > STRIDE_MAX[15:0]) ||
                    (cfg_stride_w_i > STRIDE_MAX[15:0]) ||
                    (cfg_coalesce_i && !cfg_kgen_i && (coalesce_k_bytes > 32'(ARRAY_DIM)))) begin
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

    task automatic tick_bypass_states;
        begin
            unique case (state_q)
                CH_BYPASS_PREP: begin
                    if (!bypass_in_bounds) begin
                        row_data_q <= '0;
                        row_valid_out_q <= 1'b1;
                        state_q <= CH_STREAM_EMIT;
                    end else begin
                        bypass_addr_q <= bypass_candidate_addr;
                        bypass_valid_bytes_q <= block_valid_bytes;
                        bypass_crosses_beat_q <= bypass_crosses_current;
                        pending_beat_addr_q <= beat_base(bypass_candidate_addr);
                        state_q <= CH_BYPASS_REQ0;
                    end
                end

                CH_BYPASS_REQ0: begin
                    if (obi_gnt_i) begin
                        state_q <= CH_BYPASS_WAIT0;
                    end
                end

                CH_BYPASS_WAIT0: begin
                    if (obi_rvalid_i) begin
                        bypass_beat0_q <= obi_rdata_i;
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        if (bypass_crosses_beat_q) begin
                            pending_beat_addr_q <= beat_base(bypass_addr_q) + 32'(BEAT_BYTES);
                            state_q <= CH_BYPASS_REQ1;
                        end else begin
                            row_data_q <= (c32_blocked_mode &&
                                           (bypass_addr_q[BYTE_SEL_BITS-1:0] == '0) &&
                                           (bypass_valid_bytes_q == 6'(BEAT_BYTES))) ?
                                          unpack_row(obi_rdata_i, bypass_valid_bytes_q, cfg_lane_base_i) :
                                          unpack_row(
                                              merge_beats(obi_rdata_i, '0,
                                                          bypass_addr_q[BYTE_SEL_BITS-1:0],
                                                          bypass_valid_bytes_q),
                                              bypass_valid_bytes_q,
                                              cfg_lane_base_i
                                          );
                            row_valid_out_q <= 1'b1;
                            state_q <= CH_STREAM_EMIT;
                        end
                    end
                end

                CH_BYPASS_REQ1: begin
                    if (obi_gnt_i) begin
                        state_q <= CH_BYPASS_WAIT1;
                    end
                end

                CH_BYPASS_WAIT1: begin
                    if (obi_rvalid_i) begin
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        row_data_q <= unpack_row(
                            merge_beats(bypass_beat0_q, obi_rdata_i,
                                        bypass_addr_q[BYTE_SEL_BITS-1:0],
                                        bypass_valid_bytes_q),
                            bypass_valid_bytes_q,
                            cfg_lane_base_i
                        );
                        row_valid_out_q <= 1'b1;
                        state_q <= CH_STREAM_EMIT;
                    end
                end

                default: begin
                end
            endcase
        end
    endtask

    task automatic tick_fill_states;
        begin
            unique case (state_q)
                CH_ENSURE: begin
                    window_q <= '0;
                    if (fill_kh_q == fill_done_rows) begin
                        if (prefetch_active_q) begin
                            prefetch_active_q <= 1'b0;
                            prefetch_ready_q <= 1'b1;
                            prefetched_c_base_q <= cfg_c_base_i;
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
                        fill_x_q <= '0;
                        fill_valid_bytes_q <= block_valid_bytes;
                        fill_addr_q <= row_cache_full_q ?
                                       row_tap_addr(cfg_origin_base_i +
                                                    (32'({16'd0, cfg_pad_h_i}) *
                                                     cfg_row_stride_bytes_i),
                                                    fill_kh_q) :
                                       row_tap_addr(output_row_base_addr_q, fill_kh_q);
                        pending_beat_addr_q <= beat_base(
                            row_cache_full_q ?
                            row_tap_addr(cfg_origin_base_i +
                                         (32'({16'd0, cfg_pad_h_i}) *
                                          cfg_row_stride_bytes_i),
                                         fill_kh_q) :
                            row_tap_addr(output_row_base_addr_q, fill_kh_q)
                        );
                        if (row_ring_mode && fill_row_in_bounds) begin
                            start_row_fetch(fill_row_slot, fill_ih[15:0]);
                        end
                        state_q <= CH_FILL_REQ0;
                    end
                end

                CH_FILL_REQ0: begin
                    if (obi_gnt_i && !bf_full) begin
                        push_beat_metadata(fill_addr_q[BYTE_SEL_BITS-1:0],
                                           fill_valid_bytes_q,
                                           row_ring_mode ? fill_row_slot : fill_kh_q,
                                           fill_x_q,
                                           fill_crosses_current,
                                           !fill_crosses_current);
                        if (fill_crosses_current) begin
                            pending_beat_addr_q <= beat_base(fill_addr_q) + 32'(BEAT_BYTES);
                            state_q <= CH_FILL_REQ1;
                        end else if ((fill_x_q + 16'd1) == cfg_input_w_i) begin
                            state_q <= CH_FILL_DRAIN;
                        end else begin
                            fill_x_q <= fill_x_q + 16'd1;
                            fill_addr_q <= fill_addr_q + cfg_pixel_stride_bytes_i;
                            pending_beat_addr_q <= beat_base(fill_addr_q + cfg_pixel_stride_bytes_i);
                        end
                    end
                end

                CH_FILL_REQ1: begin
                    if (obi_gnt_i && !bf_full) begin
                        push_beat_metadata(fill_addr_q[BYTE_SEL_BITS-1:0],
                                           fill_valid_bytes_q,
                                           row_ring_mode ? fill_row_slot : fill_kh_q,
                                           fill_x_q,
                                           1'b0,
                                           1'b0);
                        if ((fill_x_q + 16'd1) == cfg_input_w_i) begin
                            state_q <= CH_FILL_DRAIN;
                        end else begin
                            fill_x_q <= fill_x_q + 16'd1;
                            fill_addr_q <= fill_addr_q + cfg_pixel_stride_bytes_i;
                            pending_beat_addr_q <= beat_base(fill_addr_q + cfg_pixel_stride_bytes_i);
                            state_q <= CH_FILL_REQ0;
                        end
                    end
                end

                CH_FILL_DRAIN: begin
                    if ((row_ring_mode && fill_row_ready) ||
                        (!row_ring_mode && bf_empty)) begin
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
                    for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                        logic signed [31:0] cell_ih_ff;
                        logic signed [31:0] cell_iw_ff;
                        logic [$clog2(BANKS)-1:0] idx_ff;
                        cell_ih_ff = output_base_ih_q + $signed(32'(kh));
                        cell_iw_ff = output_base_iw_q + $signed({16'd0, window_kw_q});
                        if ((kh < cfg_kernel_h_i) &&
                            (window_kw_q < cfg_kernel_w_i) &&
                            (cell_ih_ff >= 32'sd0) &&
                            (cell_iw_ff >= 32'sd0) &&
                            (cell_ih_ff < $signed({16'd0, cfg_input_h_i})) &&
                            (cell_iw_ff < $signed({16'd0, cfg_input_w_i}))) begin
                            idx_ff = bank_index(row_ring_mode ? cache_row_slot(cell_ih_ff[15:0]) :
                                                (row_cache_full_q ? cell_ih_ff[15:0] : 16'(kh)),
                                                cell_iw_ff[15:0]);
                            window_q[kh][window_kw_q[2:0]] <= bank_rdata[idx_ff];
                        end else if (kh < K_MAX) begin
                            window_q[kh][window_kw_q[2:0]] <= pad_vector;
                        end
                    end
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
                        if (bypass_active) begin
                            bypass_vectors_q <= bypass_vectors_q + 32'd1;
                            row_valid_out_q <= 1'b0;
                        end

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
                            end else if (bypass_active) begin
                                stg1_valid_q <= 1'b0;
                                if ((ow_q + 16'd1) == cfg_output_w_i) begin
                                    advance_to_next_output_row();
                                end else begin
                                    advance_to_next_output_col();
                                end
                                state_q <= CH_BYPASS_PREP;
                            end else if ((ow_q + 16'd1) == cfg_output_w_i) begin
                                stg1_valid_q <= 1'b0;
                                advance_to_next_output_row();
                                fill_kh_q <= '0;
                                window_kw_q <= '0;
                                window_q <= '0;
                                state_q <= row_cache_full_q ? CH_WINDOW_REQ : CH_ENSURE;
                            end else if (has_next_same_row) begin
                                window_q <= slide_window;
                                stg1_tap_kh_q <= '0;
                                stg1_tap_kw_q <= '0;
                                advance_to_next_output_col();
                            end
                        end
                    end
                end

                CH_STREAM_DONE: begin
                    stg1_valid_q <= 1'b0;
                    if (!row_valid_out_q || row_ready_i) begin
                        if (more_k_tiles) begin
                            if (next_tile_i) begin
                                k_tile_idx_q <= k_tile_idx_q + 32'd1;
                                reset_spatial_walk();
                                invalidate_row_tracking_if_needed();
                                bg_state_q <= BG_IDLE;
                                bg_started_for_row_q <= 1'b0;
                                prefetch_active_q <= 1'b0;
                                prefetch_ready_q <= 1'b0;
                                if (row_cache_reuse ||
                                    (prefetch_ready_q && (prefetched_c_base_q == cfg_c_base_i))) begin
                                    state_q <= CH_WINDOW_REQ;
                                end else begin
                                    cached_c_base_q <= cfg_c_base_i;
                                    state_q <= CH_ENSURE;
                                end
                            end else if (prefetch_i && !prefetch_active_q && !prefetch_ready_q &&
                                         !row_cache_reuse) begin
                                reset_spatial_walk();
                                invalidate_row_tracking_if_needed();
                                bg_state_q <= BG_IDLE;
                                bg_started_for_row_q <= 1'b0;
                                cached_c_base_q <= cfg_c_base_i;
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
                CH_BYPASS_PREP,
                CH_BYPASS_REQ0,
                CH_BYPASS_WAIT0,
                CH_BYPASS_REQ1,
                CH_BYPASS_WAIT1: tick_bypass_states();
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

    task automatic tick_background_fsm;
        begin
            unique case (bg_state_q)
                BG_IDLE: begin
                    if (bg_can_start) begin
                        bg_base_ih_q <= output_base_ih_q + $signed({16'd0, cfg_stride_h_i});
                        bg_row_base_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                        bg_kh_q <= '0;
                        bg_x_q <= '0;
                        bg_started_for_row_q <= 1'b1;
                        bg_state_q <= BG_SCAN;
                    end
                end

                BG_SCAN: begin
                    if (bg_kh_q == cfg_kernel_h_i) begin
                        bg_state_q <= BG_IDLE;
                    end else if (!bg_row_in_bounds || bg_row_ready || bg_row_pending ||
                                 (block_valid_bytes == 6'd0)) begin
                        bg_kh_q <= bg_kh_q + 16'd1;
                    end else begin
                        bg_x_q <= '0;
                        bg_valid_bytes_q <= block_valid_bytes;
                        bg_addr_q <= row_tap_addr(bg_row_base_addr_q, bg_kh_q);
                        bg_pending_beat_addr_q <= beat_base(
                            row_tap_addr(bg_row_base_addr_q, bg_kh_q)
                        );
                        if (row_ring_mode && bg_row_in_bounds) begin
                            start_row_fetch(bg_row_slot, bg_ih[15:0]);
                        end
                        bg_state_q <= BG_REQ0;
                    end
                end

                BG_REQ0: begin
                    if (bg_obi_gnt && !bf_full) begin
                        push_beat_metadata(bg_addr_q[BYTE_SEL_BITS-1:0],
                                           bg_valid_bytes_q,
                                           bg_row_slot,
                                           bg_x_q,
                                           bg_crosses_current,
                                           !bg_crosses_current);
                        if (bg_crosses_current) begin
                            bg_pending_beat_addr_q <= beat_base(bg_addr_q) + 32'(BEAT_BYTES);
                            bg_state_q <= BG_REQ1;
                        end else if ((bg_x_q + 16'd1) == cfg_input_w_i) begin
                            bg_state_q <= BG_DRAIN;
                        end else begin
                            bg_x_q <= bg_x_q + 16'd1;
                            bg_addr_q <= bg_addr_q + cfg_pixel_stride_bytes_i;
                            bg_pending_beat_addr_q <= beat_base(bg_addr_q + cfg_pixel_stride_bytes_i);
                        end
                    end
                end

                BG_REQ1: begin
                    if (bg_obi_gnt && !bf_full) begin
                        push_beat_metadata(bg_addr_q[BYTE_SEL_BITS-1:0],
                                           bg_valid_bytes_q,
                                           bg_row_slot,
                                           bg_x_q,
                                           1'b0,
                                           1'b0);
                        if ((bg_x_q + 16'd1) == cfg_input_w_i) begin
                            bg_state_q <= BG_DRAIN;
                        end else begin
                            bg_x_q <= bg_x_q + 16'd1;
                            bg_addr_q <= bg_addr_q + cfg_pixel_stride_bytes_i;
                            bg_pending_beat_addr_q <= beat_base(bg_addr_q + cfg_pixel_stride_bytes_i);
                            bg_state_q <= BG_REQ0;
                        end
                    end
                end

                BG_DRAIN: begin
                    if (bg_row_ready) begin
                        bg_kh_q <= bg_kh_q + 16'd1;
                        bg_state_q <= BG_SCAN;
                    end
                end

                default: begin
                    bg_state_q <= BG_IDLE;
                end
            endcase
        end
    endtask

    task automatic tick_output_stage;
        begin
            if (stg1_valid_q && stg2_ready) begin
                row_data_q <= build_emit_row(window_q, stg1_tap_kh_q, stg1_tap_kw_q,
                                             stg1_lane_kh_q, stg1_lane_kw_q, stg1_lane_ic_q);
                row_valid_out_q <= 1'b1;
            end else if (!stg1_valid_q && stg2_ready && !bypass_active) begin
                row_valid_out_q <= 1'b0;
            end
        end
    endtask

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            reset_sequential_state();
        end else begin
            done_q <= 1'b0;
            tick_response_engine();
            tick_lane_pipeline();
            tick_row_fetch_accounting();
            tick_main_fsm();
            tick_background_fsm();
            tick_output_stage();
        end
    end

endmodule

`default_nettype wire
