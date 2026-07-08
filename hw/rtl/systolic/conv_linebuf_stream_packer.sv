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
    localparam int unsigned BANKS = K_MAX * STRIDE_MAX;
    localparam int unsigned BANK_DEPTH = (MAX_INPUT_W + STRIDE_MAX - 1) / STRIDE_MAX;
    localparam int unsigned BANK_ADDR_WIDTH = $clog2(BANK_DEPTH);

    typedef enum logic [4:0] {
        CH_IDLE,
        CH_ENSURE,
        CH_FILL_REQ0,
        CH_FILL_WAIT0,
        CH_FILL_REQ1,
        CH_FILL_WAIT1,
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
        CH_STREAM_REFILL,
        CH_STREAM_STALL,
        CH_STREAM_SLIDE,
        CH_SS_FILL_PREP,
        CH_SS_FILL_REQ0,
        CH_SS_FILL_WAIT0,
        CH_SS_FILL_REQ1,
        CH_SS_FILL_WAIT1,
        CH_STREAM_DONE
    } state_e;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_t;
    localparam int unsigned SS_ROWS = K_MAX;
    localparam int unsigned SS_COLS = 16;
    localparam int unsigned SS_CBLOCKS = 4;
    localparam int unsigned SS_ROW_BITS = $clog2(SS_ROWS);
    localparam int unsigned SS_ROW_COUNT_BITS = $clog2(SS_ROWS + 1);
    localparam int unsigned SS_COL_BITS = $clog2(SS_COLS);
    localparam int unsigned SS_CB_BITS = $clog2(SS_CBLOCKS);
    localparam int unsigned SS_CB_IDX_LSB = $clog2(ARRAY_DIM);  // = 5 for ARRAY_DIM=32

    state_e state_q;

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

    logic [BANKS-1:0] bank_req;
    logic [BANKS-1:0] bank_we;
    logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_addr;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_wdata;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_rdata;
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
    logic [K_MAX-1:0] row_slot_valid_q;
    logic [K_MAX-1:0][15:0] row_slot_ih_q;
    logic ss_active_q;
    logic [SS_ROW_COUNT_BITS-1:0] ss_fill_row_q;
    logic [SS_COL_BITS-1:0] ss_fill_x_q;
    logic [SS_CB_BITS-1:0] ss_fill_cb_q;
    logic [31:0] ss_fill_addr_q;
    logic [31:0] ss_pending_addr_q;
    logic [DATA_WIDTH-1:0] ss_fill_beat0_q;
    logic [5:0] ss_fill_valid_bytes_q;
    logic ss_fill_crosses_beat_q;
    logic [SS_ROWS-1:0][SS_COLS-1:0][SS_CBLOCKS-1:0][DATA_WIDTH-1:0] ss_cache_q;

    logic [15:0] fill_kh_q;
    logic [15:0] fill_x_q;
    logic [31:0] fill_addr_q;
    logic [DATA_WIDTH-1:0] fill_beat0_q;
    logic [DATA_WIDTH-1:0] fill_data_q;
    logic [31:0] pending_beat_addr_q;
    logic [5:0] fill_valid_bytes_q;
    logic fill_crosses_beat_q;

    logic [15:0] window_kw_q;

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
    logic ss_mode;
    logic row_ring_mode;
    logic [15:0] fill_row_slot;
    logic fill_row_cached;
    logic [15:0] fill_done_rows;
    logic [15:0] next_kh;
    logic [15:0] next_kw;
    logic signed [31:0] slide_from_iw;
    logic slide_req_active;
    
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

    for (genvar bank = 0; bank < BANKS; bank++) begin : gen_line_banks
        tc_sram #(
            .NumWords    (BANK_DEPTH),
            .DataWidth   (DATA_WIDTH),
            .ByteWidth   (8),
            .NumPorts    (1),
            .Latency     (1),
            .SimInit     ("none"),
            .PrintSimCfg (1'b0)
        ) i_bank_sram (
            .clk_i   (clk_i),
            .rst_ni  (rst_ni),
            .req_i   (bank_req[bank]),
            .we_i    (bank_we[bank]),
            .addr_i  (bank_addr[bank]),
            .wdata_i (bank_wdata[bank]),
            .be_i    (bank_be),
            .rdata_o (bank_rdata[bank])
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
        cache_row_slot = 16'(32'(ih) % K_MAX);
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
            if (cfg_coalesce_i && cfg_kgen_i) begin
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

    function automatic input_row_t build_ss_emit_row(
        input logic [ARRAY_DIM-1:0][7:0] l_kh,
        input logic [ARRAY_DIM-1:0][7:0] l_kw,
        input logic [ARRAY_DIM-1:0][15:0] l_ic
    );
        input_row_t row;
        logic signed [31:0] cell_ih_ss;
        logic signed [31:0] cell_iw_ss;
        logic [SS_CB_BITS-1:0] cb;
        logic [SS_CB_IDX_LSB-1:0] src_lane;
        begin
            row = '0;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                cell_ih_ss = output_base_ih_q + $signed({24'd0, l_kh[lane]});
                cell_iw_ss = output_base_iw_q + $signed({24'd0, l_kw[lane]});
                cb = l_ic[lane][SS_CB_IDX_LSB +: SS_CB_BITS];
                src_lane = l_ic[lane][SS_CB_IDX_LSB-1:0];
                if (({8'd0, l_kh[lane]} < cfg_kernel_h_i) &&
                    ({8'd0, l_kw[lane]} < cfg_kernel_w_i) &&
                    (l_ic[lane] < cfg_input_c_i) &&
                    (cell_ih_ss >= 32'sd0) &&
                    (cell_iw_ss >= 32'sd0) &&
                    (cell_ih_ss < $signed({16'd0, cfg_input_h_i})) &&
                    (cell_iw_ss < $signed({16'd0, cfg_input_w_i})) &&
                    ({30'd0, cb} < 32'(SS_CBLOCKS))) begin
                    row[lane] = ss_cache_q[cell_ih_ss[SS_ROW_BITS-1:0]]
                                          [cell_iw_ss[SS_COL_BITS-1:0]]
                                          [cb][src_lane * 8 +: 8];
                end
            end
            build_ss_emit_row = row;
        end
    endfunction

    always_comb begin
        logic [7:0] gen_kh;
        logic [7:0] gen_kw;
        logic [15:0] gen_ic;
        logic [15:0] slide_new_cols;
        logic [15:0] slide_target_kw;
        logic signed [31:0] slide_iw;
        logic signed [31:0] cell_ih;
        logic signed [31:0] cell_iw;
        logic [$clog2(BANKS)-1:0] idx;

        gen_kh = '0;
        gen_kw = '0;
        gen_ic = '0;
        slide_new_cols = '0;
        slide_target_kw = '0;
        slide_iw = '0;
        cell_ih = '0;
        cell_iw = '0;
        idx = '0;

        block_valid_bytes = valid_c_bytes(cfg_input_c_i, cfg_c_base_i, cfg_lane_base_i);
        coalesce_k_bytes = 32'(cfg_kernel_h_i) * 32'(cfg_kernel_w_i) * 32'(block_valid_bytes);

        ss_mode = cfg_coalesce_i && cfg_kgen_i &&
                  (cfg_k_tiles_i > 32'd1) &&
                  (cfg_input_h_i <= 16'(SS_ROWS)) &&
                  (cfg_input_w_i <= 16'(SS_COLS)) &&
                  (({16'd0, cfg_input_c_i} + 32'd31) >> 5 <= 32'(SS_CBLOCKS));

        gen_kh = cfg_k_seed_kh_i;
        gen_kw = cfg_k_seed_kw_i;
        gen_ic = cfg_k_seed_ic_i;
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

        row_cache_full_mode = cfg_coalesce_i && cfg_kgen_i &&
                              (cfg_k_tiles_i > 32'd1) &&
                              (cfg_input_h_i <= K_MAX[15:0]);
        row_cache_reuse = row_cache_full_q && (cfg_c_base_i == cached_c_base_q);
        row_ring_mode = cfg_coalesce_i && cfg_kgen_i &&
                        !row_cache_full_q &&
                        (cfg_kernel_h_i <= K_MAX[15:0]) &&
                        (cfg_kernel_w_i <= K_MAX[15:0]) &&
                        (cfg_stride_h_i != 16'd0) &&
                        (cfg_stride_h_i <= STRIDE_MAX[15:0]) &&
                        (cfg_stride_w_i != 16'd0) &&
                        (cfg_stride_w_i <= STRIDE_MAX[15:0]);
        fill_done_rows = row_cache_full_q ? cfg_input_h_i : cfg_kernel_h_i;
        fill_ih = row_cache_full_q ?
                  $signed({16'd0, fill_kh_q}) :
                  (output_base_ih_q + $signed({16'd0, fill_kh_q}));
        fill_row_in_bounds = row_cache_full_q ?
                             (fill_kh_q < cfg_input_h_i) :
                             ((fill_ih >= 32'sd0) &&
                              (fill_ih < $signed({16'd0, cfg_input_h_i})));
        fill_row_slot = fill_row_in_bounds ? cache_row_slot(fill_ih[15:0]) : 16'd0;
        fill_row_cached = row_ring_mode && fill_row_in_bounds &&
                          row_slot_valid_q[fill_row_slot[$clog2(K_MAX)-1:0]] &&
                          (row_slot_ih_q[fill_row_slot[$clog2(K_MAX)-1:0]] == fill_ih[15:0]);
        fill_crosses_current = ({2'b00, fill_addr_q[BYTE_SEL_BITS-1:0]} + {1'b0, fill_valid_bytes_q}) >
                               (BYTE_SEL_BITS+2)'(BEAT_BYTES);

        bypass_candidate_addr = output_spatial_addr_q + {16'd0, cfg_c_base_i};
        bypass_in_bounds = (output_base_ih_q >= 32'sd0) &&
                           (output_base_iw_q >= 32'sd0) &&
                           (output_base_ih_q < $signed({16'd0, cfg_input_h_i})) &&
                           (output_base_iw_q < $signed({16'd0, cfg_input_w_i})) &&
                           (block_valid_bytes != 6'd0);
        bypass_crosses_current = ({2'b00, bypass_candidate_addr[BYTE_SEL_BITS-1:0]} +
                                  {1'b0, block_valid_bytes}) >
                                 (BYTE_SEL_BITS+2)'(BEAT_BYTES);

        last_kernel_vector = ((kh_q + 16'd1) == cfg_kernel_h_i) &&
                             ((kw_q + 16'd1) == cfg_kernel_w_i);

        bypass_active = (cfg_kernel_h_i == 16'd1) && 
                        (cfg_kernel_w_i == 16'd1) && 
                        (cfg_pad_h_i == 16'd0) && 
                        (cfg_pad_w_i == 16'd0);
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

        bank_req = '0;
        bank_we = '0;
        bank_addr = '0;
        bank_wdata = '0;

        // --- Response engine: combinational bank write ---
        resp_write_bank = 1'b0;
        resp_bank_idx = '0;
        resp_bank_addr = '0;
        resp_bank_wdata = '0;
        if (obi_rvalid_i && !bf_empty) begin
            if (resp_meta.is_beat0_of_cross) begin
                // First beat of crossing pixel: no bank write yet
                resp_write_bank = 1'b0;
            end else if (resp_meta.is_solo) begin
                // Solo beat: merge with zeros and write
                resp_write_bank = 1'b1;
                resp_bank_idx = bank_index(resp_meta.kh, resp_meta.x);
                resp_bank_addr = bank_word_addr(resp_meta.x);
                resp_bank_wdata = merge_beats(obi_rdata_i, '0, resp_meta.addr_lsb, resp_meta.valid_bytes);
            end else begin
                // Beat1 of crossing pixel: merge beat0 + beat1
                resp_write_bank = 1'b1;
                resp_bank_idx = bank_index(resp_meta.kh, resp_meta.x);
                resp_bank_addr = bank_word_addr(resp_meta.x);
                resp_bank_wdata = merge_beats(resp_beat0_q, obi_rdata_i, resp_meta.addr_lsb, resp_meta.valid_bytes);
            end
        end

        if (resp_write_bank) begin
            idx = resp_bank_idx;
            bank_req[idx] = 1'b1;
            bank_we[idx] = 1'b1;
            bank_addr[idx] = resp_bank_addr;
            bank_wdata[idx] = resp_bank_wdata;
        end else if (state_q == CH_WINDOW_REQ) begin
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                cell_ih = output_base_ih_q + $signed(32'(kh));
                cell_iw = output_base_iw_q + $signed({16'd0, window_kw_q});
                if ((kh < cfg_kernel_h_i) &&
                    (window_kw_q < cfg_kernel_w_i) &&
                    (cell_ih >= 32'sd0) &&
                    (cell_iw >= 32'sd0) &&
                    (cell_ih < $signed({16'd0, cfg_input_h_i})) &&
                    (cell_iw < $signed({16'd0, cfg_input_w_i}))) begin
                    idx = bank_index(row_ring_mode ? cache_row_slot(cell_ih[15:0]) :
                                     (row_cache_full_q ? cell_ih[15:0] : 16'(kh)), cell_iw[15:0]);
                    bank_req[idx] = 1'b1;
                    bank_addr[idx] = bank_word_addr(cell_iw[15:0]);
                end
            end
        end else if (slide_req_active) begin
            slide_new_cols = (cfg_stride_w_i >= cfg_kernel_w_i) ? cfg_kernel_w_i : cfg_stride_w_i;
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                for (int unsigned col = 0; col < STRIDE_MAX; col++) begin
                    if ((kh < cfg_kernel_h_i) && (col < slide_new_cols)) begin
                        if (cfg_stride_w_i >= cfg_kernel_w_i) begin
                            slide_target_kw = 16'(col);
                        end else begin
                            slide_target_kw = cfg_kernel_w_i - cfg_stride_w_i + 16'(col);
                        end
                        cell_ih = output_base_ih_q + $signed(32'(kh));
                        slide_iw = slide_from_iw + $signed({16'd0, cfg_stride_w_i}) +
                                   $signed({16'd0, slide_target_kw});
                        if ((cell_ih >= 32'sd0) &&
                            (slide_iw >= 32'sd0) &&
                            (cell_ih < $signed({16'd0, cfg_input_h_i})) &&
                            (slide_iw < $signed({16'd0, cfg_input_w_i}))) begin
                            idx = bank_index(row_ring_mode ? cache_row_slot(cell_ih[15:0]) :
                                             (row_cache_full_q ? cell_ih[15:0] : 16'(kh)), slide_iw[15:0]);
                            bank_req[idx] = 1'b1;
                            bank_addr[idx] = bank_word_addr(slide_iw[15:0]);
                        end
                    end
                end
            end
        end

        obi_req_o = 1'b0;
        obi_addr_o = pending_beat_addr_q;
        if ((state_q == CH_FILL_REQ0 || state_q == CH_FILL_REQ1) && !bf_full) begin
            obi_req_o = 1'b1;
        end else if (state_q == CH_BYPASS_REQ0 || state_q == CH_BYPASS_REQ1) begin
            obi_req_o = 1'b1;
        end else if (state_q == CH_SS_FILL_REQ0 || state_q == CH_SS_FILL_REQ1) begin
            obi_req_o = 1'b1;
            obi_addr_o = ss_pending_addr_q;
        end

        slide_window = window_q;
        if (cfg_stride_w_i >= cfg_kernel_w_i) begin
            slide_window = '0;
        end else begin
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                for (int unsigned kw = 0; kw < K_MAX; kw++) begin
                    if ((kh < cfg_kernel_h_i) && ((32'(kw) + 32'(cfg_stride_w_i)) < 32'(cfg_kernel_w_i))) begin
                        slide_window[kh][kw] = window_q[kh][kw + 32'(cfg_stride_w_i)];
                    end else begin
                        slide_window[kh][kw] = '0;
                    end
                end
            end
        end
        slide_new_cols = (cfg_stride_w_i >= cfg_kernel_w_i) ? cfg_kernel_w_i : cfg_stride_w_i;
        for (int unsigned kh = 0; kh < K_MAX; kh++) begin
            for (int unsigned col = 0; col < STRIDE_MAX; col++) begin
                if ((kh < cfg_kernel_h_i) && (col < slide_new_cols)) begin
                    if (cfg_stride_w_i >= cfg_kernel_w_i) begin
                        slide_target_kw = 16'(col);
                    end else begin
                        slide_target_kw = cfg_kernel_w_i - cfg_stride_w_i + 16'(col);
                    end
                    cell_ih = output_base_ih_q + $signed(32'(kh));
                    slide_iw = output_base_iw_q + $signed({16'd0, cfg_stride_w_i}) +
                               $signed({16'd0, slide_target_kw});
                    if ((cell_ih >= 32'sd0) &&
                        (slide_iw >= 32'sd0) &&
                        (cell_ih < $signed({16'd0, cfg_input_h_i})) &&
                        (slide_iw < $signed({16'd0, cfg_input_w_i}))) begin
                        idx = bank_index(row_ring_mode ? cache_row_slot(cell_ih[15:0]) :
                                         (row_cache_full_q ? cell_ih[15:0] : 16'(kh)), slide_iw[15:0]);
                        slide_window[kh][slide_target_kw[2:0]] = bank_rdata[idx];
                    end else begin
                        slide_window[kh][slide_target_kw[2:0]] = '0;
                    end
                end
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= CH_IDLE;
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
            row_slot_ih_q <= '0;
            ss_active_q <= 1'b0;
            ss_fill_row_q <= '0;
            ss_fill_x_q <= '0;
            ss_fill_cb_q <= '0;
            ss_fill_addr_q <= '0;
            ss_pending_addr_q <= '0;
            ss_fill_beat0_q <= '0;
            ss_fill_valid_bytes_q <= '0;
            ss_fill_crosses_beat_q <= 1'b0;
            ss_cache_q <= '0;
            fill_kh_q <= '0;
            fill_x_q <= '0;
            fill_addr_q <= '0;
            fill_beat0_q <= '0;
            fill_data_q <= '0;
            pending_beat_addr_q <= '0;
            fill_valid_bytes_q <= '0;
            fill_crosses_beat_q <= 1'b0;
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
        end else begin
            done_q <= 1'b0;

            // --- Beat FIFO: response engine (runs every cycle in parallel) ---
            if (obi_rvalid_i && !bf_empty) begin
                fetch_beats_q <= fetch_beats_q + 32'd1;
                if (resp_meta.is_beat0_of_cross) begin
                    resp_beat0_q <= obi_rdata_i;
                end
                // Pop from beat FIFO
                bf_rptr_q <= bf_rptr_q + 1;
                bf_count_q <= bf_count_q - 1;
            end
            
            stg1_lane_kh_q <= lane_kh;
            stg1_lane_kw_q <= lane_kw;
            stg1_lane_ic_q <= lane_ic;

            unique case (state_q)
                CH_IDLE: begin
                    stg1_valid_q <= 1'b0;
                    row_valid_out_q <= 1'b0;
                    if (start_i) begin
                        window_q <= '0;
                        output_row_base_addr_q <= cfg_origin_base_i;
                        output_spatial_addr_q <= cfg_origin_base_i;
                        output_base_ih_q <= -$signed({16'd0, cfg_pad_h_i});
                        output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
                        ow_q <= '0;
                        kh_q <= '0;
                        kw_q <= '0;
                        spatial_rows_q <= '0;
                        emitted_vectors_q <= '0;
                        fetch_beats_q <= '0;
                        bypass_vectors_q <= '0;
                        k_tile_idx_q <= '0;
                        row_cache_full_q <= row_cache_full_mode;
                        cached_c_base_q <= cfg_c_base_i;
                        row_slot_valid_q <= '0;
                        ss_active_q <= ss_mode;
                        ss_fill_row_q <= '0;
                        ss_fill_x_q <= '0;
                        ss_fill_cb_q <= '0;
                        prefetch_active_q <= 1'b0;
                        prefetch_ready_q <= 1'b0;
                        prefetched_c_base_q <= '0;
                        fill_kh_q <= '0;
                        fill_x_q <= '0;
                        window_kw_q <= '0;
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
                        end else if (ss_mode) begin
                            state_q <= CH_SS_FILL_PREP;
                        end else begin
                            state_q <= CH_ENSURE;
                        end
                    end
                end

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
                            row_data_q <= unpack_row(
                                merge_beats(obi_rdata_i, '0, bypass_addr_q[BYTE_SEL_BITS-1:0], bypass_valid_bytes_q),
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
                            merge_beats(bypass_beat0_q, obi_rdata_i, bypass_addr_q[BYTE_SEL_BITS-1:0], bypass_valid_bytes_q),
                            bypass_valid_bytes_q,
                            cfg_lane_base_i
                        );
                        row_valid_out_q <= 1'b1;
                        state_q <= CH_STREAM_EMIT;
                    end
                end

                CH_SS_FILL_PREP: begin
                    if ({29'd0, ss_fill_row_q} >= 32'(cfg_input_h_i)) begin
                        kh_q <= '0;
                        kw_q <= '0;
                        state_q <= CH_STREAM_PRIME;
                    end else begin
                        logic [15:0] ss_c_base;
                        logic [5:0] ss_valid_bytes;
                        logic [31:0] ss_addr;
                        ss_c_base = {ss_fill_cb_q, {SS_CB_IDX_LSB{1'b0}}};
                        ss_valid_bytes = valid_c_bytes(cfg_input_c_i, ss_c_base, 6'd0);
                        ss_addr = cfg_origin_base_i +
                                  (32'(cfg_pad_h_i + {14'd0, ss_fill_row_q}) *
                                   cfg_row_stride_bytes_i) +
                                  (32'({12'd0, ss_fill_x_q}) * cfg_pixel_stride_bytes_i) +
                                  {16'd0, ss_c_base};
                        ss_fill_addr_q <= ss_addr;
                        ss_pending_addr_q <= beat_base(ss_addr);
                        ss_fill_valid_bytes_q <= ss_valid_bytes;
                        ss_fill_crosses_beat_q <=
                            ({2'b00, ss_addr[BYTE_SEL_BITS-1:0]} + {1'b0, ss_valid_bytes}) >
                            (BYTE_SEL_BITS+2)'(BEAT_BYTES);
                        if (ss_valid_bytes == 6'd0) begin
                            ss_cache_q[ss_fill_row_q[SS_ROW_BITS-1:0]][ss_fill_x_q][ss_fill_cb_q] <= '0;
                            if (ss_fill_cb_q == SS_CB_BITS'(SS_CBLOCKS - 1)) begin
                                ss_fill_cb_q <= '0;
                                if (({12'd0, ss_fill_x_q} + 16'd1) >= cfg_input_w_i) begin
                                    ss_fill_x_q <= '0;
                                    ss_fill_row_q <= ss_fill_row_q + 1'b1;
                                end else begin
                                    ss_fill_x_q <= ss_fill_x_q + 1'b1;
                                end
                            end else begin
                                ss_fill_cb_q <= ss_fill_cb_q + 1'b1;
                            end
                        end else begin
                            state_q <= CH_SS_FILL_REQ0;
                        end
                    end
                end

                CH_SS_FILL_REQ0: begin
                    if (obi_gnt_i) begin
                        state_q <= CH_SS_FILL_WAIT0;
                    end
                end

                CH_SS_FILL_WAIT0: begin
                    if (obi_rvalid_i) begin
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        if (ss_fill_crosses_beat_q) begin
                            ss_fill_beat0_q <= obi_rdata_i;
                            ss_pending_addr_q <= beat_base(ss_fill_addr_q) + 32'(BEAT_BYTES);
                            state_q <= CH_SS_FILL_REQ1;
                        end else begin
                            ss_cache_q[ss_fill_row_q[SS_ROW_BITS-1:0]][ss_fill_x_q][ss_fill_cb_q] <=
                                merge_beats(obi_rdata_i, '0, ss_fill_addr_q[BYTE_SEL_BITS-1:0], ss_fill_valid_bytes_q);
                            if (ss_fill_cb_q == SS_CB_BITS'(SS_CBLOCKS - 1)) begin
                                ss_fill_cb_q <= '0;
                                if (({12'd0, ss_fill_x_q} + 16'd1) >= cfg_input_w_i) begin
                                    ss_fill_x_q <= '0;
                                    ss_fill_row_q <= ss_fill_row_q + 1'b1;
                                end else begin
                                    ss_fill_x_q <= ss_fill_x_q + 1'b1;
                                end
                            end else begin
                                ss_fill_cb_q <= ss_fill_cb_q + 1'b1;
                            end
                            state_q <= CH_SS_FILL_PREP;
                        end
                    end
                end

                CH_SS_FILL_REQ1: begin
                    if (obi_gnt_i) begin
                        state_q <= CH_SS_FILL_WAIT1;
                    end
                end

                CH_SS_FILL_WAIT1: begin
                    if (obi_rvalid_i) begin
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        ss_cache_q[ss_fill_row_q[SS_ROW_BITS-1:0]][ss_fill_x_q][ss_fill_cb_q] <=
                            merge_beats(ss_fill_beat0_q, obi_rdata_i, ss_fill_addr_q[BYTE_SEL_BITS-1:0], ss_fill_valid_bytes_q);
                        if (ss_fill_cb_q == SS_CB_BITS'(SS_CBLOCKS - 1)) begin
                            ss_fill_cb_q <= '0;
                            if (({12'd0, ss_fill_x_q} + 16'd1) >= cfg_input_w_i) begin
                                ss_fill_x_q <= '0;
                                ss_fill_row_q <= ss_fill_row_q + 1'b1;
                            end else begin
                                ss_fill_x_q <= ss_fill_x_q + 1'b1;
                            end
                        end else begin
                            ss_fill_cb_q <= ss_fill_cb_q + 1'b1;
                        end
                        state_q <= CH_SS_FILL_PREP;
                    end
                end

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
                    end else if (!fill_row_in_bounds || fill_row_cached || (block_valid_bytes == 6'd0)) begin
                        fill_kh_q <= fill_kh_q + 16'd1;
                    end else begin
                        fill_x_q <= '0;
                        fill_valid_bytes_q <= block_valid_bytes;
                        fill_addr_q <= row_cache_full_q ?
                                       (cfg_origin_base_i +
                                        ((32'({16'd0, cfg_pad_h_i}) + 32'(fill_kh_q)) *
                                         cfg_row_stride_bytes_i) +
                                        {16'd0, cfg_c_base_i}) :
                                       (output_row_base_addr_q +
                                        (32'(fill_kh_q) * cfg_row_stride_bytes_i) +
                                        {16'd0, cfg_c_base_i});
                        pending_beat_addr_q <= beat_base(
                            row_cache_full_q ?
                            (cfg_origin_base_i +
                             ((32'({16'd0, cfg_pad_h_i}) + 32'(fill_kh_q)) *
                              cfg_row_stride_bytes_i) +
                             {16'd0, cfg_c_base_i}) :
                            (output_row_base_addr_q +
                             (32'(fill_kh_q) * cfg_row_stride_bytes_i) +
                             {16'd0, cfg_c_base_i})
                        );
                        state_q <= CH_FILL_REQ0;
                    end
                end

                CH_FILL_REQ0: begin
                    if (obi_gnt_i && !bf_full) begin
                        // Push beat0 metadata to FIFO
                        beat_fifo_q[bf_wptr_q].addr_lsb <= fill_addr_q[BYTE_SEL_BITS-1:0];
                        beat_fifo_q[bf_wptr_q].valid_bytes <= fill_valid_bytes_q;
                        beat_fifo_q[bf_wptr_q].kh <= row_ring_mode ? fill_row_slot : fill_kh_q;
                        beat_fifo_q[bf_wptr_q].x <= fill_x_q;
                        beat_fifo_q[bf_wptr_q].is_beat0_of_cross <= fill_crosses_current;
                        beat_fifo_q[bf_wptr_q].is_solo <= !fill_crosses_current;
                        bf_wptr_q <= bf_wptr_q + 1;
                        // Adjust count: +1 for push, -1 if pop also happens this cycle
                        if (obi_rvalid_i && !bf_empty)
                            bf_count_q <= bf_count_q; // push and pop cancel
                        else
                            bf_count_q <= bf_count_q + 1;
                        if (fill_crosses_current) begin
                            pending_beat_addr_q <= beat_base(fill_addr_q) + 32'(BEAT_BYTES);
                            state_q <= CH_FILL_REQ1;
                        end else if ((fill_x_q + 16'd1) == cfg_input_w_i) begin
                            state_q <= CH_FILL_DRAIN;
                        end else begin
                            fill_x_q <= fill_x_q + 16'd1;
                            fill_addr_q <= fill_addr_q + cfg_pixel_stride_bytes_i;
                            pending_beat_addr_q <= beat_base(fill_addr_q + cfg_pixel_stride_bytes_i);
                            // Stay in CH_FILL_REQ0
                        end
                    end
                end

                CH_FILL_REQ1: begin
                    if (obi_gnt_i && !bf_full) begin
                        // Push beat1 metadata to FIFO
                        beat_fifo_q[bf_wptr_q].addr_lsb <= fill_addr_q[BYTE_SEL_BITS-1:0];
                        beat_fifo_q[bf_wptr_q].valid_bytes <= fill_valid_bytes_q;
                        beat_fifo_q[bf_wptr_q].kh <= row_ring_mode ? fill_row_slot : fill_kh_q;
                        beat_fifo_q[bf_wptr_q].x <= fill_x_q;
                        beat_fifo_q[bf_wptr_q].is_beat0_of_cross <= 1'b0;
                        beat_fifo_q[bf_wptr_q].is_solo <= 1'b0; // this is beat1 of cross
                        bf_wptr_q <= bf_wptr_q + 1;
                        if (obi_rvalid_i && !bf_empty)
                            bf_count_q <= bf_count_q;
                        else
                            bf_count_q <= bf_count_q + 1;
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
                    // Wait for all outstanding responses to drain
                    if (bf_empty) begin
                        if (row_ring_mode && fill_row_in_bounds) begin
                            row_slot_valid_q[fill_row_slot[$clog2(K_MAX)-1:0]] <= 1'b1;
                            row_slot_ih_q[fill_row_slot[$clog2(K_MAX)-1:0]] <= fill_ih[15:0];
                        end
                        fill_kh_q <= fill_kh_q + 16'd1;
                        state_q <= CH_ENSURE;
                    end
                end

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
                            window_q[kh][window_kw_q[2:0]] <= '0;
                        end
                    end
                    if ((window_kw_q + 16'd1) == cfg_kernel_w_i) begin
                        kh_q <= '0;
                        kw_q <= '0;
                        state_q <= CH_STREAM_PRIME;
                    end else begin
                        window_kw_q <= window_kw_q + 16'd1;
                        state_q <= CH_WINDOW_REQ;
                    end
                end

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
                                    ow_q <= '0;
                                    output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
                                    output_base_ih_q <= output_base_ih_q + $signed({16'd0, cfg_stride_h_i});
                                    output_row_base_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                                    output_spatial_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                                end else begin
                                    ow_q <= ow_q + 16'd1;
                                    output_base_iw_q <= output_base_iw_q + $signed({16'd0, cfg_stride_w_i});
                                    output_spatial_addr_q <= output_spatial_addr_q + cfg_ow_step_bytes_i;
                                end
                                state_q <= CH_BYPASS_PREP;
                            end else if ((ow_q + 16'd1) == cfg_output_w_i) begin
                                stg1_valid_q <= 1'b0;
                                ow_q <= '0;
                                output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
                                output_base_ih_q <= output_base_ih_q + $signed({16'd0, cfg_stride_h_i});
                                output_row_base_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                                output_spatial_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                                fill_kh_q <= '0;
                                window_kw_q <= '0;
                                window_q <= '0;
                                state_q <= row_cache_full_q ? CH_WINDOW_REQ : CH_ENSURE;
                            end else if (has_next_same_row) begin
                                window_q <= slide_window;
                                stg1_tap_kh_q <= '0;
                                stg1_tap_kw_q <= '0;
                                ow_q <= ow_q + 16'd1;
                                output_base_iw_q <= output_base_iw_q + $signed({16'd0, cfg_stride_w_i});
                                output_spatial_addr_q <= output_spatial_addr_q + cfg_ow_step_bytes_i;
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
                                if (!(row_ring_mode && (cfg_c_base_i == cached_c_base_q))) begin
                                    row_slot_valid_q <= '0;
                                end
                                prefetch_active_q <= 1'b0;
                                prefetch_ready_q <= 1'b0;
                                if (ss_active_q) begin
                                    state_q <= CH_STREAM_PRIME;
                                end else if (row_cache_reuse ||
                                    (prefetch_ready_q && (prefetched_c_base_q == cfg_c_base_i))) begin
                                    state_q <= CH_WINDOW_REQ;
                                end else begin
                                    cached_c_base_q <= cfg_c_base_i;
                                    state_q <= CH_ENSURE;
                                end
                            end else if (prefetch_i && !prefetch_active_q && !prefetch_ready_q &&
                                         !row_cache_reuse) begin
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
                                if (!(row_ring_mode && (cfg_c_base_i == cached_c_base_q))) begin
                                    row_slot_valid_q <= '0;
                                end
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
                    state_q <= CH_IDLE;
                end
            endcase

            // -------------------------------------------------------------
            // Pipeline Stage 2: Output Update
            // -------------------------------------------------------------
            if (stg1_valid_q && stg2_ready) begin
                row_data_q <= ss_active_q ?
                              build_ss_emit_row(stg1_lane_kh_q, stg1_lane_kw_q, stg1_lane_ic_q) :
                              build_emit_row(window_q, stg1_tap_kh_q, stg1_tap_kw_q, stg1_lane_kh_q, stg1_lane_kw_q, stg1_lane_ic_q);
                row_valid_out_q <= 1'b1;
            end else if (!stg1_valid_q && stg2_ready && !bypass_active) begin
                row_valid_out_q <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
