`default_nettype none

module conv_channel_linebuf_packer #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned K_MAX = 5,
    parameter int unsigned MAX_INPUT_W = 640
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      start_i,
    input  logic [31:0]               dim_m_i,

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
    output logic [31:0]               emitted_vectors_o,
    output logic [31:0]               fetch_beats_o,
    output logic [31:0]               bypass_vectors_o
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BYTE_SEL_BITS = $clog2(BEAT_BYTES);
    localparam int unsigned LINE_ADDR_WIDTH = $clog2(MAX_INPUT_W);
    localparam int unsigned K_SLOT_WIDTH = $clog2(K_MAX);

    typedef enum logic [4:0] {
        CH_IDLE,
        CH_ENSURE,
        CH_FILL_REQ0,
        CH_FILL_WAIT0,
        CH_FILL_REQ1,
        CH_FILL_WAIT1,
        CH_FILL_WRITE,
        CH_EMIT_PREP,
        CH_READ_REQ,
        CH_READ_WAIT,
        CH_EMIT,
        CH_BYPASS_PREP,
        CH_BYPASS_REQ0,
        CH_BYPASS_WAIT0,
        CH_BYPASS_REQ1,
        CH_BYPASS_WAIT1,
        CH_COAL_PREP,
        CH_COAL_READ_REQ,
        CH_COAL_READ_WAIT,
        CH_COAL_EMIT
    } state_e;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;

    state_e state_q;

    logic [K_MAX-1:0] ring_req;
    logic [K_MAX-1:0] ring_we;
    logic [K_MAX-1:0][LINE_ADDR_WIDTH-1:0] ring_addr;
    logic [K_MAX-1:0][DATA_WIDTH-1:0] ring_wdata;
    logic [K_MAX-1:0][DATA_WIDTH-1:0] ring_rdata;
    logic [DATA_WIDTH/8-1:0] ring_be;

    logic [K_MAX-1:0] row_valid_q;
    logic [K_MAX-1:0][15:0] row_tag_q;
    logic [K_SLOT_WIDTH-1:0] fill_slot_q;

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
    logic bypass_active_q;

    logic [15:0] ensure_kh_q;
    logic [31:0] ensure_kh_addr_q;
    logic signed [31:0] ensure_ih;
    logic ensure_row_in_bounds;
    logic ensure_row_hit;
    logic [K_SLOT_WIDTH-1:0] ensure_hit_slot;

    logic [15:0] fill_x_q;
    logic [31:0] fill_addr_q;
    logic [K_SLOT_WIDTH-1:0] active_fill_slot_q;
    logic [15:0] active_fill_ih_q;
    logic [DATA_WIDTH-1:0] fill_beat0_q;
    logic [DATA_WIDTH-1:0] fill_data_q;
    logic [31:0] pending_beat_addr_q;
    logic [5:0] fill_valid_bytes_q;
    logic fill_crosses_beat_q;

    logic signed [31:0] emit_ih;
    logic signed [31:0] emit_iw;
    logic emit_in_bounds;
    logic emit_row_hit;
    logic [K_SLOT_WIDTH-1:0] emit_hit_slot;
    logic [5:0] emit_valid_bytes;
    logic [DATA_WIDTH-1:0] emit_read_data_q;

    logic [31:0] bypass_addr_q;
    logic [31:0] bypass_candidate_addr;
    logic [DATA_WIDTH-1:0] bypass_beat0_q;
    logic [5:0] bypass_valid_bytes_q;
    logic bypass_crosses_beat_q;
    logic bypass_in_bounds;
    logic bypass_crosses_current;

    input_row_t row_data_q;
    logic row_valid_out_q;
    logic done_q;
    logic [5:0] coalesce_lane_q;

    logic [15:0] remaining_c;
    logic [5:0] block_valid_bytes;
    logic [31:0] coalesce_k_bytes;
    logic [ARRAY_DIM-1:0][7:0] lane_kh;
    logic [ARRAY_DIM-1:0][7:0] lane_kw;
    logic [ARRAY_DIM-1:0][15:0] lane_ic;
    logic fill_crosses_current;
    logic emit_fire;
    logic last_kernel_vector;
    logic last_spatial;

    assign row_data_o = row_data_q;
    assign row_valid_o = row_valid_out_q;
    assign busy_o = state_q != CH_IDLE;
    assign done_o = done_q;
    assign emitted_vectors_o = emitted_vectors_q;
    assign fetch_beats_o = fetch_beats_q;
    assign bypass_vectors_o = bypass_vectors_q;
    assign ring_be = '1;

    for (genvar slot = 0; slot < K_MAX; slot++) begin : gen_line_ring
        tc_sram #(
            .NumWords    (MAX_INPUT_W),
            .DataWidth   (DATA_WIDTH),
            .ByteWidth   (8),
            .NumPorts    (1),
            .Latency     (1),
            .SimInit     ("none"),
            .PrintSimCfg (1'b0)
        ) i_row_sram (
            .clk_i   (clk_i),
            .rst_ni  (rst_ni),
            .req_i   (ring_req[slot]),
            .we_i    (ring_we[slot]),
            .addr_i  (ring_addr[slot]),
            .wdata_i (ring_wdata[slot]),
            .be_i    (ring_be),
            .rdata_o (ring_rdata[slot])
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

    function automatic input_row_t merge_row_lanes(
        input input_row_t old_row,
        input logic [DATA_WIDTH-1:0] data,
        input logic [5:0] valid_bytes,
        input logic [5:0] lane_base
    );
        input_row_t row;
        logic [6:0] dst_lane;
        begin
            row = old_row;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                dst_lane = {1'b0, lane_base} + 7'(lane);
                if ((lane < valid_bytes) && (dst_lane < 7'(ARRAY_DIM))) begin
                    row[dst_lane[4:0]] = data[lane * 8 +: 8];
                end
            end
            merge_row_lanes = row;
        end
    endfunction

    function automatic input_row_t merge_kgen_lanes(
        input input_row_t old_row,
        input logic [DATA_WIDTH-1:0] data,
        input logic [15:0] emit_kh_u,
        input logic [15:0] emit_kw_u,
        input logic [15:0] c_base
    );
        input_row_t row;
        logic [15:0] src_lane;
        begin
            row = old_row;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                if (({8'd0, lane_kh[lane]} == emit_kh_u) &&
                    ({8'd0, lane_kw[lane]} == emit_kw_u) &&
                    (lane_ic[lane] >= c_base) &&
                    (lane_ic[lane] < (c_base + 16'(ARRAY_DIM)))) begin
                    src_lane = lane_ic[lane] - c_base;
                    row[lane] = data[src_lane[4:0] * 8 +: 8];
                end
            end
            merge_kgen_lanes = row;
        end
    endfunction

    always_comb begin
        logic [7:0] gen_kh;
        logic [7:0] gen_kw;
        logic [15:0] gen_ic;

        remaining_c = cfg_input_c_i - cfg_c_base_i;
        block_valid_bytes = valid_c_bytes(cfg_input_c_i, cfg_c_base_i, cfg_lane_base_i);
        coalesce_k_bytes = 32'(cfg_kernel_h_i) * 32'(cfg_kernel_w_i) * 32'(block_valid_bytes);

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

        ensure_ih = output_base_ih_q + $signed({16'd0, ensure_kh_q});
        ensure_row_in_bounds = (ensure_ih >= 32'sd0) &&
                               (ensure_ih < $signed({16'd0, cfg_input_h_i}));

        ensure_row_hit = 1'b0;
        ensure_hit_slot = '0;
        for (int unsigned slot = 0; slot < K_MAX; slot++) begin
            if (row_valid_q[slot] && (row_tag_q[slot] == ensure_ih[15:0])) begin
                ensure_row_hit = 1'b1;
                ensure_hit_slot = K_SLOT_WIDTH'(slot);
            end
        end

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

        emit_ih = output_base_ih_q + $signed({16'd0, kh_q});
        emit_iw = output_base_iw_q + $signed({16'd0, kw_q});
        emit_in_bounds = (emit_ih >= 32'sd0) &&
                         (emit_iw >= 32'sd0) &&
                         (emit_ih < $signed({16'd0, cfg_input_h_i})) &&
                         (emit_iw < $signed({16'd0, cfg_input_w_i})) &&
                         (block_valid_bytes != 6'd0);

        emit_row_hit = 1'b0;
        emit_hit_slot = '0;
        for (int unsigned slot = 0; slot < K_MAX; slot++) begin
            if (row_valid_q[slot] && (row_tag_q[slot] == emit_ih[15:0])) begin
                emit_row_hit = 1'b1;
                emit_hit_slot = K_SLOT_WIDTH'(slot);
            end
        end

        emit_valid_bytes = emit_in_bounds ? block_valid_bytes : 6'd0;
        emit_fire = row_valid_out_q && row_ready_i;
        last_kernel_vector = ((kh_q + 16'd1) == cfg_kernel_h_i) &&
                             ((kw_q + 16'd1) == cfg_kernel_w_i);
        last_spatial = (spatial_rows_q + 32'd1) == dim_m_i;

        ring_req = '0;
        ring_we = '0;
        ring_addr = '0;
        ring_wdata = '0;

        if (state_q == CH_FILL_WRITE) begin
            ring_req[active_fill_slot_q] = 1'b1;
            ring_we[active_fill_slot_q] = 1'b1;
            ring_addr[active_fill_slot_q] = fill_x_q[LINE_ADDR_WIDTH-1:0];
            ring_wdata[active_fill_slot_q] = fill_data_q;
        end else if ((state_q == CH_READ_REQ || state_q == CH_COAL_READ_REQ) && emit_in_bounds && emit_row_hit) begin
            ring_req[emit_hit_slot] = 1'b1;
            ring_addr[emit_hit_slot] = emit_iw[LINE_ADDR_WIDTH-1:0];
        end

        obi_req_o = 1'b0;
        obi_addr_o = pending_beat_addr_q;
        if (state_q == CH_FILL_REQ0 || state_q == CH_FILL_REQ1 ||
            state_q == CH_BYPASS_REQ0 || state_q == CH_BYPASS_REQ1) begin
            obi_req_o = 1'b1;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= CH_IDLE;
            row_valid_q <= '0;
            row_tag_q <= '0;
            fill_slot_q <= '0;
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
            bypass_active_q <= 1'b0;
            ensure_kh_q <= '0;
            ensure_kh_addr_q <= '0;
            fill_x_q <= '0;
            fill_addr_q <= '0;
            active_fill_slot_q <= '0;
            active_fill_ih_q <= '0;
            fill_beat0_q <= '0;
            fill_data_q <= '0;
            pending_beat_addr_q <= '0;
            fill_valid_bytes_q <= '0;
            fill_crosses_beat_q <= 1'b0;
            emit_read_data_q <= '0;
            bypass_addr_q <= '0;
            bypass_beat0_q <= '0;
            bypass_valid_bytes_q <= '0;
            bypass_crosses_beat_q <= 1'b0;
            row_data_q <= '0;
            row_valid_out_q <= 1'b0;
            done_q <= 1'b0;
            coalesce_lane_q <= '0;
        end else begin
            done_q <= 1'b0;

            unique case (state_q)
                CH_IDLE: begin
                    row_valid_out_q <= 1'b0;
                    if (start_i) begin
                        row_valid_q <= '0;
                        fill_slot_q <= '0;
                        output_row_base_addr_q <= cfg_origin_base_i;
                        output_spatial_addr_q <= cfg_origin_base_i;
                        output_base_ih_q <= -$signed({16'd0, cfg_pad_h_i});
                        output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
                        ow_q <= '0;
                        kh_q <= '0;
                        kw_q <= '0;
                        spatial_rows_q <= '0;
                        coalesce_lane_q <= '0;
                        emitted_vectors_q <= '0;
                        fetch_beats_q <= '0;
                        bypass_vectors_q <= '0;
                        bypass_active_q <= (cfg_kernel_h_i == 16'd1) &&
                                           (cfg_kernel_w_i == 16'd1) &&
                                           (cfg_pad_h_i == 16'd0) &&
                                           (cfg_pad_w_i == 16'd0);
                        ensure_kh_q <= '0;
                        ensure_kh_addr_q <= '0;
                        if ((dim_m_i == 32'd0) ||
                            (cfg_output_w_i == 16'd0) ||
                            (cfg_kernel_h_i == 16'd0) ||
                            (cfg_kernel_w_i == 16'd0) ||
                            (cfg_kernel_h_i > K_MAX[15:0]) ||
                            (cfg_kernel_w_i > K_MAX[15:0]) ||
                            (cfg_input_w_i > MAX_INPUT_W[15:0]) ||
                            (cfg_stride_h_i == 16'd0) ||
                            (cfg_stride_w_i == 16'd0) ||
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

                CH_BYPASS_PREP: begin
                    if (!bypass_in_bounds) begin
                        row_data_q <= '0;
                        row_valid_out_q <= 1'b1;
                        state_q <= CH_EMIT;
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
                                merge_beats(
                                    obi_rdata_i,
                                    '0,
                                    bypass_addr_q[BYTE_SEL_BITS-1:0],
                                    bypass_valid_bytes_q
                                ),
                                bypass_valid_bytes_q,
                                cfg_lane_base_i
                            );
                            row_valid_out_q <= 1'b1;
                            state_q <= CH_EMIT;
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
                            merge_beats(
                                bypass_beat0_q,
                                obi_rdata_i,
                                bypass_addr_q[BYTE_SEL_BITS-1:0],
                                bypass_valid_bytes_q
                            ),
                            bypass_valid_bytes_q,
                            cfg_lane_base_i
                        );
                        row_valid_out_q <= 1'b1;
                        state_q <= CH_EMIT;
                    end
                end

                CH_ENSURE: begin
                    if (ensure_kh_q == cfg_kernel_h_i) begin
                        kh_q <= '0;
                        kw_q <= '0;
                        coalesce_lane_q <= cfg_lane_base_i;
                        if (cfg_coalesce_i) begin
                            row_data_q <= '0;
                            state_q <= CH_COAL_PREP;
                        end else begin
                            state_q <= CH_EMIT_PREP;
                        end
                    end else if (!ensure_row_in_bounds || ensure_row_hit || (block_valid_bytes == 6'd0)) begin
                        ensure_kh_q <= ensure_kh_q + 16'd1;
                        ensure_kh_addr_q <= ensure_kh_addr_q + cfg_row_stride_bytes_i;
                    end else begin
                        active_fill_slot_q <= fill_slot_q;
                        active_fill_ih_q <= ensure_ih[15:0];
                        fill_x_q <= '0;
                        fill_valid_bytes_q <= block_valid_bytes;
                        fill_addr_q <= output_row_base_addr_q + ensure_kh_addr_q + {16'd0, cfg_c_base_i};
                        pending_beat_addr_q <= beat_base(output_row_base_addr_q + ensure_kh_addr_q + {16'd0, cfg_c_base_i});
                        row_valid_q[fill_slot_q] <= 1'b0;
                        state_q <= CH_FILL_REQ0;
                    end
                end

                CH_FILL_REQ0: begin
                    if (obi_gnt_i) begin
                        state_q <= CH_FILL_WAIT0;
                    end
                end

                CH_FILL_WAIT0: begin
                    if (obi_rvalid_i) begin
                        fill_beat0_q <= obi_rdata_i;
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        fill_crosses_beat_q <= fill_crosses_current;
                        if (fill_crosses_current) begin
                            pending_beat_addr_q <= beat_base(fill_addr_q) + 32'(BEAT_BYTES);
                            state_q <= CH_FILL_REQ1;
                        end else begin
                            fill_data_q <= merge_beats(
                                obi_rdata_i,
                                '0,
                                fill_addr_q[BYTE_SEL_BITS-1:0],
                                fill_valid_bytes_q
                            );
                            state_q <= CH_FILL_WRITE;
                        end
                    end
                end

                CH_FILL_REQ1: begin
                    if (obi_gnt_i) begin
                        state_q <= CH_FILL_WAIT1;
                    end
                end

                CH_FILL_WAIT1: begin
                    if (obi_rvalid_i) begin
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        fill_data_q <= merge_beats(
                            fill_beat0_q,
                            obi_rdata_i,
                            fill_addr_q[BYTE_SEL_BITS-1:0],
                            fill_valid_bytes_q
                        );
                        state_q <= CH_FILL_WRITE;
                    end
                end

                CH_FILL_WRITE: begin
                    if ((fill_x_q + 16'd1) == cfg_input_w_i) begin
                        row_valid_q[active_fill_slot_q] <= 1'b1;
                        row_tag_q[active_fill_slot_q] <= active_fill_ih_q;
                        if (fill_slot_q == K_SLOT_WIDTH'(K_MAX - 1)) begin
                            fill_slot_q <= '0;
                        end else begin
                            fill_slot_q <= fill_slot_q + K_SLOT_WIDTH'(1);
                        end
                        ensure_kh_q <= ensure_kh_q + 16'd1;
                        ensure_kh_addr_q <= ensure_kh_addr_q + cfg_row_stride_bytes_i;
                        state_q <= CH_ENSURE;
                    end else begin
                        fill_x_q <= fill_x_q + 16'd1;
                        fill_addr_q <= fill_addr_q + cfg_pixel_stride_bytes_i;
                        pending_beat_addr_q <= beat_base(fill_addr_q + cfg_pixel_stride_bytes_i);
                        state_q <= CH_FILL_REQ0;
                    end
                end

                CH_EMIT_PREP: begin
                    if (!emit_in_bounds || !emit_row_hit) begin
                        row_data_q <= '0;
                        row_valid_out_q <= 1'b1;
                        state_q <= CH_EMIT;
                    end else begin
                        state_q <= CH_READ_REQ;
                    end
                end

                CH_READ_REQ: begin
                    state_q <= CH_READ_WAIT;
                end

                CH_READ_WAIT: begin
                    emit_read_data_q <= ring_rdata[emit_hit_slot];
                    row_data_q <= unpack_row(ring_rdata[emit_hit_slot], emit_valid_bytes, cfg_lane_base_i);
                    row_valid_out_q <= 1'b1;
                    state_q <= CH_EMIT;
                end

                CH_COAL_PREP: begin
                    if (!emit_in_bounds || !emit_row_hit) begin
                        if (last_kernel_vector) begin
                            row_valid_out_q <= 1'b1;
                            state_q <= CH_COAL_EMIT;
                        end else if ((kw_q + 16'd1) != cfg_kernel_w_i) begin
                            kw_q <= kw_q + 16'd1;
                            coalesce_lane_q <= coalesce_lane_q + block_valid_bytes;
                            state_q <= CH_COAL_PREP;
                        end else begin
                            kw_q <= '0;
                            kh_q <= kh_q + 16'd1;
                            coalesce_lane_q <= coalesce_lane_q + block_valid_bytes;
                            state_q <= CH_COAL_PREP;
                        end
                    end else begin
                        state_q <= CH_COAL_READ_REQ;
                    end
                end

                CH_COAL_READ_REQ: begin
                    state_q <= CH_COAL_READ_WAIT;
                end

                CH_COAL_READ_WAIT: begin
                    emit_read_data_q <= ring_rdata[emit_hit_slot];
                    if (cfg_kgen_i) begin
                        row_data_q <= merge_kgen_lanes(row_data_q,
                                                       ring_rdata[emit_hit_slot],
                                                       kh_q,
                                                       kw_q,
                                                       cfg_c_base_i);
                    end else begin
                        row_data_q <= merge_row_lanes(row_data_q, ring_rdata[emit_hit_slot], emit_valid_bytes, coalesce_lane_q);
                    end
                    if (last_kernel_vector) begin
                        row_valid_out_q <= 1'b1;
                        state_q <= CH_COAL_EMIT;
                    end else if ((kw_q + 16'd1) != cfg_kernel_w_i) begin
                        kw_q <= kw_q + 16'd1;
                        coalesce_lane_q <= coalesce_lane_q + block_valid_bytes;
                        state_q <= CH_COAL_PREP;
                    end else begin
                        kw_q <= '0;
                        kh_q <= kh_q + 16'd1;
                        coalesce_lane_q <= coalesce_lane_q + block_valid_bytes;
                        state_q <= CH_COAL_PREP;
                    end
                end

                CH_COAL_EMIT: begin
                    if (emit_fire) begin
                        row_valid_out_q <= 1'b0;
                        emitted_vectors_q <= emitted_vectors_q + 32'd1;
                        spatial_rows_q <= spatial_rows_q + 32'd1;
                        if (last_spatial) begin
                            done_q <= 1'b1;
                            state_q <= CH_IDLE;
                        end else if ((ow_q + 16'd1) == cfg_output_w_i) begin
                            ow_q <= '0;
                            kh_q <= '0;
                            kw_q <= '0;
                            coalesce_lane_q <= cfg_lane_base_i;
                            output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
                            output_base_ih_q <= output_base_ih_q + $signed({16'd0, cfg_stride_h_i});
                            output_row_base_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                            output_spatial_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                            ensure_kh_q <= '0;
                            ensure_kh_addr_q <= '0;
                            row_data_q <= '0;
                            state_q <= CH_ENSURE;
                        end else begin
                            ow_q <= ow_q + 16'd1;
                            kh_q <= '0;
                            kw_q <= '0;
                            coalesce_lane_q <= cfg_lane_base_i;
                            output_base_iw_q <= output_base_iw_q + $signed({16'd0, cfg_stride_w_i});
                            output_spatial_addr_q <= output_spatial_addr_q + cfg_ow_step_bytes_i;
                            row_data_q <= '0;
                            state_q <= CH_COAL_PREP;
                        end
                    end
                end

                CH_EMIT: begin
                    if (emit_fire) begin
                        row_valid_out_q <= 1'b0;
                        emitted_vectors_q <= emitted_vectors_q + 32'd1;
                        if (last_kernel_vector) begin
                            spatial_rows_q <= spatial_rows_q + 32'd1;
                            if (bypass_active_q) begin
                                bypass_vectors_q <= bypass_vectors_q + 32'd1;
                            end
                            if (last_spatial) begin
                                done_q <= 1'b1;
                                state_q <= CH_IDLE;
                            end else if ((ow_q + 16'd1) == cfg_output_w_i) begin
                                ow_q <= '0;
                                output_base_iw_q <= -$signed({16'd0, cfg_pad_w_i});
                                output_base_ih_q <= output_base_ih_q + $signed({16'd0, cfg_stride_h_i});
                                output_row_base_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                                output_spatial_addr_q <= output_row_base_addr_q + cfg_oh_step_bytes_i;
                                ensure_kh_q <= '0;
                                ensure_kh_addr_q <= '0;
                                if (bypass_active_q) begin
                                    state_q <= CH_BYPASS_PREP;
                                end else begin
                                    state_q <= CH_ENSURE;
                                end
                            end else begin
                                ow_q <= ow_q + 16'd1;
                                output_base_iw_q <= output_base_iw_q + $signed({16'd0, cfg_stride_w_i});
                                output_spatial_addr_q <= output_spatial_addr_q + cfg_ow_step_bytes_i;
                                kh_q <= '0;
                                kw_q <= '0;
                                if (bypass_active_q) begin
                                    state_q <= CH_BYPASS_PREP;
                                end else begin
                                    state_q <= CH_EMIT_PREP;
                                end
                            end
                        end else if ((kw_q + 16'd1) != cfg_kernel_w_i) begin
                            kw_q <= kw_q + 16'd1;
                            state_q <= CH_EMIT_PREP;
                        end else begin
                            kw_q <= '0;
                            kh_q <= kh_q + 16'd1;
                            state_q <= CH_EMIT_PREP;
                        end
                    end
                end

                default: begin
                    state_q <= CH_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
