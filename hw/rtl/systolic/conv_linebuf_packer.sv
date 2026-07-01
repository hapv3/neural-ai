`default_nettype none

module conv_linebuf_packer #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned CACHE_ENTRIES = 128,
    parameter int unsigned MAX_PADDED_WIDTH = 4112
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      start_i,
    input  logic [31:0]               dim_m_i,

    input  logic [31:0]               cfg_input_base_i,
    input  logic [15:0]               cfg_input_h_i,
    input  logic [15:0]               cfg_input_w_i,
    input  logic [15:0]               cfg_input_c_i,
    input  logic [15:0]               cfg_output_w_i,
    input  logic [15:0]               cfg_stride_h_i,
    input  logic [15:0]               cfg_stride_w_i,
    input  logic [15:0]               cfg_pad_h_i,
    input  logic [15:0]               cfg_pad_w_i,
    input  logic [15:0]               cfg_tile_oh_base_i,
    input  logic [15:0]               cfg_tile_ow_base_i,
    input  logic [ARRAY_DIM-1:0]      cfg_lane_valid_i,
    input  logic [ARRAY_DIM-1:0][7:0] cfg_lane_kh_i,
    input  logic [ARRAY_DIM-1:0][7:0] cfg_lane_kw_i,
    input  logic [ARRAY_DIM-1:0][15:0] cfg_lane_ic_i,

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
    output logic [31:0]               cache_hits_o,
    output logic [31:0]               cache_misses_o
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BYTE_SEL_BITS = $clog2(BEAT_BYTES);
    localparam int unsigned LINE_ADDR_WIDTH = $clog2(MAX_PADDED_WIDTH);

    typedef enum logic [1:0] {
        LB_IDLE,
        LB_SCAN,
        LB_FETCH_REQ,
        LB_FETCH_WAIT
    } state_e;

    typedef logic [23:0] rgb_pixel_t;
    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;

    state_e state_q;

    rgb_pixel_t window_q [3][3];
    rgb_pixel_t window_next [3][3];
    rgb_pixel_t line1_rdata [2];
    rgb_pixel_t line2_rdata [2];
    rgb_pixel_t line1_wdata [2];
    rgb_pixel_t line2_wdata [2];
    logic [1:0] line1_req;
    logic [1:0] line2_req;
    logic [1:0] line1_we;
    logic [1:0] line2_we;
    logic [LINE_ADDR_WIDTH-1:0] line1_addr [2];
    logic [LINE_ADDR_WIDTH-1:0] line2_addr [2];
    logic [1:0][2:0] line_be;

    logic [15:0] scan_x_q;
    logic [15:0] scan_y_q;
    logic [15:0] scan_w_q;
    logic [31:0] emitted_rows_q;
    logic [31:0] candidate_rows_q;
    logic [15:0] emit_target_x_q;
    logic [15:0] emit_target_y_q;
    logic [15:0] emit_oh_q;
    logic [15:0] emit_ow_q;
    logic [31:0] pending_beat_addr_q;
    logic        cache_fill_way_q;
    logic [31:BYTE_SEL_BITS] cache_tag_q [2];
    logic [DATA_WIDTH-1:0] cache_data_q [2];
    logic [1:0] cache_valid_q;
    logic [31:0] cache_hits_q;
    logic [31:0] cache_misses_q;
    logic        row_valid_q;
    input_row_t  row_data_q;
    logic        done_q;
    logic        pipe_valid_q;
    logic [15:0] pipe_x_q;
    logic [15:0] pipe_y_q;
    rgb_pixel_t  pipe_pixel_q;
    logic        pipe_line_hold_valid_q;
    rgb_pixel_t  pipe_line1_q;
    rgb_pixel_t  pipe_line2_q;

    logic        scan_is_input;
    logic [31:0] input_byte_addr;
    logic        cache_have_pixel;
    logic [31:0] missing_beat_addr;
    rgb_pixel_t  current_pixel;
    logic        consume_pipe;
    logic        issue_pixel;
    logic        emit_candidate;
    logic        last_candidate;
    logic        emit_window;
    logic        output_blocked;
    rgb_pixel_t  pipe_line1_data;
    rgb_pixel_t  pipe_line2_data;
    input_row_t  packed_window;

    assign row_data_o = row_data_q;
    assign row_valid_o = row_valid_q;
    assign busy_o = (state_q != LB_IDLE) || row_valid_q || pipe_valid_q;
    assign done_o = done_q;
    assign cache_hits_o = cache_hits_q;
    assign cache_misses_o = cache_misses_q;
    assign line_be = '{3'b111, 3'b111};

    tc_sram #(
        .NumWords    (MAX_PADDED_WIDTH),
        .DataWidth   (24),
        .ByteWidth   (8),
        .NumPorts    (2),
        .Latency     (1),
        .SimInit     ("none"),
        .PrintSimCfg (1'b0)
    ) i_line_prev1 (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .req_i   (line1_req),
        .we_i    (line1_we),
        .addr_i  ({line1_addr[1], line1_addr[0]}),
        .wdata_i ({line1_wdata[1], line1_wdata[0]}),
        .be_i    ({line_be[1], line_be[0]}),
        .rdata_o ({line1_rdata[1], line1_rdata[0]})
    );

    tc_sram #(
        .NumWords    (MAX_PADDED_WIDTH),
        .DataWidth   (24),
        .ByteWidth   (8),
        .NumPorts    (2),
        .Latency     (1),
        .SimInit     ("none"),
        .PrintSimCfg (1'b0)
    ) i_line_prev2 (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .req_i   (line2_req),
        .we_i    (line2_we),
        .addr_i  ({line2_addr[1], line2_addr[0]}),
        .wdata_i ({line2_wdata[1], line2_wdata[0]}),
        .be_i    ({line_be[1], line_be[0]}),
        .rdata_o ({line2_rdata[1], line2_rdata[0]})
    );

    function automatic logic [31:BYTE_SEL_BITS] beat_tag(input logic [31:0] addr);
        beat_tag = addr[31:BYTE_SEL_BITS];
    endfunction

    function automatic logic [31:0] beat_base(input logic [31:0] addr);
        beat_base = {addr[31:BYTE_SEL_BITS], {BYTE_SEL_BITS{1'b0}}};
    endfunction

    function automatic logic cache_has_addr(input logic [31:0] addr);
        logic [31:BYTE_SEL_BITS] tag;
        begin
            tag = beat_tag(addr);
            cache_has_addr = (cache_valid_q[0] && (cache_tag_q[0] == tag)) ||
                             (cache_valid_q[1] && (cache_tag_q[1] == tag));
        end
    endfunction

    function automatic logic [7:0] cache_read_byte(input logic [31:0] addr);
        logic [31:BYTE_SEL_BITS] tag;
        logic [BYTE_SEL_BITS-1:0] sel;
        begin
            tag = beat_tag(addr);
            sel = addr[BYTE_SEL_BITS-1:0];
            cache_read_byte = 8'h00;
            if (cache_valid_q[0] && (cache_tag_q[0] == tag)) begin
                cache_read_byte = cache_data_q[0][sel * 8 +: 8];
            end else if (cache_valid_q[1] && (cache_tag_q[1] == tag)) begin
                cache_read_byte = cache_data_q[1][sel * 8 +: 8];
            end
        end
    endfunction

    always_comb begin
        scan_is_input = (scan_y_q >= cfg_pad_h_i) &&
                        (scan_x_q >= cfg_pad_w_i) &&
                        ((scan_y_q - cfg_pad_h_i) < cfg_input_h_i) &&
                        ((scan_x_q - cfg_pad_w_i) < cfg_input_w_i);

        input_byte_addr = cfg_input_base_i +
                          ((({16'd0, scan_y_q - cfg_pad_h_i} * {16'd0, cfg_input_w_i}) +
                            {16'd0, scan_x_q - cfg_pad_w_i}) * {16'd0, cfg_input_c_i});

        cache_have_pixel = 1'b1;
        missing_beat_addr = beat_base(input_byte_addr);
        if (scan_is_input) begin
            if (!cache_has_addr(input_byte_addr)) begin
                cache_have_pixel = 1'b0;
                missing_beat_addr = beat_base(input_byte_addr);
            end else if ((cfg_input_c_i > 16'd1) && !cache_has_addr(input_byte_addr + 32'd1)) begin
                cache_have_pixel = 1'b0;
                missing_beat_addr = beat_base(input_byte_addr + 32'd1);
            end else if ((cfg_input_c_i > 16'd2) && !cache_has_addr(input_byte_addr + 32'd2)) begin
                cache_have_pixel = 1'b0;
                missing_beat_addr = beat_base(input_byte_addr + 32'd2);
            end
        end

        current_pixel = 24'h000000;
        if (scan_is_input && cache_have_pixel) begin
            current_pixel[7:0] = cache_read_byte(input_byte_addr);
            if (cfg_input_c_i > 16'd1) begin
                current_pixel[15:8] = cache_read_byte(input_byte_addr + 32'd1);
            end
            if (cfg_input_c_i > 16'd2) begin
                current_pixel[23:16] = cache_read_byte(input_byte_addr + 32'd2);
            end
        end

        output_blocked = row_valid_q && !row_ready_i;
        pipe_line1_data = pipe_line_hold_valid_q ? pipe_line1_q : line1_rdata[1];
        pipe_line2_data = pipe_line_hold_valid_q ? pipe_line2_q : line2_rdata[1];
        consume_pipe = pipe_valid_q && !output_blocked;
        emit_candidate = consume_pipe &&
                         (cfg_stride_h_i != 16'd0) &&
                         (cfg_stride_w_i != 16'd0) &&
                         (candidate_rows_q < dim_m_i) &&
                         (pipe_x_q == emit_target_x_q) &&
                         (pipe_y_q == emit_target_y_q);
        last_candidate = emit_candidate && ((candidate_rows_q + 32'd1) == dim_m_i);
        issue_pixel = (state_q == LB_SCAN) && (!pipe_valid_q || consume_pipe) &&
                      (candidate_rows_q < dim_m_i) && cache_have_pixel &&
                      !output_blocked && !last_candidate;

        for (int unsigned wr = 0; wr < 3; wr++) begin
            window_next[wr][0] = window_q[wr][1];
            window_next[wr][1] = window_q[wr][2];
        end
        window_next[0][2] = (pipe_y_q >= 16'd2) ? pipe_line2_data : 24'h000000;
        window_next[1][2] = (pipe_y_q >= 16'd1) ? pipe_line1_data : 24'h000000;
        window_next[2][2] = pipe_pixel_q;

        emit_window = emit_candidate &&
                      (emit_oh_q >= cfg_tile_oh_base_i) &&
                      (emit_ow_q >= cfg_tile_ow_base_i) &&
                      (emitted_rows_q < dim_m_i);

        packed_window = '0;
        for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
            if (cfg_lane_valid_i[lane] &&
                (cfg_lane_kh_i[lane] < 8'd3) &&
                (cfg_lane_kw_i[lane] < 8'd3) &&
                (cfg_lane_ic_i[lane] < 16'd3)) begin
                packed_window[lane] =
                    window_next[cfg_lane_kh_i[lane]][cfg_lane_kw_i[lane]][cfg_lane_ic_i[lane] * 8 +: 8];
            end
        end

        line1_req = '0;
        line1_we = '0;
        for (int unsigned port = 0; port < 2; port++) begin
            line1_addr[port] = '0;
            line1_wdata[port] = '0;
        end
        line2_req = '0;
        line2_we = '0;
        for (int unsigned port = 0; port < 2; port++) begin
            line2_addr[port] = '0;
            line2_wdata[port] = '0;
        end

        if (issue_pixel) begin
            line1_req[0] = 1'b1;
            line1_we[0] = 1'b1;
            line1_addr[0] = scan_x_q[LINE_ADDR_WIDTH-1:0];
            line1_wdata[0] = current_pixel;
            line1_req[1] = 1'b1;
            line1_addr[1] = scan_x_q[LINE_ADDR_WIDTH-1:0];

            line2_req[1] = 1'b1;
            line2_addr[1] = scan_x_q[LINE_ADDR_WIDTH-1:0];
        end

        if (consume_pipe) begin
            line2_req[0] = 1'b1;
            line2_we[0] = 1'b1;
            line2_addr[0] = pipe_x_q[LINE_ADDR_WIDTH-1:0];
            line2_wdata[0] = pipe_line1_data;
        end
    end

    always_comb begin
        obi_req_o = 1'b0;
        obi_addr_o = pending_beat_addr_q;
        if (state_q == LB_FETCH_REQ) begin
            obi_req_o = 1'b1;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= LB_IDLE;
            scan_x_q <= '0;
            scan_y_q <= '0;
            scan_w_q <= '0;
            emitted_rows_q <= '0;
            candidate_rows_q <= '0;
            emit_target_x_q <= 16'd2;
            emit_target_y_q <= 16'd2;
            emit_oh_q <= '0;
            emit_ow_q <= '0;
            pending_beat_addr_q <= '0;
            cache_fill_way_q <= 1'b0;
            cache_valid_q <= '0;
            cache_hits_q <= '0;
            cache_misses_q <= '0;
            row_valid_q <= 1'b0;
            row_data_q <= '0;
            done_q <= 1'b0;
            pipe_valid_q <= 1'b0;
            pipe_x_q <= '0;
            pipe_y_q <= '0;
            pipe_pixel_q <= '0;
            pipe_line_hold_valid_q <= 1'b0;
            pipe_line1_q <= '0;
            pipe_line2_q <= '0;
            for (int unsigned wy = 0; wy < 3; wy++) begin
                for (int unsigned wx = 0; wx < 3; wx++) begin
                    window_q[wy][wx] <= '0;
                end
            end
        end else begin
            done_q <= 1'b0;

            if (row_valid_q && row_ready_i) begin
                row_valid_q <= 1'b0;
            end

            if (consume_pipe) begin
                pipe_valid_q <= 1'b0;
                pipe_line_hold_valid_q <= 1'b0;
                if (emit_candidate) begin
                    candidate_rows_q <= candidate_rows_q + 32'd1;
                    if ((emit_ow_q + 16'd1) == cfg_output_w_i) begin
                        emit_ow_q <= '0;
                        emit_oh_q <= emit_oh_q + 16'd1;
                        emit_target_x_q <= 16'd2;
                        emit_target_y_q <= emit_target_y_q + cfg_stride_h_i;
                    end else begin
                        emit_ow_q <= emit_ow_q + 16'd1;
                        emit_target_x_q <= emit_target_x_q + cfg_stride_w_i;
                    end
                end
                for (int unsigned wy = 0; wy < 3; wy++) begin
                    for (int unsigned wx = 0; wx < 3; wx++) begin
                        window_q[wy][wx] <= window_next[wy][wx];
                    end
                end
                if (emit_window) begin
                    row_data_q <= packed_window;
                    row_valid_q <= 1'b1;
                    emitted_rows_q <= emitted_rows_q + 32'd1;
                end
            end else if (pipe_valid_q && !pipe_line_hold_valid_q) begin
                pipe_line_hold_valid_q <= 1'b1;
                pipe_line1_q <= line1_rdata[1];
                pipe_line2_q <= line2_rdata[1];
            end

            if (issue_pixel) begin
                pipe_valid_q <= 1'b1;
                pipe_line_hold_valid_q <= 1'b0;
                pipe_x_q <= scan_x_q;
                pipe_y_q <= scan_y_q;
                pipe_pixel_q <= current_pixel;
                if (scan_is_input) begin
                    cache_hits_q <= cache_hits_q + 32'd1;
                end
                if ((scan_x_q + 16'd1) == scan_w_q) begin
                    scan_x_q <= '0;
                    scan_y_q <= scan_y_q + 16'd1;
                end else begin
                    scan_x_q <= scan_x_q + 16'd1;
                end
            end

            unique case (state_q)
                LB_IDLE: begin
                    if (start_i) begin
                        scan_x_q <= '0;
                        scan_y_q <= '0;
                        emitted_rows_q <= '0;
                        candidate_rows_q <= '0;
                        emit_target_x_q <= 16'd2;
                        emit_target_y_q <= 16'd2;
                        emit_oh_q <= '0;
                        emit_ow_q <= '0;
                        scan_w_q <= (cfg_output_w_i == 16'd0) ?
                                     16'd0 :
                                     ((cfg_output_w_i - 16'd1) * cfg_stride_w_i) + 16'd3;
                        cache_valid_q <= '0;
                        cache_hits_q <= '0;
                        cache_misses_q <= '0;
                        row_valid_q <= 1'b0;
                        row_data_q <= '0;
                        pipe_valid_q <= 1'b0;
                        pipe_line_hold_valid_q <= 1'b0;
                        for (int unsigned wy = 0; wy < 3; wy++) begin
                            for (int unsigned wx = 0; wx < 3; wx++) begin
                                window_q[wy][wx] <= '0;
                            end
                        end
                        if ((dim_m_i == 32'd0) || (cfg_output_w_i == 16'd0) ||
                            (cfg_stride_h_i == 16'd0) || (cfg_stride_w_i == 16'd0)) begin
                            done_q <= 1'b1;
                            state_q <= LB_IDLE;
                        end else begin
                            state_q <= LB_SCAN;
                        end
                    end
                end

                LB_SCAN: begin
                    if (!row_valid_q || row_ready_i) begin
                        if ((candidate_rows_q == dim_m_i) && !pipe_valid_q) begin
                            done_q <= 1'b1;
                            state_q <= LB_IDLE;
                        end else if ((candidate_rows_q < dim_m_i) && !cache_have_pixel &&
                                     (!pipe_valid_q || consume_pipe)) begin
                            pending_beat_addr_q <= missing_beat_addr;
                            cache_misses_q <= cache_misses_q + 32'd1;
                            state_q <= LB_FETCH_REQ;
                        end
                    end
                end

                LB_FETCH_REQ: begin
                    if (obi_gnt_i) begin
                        state_q <= LB_FETCH_WAIT;
                    end
                end

                LB_FETCH_WAIT: begin
                    if (obi_rvalid_i) begin
                        cache_valid_q[cache_fill_way_q] <= 1'b1;
                        cache_tag_q[cache_fill_way_q] <= pending_beat_addr_q[31:BYTE_SEL_BITS];
                        cache_data_q[cache_fill_way_q] <= obi_rdata_i;
                        cache_fill_way_q <= ~cache_fill_way_q;
                        state_q <= LB_SCAN;
                    end
                end

                default: begin
                    state_q <= LB_IDLE;
                end
            endcase
        end
    end

endmodule
