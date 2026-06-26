`default_nettype none

module conv2d_feeder #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned K_TILE     = 32,
    parameter int unsigned CACHE_ENTRIES = 128,
    parameter int unsigned CACHE_BANKS   = 4
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic [31:0] cfg_input_ptr_i,
    input  logic [31:0] cfg_output_ptr_i,
    input  logic [31:0] cfg_rows_i,
    input  logic [31:0] cfg_output_w_i,
    input  logic [31:0] cfg_input_h_i,
    input  logic [31:0] cfg_input_w_i,
    input  logic [31:0] cfg_input_c_i,
    input  logic [15:0] cfg_kernel_h_i,
    input  logic [15:0] cfg_kernel_w_i,
    input  logic [15:0] cfg_stride_h_i,
    input  logic [15:0] cfg_stride_w_i,
    input  logic [15:0] cfg_pad_h_i,
    input  logic [15:0] cfg_pad_w_i,
    input  logic [15:0] cfg_dilation_h_i,
    input  logic [15:0] cfg_dilation_w_i,
    input  logic [31:0] cfg_k_base_i,
    input  logic        cfg_stream_en_i,
    input  logic        start_i,
    output logic        done_o,
    output logic        busy_o,

    output logic                      stream_ifm_valid_o,
    input  logic                      stream_ifm_ready_i,
    output logic [K_TILE-1:0][7:0]    stream_ifm_data_o,

    output logic                      obi_req_o,
    input  logic                      obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_addr_o,
    output logic                      obi_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_be_o,
    output logic [DATA_WIDTH-1:0]     obi_wdata_o,
    input  logic                      obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_rdata_i
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BEAT_OFF_W = $clog2(BEAT_BYTES);
    localparam int unsigned LANE_WIDTH = $clog2(K_TILE);
    localparam logic [LANE_WIDTH-1:0] LANE_LAST = LANE_WIDTH'(K_TILE - 1);
    localparam int unsigned CACHE_LINES_PER_BANK = CACHE_ENTRIES / CACHE_BANKS;
    localparam int unsigned CACHE_BANK_W = $clog2(CACHE_BANKS);
    localparam int unsigned CACHE_INDEX_W = $clog2(CACHE_LINES_PER_BANK);
    localparam int unsigned CACHE_LINE_W = ADDR_WIDTH - BEAT_OFF_W;
    localparam int unsigned CACHE_TAG_W = CACHE_LINE_W - CACHE_BANK_W - CACHE_INDEX_W;

    typedef enum logic [4:0] {
        IDLE,
        MUL_ROW_STRIDE,
        MUL_PAD_H,
        MUL_PAD_W,
        MUL_STRIDE_H,
        MUL_STRIDE_W,
        MUL_DILATION_H,
        MUL_DILATION_W,
        MUL_KERNEL_TAIL_W,
        DECODE_K_BASE,
        CACHE_FLUSH_REQ,
        CACHE_FLUSH_WAIT,
        PREP_ROW,
        FILL_LANE,
        CACHE_LOOKUP,
        READ_REQ,
        READ_WAIT,
        WRITE_ROW,
        DONE
    } state_e;

    typedef struct packed {
        logic               valid;
        logic [31:0]        kh;
        logic [31:0]        kw;
        logic [31:0]        ic;
        logic signed [31:0] h_pos;
        logic signed [31:0] w_pos;
        logic [31:0]        addr;
    } cursor_t;

    state_e state_q;

    logic [31:0] row_count_q;
    logic [31:0] ow_count_q;
    logic signed [31:0] h_origin_q;
    logic signed [31:0] w_origin_q;
    logic [31:0] pixel_row_base_q;
    logic [31:0] pixel_base_q;
    logic [31:0] output_addr_q;

    logic [31:0] row_stride_q;
    logic [31:0] pad_h_row_q;
    logic [31:0] pad_w_c_q;
    logic [31:0] stride_h_row_q;
    logic [31:0] stride_w_c_q;
    logic [31:0] dilation_h_row_q;
    logic [31:0] dilation_w_c_q;
    logic [31:0] kernel_tail_w_q;

    logic [31:0] mul_acc_q;
    logic [31:0] mul_count_q;
    logic [31:0] mul_addend_q;
    logic [31:0] decode_count_q;

    cursor_t start_cursor_q;
    cursor_t lane_cursor_q;

    logic [LANE_WIDTH-1:0] lane_q;
    logic [DATA_WIDTH-1:0] row_buf_q;

    logic [CACHE_LINE_W-1:0] lane_cache_line;
    logic cache_flush_req;
    logic cache_flush_done;
    logic cache_lookup_req;
    logic cache_lookup_valid;
    logic cache_lookup_hit;
    logic [7:0] cache_lookup_byte;
    logic cache_fill_req;

    logic lane_in_bounds;
    logic lane_cache_hit;
    logic [BEAT_OFF_W-1:0] lane_byte_offset;
    logic [7:0] cache_byte;
    logic [7:0] read_byte;
    logic signed [32:0] lane_h_ext;
    logic signed [32:0] lane_w_ext;
    logic signed [32:0] input_h_ext;
    logic signed [32:0] input_w_ext;
    cursor_t decode_next;
    cursor_t lane_next;

    function automatic cursor_t advance_cursor(
        input cursor_t              cur,
        input logic signed [31:0]   reset_w
    );
        cursor_t nxt;
        logic [31:0] input_c_minus_one;
        logic [31:0] next_kw_delta;
        logic [31:0] next_kh_delta;
        begin
            nxt = cur;
            input_c_minus_one = (cfg_input_c_i == 32'd0) ? 32'd0 : (cfg_input_c_i - 32'd1);
            next_kw_delta = dilation_w_c_q - input_c_minus_one;
            next_kh_delta = dilation_h_row_q - kernel_tail_w_q - input_c_minus_one;

            if (cur.valid) begin
                if ((cur.ic + 32'd1) < cfg_input_c_i) begin
                    nxt.ic = cur.ic + 32'd1;
                    nxt.addr = cur.addr + 32'd1;
                end else begin
                    nxt.ic = 32'd0;
                    if ((cur.kw + 32'd1) < 32'(cfg_kernel_w_i)) begin
                        nxt.kw = cur.kw + 32'd1;
                        nxt.w_pos = cur.w_pos + $signed({16'd0, cfg_dilation_w_i});
                        nxt.addr = cur.addr + next_kw_delta;
                    end else begin
                        nxt.kw = 32'd0;
                        nxt.w_pos = reset_w;
                        if ((cur.kh + 32'd1) < 32'(cfg_kernel_h_i)) begin
                            nxt.kh = cur.kh + 32'd1;
                            nxt.h_pos = cur.h_pos + $signed({16'd0, cfg_dilation_h_i});
                            nxt.addr = cur.addr + next_kh_delta;
                        end else begin
                            nxt.valid = 1'b0;
                        end
                    end
                end
            end
            return nxt;
        end
    endfunction

    assign done_o = (state_q == DONE);
    assign busy_o = (state_q != IDLE) && (state_q != DONE);

    assign lane_h_ext = $signed({lane_cursor_q.h_pos[31], lane_cursor_q.h_pos});
    assign lane_w_ext = $signed({lane_cursor_q.w_pos[31], lane_cursor_q.w_pos});
    assign input_h_ext = $signed({1'b0, cfg_input_h_i});
    assign input_w_ext = $signed({1'b0, cfg_input_w_i});
    assign lane_in_bounds = lane_cursor_q.valid &&
                            (lane_h_ext >= 33'sd0) &&
                            (lane_w_ext >= 33'sd0) &&
                            (lane_h_ext < input_h_ext) &&
                            (lane_w_ext < input_w_ext);
    assign lane_byte_offset = lane_cursor_q.addr[BEAT_OFF_W-1:0];
    assign lane_cache_line = lane_cursor_q.addr[ADDR_WIDTH-1:BEAT_OFF_W];
    assign lane_cache_hit = cache_lookup_valid && cache_lookup_hit;
    assign cache_byte = cache_lookup_byte;
    assign read_byte = obi_rdata_i[lane_byte_offset * 8 +: 8];
    assign decode_next = advance_cursor(start_cursor_q, 32'sd0);
    assign lane_next = advance_cursor(lane_cursor_q, w_origin_q);
    assign cache_fill_req = (state_q == READ_WAIT) && obi_rvalid_i;

    conv2d_feeder_cache #(
        .ADDR_WIDTH    (ADDR_WIDTH),
        .DATA_WIDTH    (DATA_WIDTH),
        .CACHE_ENTRIES (CACHE_ENTRIES),
        .CACHE_BANKS   (CACHE_BANKS)
    ) i_feeder_cache (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .flush_req_i          (cache_flush_req),
        .flush_done_o         (cache_flush_done),
        .lookup_req_i         (cache_lookup_req),
        .lookup_line_i        (lane_cache_line),
        .lookup_byte_offset_i (lane_byte_offset),
        .lookup_valid_o       (cache_lookup_valid),
        .lookup_hit_o         (cache_lookup_hit),
        .lookup_byte_o        (cache_lookup_byte),
        .fill_req_i           (cache_fill_req),
        .fill_line_i          (lane_cache_line),
        .fill_data_i          (obi_rdata_i)
    );

    always_comb begin
        obi_req_o = 1'b0;
        obi_addr_o = '0;
        obi_we_o = 1'b0;
        obi_be_o = '1;
        obi_wdata_o = row_buf_q;
        stream_ifm_valid_o = 1'b0;
        stream_ifm_data_o = row_buf_q;
        cache_flush_req = 1'b0;
        cache_lookup_req = 1'b0;

        unique case (state_q)
            CACHE_FLUSH_REQ: begin
                cache_flush_req = 1'b1;
            end

            FILL_LANE: begin
                cache_lookup_req = lane_in_bounds;
            end

            READ_REQ: begin
                obi_req_o = 1'b1;
                obi_addr_o = {lane_cursor_q.addr[ADDR_WIDTH-1:BEAT_OFF_W],
                              {BEAT_OFF_W{1'b0}}};
                obi_we_o = 1'b0;
            end

            WRITE_ROW: begin
                if (cfg_stream_en_i) begin
                    stream_ifm_valid_o = 1'b1;
                end else begin
                    obi_req_o = 1'b1;
                    obi_addr_o = output_addr_q;
                    obi_we_o = 1'b1;
                end
            end

            default: begin
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
            row_count_q <= '0;
            ow_count_q <= '0;
            h_origin_q <= '0;
            w_origin_q <= '0;
            pixel_row_base_q <= '0;
            pixel_base_q <= '0;
            output_addr_q <= '0;
            row_stride_q <= '0;
            pad_h_row_q <= '0;
            pad_w_c_q <= '0;
            stride_h_row_q <= '0;
            stride_w_c_q <= '0;
            dilation_h_row_q <= '0;
            dilation_w_c_q <= '0;
            kernel_tail_w_q <= '0;
            mul_acc_q <= '0;
            mul_count_q <= '0;
            mul_addend_q <= '0;
            decode_count_q <= '0;
            start_cursor_q <= '0;
            lane_cursor_q <= '0;
            lane_q <= '0;
            row_buf_q <= '0;
        end else begin
            unique case (state_q)
                IDLE: begin
                    if (start_i) begin
                        row_count_q <= '0;
                        ow_count_q <= '0;
                        row_buf_q <= '0;
                        lane_q <= '0;
                        if ((cfg_rows_i == 32'd0) || (cfg_output_w_i == 32'd0)) begin
                            state_q <= DONE;
                        end else begin
                            state_q <= CACHE_FLUSH_REQ;
                        end
                    end
                end

                CACHE_FLUSH_REQ: begin
                    state_q <= CACHE_FLUSH_WAIT;
                end

                CACHE_FLUSH_WAIT: begin
                    if (cache_flush_done) begin
                        mul_acc_q <= 32'd0;
                        mul_count_q <= cfg_input_w_i;
                        mul_addend_q <= cfg_input_c_i;
                        state_q <= MUL_ROW_STRIDE;
                    end
                end

                MUL_ROW_STRIDE: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        row_stride_q <= mul_acc_q;
                        mul_acc_q <= 32'd0;
                        mul_count_q <= 32'(cfg_pad_h_i);
                        mul_addend_q <= mul_acc_q;
                        state_q <= MUL_PAD_H;
                    end
                end

                MUL_PAD_H: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        pad_h_row_q <= mul_acc_q;
                        mul_acc_q <= 32'd0;
                        mul_count_q <= 32'(cfg_pad_w_i);
                        mul_addend_q <= cfg_input_c_i;
                        state_q <= MUL_PAD_W;
                    end
                end

                MUL_PAD_W: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        pad_w_c_q <= mul_acc_q;
                        mul_acc_q <= 32'd0;
                        mul_count_q <= 32'(cfg_stride_h_i);
                        mul_addend_q <= row_stride_q;
                        state_q <= MUL_STRIDE_H;
                    end
                end

                MUL_STRIDE_H: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        stride_h_row_q <= mul_acc_q;
                        mul_acc_q <= 32'd0;
                        mul_count_q <= 32'(cfg_stride_w_i);
                        mul_addend_q <= cfg_input_c_i;
                        state_q <= MUL_STRIDE_W;
                    end
                end

                MUL_STRIDE_W: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        stride_w_c_q <= mul_acc_q;
                        mul_acc_q <= 32'd0;
                        mul_count_q <= 32'(cfg_dilation_h_i);
                        mul_addend_q <= row_stride_q;
                        state_q <= MUL_DILATION_H;
                    end
                end

                MUL_DILATION_H: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        dilation_h_row_q <= mul_acc_q;
                        mul_acc_q <= 32'd0;
                        mul_count_q <= 32'(cfg_dilation_w_i);
                        mul_addend_q <= cfg_input_c_i;
                        state_q <= MUL_DILATION_W;
                    end
                end

                MUL_DILATION_W: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        dilation_w_c_q <= mul_acc_q;
                        mul_acc_q <= 32'd0;
                        mul_count_q <= (cfg_kernel_w_i == 16'd0) ? 32'd0 : (32'(cfg_kernel_w_i) - 32'd1);
                        mul_addend_q <= mul_acc_q;
                        state_q <= MUL_KERNEL_TAIL_W;
                    end
                end

                MUL_KERNEL_TAIL_W: begin
                    if (mul_count_q != 32'd0) begin
                        mul_acc_q <= mul_acc_q + mul_addend_q;
                        mul_count_q <= mul_count_q - 32'd1;
                    end else begin
                        kernel_tail_w_q <= mul_acc_q;
                        decode_count_q <= cfg_k_base_i;
                        start_cursor_q.valid <= (cfg_input_c_i != 32'd0) &&
                                                (cfg_kernel_h_i != 16'd0) &&
                                                (cfg_kernel_w_i != 16'd0);
                        start_cursor_q.kh <= 32'd0;
                        start_cursor_q.kw <= 32'd0;
                        start_cursor_q.ic <= 32'd0;
                        start_cursor_q.h_pos <= 32'sd0;
                        start_cursor_q.w_pos <= 32'sd0;
                        start_cursor_q.addr <= 32'd0;
                        state_q <= DECODE_K_BASE;
                    end
                end

                DECODE_K_BASE: begin
                    if ((decode_count_q != 32'd0) && start_cursor_q.valid) begin
                        start_cursor_q <= decode_next;
                        decode_count_q <= decode_count_q - 32'd1;
                    end else begin
                        decode_count_q <= 32'd0;
                        h_origin_q <= -$signed({16'd0, cfg_pad_h_i});
                        w_origin_q <= -$signed({16'd0, cfg_pad_w_i});
                        pixel_row_base_q <= cfg_input_ptr_i - pad_h_row_q - pad_w_c_q;
                        pixel_base_q <= cfg_input_ptr_i - pad_h_row_q - pad_w_c_q;
                        output_addr_q <= cfg_output_ptr_i;
                        row_count_q <= 32'd0;
                        ow_count_q <= 32'd0;
                        state_q <= PREP_ROW;
                    end
                end

                PREP_ROW: begin
                    lane_q <= '0;
                    row_buf_q <= '0;
                    lane_cursor_q.valid <= start_cursor_q.valid;
                    lane_cursor_q.kh <= start_cursor_q.kh;
                    lane_cursor_q.kw <= start_cursor_q.kw;
                    lane_cursor_q.ic <= start_cursor_q.ic;
                    lane_cursor_q.h_pos <= h_origin_q + start_cursor_q.h_pos;
                    lane_cursor_q.w_pos <= w_origin_q + start_cursor_q.w_pos;
                    lane_cursor_q.addr <= pixel_base_q + start_cursor_q.addr;
                    state_q <= FILL_LANE;
                end

                FILL_LANE: begin
                    if (!lane_in_bounds) begin
                        row_buf_q[lane_q * 8 +: 8] <= 8'd0;
                        if (lane_q == LANE_LAST) begin
                            state_q <= WRITE_ROW;
                        end else begin
                            lane_q <= lane_q + 1'b1;
                            lane_cursor_q <= lane_next;
                        end
                    end else begin
                        state_q <= CACHE_LOOKUP;
                    end
                end

                CACHE_LOOKUP: begin
                    if (lane_cache_hit) begin
                        row_buf_q[lane_q * 8 +: 8] <= cache_byte;
                        if (lane_q == LANE_LAST) begin
                            state_q <= WRITE_ROW;
                        end else begin
                            lane_q <= lane_q + 1'b1;
                            lane_cursor_q <= lane_next;
                        end
                    end else if (cache_lookup_valid) begin
                        state_q <= READ_REQ;
                    end
                end

                READ_REQ: begin
                    if (obi_gnt_i) begin
                        state_q <= READ_WAIT;
                    end
                end

                READ_WAIT: begin
                    if (obi_rvalid_i) begin
                        row_buf_q[lane_q * 8 +: 8] <= read_byte;
                        if (lane_q == LANE_LAST) begin
                            state_q <= WRITE_ROW;
                        end else begin
                            lane_q <= lane_q + 1'b1;
                            lane_cursor_q <= lane_next;
                            state_q <= FILL_LANE;
                        end
                    end
                end

                WRITE_ROW: begin
                    if ((cfg_stream_en_i && stream_ifm_ready_i) ||
                        (!cfg_stream_en_i && obi_gnt_i)) begin
                        if (row_count_q == (cfg_rows_i - 32'd1)) begin
                            state_q <= DONE;
                        end else begin
                            row_count_q <= row_count_q + 32'd1;
                            output_addr_q <= output_addr_q + 32'd32;
                            if (ow_count_q == (cfg_output_w_i - 32'd1)) begin
                                ow_count_q <= 32'd0;
                                h_origin_q <= h_origin_q + $signed({16'd0, cfg_stride_h_i});
                                w_origin_q <= -$signed({16'd0, cfg_pad_w_i});
                                pixel_row_base_q <= pixel_row_base_q + stride_h_row_q;
                                pixel_base_q <= pixel_row_base_q + stride_h_row_q;
                            end else begin
                                ow_count_q <= ow_count_q + 32'd1;
                                w_origin_q <= w_origin_q + $signed({16'd0, cfg_stride_w_i});
                                pixel_base_q <= pixel_base_q + stride_w_c_q;
                            end
                            state_q <= PREP_ROW;
                        end
                    end
                end

                DONE: begin
                    state_q <= IDLE;
                end

                default: begin
                    state_q <= IDLE;
                end
            endcase
        end
    end

endmodule
