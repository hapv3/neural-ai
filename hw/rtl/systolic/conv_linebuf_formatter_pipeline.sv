`default_nettype none

module conv_linebuf_formatter_pipeline #(
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned K_MAX = 5
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic flush_i,
    input  logic advance_i,

    input  logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_i,
    input  logic [ARRAY_DIM-1:0][7:0] lane_kh_i,
    input  logic [ARRAY_DIM-1:0][7:0] lane_kw_i,
    input  logic [ARRAY_DIM-1:0][15:0] lane_ic_i,
    input  logic [15:0] tap_kh_i,
    input  logic [15:0] tap_kw_i,
    input  logic [15:0] kernel_h_i,
    input  logic [15:0] kernel_w_i,
    input  logic [15:0] c_base_i,
    input  logic [15:0] input_c_i,
    input  logic [5:0]  lane_base_i,
    input  logic [5:0]  block_valid_bytes_i,
    input  logic        coalesce_i,
    input  logic        kgen_i,
    input  logic        c32_kgen_fast_i,
    input  logic        valid_i,

    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] row_o,
    output logic valid_o,
    output logic empty_o
);

    typedef logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_t;
    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;

    typedef struct packed {
        logic       valid;
        logic [2:0] kh;
        logic [2:0] kw;
        logic [4:0] src_lane;
    } lane_desc_t;

    // S1 snapshots the window and format controls. S2 builds lane descriptors,
    // S3 selects one tap vector per lane, and S4 packs the output bytes.
    window_t s1_window_q;
    logic [ARRAY_DIM-1:0][7:0] s1_lane_kh_q;
    logic [ARRAY_DIM-1:0][7:0] s1_lane_kw_q;
    logic [ARRAY_DIM-1:0][15:0] s1_lane_ic_q;
    logic [15:0] s1_tap_kh_q;
    logic [15:0] s1_tap_kw_q;
    logic [15:0] s1_kernel_h_q;
    logic [15:0] s1_kernel_w_q;
    logic [15:0] s1_c_base_q;
    logic [15:0] s1_input_c_q;
    logic [5:0]  s1_lane_base_q;
    logic [5:0]  s1_block_valid_bytes_q;
    logic        s1_coalesce_q;
    logic        s1_kgen_q;
    logic        s1_c32_kgen_fast_q;
    logic        s1_valid_q;

    window_t s2_window_q;
    lane_desc_t s2_desc_q [ARRAY_DIM];
    logic s2_valid_q;

    logic [ARRAY_DIM-1:0][DATA_WIDTH-1:0] s3_tap_vec_q;
    lane_desc_t s3_desc_q [ARRAY_DIM];
    logic s3_valid_q;

    input_row_t s4_row_q;
    logic s4_valid_q;

    lane_desc_t desc_next [ARRAY_DIM];
    logic [ARRAY_DIM-1:0][DATA_WIDTH-1:0] tap_vec_next;
    input_row_t row_next;

    task automatic build_lane_desc(
        input logic [15:0] tap_kh,
        input logic [15:0] tap_kw,
        input logic [ARRAY_DIM-1:0][7:0] lane_kh,
        input logic [ARRAY_DIM-1:0][7:0] lane_kw,
        input logic [ARRAY_DIM-1:0][15:0] lane_ic,
        input logic [15:0] kernel_h,
        input logic [15:0] kernel_w,
        input logic [15:0] c_base,
        input logic [15:0] input_c,
        input logic [5:0] lane_base,
        input logic [5:0] valid_bytes,
        input logic coalesce,
        input logic kgen,
        input logic c32_fast,
        output lane_desc_t desc_o [ARRAY_DIM]
    );
        logic [6:0] dst_lane;
        logic [15:0] src_lane;
        logic [15:0] dst_count;
        begin
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                desc_o[lane] = '0;
            end

            if (c32_fast) begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    desc_o[lane].valid = 1'b1;
                    desc_o[lane].kh = lane_kh[lane][2:0];
                    desc_o[lane].kw = lane_kw[lane][2:0];
                    desc_o[lane].src_lane = 5'(lane);
                end
            end else if (coalesce && kgen) begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    if (({8'd0, lane_kh[lane]} < kernel_h) &&
                        ({8'd0, lane_kw[lane]} < kernel_w) &&
                        (lane_ic[lane] >= c_base) &&
                        (lane_ic[lane] < (c_base + 16'(ARRAY_DIM))) &&
                        (lane_ic[lane] < input_c)) begin
                        src_lane = lane_ic[lane] - c_base;
                        desc_o[lane].valid = 1'b1;
                        desc_o[lane].kh = lane_kh[lane][2:0];
                        desc_o[lane].kw = lane_kw[lane][2:0];
                        desc_o[lane].src_lane = src_lane[4:0];
                    end
                end
            end else if (coalesce) begin
                dst_count = {10'd0, lane_base};
                for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                    for (int unsigned kw = 0; kw < K_MAX; kw++) begin
                        for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                            if ((kh < kernel_h) &&
                                (kw < kernel_w) &&
                                (lane < valid_bytes) &&
                                (dst_count < 16'(ARRAY_DIM))) begin
                                desc_o[dst_count[4:0]].valid = 1'b1;
                                desc_o[dst_count[4:0]].kh = 3'(kh);
                                desc_o[dst_count[4:0]].kw = 3'(kw);
                                desc_o[dst_count[4:0]].src_lane = 5'(lane);
                                dst_count = dst_count + 16'd1;
                            end
                        end
                    end
                end
            end else begin
                for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                    dst_lane = {1'b0, lane_base} + 7'(lane);
                    if ((lane < valid_bytes) && (dst_lane < 7'(ARRAY_DIM))) begin
                        desc_o[dst_lane[4:0]].valid = 1'b1;
                        desc_o[dst_lane[4:0]].kh = tap_kh[2:0];
                        desc_o[dst_lane[4:0]].kw = tap_kw[2:0];
                        desc_o[dst_lane[4:0]].src_lane = 5'(lane);
                    end
                end
            end
        end
    endtask

    always_comb begin
        build_lane_desc(s1_tap_kh_q,
                        s1_tap_kw_q,
                        s1_lane_kh_q,
                        s1_lane_kw_q,
                        s1_lane_ic_q,
                        s1_kernel_h_q,
                        s1_kernel_w_q,
                        s1_c_base_q,
                        s1_input_c_q,
                        s1_lane_base_q,
                        s1_block_valid_bytes_q,
                        s1_coalesce_q,
                        s1_kgen_q,
                        s1_c32_kgen_fast_q,
                        desc_next);

        tap_vec_next = '0;
        for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
            if (s2_desc_q[lane].valid) begin
                tap_vec_next[lane] = s2_window_q[s2_desc_q[lane].kh][s2_desc_q[lane].kw];
            end
        end

        row_next = '0;
        for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
            if (s3_desc_q[lane].valid) begin
                row_next[lane] =
                    s3_tap_vec_q[lane][{s3_desc_q[lane].src_lane, 3'b000} +: INPUT_ELEM_WIDTH];
            end
        end
    end

    assign row_o = s4_row_q;
    assign valid_o = s4_valid_q;
    assign empty_o = !(s1_valid_q || s2_valid_q || s3_valid_q || s4_valid_q);

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            s1_window_q <= '0;
            s1_lane_kh_q <= '0;
            s1_lane_kw_q <= '0;
            s1_lane_ic_q <= '0;
            s1_tap_kh_q <= '0;
            s1_tap_kw_q <= '0;
            s1_kernel_h_q <= '0;
            s1_kernel_w_q <= '0;
            s1_c_base_q <= '0;
            s1_input_c_q <= '0;
            s1_lane_base_q <= '0;
            s1_block_valid_bytes_q <= '0;
            s1_coalesce_q <= 1'b0;
            s1_kgen_q <= 1'b0;
            s1_c32_kgen_fast_q <= 1'b0;
            s1_valid_q <= 1'b0;
            s2_window_q <= '0;
            s2_valid_q <= 1'b0;
            s3_tap_vec_q <= '0;
            s3_valid_q <= 1'b0;
            s4_row_q <= '0;
            s4_valid_q <= 1'b0;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                s2_desc_q[lane] <= '0;
                s3_desc_q[lane] <= '0;
            end
        end else if (flush_i) begin
            s1_valid_q <= 1'b0;
            s2_valid_q <= 1'b0;
            s3_valid_q <= 1'b0;
            s4_valid_q <= 1'b0;
        end else if (advance_i) begin
            s4_row_q <= row_next;
            s4_valid_q <= s3_valid_q;

            s3_tap_vec_q <= tap_vec_next;
            s3_valid_q <= s2_valid_q;

            s2_window_q <= s1_window_q;
            s2_valid_q <= s1_valid_q;

            s1_window_q <= window_i;
            s1_lane_kh_q <= lane_kh_i;
            s1_lane_kw_q <= lane_kw_i;
            s1_lane_ic_q <= lane_ic_i;
            s1_tap_kh_q <= tap_kh_i;
            s1_tap_kw_q <= tap_kw_i;
            s1_kernel_h_q <= kernel_h_i;
            s1_kernel_w_q <= kernel_w_i;
            s1_c_base_q <= c_base_i;
            s1_input_c_q <= input_c_i;
            s1_lane_base_q <= lane_base_i;
            s1_block_valid_bytes_q <= block_valid_bytes_i;
            s1_coalesce_q <= coalesce_i;
            s1_kgen_q <= kgen_i;
            s1_c32_kgen_fast_q <= c32_kgen_fast_i;
            s1_valid_q <= valid_i;

            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                s2_desc_q[lane] <= desc_next[lane];
                s3_desc_q[lane] <= s2_desc_q[lane];
            end
        end
    end

endmodule

`default_nettype wire
