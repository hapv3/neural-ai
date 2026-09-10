`default_nettype none

module systolic_k_tile_scheduler #(
    parameter int unsigned ARRAY_DIM = 32
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic job_start_i,
    input  logic advance_i,
    input  logic kgen_multi_i,
    input  logic c32_group_stationary_i,
    input  logic generic_linear_k32_i,
    input  logic [31:0] k_tiles_i,
    input  logic [15:0] input_c_i,
    input  logic [15:0] kernel_h_i,
    input  logic [15:0] kernel_w_i,
    input  logic [15:0] initial_seed_ic_i,
    input  logic [7:0]  initial_seed_kw_i,
    input  logic [7:0]  initial_seed_kh_i,
    input  logic [31:0] channel_addr_offset_i,

    output logic        has_next_o,
    output logic [31:0] tile_index_o,
    output logic [15:0] seed_ic_o,
    output logic [7:0]  seed_kw_o,
    output logic [7:0]  seed_kh_o,
    output logic [31:0] channel_offset_o,
    output logic [15:0] next_seed_ic_o,
    output logic [7:0]  next_seed_kw_o,
    output logic [7:0]  next_seed_kh_o,
    output logic [31:0] next_channel_offset_o
);

    logic [31:0] tile_index_q, tile_index_d;
    logic [15:0] seed_ic_q, seed_ic_d;
    logic [7:0]  seed_kw_q, seed_kw_d;
    logic [7:0]  seed_kh_q, seed_kh_d;
    logic [31:0] channel_offset_q, channel_offset_d;

    function automatic void advance_c32_group_stationary(
        input  logic [7:0]  kh_i,
        input  logic [7:0]  kw_i,
        input  logic [15:0] ic_i,
        input  logic [15:0] input_c,
        input  logic [15:0] kernel_h,
        input  logic [15:0] kernel_w,
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
            if ((kw + 8'd1) == kernel_w[7:0]) begin
                kw = '0;
                if ((kh + 8'd1) == kernel_h[7:0]) begin
                    kh = '0;
                    if ((ic + 16'(ARRAY_DIM)) >= input_c) begin
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

    function automatic void advance_generic_linear_k32(
        input  logic [7:0]  kh_i,
        input  logic [7:0]  kw_i,
        input  logic [15:0] ic_i,
        input  logic [15:0] kernel_h,
        input  logic [15:0] kernel_w,
        output logic [7:0]  kh_o,
        output logic [7:0]  kw_o,
        output logic [15:0] ic_o
    );
        begin
            kh_o = kh_i;
            kw_o = kw_i;
            ic_o = ic_i;
            if ((kw_i + 8'd1) == kernel_w[7:0]) begin
                kw_o = '0;
                if ((kh_i + 8'd1) == kernel_h[7:0]) begin
                    kh_o = '0;
                end else begin
                    kh_o = kh_i + 8'd1;
                end
            end else begin
                kw_o = kw_i + 8'd1;
            end
        end
    endfunction

    assign has_next_o = kgen_multi_i && ((tile_index_q + 32'd1) < k_tiles_i);
    assign tile_index_o = tile_index_q;
    assign seed_ic_o = seed_ic_q;
    assign seed_kw_o = seed_kw_q;
    assign seed_kh_o = seed_kh_q;
    assign channel_offset_o = channel_offset_q;

    always_comb begin
        if (c32_group_stationary_i) begin
            advance_c32_group_stationary(seed_kh_q,
                                         seed_kw_q,
                                         seed_ic_q,
                                         input_c_i,
                                         kernel_h_i,
                                         kernel_w_i,
                                         next_seed_kh_o,
                                         next_seed_kw_o,
                                         next_seed_ic_o);
        end else if (kgen_multi_i && generic_linear_k32_i) begin
            advance_generic_linear_k32(seed_kh_q,
                                       seed_kw_q,
                                       seed_ic_q,
                                       kernel_h_i,
                                       kernel_w_i,
                                       next_seed_kh_o,
                                       next_seed_kw_o,
                                       next_seed_ic_o);
        end else begin
            next_seed_kh_o = initial_seed_kh_i;
            next_seed_kw_o = initial_seed_kw_i;
            next_seed_ic_o = initial_seed_ic_i;
        end

        next_channel_offset_o = channel_offset_q;
        if (c32_group_stationary_i && (next_seed_ic_o[15:5] != seed_ic_q[15:5])) begin
            if (next_seed_ic_o[15:5] == 11'd0) begin
                next_channel_offset_o = 32'd0;
            end else begin
                next_channel_offset_o = channel_offset_q + channel_addr_offset_i;
            end
        end
    end

    always_comb begin
        tile_index_d = tile_index_q;
        seed_ic_d = seed_ic_q;
        seed_kw_d = seed_kw_q;
        seed_kh_d = seed_kh_q;
        channel_offset_d = channel_offset_q;

        if (advance_i) begin
            tile_index_d = tile_index_q + 32'd1;
            seed_ic_d = next_seed_ic_o;
            seed_kw_d = next_seed_kw_o;
            seed_kh_d = next_seed_kh_o;
            channel_offset_d = next_channel_offset_o;
        end

        if (job_start_i) begin
            tile_index_d = '0;
            seed_ic_d = initial_seed_ic_i;
            seed_kw_d = initial_seed_kw_i;
            seed_kh_d = initial_seed_kh_i;
            channel_offset_d = '0;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            tile_index_q <= '0;
            seed_ic_q <= '0;
            seed_kw_q <= '0;
            seed_kh_q <= '0;
            channel_offset_q <= '0;
        end else begin
            tile_index_q <= tile_index_d;
            seed_ic_q <= seed_ic_d;
            seed_kw_q <= seed_kw_d;
            seed_kh_q <= seed_kh_d;
            channel_offset_q <= channel_offset_d;
        end
    end

endmodule

`default_nettype wire
