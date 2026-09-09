`default_nettype none

module conv_linebuf_config_decoder #(
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32
)(
    input  logic [31:0] cfg_k_tiles_i,
    input  logic [31:0] cfg_row_stride_bytes_i,
    input  logic [15:0] cfg_input_h_i,
    input  logic [15:0] cfg_input_c_i,
    input  logic [15:0] cfg_kernel_h_i,
    input  logic [15:0] cfg_kernel_w_i,
    input  logic [15:0] cfg_stride_h_i,
    input  logic [15:0] cfg_stride_w_i,
    input  logic [15:0] cfg_pad_h_i,
    input  logic [15:0] cfg_c_base_i,
    input  logic [5:0]  cfg_lane_base_i,
    input  logic        cfg_coalesce_i,
    input  logic        cfg_kgen_i,
    input  logic        cfg_pool_i,
    input  logic        cfg_c32_fast_i,
    input  logic        cfg_depthwise_i,
    input  logic [5:0]  cfg_block_valid_bytes_i,
    input  logic [31:0] cfg_channel_addr_offset_i,
    input  logic [31:0] cfg_coalesce_k_bytes_i,
    input  logic [7:0]  cfg_k_seed_kh_i,
    input  logic [7:0]  cfg_k_seed_kw_i,
    input  logic [15:0] cfg_k_seed_ic_i,

    input  logic        row_cache_full_i,
    input  logic [15:0] cached_c_base_i,

    output logic [5:0]  block_valid_bytes_o,
    output logic [31:0] coalesce_k_bytes_o,
    output logic [ARRAY_DIM-1:0][7:0]  lane_kh_o,
    output logic [ARRAY_DIM-1:0][7:0]  lane_kw_o,
    output logic [ARRAY_DIM-1:0][15:0] lane_ic_o,
    output logic [15:0] effective_c_base_o,
    output logic [31:0] channel_addr_offset_o,
    output logic        c32_blocked_mode_o,
    output logic        c32_kgen_fast_o,
    output logic        row_cache_full_mode_o,
    output logic        row_cache_reuse_o,
    output logic        row_ring_mode_o,
    output logic [15:0] fill_done_rows_o,
    output logic [DATA_WIDTH-1:0] pad_vector_o,
    output logic [31:0] pad_row_offset_o
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BYTE_SEL_BITS = $clog2(BEAT_BYTES);
    localparam int unsigned K_MAX = 5;
    localparam int unsigned STRIDE_MAX = 2;

    function automatic logic [5:0] valid_c_bytes(
        input logic [15:0] input_c,
        input logic [15:0] c_base,
        input logic [5:0]  lane_base
    );
        logic [15:0] rem;
        logic [6:0] lane_room;
        begin
            lane_room = (lane_base >= 6'(ARRAY_DIM)) ? 7'd0 :
                        (7'(ARRAY_DIM) - {1'b0, lane_base});
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

    function automatic logic [31:0] scale_u32_by_0_to_5(
        input logic [31:0] value,
        input logic [15:0] factor
    );
        logic [31:0] value_x2;
        logic [31:0] value_x4;
        begin
            value_x2 = value << 1;
            value_x4 = value << 2;
            unique case (factor[2:0])
                3'd0: scale_u32_by_0_to_5 = 32'd0;
                3'd1: scale_u32_by_0_to_5 = value;
                3'd2: scale_u32_by_0_to_5 = value_x2;
                3'd3: scale_u32_by_0_to_5 = value_x2 + value;
                3'd4: scale_u32_by_0_to_5 = value_x4;
                3'd5: scale_u32_by_0_to_5 = value_x4 + value;
                default: scale_u32_by_0_to_5 = 32'd0;
            endcase
        end
    endfunction

    function automatic logic [31:0] coalesce_kernel_bytes(
        input logic [15:0] kernel_h,
        input logic [15:0] kernel_w,
        input logic [5:0] valid_bytes
    );
        logic [31:0] row_bytes;
        begin
            row_bytes = scale_u32_by_0_to_5({26'd0, valid_bytes}, kernel_w);
            coalesce_kernel_bytes = scale_u32_by_0_to_5(row_bytes, kernel_h);
        end
    endfunction

    task automatic divmod_small_q6(
        input  logic [21:0] value_i,
        input  logic [15:0] divisor_i,
        output logic [5:0]  quotient_o,
        output logic [21:0] remainder_o
    );
        logic [21:0] divisor;
        logic [21:0] remainder;
        logic [5:0] quotient;
        begin
            divisor = (divisor_i == 16'd0) ? 22'd1 : {6'd0, divisor_i};
            remainder = value_i;
            quotient = '0;

            if (remainder >= (divisor << 5)) begin
                remainder = remainder - (divisor << 5);
                quotient = quotient | 6'd32;
            end
            if (remainder >= (divisor << 4)) begin
                remainder = remainder - (divisor << 4);
                quotient = quotient | 6'd16;
            end
            if (remainder >= (divisor << 3)) begin
                remainder = remainder - (divisor << 3);
                quotient = quotient | 6'd8;
            end
            if (remainder >= (divisor << 2)) begin
                remainder = remainder - (divisor << 2);
                quotient = quotient | 6'd4;
            end
            if (remainder >= (divisor << 1)) begin
                remainder = remainder - (divisor << 1);
                quotient = quotient | 6'd2;
            end
            if (remainder >= divisor) begin
                remainder = remainder - divisor;
                quotient = quotient | 6'd1;
            end

            quotient_o = quotient;
            remainder_o = remainder;
        end
    endtask

    always_comb begin : decode_config
        logic [21:0] ic_index;
        logic [21:0] ic_remainder;
        logic [21:0] kw_index;
        logic [21:0] kw_remainder;
        logic [5:0] ic_wrap_count;
        logic [5:0] kw_wrap_count;

        ic_index = '0;
        ic_remainder = '0;
        kw_index = '0;
        kw_remainder = '0;
        ic_wrap_count = '0;
        kw_wrap_count = '0;

        effective_c_base_o = (cfg_c32_fast_i && cfg_kgen_i) ?
                             cfg_k_seed_ic_i : cfg_c_base_i;
        block_valid_bytes_o = (cfg_block_valid_bytes_i != 6'd0) ?
                              cfg_block_valid_bytes_i :
                              ((cfg_c32_fast_i && cfg_kgen_i) ?
                               valid_c_bytes(cfg_input_c_i, cfg_k_seed_ic_i,
                                             cfg_lane_base_i) :
                               valid_c_bytes(cfg_input_c_i, cfg_c_base_i,
                                             cfg_lane_base_i));
        coalesce_k_bytes_o = (cfg_coalesce_k_bytes_i != 32'd0) ?
                             cfg_coalesce_k_bytes_i :
                             coalesce_kernel_bytes(cfg_kernel_h_i,
                                                   cfg_kernel_w_i,
                                                   block_valid_bytes_o);

        if (cfg_c32_fast_i && cfg_kgen_i) begin
            channel_addr_offset_o = cfg_channel_addr_offset_i;
        end else begin
            channel_addr_offset_o = (cfg_channel_addr_offset_i != 32'd0) ?
                                    cfg_channel_addr_offset_i :
                                    {16'd0, cfg_c_base_i};
        end

        c32_blocked_mode_o = cfg_c32_fast_i &&
                             (cfg_block_valid_bytes_i == 6'(BEAT_BYTES)) &&
                             (channel_addr_offset_o[BYTE_SEL_BITS-1:0] == '0);
        c32_kgen_fast_o = c32_blocked_mode_o && cfg_coalesce_i && cfg_kgen_i &&
                          (cfg_lane_base_i == 6'd0) &&
                          (cfg_c_base_i[4:0] == 5'd0) &&
                          (cfg_k_seed_ic_i[4:0] == 5'd0);

        lane_kh_o = '0;
        lane_kw_o = '0;
        lane_ic_o = '0;
        if (c32_kgen_fast_o) begin
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                lane_kh_o[lane] = cfg_k_seed_kh_i;
                lane_kw_o[lane] = cfg_k_seed_kw_i;
                lane_ic_o[lane] = cfg_k_seed_ic_i + 16'(lane);
            end
        end else begin
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                ic_index = {6'd0, cfg_k_seed_ic_i} + 22'(lane);
                divmod_small_q6(ic_index, cfg_input_c_i,
                                ic_wrap_count, ic_remainder);

                kw_index = {14'd0, cfg_k_seed_kw_i} + {16'd0, ic_wrap_count};
                divmod_small_q6(kw_index, cfg_kernel_w_i,
                                kw_wrap_count, kw_remainder);

                lane_kh_o[lane] = cfg_k_seed_kh_i + 8'(kw_wrap_count);
                lane_kw_o[lane] = kw_remainder[7:0];
                lane_ic_o[lane] = ic_remainder[15:0];
            end
        end

        row_cache_full_mode_o = cfg_coalesce_i && cfg_kgen_i &&
                                (cfg_k_tiles_i > 32'd1) &&
                                (cfg_input_h_i <= K_MAX[15:0]);
        row_cache_reuse_o = row_cache_full_i &&
                            (effective_c_base_o == cached_c_base_i);
        row_ring_mode_o = (cfg_depthwise_i || (cfg_coalesce_i && cfg_kgen_i)) &&
                          !row_cache_full_i &&
                          (cfg_kernel_h_i <= K_MAX[15:0]) &&
                          (cfg_kernel_w_i <= K_MAX[15:0]) &&
                          (cfg_stride_h_i != 16'd0) &&
                          (cfg_stride_h_i <= STRIDE_MAX[15:0]) &&
                          (cfg_stride_w_i != 16'd0) &&
                          (cfg_stride_w_i <= STRIDE_MAX[15:0]);
        fill_done_rows_o = row_cache_full_i ? cfg_input_h_i : cfg_kernel_h_i;
        pad_vector_o = cfg_pool_i ? {BEAT_BYTES{8'h80}} : DATA_WIDTH'(0);
        pad_row_offset_o = scale_u32_by_0_to_5(cfg_row_stride_bytes_i,
                                               cfg_pad_h_i);
    end

endmodule
