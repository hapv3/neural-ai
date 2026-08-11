// Copyright (c) 2026
// AFU Core - Shift-and-OR parallel processing with SRAM LUT Pipeline

module afu_core #(
    parameter int unsigned LUT_LANES = 4
)(
    input  logic         clk_i,
    input  logic         rst_ni,

    // CSRs
    input  logic [31:0]  cfg_src_ptr_i,
    input  logic [31:0]  cfg_src2_ptr_i,
    input  logic [31:0]  cfg_dst_ptr_i,
    input  logic [31:0]  cfg_length_i,
    input  logic [2:0]   cfg_mode_i,
    input  logic         cfg_start_i,

    // LUT write interface
    input  logic         lut_we_i,
    input  logic [7:0]   lut_addr_i,
    input  logic [31:0]  lut_wdata_i,
    input  logic [3:0]   lut_be_i,
    input  logic         lut_fixed_bank_i,
    input  logic         lut_bank_i,

    // Read FIFO
    input  logic         rfifo_empty_i,
    output logic         rfifo_pop_o,
    input  logic [255:0] rfifo_data_i,

    // RHS Read FIFO (used by binary modes)
    input  logic         rhs_rfifo_empty_i,
    output logic         rhs_rfifo_pop_o,
    input  logic [255:0] rhs_rfifo_data_i,

    // Write FIFO
    input  logic         wfifo_full_i,
    output logic         wfifo_push_o,
    output logic [287:0] wfifo_data_o,

    output logic         done_o,
    output logic         busy_o
);

    localparam logic [2:0] MODE_8BIT  = 3'd0;
    localparam logic [2:0] MODE_16BIT = 3'd1;
    localparam logic [2:0] MODE_32BIT = 3'd2;
    localparam logic [2:0] MODE_MUL_Q7 = 3'd3;
    localparam logic [2:0] MODE_ADD_I8 = 3'd4;
    localparam logic [2:0] MODE_DFL4_ROW32_Q8 = 3'd5;
    localparam logic [2:0] MODE_CLASS_SIGMOID_ROW32_HIGH16 = 3'd6;
    localparam logic [2:0] MODE_GLOBAL_AVGPOOL_C32 = 3'd7;

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_READ_IN,
        ST_PROCESS,
        ST_WAIT_FLUSH,
        ST_DFL_EXP_REQ,
        ST_DFL_EXP_WAIT,
        ST_DFL_RECIP_REQ,
        ST_DFL_RECIP_WAIT,
        ST_DFL_MUL_WRITE,
        ST_DFL_WRITE,
        ST_DFL_PUSH,
        ST_CLASS_LUT_REQ,
        ST_CLASS_LUT_WAIT,
        ST_CLASS_PUSH,
        ST_GAP_ACCUM,
        ST_GAP_RECIP_REQ,
        ST_GAP_RECIP_WAIT,
        ST_GAP_MUL_WRITE,
        ST_GAP_WRITE,
        ST_GAP_PUSH,
        ST_DONE
    } state_e;

    state_e state_q, state_n;

    assign done_o = (state_q == ST_DONE);
    assign busy_o = (state_q != ST_IDLE) && (state_q != ST_DONE);

    logic [31:0] src_addr_q, src_addr_n;
    logic [31:0] rhs_addr_q, rhs_addr_n;
    logic [31:0] dst_addr_q, dst_addr_n;
    logic [31:0] elem_cnt_q, elem_cnt_n;
    logic [31:0] out_base_q, out_base_n;

    logic [255:0] in_buf_q,  in_buf_n;
    logic [255:0] rhs_buf_q, rhs_buf_n;
    logic [255:0] out_buf_q, out_buf_n;
    logic [31:0]  out_be_q,  out_be_n;

    // Pipeline Registers
    logic       p1_valid_q, p1_valid_n;
    logic [5:0] p1_num_valid_lanes_q, p1_num_valid_lanes_n;
    logic [255:0] p1_lhs_data_q, p1_lhs_data_n;
    logic [255:0] p1_rhs_data_q, p1_rhs_data_n;
    logic [31:0] p1_src_addr_q, p1_src_addr_n;
    logic [31:0] p1_rhs_addr_q, p1_rhs_addr_n;
    logic [31:0] p1_dst_addr_q, p1_dst_addr_n;
    logic       p1_flush_mid_q, p1_flush_mid_n;
    logic       p1_flush_done_q, p1_flush_done_n;

    logic       mul_p2_valid_q, mul_p2_valid_n;
    logic [5:0] mul_p2_num_valid_lanes_q, mul_p2_num_valid_lanes_n;
    logic signed [15:0] mul_p2_product_q [32];
    logic signed [15:0] mul_p2_product_n [32];
    logic [31:0] mul_p2_dst_addr_q, mul_p2_dst_addr_n;
    logic        mul_p2_flush_mid_q, mul_p2_flush_mid_n;
    logic        mul_p2_flush_done_q, mul_p2_flush_done_n;

    logic       mul_p3_valid_q, mul_p3_valid_n;
    logic [5:0] mul_p3_num_valid_lanes_q, mul_p3_num_valid_lanes_n;
    logic signed [15:0] mul_p3_shifted_q [32];
    logic signed [15:0] mul_p3_shifted_n [32];
    logic [31:0] mul_p3_dst_addr_q, mul_p3_dst_addr_n;
    logic        mul_p3_flush_mid_q, mul_p3_flush_mid_n;
    logic        mul_p3_flush_done_q, mul_p3_flush_done_n;

    logic s2_stall;
    logic s2_flush_mid_completed;
    logic s2_flush_done_completed;

    logic [31:0] remaining_elems;
    logic [5:0]  in_avail;
    logic [5:0]  out_avail_bytes;
    logic [5:0]  out_avail_elems;
    logic [5:0]  rhs_avail;
    logic [5:0]  max_lanes_1, max_lanes_2, max_lanes_3, max_lanes_4;
    logic [5:0]  num_valid_lanes;
    logic        mul_p3_will_pop;
    logic        mul_p2_can_advance;

    assign mul_p3_will_pop = mul_p3_valid_q && !wfifo_full_i;
    assign mul_p2_can_advance = mul_p2_valid_q && (!mul_p3_valid_q || mul_p3_will_pop);
    assign s2_stall = (cfg_mode_i == MODE_MUL_Q7) ?
                      (p1_valid_q && mul_p2_valid_q && !mul_p2_can_advance) :
                      (wfifo_full_i && p1_valid_q &&
                       ((cfg_mode_i == MODE_ADD_I8) ||
                        p1_flush_mid_q || (p1_flush_done_q && out_be_q != 0)));

    // SRAM LUT Instances
    logic s1_sram_req;
    logic [7:0] lut_idx_s1 [LUT_LANES];
    logic [31:0] lut_rdata_ports [LUT_LANES];
    logic [31:0] lut_rdata_bank0 [LUT_LANES];
    logic [31:0] lut_rdata_bank1 [LUT_LANES];
    logic [31:0] lut_rdata_dummy_bank0 [LUT_LANES];
    logic [31:0] lut_rdata_dummy_bank1 [LUT_LANES];
    logic        active_lut_bank_q;
    logic        stage_lut_bank_q;
    logic        lut_pending_q;
    logic        dfl_exp_req;
    logic        dfl_recip_req;
    logic        gap_recip_req;
    logic        class_lut_req;
    logic [7:0]  dfl_exp_idx [LUT_LANES];
    logic [7:0]  dfl_recip_idx;
    logic [7:0]  class_lut_idx [LUT_LANES];
    // DFL4 uses this as the side index. DFL16 uses it as the four-bin group
    // index within one 16-bin ROW32 record.
    logic [1:0]  dfl_side_q, dfl_side_n;
    logic [1:0]  class_group_q, class_group_n;
    logic [19:0] dfl_sum_q, dfl_sum_n;
    logic [22:0] dfl_weighted_q, dfl_weighted_n;
    logic [4:0]  dfl_shift_q, dfl_shift_n;
    logic signed [7:0] dfl_s1_max_value;
    logic [19:0] dfl_s1_sum_value;
    logic [22:0] dfl_s1_weighted_value;
    logic [31:0] dfl_s1_next_elem_cnt;
    logic [31:0] dfl_s1_next_dst_addr;
    logic        dfl_s1_flush_output;
    logic        dfl_s1_final_location;
    logic [15:0] dfl_s2_out_value;
    logic [4:0]  dfl_s2_out_off;
    logic        dfl_s2_final_location;
    logic [31:0] dfl_s2_next_elem_cnt;
    logic [31:0] dfl_s2_next_dst_addr;
    logic        dfl_s2_flush_output;
    logic [15:0] dfl_round_value_q, dfl_round_value_n;
    logic        class_s1_final_group;
    logic [31:0] class_s1_next_elem_cnt;
    logic [31:0] class_s1_next_dst_addr;
    logic        class_s1_flush_output;
    logic        class_s2_final_group;
    logic [31:0] class_s2_next_elem_cnt;
    logic [31:0] class_s2_next_dst_addr;
    logic        class_s2_flush_output;
    logic signed [31:0] gap_acc_q [32];
    logic signed [31:0] gap_acc_n [32];
    logic [31:0] gap_row_count_q, gap_row_count_n;
    logic [31:0] gap_spatial_count;
    logic [31:0] gap_next_elem_cnt;
    logic        gap_group_done;
    logic        gap_final_input;
    logic [63:0] gap_mul_product_q [32];
    logic [63:0] gap_mul_product_n [32];
    logic        gap_mul_negative_q [32];
    logic        gap_mul_negative_n [32];
    logic signed [31:0] gap_avg_q [32];
    logic signed [31:0] gap_avg_n [32];
    logic [62:0] dfl_mul_product_q, dfl_mul_product_n;

    localparam int unsigned GAP_RECIP_SHIFT = 31;

    assign gap_spatial_count = cfg_src2_ptr_i;
    // Preserve the deployed DFL4 ABI (SRC2=0). SRC2=16 selects one 16-bin
    // distribution per ROW32 record while reusing the same AFU mode.
    logic dfl16_mode;
    assign dfl16_mode = cfg_mode_i == MODE_DFL4_ROW32_Q8 && cfg_src2_ptr_i == 32'd16;
    assign gap_next_elem_cnt = elem_cnt_q + 32'd32;
    assign gap_group_done = ((gap_row_count_q + 32'd1) >= gap_spatial_count);
    assign gap_final_input = (elem_cnt_q >= cfg_length_i);

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            active_lut_bank_q <= 1'b0;
            stage_lut_bank_q <= 1'b1;
            lut_pending_q <= 1'b0;
        end else begin
            if (lut_we_i) begin
                lut_pending_q <= 1'b1;
            end
            if (cfg_start_i && lut_pending_q && cfg_mode_i != MODE_DFL4_ROW32_Q8) begin
                active_lut_bank_q <= stage_lut_bank_q;
                stage_lut_bank_q <= ~stage_lut_bank_q;
                lut_pending_q <= 1'b0;
            end
        end
    end

    // Read byte extraction for S1.  Keep this as per-lane byte muxing instead
    // of shifting the full 256-bit read beat by a dynamic byte offset.
    logic [4:0] in_off_s1;
    logic [5:0] lut_byte_idx_s1 [LUT_LANES];
    assign in_off_s1 = src_addr_q[4:0];

    function automatic logic [7:0] select_input_byte(
        input logic [255:0] data,
        input logic [5:0] byte_idx
    );
        logic [7:0] selected;
        begin
            selected = 8'h00;
            for (int unsigned b = 0; b < 32; b++) begin
                if (byte_idx == 6'(b)) begin
                    selected = data[b*8 +: 8];
                end
            end
            select_input_byte = selected;
        end
    endfunction

    function automatic logic signed [8:0] clamp_i8(
        input logic signed [15:0] value
    );
        begin
            if (value > 16'sd127) begin
                clamp_i8 = 9'sd127;
            end else if (value < -16'sd128) begin
                clamp_i8 = -9'sd128;
            end else begin
                clamp_i8 = value[8:0];
            end
        end
    endfunction

    function automatic logic signed [15:0] mul_q7_product(
        input logic [7:0] lhs_u8,
        input logic [7:0] rhs_u8
    );
        logic signed [7:0] lhs_i8;
        logic signed [7:0] rhs_i8;
        begin
            lhs_i8 = $signed(lhs_u8);
            rhs_i8 = $signed(rhs_u8);
            mul_q7_product = lhs_i8 * rhs_i8;
        end
    endfunction

    function automatic logic signed [15:0] mul_q7_shifted(
        input logic signed [15:0] product_i16
    );
        begin
            mul_q7_shifted = product_i16 >>> 7;
        end
    endfunction

    function automatic logic [7:0] i8_byte_from_i16(
        input logic signed [15:0] value
    );
        logic signed [8:0] clamped_i9;
        begin
            clamped_i9 = clamp_i8(value);
            i8_byte_from_i16 = clamped_i9[7:0];
        end
    endfunction

    function automatic logic [7:0] add_i8_byte(
        input logic [7:0] lhs_u8,
        input logic [7:0] rhs_u8
    );
        logic signed [7:0] lhs_i8;
        logic signed [7:0] rhs_i8;
        logic signed [15:0] sum_i16;
        logic signed [8:0] clamped_i9;
        begin
            lhs_i8 = $signed(lhs_u8);
            rhs_i8 = $signed(rhs_u8);
            sum_i16 = lhs_i8 + rhs_i8;
            clamped_i9 = clamp_i8(sum_i16);
            add_i8_byte = clamped_i9[7:0];
        end
    endfunction

    function automatic logic signed [31:0] avg_i32_from_product(
        input logic [63:0] product,
        input logic        negative,
        input logic [31:0] abs_sum,
        input logic [31:0] count
    );
        logic [31:0] quotient;
        logic [31:0] product_back;
        logic        correction;
        begin
            quotient = 32'(product >> GAP_RECIP_SHIFT);
            product_back = quotient * count;
            correction = (count != 32'd0) && ((product_back + count) <= abs_sum);
            quotient = quotient + {31'd0, correction};
            if (negative) begin
                avg_i32_from_product = -$signed({1'b0, quotient[30:0]});
            end else begin
                avg_i32_from_product = $signed({1'b0, quotient[30:0]});
            end
        end
    endfunction

    function automatic logic [7:0] select_dfl_byte(
        input logic [255:0] data,
        input logic [1:0] side,
        input logic [1:0] bin
    );
        logic [5:0] byte_idx;
        begin
            byte_idx = {2'd0, side, 2'b00} + {4'd0, bin};
            select_dfl_byte = data[{byte_idx[4:0], 3'b000} +: 8];
        end
    endfunction

    function automatic logic [7:0] select_dfl16_byte(
        input logic [255:0] data,
        input logic [1:0] group,
        input logic [1:0] bin
    );
        logic [4:0] byte_idx;
        begin
            byte_idx = {group, 2'b00} + {3'd0, bin};
            select_dfl16_byte = data[{byte_idx, 3'b000} +: 8];
        end
    endfunction

    function automatic logic signed [7:0] max4_i8(
        input logic [7:0] a,
        input logic [7:0] b,
        input logic [7:0] c,
        input logic [7:0] d
    );
        logic signed [7:0] ma;
        begin
            ma = $signed(a);
            if ($signed(b) > ma) ma = $signed(b);
            if ($signed(c) > ma) ma = $signed(c);
            if ($signed(d) > ma) ma = $signed(d);
            max4_i8 = ma;
        end
    endfunction

    function automatic logic signed [7:0] max2_i8(
        input logic [7:0] a,
        input logic [7:0] b
    );
        begin
            max2_i8 = ($signed(a) > $signed(b)) ? $signed(a) : $signed(b);
        end
    endfunction

    function automatic logic signed [7:0] max16_i8(input logic [255:0] data);
        logic signed [7:0] max01, max23, max45, max67;
        logic signed [7:0] max89, maxab, maxcd, maxef;
        logic signed [7:0] max03, max47, max8b, maxcf;
        logic signed [7:0] max07, max8f;
        begin
            // Explicit balanced tree: four comparator levels, independent of
            // synthesis-loop unrolling heuristics.
            max01 = max2_i8(data[7:0],     data[15:8]);
            max23 = max2_i8(data[23:16],   data[31:24]);
            max45 = max2_i8(data[39:32],   data[47:40]);
            max67 = max2_i8(data[55:48],   data[63:56]);
            max89 = max2_i8(data[71:64],   data[79:72]);
            maxab = max2_i8(data[87:80],   data[95:88]);
            maxcd = max2_i8(data[103:96],  data[111:104]);
            maxef = max2_i8(data[119:112], data[127:120]);
            max03 = max2_i8(max01, max23);
            max47 = max2_i8(max45, max67);
            max8b = max2_i8(max89, maxab);
            maxcf = max2_i8(maxcd, maxef);
            max07 = max2_i8(max03, max47);
            max8f = max2_i8(max8b, maxcf);
            max16_i8 = max2_i8(max07, max8f);
        end
    endfunction

    function automatic logic [4:0] msb_pos20(input logic [19:0] value);
        logic [4:0] pos;
        begin
            pos = 5'd0;
            for (int i = 0; i < 20; i++) begin
                if (value[i]) pos = 5'(i);
            end
            msb_pos20 = pos;
        end
    endfunction

    function automatic logic [7:0] recip_index_from_sum(input logic [19:0] sum);
        logic [4:0]  shift;
        logic [27:0] shifted;
        logic [8:0]  norm_q8;
        begin
            shift = msb_pos20(sum);
            shifted = {8'd0, sum} << 8;
            norm_q8 = shifted >> shift;
            if (norm_q8 < 9'd256) begin
                recip_index_from_sum = 8'd0;
            end else if (norm_q8 > 9'd511) begin
                recip_index_from_sum = 8'd255;
            end else begin
                recip_index_from_sum = norm_q8[7:0];
            end
        end
    endfunction

    function automatic logic [15:0] dfl_q8_from_product(
        input logic [62:0] product,
        input logic [4:0]  sum_shift
    );
        logic [62:0] rounded;
        logic [5:0]  total_shift;
        logic [62:0] round_add;
        begin
            total_shift = 6'd28 + {1'b0, sum_shift};
            round_add = 63'd1 << (total_shift - 6'd1);
            rounded = (product + round_add) >> total_shift;
            if (rounded > 63'hFFFF) begin
                dfl_q8_from_product = 16'hFFFF;
            end else begin
                dfl_q8_from_product = rounded[15:0];
            end
        end
    endfunction

    generate
        for (genvar i = 0; i < LUT_LANES; i++) begin : gen_lut_sram
            assign lut_byte_idx_s1[i] = {1'b0, in_off_s1} + 6'(i);
            assign lut_idx_s1[i] = select_input_byte(in_buf_q, lut_byte_idx_s1[i]);

            assign lut_rdata_ports[i] = active_lut_bank_q ? lut_rdata_bank1[i] : lut_rdata_bank0[i];

            tc_sram #(
                .NumWords  (256),
                .DataWidth (32),
                .NumPorts  (2),
                .Latency   (1)
            ) i_lut_sram_bank0 (
                .clk_i   (clk_i),
                .rst_ni  (rst_ni),
                .req_i   ({(dfl_exp_req && i < 4) ||
                           (gap_recip_req && i == 0 && !active_lut_bank_q) ||
                           (class_lut_req && !active_lut_bank_q) ||
                           (s1_sram_req && !active_lut_bank_q),
                           lut_we_i && ((lut_fixed_bank_i && !lut_bank_i) ||
                                        (!lut_fixed_bank_i && !stage_lut_bank_q))}),
                .we_i    ({1'b0,        1'b1}),
                .addr_i  ({dfl_exp_req ? dfl_exp_idx[i] :
                            (gap_recip_req ? 8'd0 :
                             (class_lut_req ? class_lut_idx[i] : lut_idx_s1[i])), lut_addr_i}),
                .wdata_i ({32'd0,       lut_wdata_i}),
                .be_i    ({4'b1111,     lut_be_i}),
                .rdata_o ({lut_rdata_bank0[i], lut_rdata_dummy_bank0[i]})
            );

            tc_sram #(
                .NumWords  (256),
                .DataWidth (32),
                .NumPorts  (2),
                .Latency   (1)
            ) i_lut_sram_bank1 (
                .clk_i   (clk_i),
                .rst_ni  (rst_ni),
                .req_i   ({(dfl_recip_req && i == 0) ||
                           (gap_recip_req && i == 0 && active_lut_bank_q) ||
                           (class_lut_req && active_lut_bank_q) ||
                           (s1_sram_req && active_lut_bank_q),
                           lut_we_i && ((lut_fixed_bank_i && lut_bank_i) ||
                                        (!lut_fixed_bank_i && stage_lut_bank_q))}),
                .we_i    ({1'b0,        1'b1}),
                .addr_i  ({dfl_recip_req && i == 0 ? dfl_recip_idx :
                            (gap_recip_req ? 8'd0 :
                             (class_lut_req ? class_lut_idx[i] : lut_idx_s1[i])), lut_addr_i}),
                .wdata_i ({32'd0,       lut_wdata_i}),
                .be_i    ({4'b1111,     lut_be_i}),
                .rdata_o ({lut_rdata_bank1[i], lut_rdata_dummy_bank1[i]})
            );
        end
    endgenerate

    // removed duplicate wfifo_data_o assignment

    state_e stream_state_n;
    logic [31:0] stream_src_addr_n, stream_rhs_addr_n, stream_dst_addr_n, stream_elem_cnt_n;
    logic [255:0] stream_in_buf_n, stream_rhs_buf_n;
    logic stream_rfifo_pop, stream_rhs_rfifo_pop, stream_s1_sram_req;
    logic stream_p1_valid_n, stream_p1_flush_mid_n, stream_p1_flush_done_n;
    logic [5:0] stream_p1_num_valid_lanes_n;
    logic [255:0] stream_p1_lhs_data_n, stream_p1_rhs_data_n;
    logic [31:0] stream_p1_src_addr_n, stream_p1_rhs_addr_n, stream_p1_dst_addr_n;

    state_e dfl_state_n;
    logic [31:0] dfl_elem_cnt_n, dfl_dst_addr_n;
    logic [1:0]  dfl_side_next;
    logic [19:0] dfl_sum_next;
    logic [22:0] dfl_weighted_next;
    logic [4:0]  dfl_shift_next;
    logic [62:0] dfl_mul_product_next;
    logic [15:0] dfl_round_value_next;
    logic        dfl_exp_req_next, dfl_recip_req_next;
    logic [7:0]  dfl_recip_idx_next;
    logic [7:0]  dfl_exp_idx_next [LUT_LANES];

    state_e class_state_n;
    logic [31:0] class_elem_cnt_n, class_dst_addr_n;
    logic [1:0]  class_group_next;
    logic        class_lut_req_next;
    logic [7:0]  class_lut_idx_next [LUT_LANES];

    state_e gap_state_n;
    logic [31:0] gap_elem_cnt_n, gap_row_count_next;
    logic [255:0] gap_in_buf_next;
    logic         gap_rfifo_pop_next;
    logic signed [31:0] gap_acc_next [32];
    logic [63:0] gap_mul_product_next [32];
    logic        gap_mul_negative_next [32];
    logic signed [31:0] gap_avg_next [32];
    logic        gap_recip_req_next;

    // Stream/process path: generic LUT, ADD_I8 and MUL_Q7 input scheduling.
    always_comb begin
        stream_state_n = state_q;
        stream_src_addr_n = src_addr_q;
        stream_rhs_addr_n = rhs_addr_q;
        stream_dst_addr_n = dst_addr_q;
        stream_elem_cnt_n = elem_cnt_q;
        stream_in_buf_n = in_buf_q;
        stream_rhs_buf_n = rhs_buf_q;
        stream_rfifo_pop = 1'b0;
        stream_rhs_rfifo_pop = 1'b0;
        stream_s1_sram_req = 1'b0;

        stream_p1_valid_n = 1'b0;
        stream_p1_flush_mid_n = 1'b0;
        stream_p1_flush_done_n = 1'b0;
        stream_p1_num_valid_lanes_n = '0;
        stream_p1_lhs_data_n = p1_lhs_data_q;
        stream_p1_rhs_data_n = p1_rhs_data_q;
        stream_p1_src_addr_n = src_addr_q;
        stream_p1_rhs_addr_n = rhs_addr_q;
        stream_p1_dst_addr_n = dst_addr_q;

        remaining_elems = '0;
        in_avail = '0;
        out_avail_bytes = '0;
        out_avail_elems = '0;
        rhs_avail = '0;
        max_lanes_1 = '0;
        max_lanes_2 = '0;
        max_lanes_3 = '0;
        max_lanes_4 = '0;
        num_valid_lanes = '0;

        unique case (state_q)
            ST_IDLE: begin
                // waiting for start
            end

            ST_READ_IN: begin
                if (cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                    if (!rfifo_empty_i && !rhs_rfifo_empty_i) begin
                        stream_in_buf_n = rfifo_data_i;
                        stream_rhs_buf_n = rhs_rfifo_data_i;
                        stream_rfifo_pop = 1'b1;
                        stream_rhs_rfifo_pop = 1'b1;
                        stream_state_n = ST_PROCESS;
                    end
                end else if (!rfifo_empty_i) begin
                    stream_in_buf_n = rfifo_data_i;
                    stream_rfifo_pop = 1'b1;
                    if (cfg_mode_i == MODE_DFL4_ROW32_Q8) begin
                        stream_state_n = ST_DFL_EXP_REQ;
                    end else if (cfg_mode_i == MODE_CLASS_SIGMOID_ROW32_HIGH16) begin
                        stream_state_n = ST_CLASS_LUT_REQ;
                    end else if (cfg_mode_i == MODE_GLOBAL_AVGPOOL_C32) begin
                        stream_state_n = (gap_spatial_count == 32'd0) ? ST_DONE : ST_GAP_ACCUM;
                    end else begin
                        stream_state_n = ST_PROCESS;
                    end
                end
            end

            ST_PROCESS: begin
                remaining_elems = cfg_length_i - elem_cnt_q;
                stream_p1_src_addr_n = src_addr_q;
                stream_p1_rhs_addr_n = rhs_addr_q;
                stream_p1_dst_addr_n = dst_addr_q;

                if (remaining_elems == 0) begin
                    stream_state_n = ST_DONE;
                end else if (!s2_stall) begin
                    in_avail = 6'd32 - {1'b0, src_addr_q[4:0]};
                    rhs_avail = 6'd32 - {1'b0, rhs_addr_q[4:0]};
                    out_avail_bytes = 6'd32 - {1'b0, dst_addr_q[4:0]};

                    unique case (cfg_mode_i)
                        MODE_8BIT:  out_avail_elems = out_avail_bytes;
                        MODE_16BIT: out_avail_elems = {1'b0, out_avail_bytes[5:1]};
                        MODE_32BIT: out_avail_elems = {2'b0, out_avail_bytes[5:2]};
                        MODE_MUL_Q7: out_avail_elems = out_avail_bytes;
                        MODE_ADD_I8: out_avail_elems = out_avail_bytes;
                        default:    out_avail_elems = out_avail_bytes;
                    endcase

                    if (cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                        max_lanes_1 = (remaining_elems > 32) ? 6'd32 : 6'(remaining_elems);
                    end else begin
                        max_lanes_1 = (LUT_LANES < remaining_elems) ? LUT_LANES[5:0] :
                                      (remaining_elems > 6'd31 ? 6'd31 : 6'(remaining_elems));
                    end
                    max_lanes_2 = (max_lanes_1 < in_avail) ? max_lanes_1 : in_avail;
                    max_lanes_3 = ((cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) &&
                                   rhs_avail < max_lanes_2) ? rhs_avail : max_lanes_2;
                    max_lanes_4 = (max_lanes_3 < out_avail_elems) ? max_lanes_3 : out_avail_elems;
                    num_valid_lanes = max_lanes_4;

                    if (num_valid_lanes > 0) begin
                        stream_s1_sram_req = (cfg_mode_i != MODE_MUL_Q7 && cfg_mode_i != MODE_ADD_I8);
                        stream_p1_num_valid_lanes_n = num_valid_lanes;
                        stream_p1_valid_n = 1'b1;
                        stream_p1_lhs_data_n = in_buf_q;
                        stream_p1_rhs_data_n = rhs_buf_q;

                        stream_src_addr_n = src_addr_q + 32'(num_valid_lanes);
                        stream_rhs_addr_n = rhs_addr_q + 32'(num_valid_lanes);
                        stream_elem_cnt_n = elem_cnt_q + 32'(num_valid_lanes);

                        if (cfg_mode_i == MODE_8BIT || cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                            stream_dst_addr_n = dst_addr_q + 32'(num_valid_lanes);
                        end else if (cfg_mode_i == MODE_16BIT) begin
                            stream_dst_addr_n = dst_addr_q + 32'(num_valid_lanes * 2);
                        end else begin
                            stream_dst_addr_n = dst_addr_q + 32'(num_valid_lanes * 4);
                        end

                        if (cfg_mode_i == MODE_MUL_Q7) begin
                            if (stream_elem_cnt_n == cfg_length_i) begin
                                stream_p1_flush_done_n = 1'b1;
                                stream_state_n = ST_WAIT_FLUSH;
                            end else begin
                                if (stream_dst_addr_n[4:0] == 5'd0) begin
                                    stream_p1_flush_mid_n = 1'b1;
                                end
                                if ((stream_src_addr_n[4:0] == 5'd0) || (stream_rhs_addr_n[4:0] == 5'd0)) begin
                                    stream_state_n = ST_READ_IN;
                                end else begin
                                    stream_state_n = ST_PROCESS;
                                end
                            end
                        end else if (cfg_mode_i == MODE_ADD_I8) begin
                            if (stream_elem_cnt_n == cfg_length_i) begin
                                stream_p1_flush_done_n = 1'b1;
                                stream_state_n = ST_WAIT_FLUSH;
                            end else begin
                                if (stream_dst_addr_n[4:0] == 5'd0) begin
                                    stream_p1_flush_mid_n = 1'b1;
                                end
                                if ((stream_src_addr_n[4:0] == 5'd0) || (stream_rhs_addr_n[4:0] == 5'd0)) begin
                                    stream_state_n = ST_READ_IN;
                                end else begin
                                    stream_state_n = ST_PROCESS;
                                end
                            end
                        end else if (stream_elem_cnt_n == cfg_length_i) begin
                            stream_p1_flush_done_n = 1'b1;
                            stream_state_n = ST_WAIT_FLUSH;
                        end else if (stream_dst_addr_n[4:0] == 0) begin
                            stream_p1_flush_mid_n = 1'b1;
                            stream_state_n = (stream_src_addr_n[4:0] == 0) ? ST_READ_IN : ST_PROCESS;
                        end else if (stream_src_addr_n[4:0] == 0) begin
                            stream_state_n = ST_READ_IN;
                        end
                    end
                end
            end

            ST_WAIT_FLUSH: begin
                if (cfg_mode_i == MODE_MUL_Q7) begin
                    if (s2_flush_done_completed) begin
                        stream_state_n = ST_DONE;
                    end
                end else if (s2_flush_mid_completed) begin
                    if (cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                        if (((src_addr_q[4:0] == 0) || (rhs_addr_q[4:0] == 0)) && elem_cnt_q < cfg_length_i) begin
                            stream_state_n = ST_READ_IN;
                        end else begin
                            stream_state_n = ST_PROCESS;
                        end
                    end else if (src_addr_q[4:0] == 0 && elem_cnt_q < cfg_length_i) begin
                        stream_state_n = ST_READ_IN;
                    end else begin
                        stream_state_n = ST_PROCESS;
                    end
                end else if (s2_flush_done_completed) begin
                    stream_state_n = ST_DONE;
                end
            end

            ST_DONE: begin
                // waiting for next start
            end

            default: ;
        endcase
    end

    // DFL fused-softmax path. DFL4 consumes four 4-bin sides from one ROW32
    // record. DFL16 consumes one 16-bin distribution from each ROW32 record,
    // evaluating four bins per LUT cycle and accumulating across four groups.
    always_comb begin
        logic [1:0] dfl_next_side;
        logic signed [7:0] dfl_next_max_value;
        logic [19:0] dfl_chunk_sum;
        logic [22:0] dfl_chunk_weighted;

        dfl_next_side = dfl_side_q + 2'd1;
        dfl_next_max_value = '0;
        dfl_chunk_sum = '0;
        dfl_chunk_weighted = '0;
        dfl_state_n = state_q;
        dfl_elem_cnt_n = elem_cnt_q;
        dfl_dst_addr_n = dst_addr_q;
        dfl_side_next = dfl_side_q;
        dfl_sum_next = dfl_sum_q;
        dfl_weighted_next = dfl_weighted_q;
        dfl_shift_next = dfl_shift_q;
        dfl_mul_product_next = dfl_mul_product_q;
        dfl_round_value_next = dfl_round_value_q;
        dfl_exp_req_next = 1'b0;
        dfl_recip_req_next = 1'b0;
        dfl_recip_idx_next = 8'd0;
        dfl_s1_max_value = '0;
        dfl_s1_sum_value = '0;
        dfl_s1_weighted_value = '0;
        dfl_s1_next_elem_cnt = '0;
        dfl_s1_next_dst_addr = '0;
        dfl_s1_flush_output = 1'b0;
        dfl_s1_final_location = 1'b0;
        for (int i = 0; i < LUT_LANES; i++) begin
            dfl_exp_idx_next[i] = 8'd0;
        end

        unique case (state_q)
            ST_DFL_EXP_REQ: begin
                if (dfl16_mode) begin
                    dfl_s1_max_value = max16_i8(in_buf_q);
                    for (int i = 0; i < LUT_LANES; i++) begin
                        dfl_exp_idx_next[i] = select_dfl16_byte(in_buf_q, dfl_side_q, 2'(i)) -
                                              dfl_s1_max_value[7:0];
                    end
                end else begin
                    dfl_s1_max_value = max4_i8(select_dfl_byte(in_buf_q, dfl_side_q, 2'd0),
                                               select_dfl_byte(in_buf_q, dfl_side_q, 2'd1),
                                               select_dfl_byte(in_buf_q, dfl_side_q, 2'd2),
                                               select_dfl_byte(in_buf_q, dfl_side_q, 2'd3));
                    for (int i = 0; i < LUT_LANES; i++) begin
                        dfl_exp_idx_next[i] = select_dfl_byte(in_buf_q, dfl_side_q, 2'(i)) -
                                              dfl_s1_max_value[7:0];
                    end
                end
                dfl_exp_req_next = 1'b1;
                dfl_state_n = ST_DFL_EXP_WAIT;
            end

            ST_DFL_EXP_WAIT: begin
                dfl_chunk_sum = {4'd0, lut_rdata_bank0[0][15:0]} +
                                {4'd0, lut_rdata_bank0[1][15:0]} +
                                {4'd0, lut_rdata_bank0[2][15:0]} +
                                {4'd0, lut_rdata_bank0[3][15:0]};
                if (dfl16_mode) begin
                    // Local weights 0,1,2,3 plus the group base 0,4,8,12.
                    // Keep this multiplier-free so the LUT response path is
                    // add/shift only at the 1 GHz target.
                    dfl_chunk_weighted =
                        {7'd0, lut_rdata_bank0[1][15:0]} +
                        ({7'd0, lut_rdata_bank0[2][15:0]} << 1) +
                        ({7'd0, lut_rdata_bank0[3][15:0]} << 1) +
                        {7'd0, lut_rdata_bank0[3][15:0]};
                    unique case (dfl_side_q)
                        2'd1: dfl_chunk_weighted = dfl_chunk_weighted +
                                                          ({3'd0, dfl_chunk_sum} << 2);
                        2'd2: dfl_chunk_weighted = dfl_chunk_weighted +
                                                          ({3'd0, dfl_chunk_sum} << 3);
                        2'd3: dfl_chunk_weighted = dfl_chunk_weighted +
                                                          ({3'd0, dfl_chunk_sum} << 3) +
                                                          ({3'd0, dfl_chunk_sum} << 2);
                        default: ;
                    endcase
                    dfl_s1_sum_value = dfl_sum_q + dfl_chunk_sum;
                    dfl_s1_weighted_value = dfl_weighted_q + dfl_chunk_weighted;
                    dfl_sum_next = dfl_s1_sum_value;
                    dfl_weighted_next = dfl_s1_weighted_value;

                    if (dfl_side_q != 2'd3) begin
                        dfl_next_max_value = max16_i8(in_buf_q);
                        for (int i = 0; i < LUT_LANES; i++) begin
                            dfl_exp_idx_next[i] = select_dfl16_byte(in_buf_q, dfl_next_side, 2'(i)) -
                                                  dfl_next_max_value[7:0];
                        end
                        dfl_exp_req_next = 1'b1;
                        dfl_side_next = dfl_next_side;
                        dfl_state_n = ST_DFL_EXP_WAIT;
                    end else begin
                        // Register the final accumulators before normalization.
                        // This keeps the four-lane sum/weighted adders out of
                        // the reciprocal-address path at the 1 GHz target.
                        dfl_state_n = ST_DFL_RECIP_REQ;
                    end
                end else begin
                    dfl_chunk_weighted = {7'd0, lut_rdata_bank0[1][15:0]} +
                                         ({7'd0, lut_rdata_bank0[2][15:0]} << 1) +
                                         ({7'd0, lut_rdata_bank0[3][15:0]} << 1) +
                                         {7'd0, lut_rdata_bank0[3][15:0]};
                    dfl_s1_sum_value = dfl_chunk_sum;
                    dfl_s1_weighted_value = dfl_chunk_weighted;
                    dfl_sum_next = dfl_s1_sum_value;
                    dfl_weighted_next = dfl_s1_weighted_value;
                    dfl_state_n = ST_DFL_RECIP_REQ;
                end
            end

            ST_DFL_RECIP_REQ: begin
                dfl_shift_next = msb_pos20(dfl_sum_q);
                dfl_recip_idx_next = recip_index_from_sum(dfl_sum_q);
                dfl_recip_req_next = 1'b1;
                dfl_state_n = ST_DFL_RECIP_WAIT;
            end

            ST_DFL_RECIP_WAIT: begin
                // 31x32-bit registered multiply; do not zero-extend operands
                // to the 63-bit result width before inference.
                dfl_mul_product_next = {dfl_weighted_q, 8'd0} * lut_rdata_bank1[0];
                dfl_state_n = ST_DFL_MUL_WRITE;
            end

            ST_DFL_MUL_WRITE: begin
                dfl_round_value_next = dfl_q8_from_product(dfl_mul_product_q, dfl_shift_q);
                dfl_state_n = ST_DFL_WRITE;
            end

            ST_DFL_WRITE: begin
                dfl_s1_final_location = dfl16_mode || (dfl_side_q == 2'd3);
                dfl_s1_next_elem_cnt = elem_cnt_q + (dfl_s1_final_location ? 32'd32 : 32'd0);
                dfl_s1_next_dst_addr = dst_addr_q +
                                       (dfl16_mode ? 32'd2 :
                                        (dfl_s1_final_location ? 32'd8 : 32'd0));
                dfl_s1_flush_output = dfl_s1_final_location &&
                                      ((dfl_s1_next_dst_addr[4:0] == 5'd0) ||
                                       (dfl_s1_next_elem_cnt >= cfg_length_i));

                if (dfl_s1_flush_output && wfifo_full_i) begin
                    dfl_elem_cnt_n = dfl_s1_next_elem_cnt;
                    dfl_dst_addr_n = dfl_s1_next_dst_addr;
                    dfl_side_next = 2'd0;
                    dfl_sum_next = '0;
                    dfl_weighted_next = '0;
                    dfl_state_n = ST_DFL_PUSH;
                end else if (dfl_s1_final_location) begin
                    dfl_elem_cnt_n = dfl_s1_next_elem_cnt;
                    dfl_dst_addr_n = dfl_s1_next_dst_addr;
                    dfl_side_next = 2'd0;
                    dfl_sum_next = '0;
                    dfl_weighted_next = '0;
                    dfl_state_n = (dfl_s1_next_elem_cnt >= cfg_length_i) ? ST_DONE : ST_READ_IN;
                end else begin
                    dfl_next_max_value = max4_i8(select_dfl_byte(in_buf_q, dfl_next_side, 2'd0),
                                                 select_dfl_byte(in_buf_q, dfl_next_side, 2'd1),
                                                 select_dfl_byte(in_buf_q, dfl_next_side, 2'd2),
                                                 select_dfl_byte(in_buf_q, dfl_next_side, 2'd3));
                    for (int i = 0; i < LUT_LANES; i++) begin
                        dfl_exp_idx_next[i] = select_dfl_byte(in_buf_q, dfl_next_side, 2'(i)) -
                                              dfl_next_max_value[7:0];
                    end
                    dfl_exp_req_next = 1'b1;
                    dfl_side_next = dfl_next_side;
                    dfl_state_n = ST_DFL_EXP_WAIT;
                end
            end

            ST_DFL_PUSH: begin
                if (!wfifo_full_i) begin
                    dfl_state_n = (elem_cnt_q >= cfg_length_i) ? ST_DONE : ST_READ_IN;
                end
            end

            default: ;
        endcase
    end

    // Class sigmoid row32 path.
    always_comb begin
        logic [1:0] class_next_group;

        class_next_group = class_group_q + 2'd1;
        class_state_n = state_q;
        class_elem_cnt_n = elem_cnt_q;
        class_dst_addr_n = dst_addr_q;
        class_group_next = class_group_q;
        class_lut_req_next = 1'b0;
        class_s1_final_group = 1'b0;
        class_s1_next_elem_cnt = '0;
        class_s1_next_dst_addr = '0;
        class_s1_flush_output = 1'b0;
        for (int i = 0; i < LUT_LANES; i++) begin
            class_lut_idx_next[i] = 8'd0;
        end

        unique case (state_q)
            ST_CLASS_LUT_REQ: begin
                for (int i = 0; i < LUT_LANES; i++) begin
                    class_lut_idx_next[i] = select_input_byte(
                        in_buf_q,
                        {1'b0, 5'd16 + {1'b0, class_group_q, 2'b00} + 5'(i)}
                    );
                end
                class_lut_req_next = 1'b1;
                class_state_n = ST_CLASS_LUT_WAIT;
            end

            ST_CLASS_LUT_WAIT: begin
                class_s1_final_group = (class_group_q == 2'd3);
                class_s1_next_elem_cnt = elem_cnt_q + (class_s1_final_group ? 32'd32 : 32'd0);
                class_s1_next_dst_addr = dst_addr_q + (class_s1_final_group ? 32'd16 : 32'd0);
                class_s1_flush_output = class_s1_final_group &&
                                        ((class_s1_next_dst_addr[4:0] == 5'd0) ||
                                         (class_s1_next_elem_cnt >= cfg_length_i));

                if (class_s1_flush_output && wfifo_full_i) begin
                    class_elem_cnt_n = class_s1_next_elem_cnt;
                    class_dst_addr_n = class_s1_next_dst_addr;
                    class_group_next = 2'd0;
                    class_state_n = ST_CLASS_PUSH;
                end else if (class_s1_final_group) begin
                    class_elem_cnt_n = class_s1_next_elem_cnt;
                    class_dst_addr_n = class_s1_next_dst_addr;
                    class_group_next = 2'd0;
                    class_state_n = (class_s1_next_elem_cnt >= cfg_length_i) ? ST_DONE : ST_READ_IN;
                end else begin
                    for (int i = 0; i < LUT_LANES; i++) begin
                        class_lut_idx_next[i] = select_input_byte(
                            in_buf_q,
                            {1'b0, 5'd16 + {1'b0, class_next_group, 2'b00} + 5'(i)}
                        );
                    end
                    class_lut_req_next = 1'b1;
                    class_group_next = class_next_group;
                    class_state_n = ST_CLASS_LUT_WAIT;
                end
            end

            ST_CLASS_PUSH: begin
                if (!wfifo_full_i) begin
                    class_state_n = (elem_cnt_q >= cfg_length_i) ? ST_DONE : ST_READ_IN;
                end
            end

            default: ;
        endcase
    end

    // GlobalAvgPool C32 path.
    always_comb begin
        gap_state_n = state_q;
        gap_elem_cnt_n = elem_cnt_q;
        gap_row_count_next = gap_row_count_q;
        gap_in_buf_next = in_buf_q;
        gap_rfifo_pop_next = 1'b0;
        gap_recip_req_next = 1'b0;
        for (int i = 0; i < 32; i++) begin
            gap_acc_next[i] = gap_acc_q[i];
            gap_mul_product_next[i] = gap_mul_product_q[i];
            gap_mul_negative_next[i] = gap_mul_negative_q[i];
            gap_avg_next[i] = gap_avg_q[i];
        end

        unique case (state_q)
            ST_GAP_ACCUM: begin
                for (int i = 0; i < 32; i++) begin
                    gap_acc_next[i] = gap_acc_q[i] +
                                      {{24{in_buf_q[i * 8 + 7]}}, in_buf_q[i * 8 +: 8]};
                end
                gap_elem_cnt_n = gap_next_elem_cnt;

                if (gap_group_done) begin
                    gap_row_count_next = '0;
                    gap_recip_req_next = 1'b1;
                    gap_state_n = ST_GAP_RECIP_WAIT;
                end else begin
                    gap_row_count_next = gap_row_count_q + 32'd1;
                    if (!rfifo_empty_i) begin
                        gap_in_buf_next = rfifo_data_i;
                        gap_rfifo_pop_next = 1'b1;
                        gap_state_n = ST_GAP_ACCUM;
                    end else begin
                        gap_state_n = ST_READ_IN;
                    end
                end
            end

            ST_GAP_RECIP_REQ: begin
                gap_recip_req_next = 1'b1;
                gap_state_n = ST_GAP_RECIP_WAIT;
            end

            ST_GAP_RECIP_WAIT: begin
                for (int i = 0; i < 32; i++) begin
                    gap_mul_negative_next[i] = gap_acc_q[i][31];
                    gap_mul_product_next[i] = (gap_acc_q[i][31] ?
                                               64'(-gap_acc_q[i]) :
                                               64'(gap_acc_q[i])) *
                                              64'(active_lut_bank_q ? lut_rdata_bank1[0] :
                                                                     lut_rdata_bank0[0]);
                end
                gap_state_n = ST_GAP_MUL_WRITE;
            end

            ST_GAP_MUL_WRITE: begin
                if (gap_spatial_count != 32'd0) begin
                    for (int i = 0; i < 32; i++) begin
                        gap_avg_next[i] = avg_i32_from_product(
                            gap_mul_product_q[i],
                            gap_mul_negative_q[i],
                            gap_acc_q[i][31] ? 32'(-gap_acc_q[i]) : 32'(gap_acc_q[i]),
                            gap_spatial_count
                        );
                    end
                end
                gap_state_n = ST_GAP_WRITE;
            end

            ST_GAP_WRITE: begin
                for (int i = 0; i < 32; i++) begin
                    gap_acc_next[i] = '0;
                end
                if (wfifo_full_i) begin
                    gap_state_n = ST_GAP_PUSH;
                end else if (gap_final_input) begin
                    gap_state_n = ST_DONE;
                end else if (!rfifo_empty_i) begin
                    gap_in_buf_next = rfifo_data_i;
                    gap_rfifo_pop_next = 1'b1;
                    gap_state_n = ST_GAP_ACCUM;
                end else begin
                    gap_state_n = ST_READ_IN;
                end
            end

            ST_GAP_PUSH: begin
                if (!wfifo_full_i) begin
                    if (gap_final_input) begin
                        gap_state_n = ST_DONE;
                    end else if (!rfifo_empty_i) begin
                        gap_in_buf_next = rfifo_data_i;
                        gap_rfifo_pop_next = 1'b1;
                        gap_state_n = ST_GAP_ACCUM;
                    end else begin
                        gap_state_n = ST_READ_IN;
                    end
                end
            end

            default: ;
        endcase
    end

    // Stage 1 feature mux.
    always_comb begin
        state_n = state_q;
        src_addr_n = src_addr_q;
        rhs_addr_n = rhs_addr_q;
        dst_addr_n = dst_addr_q;
        elem_cnt_n = elem_cnt_q;
        in_buf_n   = in_buf_q;
        rhs_buf_n  = rhs_buf_q;
        rfifo_pop_o = 1'b0;
        rhs_rfifo_pop_o = 1'b0;
        s1_sram_req = 1'b0;

        p1_valid_n = 1'b0;
        p1_flush_mid_n = 1'b0;
        p1_flush_done_n = 1'b0;
        p1_num_valid_lanes_n = '0;
        p1_lhs_data_n = p1_lhs_data_q;
        p1_rhs_data_n = p1_rhs_data_q;
        p1_src_addr_n = src_addr_q;
        p1_rhs_addr_n = rhs_addr_q;
        p1_dst_addr_n = dst_addr_q;
        dfl_side_n = dfl_side_q;
        class_group_n = class_group_q;
        dfl_sum_n = dfl_sum_q;
        dfl_weighted_n = dfl_weighted_q;
        dfl_shift_n = dfl_shift_q;
        dfl_mul_product_n = dfl_mul_product_q;
        dfl_round_value_n = dfl_round_value_q;
        dfl_exp_req = 1'b0;
        dfl_recip_req = 1'b0;
        gap_recip_req = 1'b0;
        class_lut_req = 1'b0;
        dfl_recip_idx = 8'd0;
        gap_row_count_n = gap_row_count_q;
        for (int i = 0; i < 32; i++) begin
            gap_acc_n[i] = gap_acc_q[i];
            gap_mul_product_n[i] = gap_mul_product_q[i];
            gap_mul_negative_n[i] = gap_mul_negative_q[i];
            gap_avg_n[i] = gap_avg_q[i];
        end
        for (int i = 0; i < LUT_LANES; i++) begin
            dfl_exp_idx[i] = 8'd0;
            class_lut_idx[i] = 8'd0;
        end

        if (cfg_start_i) begin
            elem_cnt_n = '0;
            src_addr_n = cfg_src_ptr_i;
            rhs_addr_n = cfg_src2_ptr_i;
            dst_addr_n = cfg_dst_ptr_i;
            dfl_side_n = 2'd0;
            class_group_n = 2'd0;
            dfl_sum_n = '0;
            dfl_weighted_n = '0;
            dfl_shift_n = '0;
            dfl_mul_product_n = '0;
            dfl_round_value_n = '0;
            gap_row_count_n = '0;
            for (int i = 0; i < 32; i++) begin
                gap_acc_n[i] = '0;
                gap_mul_product_n[i] = '0;
                gap_mul_negative_n[i] = 1'b0;
                gap_avg_n[i] = '0;
            end
            if (cfg_length_i == 0) begin
                state_n = ST_DONE;
            end else begin
                state_n = ST_READ_IN;
            end
        end else begin
            unique case (state_q)
                ST_IDLE, ST_READ_IN, ST_PROCESS, ST_WAIT_FLUSH, ST_DONE: begin
                    state_n = stream_state_n;
                    src_addr_n = stream_src_addr_n;
                    rhs_addr_n = stream_rhs_addr_n;
                    dst_addr_n = stream_dst_addr_n;
                    elem_cnt_n = stream_elem_cnt_n;
                    in_buf_n = stream_in_buf_n;
                    rhs_buf_n = stream_rhs_buf_n;
                    rfifo_pop_o = stream_rfifo_pop;
                    rhs_rfifo_pop_o = stream_rhs_rfifo_pop;
                    s1_sram_req = stream_s1_sram_req;
                    p1_valid_n = stream_p1_valid_n;
                    p1_flush_mid_n = stream_p1_flush_mid_n;
                    p1_flush_done_n = stream_p1_flush_done_n;
                    p1_num_valid_lanes_n = stream_p1_num_valid_lanes_n;
                    p1_lhs_data_n = stream_p1_lhs_data_n;
                    p1_rhs_data_n = stream_p1_rhs_data_n;
                    p1_src_addr_n = stream_p1_src_addr_n;
                    p1_rhs_addr_n = stream_p1_rhs_addr_n;
                    p1_dst_addr_n = stream_p1_dst_addr_n;
                end

                ST_DFL_EXP_REQ, ST_DFL_EXP_WAIT, ST_DFL_RECIP_REQ, ST_DFL_RECIP_WAIT,
                ST_DFL_MUL_WRITE, ST_DFL_WRITE, ST_DFL_PUSH: begin
                    state_n = dfl_state_n;
                    elem_cnt_n = dfl_elem_cnt_n;
                    dst_addr_n = dfl_dst_addr_n;
                    dfl_side_n = dfl_side_next;
                    dfl_sum_n = dfl_sum_next;
                    dfl_weighted_n = dfl_weighted_next;
                    dfl_shift_n = dfl_shift_next;
                    dfl_mul_product_n = dfl_mul_product_next;
                    dfl_round_value_n = dfl_round_value_next;
                    dfl_exp_req = dfl_exp_req_next;
                    dfl_recip_req = dfl_recip_req_next;
                    dfl_recip_idx = dfl_recip_idx_next;
                    for (int i = 0; i < LUT_LANES; i++) begin
                        dfl_exp_idx[i] = dfl_exp_idx_next[i];
                    end
                end

                ST_CLASS_LUT_REQ, ST_CLASS_LUT_WAIT, ST_CLASS_PUSH: begin
                    state_n = class_state_n;
                    elem_cnt_n = class_elem_cnt_n;
                    dst_addr_n = class_dst_addr_n;
                    class_group_n = class_group_next;
                    class_lut_req = class_lut_req_next;
                    for (int i = 0; i < LUT_LANES; i++) begin
                        class_lut_idx[i] = class_lut_idx_next[i];
                    end
                end

                ST_GAP_ACCUM, ST_GAP_RECIP_REQ, ST_GAP_RECIP_WAIT,
                ST_GAP_MUL_WRITE, ST_GAP_WRITE, ST_GAP_PUSH: begin
                    state_n = gap_state_n;
                    elem_cnt_n = gap_elem_cnt_n;
                    in_buf_n = gap_in_buf_next;
                    rfifo_pop_o = gap_rfifo_pop_next;
                    gap_row_count_n = gap_row_count_next;
                    gap_recip_req = gap_recip_req_next;
                    for (int i = 0; i < 32; i++) begin
                        gap_acc_n[i] = gap_acc_next[i];
                        gap_mul_product_n[i] = gap_mul_product_next[i];
                        gap_mul_negative_n[i] = gap_mul_negative_next[i];
                        gap_avg_n[i] = gap_avg_next[i];
                    end
                end

                default: ;
            endcase
        end
    end

    // Stage 2 Logic
    logic [255:0] s2_out_buf_comb;
    logic [31:0]  s2_out_be_comb;

    assign wfifo_data_o = {s2_out_be_comb, s2_out_buf_comb};
    
    // Pipeline hazard fix: if S2 stalls, SRAM output will change in the next cycle.
    // We must save lut_rdata_ports when S2 stalls.
    logic [31:0] s2_lut_rdata_saved_q [LUT_LANES];
    logic        s2_lut_rdata_saved_valid_q;
    logic [255:0] gap_wb_out_buf;
    logic [31:0]  gap_wb_out_be;
    logic         gap_wb_push_now;
    logic [255:0] dfl_wb_out_buf;
    logic [31:0]  dfl_wb_out_be;
    logic         dfl_wb_push_now;
    logic [255:0] class_wb_out_buf;
    logic [31:0]  class_wb_out_be;
    logic         class_wb_push_now;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            s2_lut_rdata_saved_valid_q <= 1'b0;
            for (int i=0; i<LUT_LANES; i++) s2_lut_rdata_saved_q[i] <= '0;
        end else begin
            if (p1_valid_q && s2_stall && !s2_lut_rdata_saved_valid_q) begin
                for (int i=0; i<LUT_LANES; i++) s2_lut_rdata_saved_q[i] <= lut_rdata_ports[i];
                s2_lut_rdata_saved_valid_q <= 1'b1;
            end else if (!s2_stall) begin
                s2_lut_rdata_saved_valid_q <= 1'b0;
            end
        end
    end

    always_comb begin : p_gap_writeback_comb
        gap_wb_out_buf = out_buf_q;
        gap_wb_out_be = out_be_q;
        gap_wb_push_now = 1'b0;

        if (state_q == ST_GAP_WRITE && gap_spatial_count != 32'd0) begin
            for (int i = 0; i < 32; i++) begin
                gap_wb_out_buf[i * 8 +: 8] = i8_byte_from_i16(gap_avg_q[i][15:0]);
                gap_wb_out_be[i] = 1'b1;
            end
            gap_wb_push_now = !wfifo_full_i;
        end else if (state_q == ST_GAP_PUSH) begin
            gap_wb_push_now = !wfifo_full_i;
        end
    end

    always_comb begin : p_dfl_writeback_comb
        dfl_s2_out_value = dfl_round_value_q;
        dfl_s2_out_off = dfl16_mode ? dst_addr_q[4:0] :
                         dst_addr_q[4:0] + {2'd0, dfl_side_q, 1'b0};
        dfl_s2_final_location = dfl16_mode || (dfl_side_q == 2'd3);
        dfl_s2_next_elem_cnt = elem_cnt_q + (dfl_s2_final_location ? 32'd32 : 32'd0);
        dfl_s2_next_dst_addr = dst_addr_q +
                               (dfl16_mode ? 32'd2 :
                                (dfl_s2_final_location ? 32'd8 : 32'd0));
        dfl_s2_flush_output = dfl_s2_final_location &&
                              ((dfl_s2_next_dst_addr[4:0] == 5'd0) ||
                               (dfl_s2_next_elem_cnt >= cfg_length_i));

        dfl_wb_out_buf = out_buf_q;
        dfl_wb_out_be = out_be_q;
        dfl_wb_push_now = 1'b0;

        if (state_q == ST_DFL_WRITE) begin
            dfl_wb_out_buf[dfl_s2_out_off * 8 +: 16] = dfl_s2_out_value;
            dfl_wb_out_be[dfl_s2_out_off +: 2] = 2'b11;
            dfl_wb_push_now = dfl_s2_flush_output && !wfifo_full_i;
        end else if (state_q == ST_DFL_PUSH) begin
            dfl_wb_push_now = !wfifo_full_i;
        end
    end

    always_comb begin : p_class_writeback_comb
        logic [31:0] lut_val_class;

        lut_val_class = '0;
        class_s2_final_group = (class_group_q == 2'd3);
        class_s2_next_elem_cnt = elem_cnt_q + (class_s2_final_group ? 32'd32 : 32'd0);
        class_s2_next_dst_addr = dst_addr_q + (class_s2_final_group ? 32'd16 : 32'd0);
        class_s2_flush_output = class_s2_final_group &&
                                ((class_s2_next_dst_addr[4:0] == 5'd0) ||
                                 (class_s2_next_elem_cnt >= cfg_length_i));

        class_wb_out_buf = out_buf_q;
        class_wb_out_be = out_be_q;
        class_wb_push_now = 1'b0;

        if (state_q == ST_CLASS_LUT_WAIT) begin
            for (int i = 0; i < LUT_LANES; i++) begin
                lut_val_class = active_lut_bank_q ? lut_rdata_bank1[i] : lut_rdata_bank0[i];
                class_wb_out_buf[(dst_addr_q[4:0] + {1'b0, class_group_q, 2'b00} + 5'(i)) * 8 +: 8] =
                    lut_val_class[7:0];
                class_wb_out_be[dst_addr_q[4:0] + {1'b0, class_group_q, 2'b00} + 5'(i)] = 1'b1;
            end
            class_wb_push_now = class_s2_flush_output && !wfifo_full_i;
        end else if (state_q == ST_CLASS_PUSH) begin
            class_wb_push_now = !wfifo_full_i;
        end
    end

    always_comb begin
        logic [31:0] lut_val;
        logic [4:0]  cur_out_off;
        logic        mul_p3_pop;
        logic        mul_p2_advance_s2;
        logic        mul_p1_advance_s2;
        lut_val = '0;
        cur_out_off = '0;
        mul_p3_pop = 1'b0;
        mul_p2_advance_s2 = 1'b0;
        mul_p1_advance_s2 = 1'b0;
        
        s2_flush_mid_completed = 1'b0;
        s2_flush_done_completed = 1'b0;
        wfifo_push_o = 1'b0;
        out_buf_n = out_buf_q;
        out_be_n  = out_be_q;
        out_base_n = out_base_q;
        mul_p2_valid_n = mul_p2_valid_q;
        mul_p2_num_valid_lanes_n = mul_p2_num_valid_lanes_q;
        mul_p2_dst_addr_n = mul_p2_dst_addr_q;
        mul_p2_flush_mid_n = mul_p2_flush_mid_q;
        mul_p2_flush_done_n = mul_p2_flush_done_q;
        mul_p3_valid_n = mul_p3_valid_q;
        mul_p3_num_valid_lanes_n = mul_p3_num_valid_lanes_q;
        mul_p3_dst_addr_n = mul_p3_dst_addr_q;
        mul_p3_flush_mid_n = mul_p3_flush_mid_q;
        mul_p3_flush_done_n = mul_p3_flush_done_q;
        for (int i = 0; i < 32; i++) begin
            mul_p2_product_n[i] = mul_p2_product_q[i];
            mul_p3_shifted_n[i] = mul_p3_shifted_q[i];
        end
        
        s2_out_buf_comb = out_buf_q;
        s2_out_be_comb  = out_be_q;

        if (cfg_start_i) begin
            out_base_n = cfg_dst_ptr_i;
            out_buf_n  = '0;
            out_be_n   = '0;
            mul_p2_valid_n = 1'b0;
            mul_p2_num_valid_lanes_n = '0;
            mul_p2_dst_addr_n = '0;
            mul_p2_flush_mid_n = 1'b0;
            mul_p2_flush_done_n = 1'b0;
            mul_p3_valid_n = 1'b0;
            mul_p3_num_valid_lanes_n = '0;
            mul_p3_dst_addr_n = '0;
            mul_p3_flush_mid_n = 1'b0;
            mul_p3_flush_done_n = 1'b0;
            for (int i = 0; i < 32; i++) begin
                mul_p2_product_n[i] = '0;
                mul_p3_shifted_n[i] = '0;
            end
        end else if (state_q == ST_GAP_WRITE) begin
            if (gap_spatial_count != 32'd0) begin
                s2_out_buf_comb = gap_wb_out_buf;
                s2_out_be_comb = gap_wb_out_be;
                if (!wfifo_full_i) begin
                    wfifo_push_o = gap_wb_push_now;
                    out_buf_n = '0;
                    out_be_n = '0;
                end else begin
                    out_buf_n = gap_wb_out_buf;
                    out_be_n = gap_wb_out_be;
                end
            end
        end else if (state_q == ST_GAP_PUSH) begin
            s2_out_buf_comb = gap_wb_out_buf;
            s2_out_be_comb = gap_wb_out_be;
            if (!wfifo_full_i) begin
                wfifo_push_o = gap_wb_push_now;
                out_buf_n = '0;
                out_be_n = '0;
            end
        end else if (state_q == ST_CLASS_LUT_WAIT) begin
            s2_out_buf_comb = class_wb_out_buf;
            s2_out_be_comb = class_wb_out_be;
            if (class_s2_flush_output && !wfifo_full_i) begin
                wfifo_push_o = class_wb_push_now;
                out_buf_n = '0;
                out_be_n = '0;
            end else begin
                out_buf_n = class_wb_out_buf;
                out_be_n = class_wb_out_be;
            end
        end else if (state_q == ST_DFL_WRITE) begin
            s2_out_buf_comb = dfl_wb_out_buf;
            s2_out_be_comb = dfl_wb_out_be;
            if (dfl_s2_flush_output && !wfifo_full_i) begin
                wfifo_push_o = dfl_wb_push_now;
                out_buf_n = '0;
                out_be_n = '0;
            end else begin
                out_buf_n = dfl_wb_out_buf;
                out_be_n = dfl_wb_out_be;
            end
        end else if (state_q == ST_DFL_PUSH) begin
            s2_out_buf_comb = dfl_wb_out_buf;
            s2_out_be_comb = dfl_wb_out_be;
            if (!wfifo_full_i) begin
                wfifo_push_o = dfl_wb_push_now;
                out_buf_n = '0;
                out_be_n = '0;
            end
        end else if (state_q == ST_CLASS_PUSH) begin
            s2_out_buf_comb = class_wb_out_buf;
            s2_out_be_comb = class_wb_out_be;
            if (!wfifo_full_i) begin
                wfifo_push_o = class_wb_push_now;
                out_buf_n = '0;
                out_be_n = '0;
            end
        end else if (cfg_mode_i == MODE_MUL_Q7) begin
            if (mul_p3_valid_q) begin
                s2_out_buf_comb = out_buf_q;
                s2_out_be_comb = out_be_q;
                for (int i = 0; i < 32; i++) begin
                    if (i < mul_p3_num_valid_lanes_q) begin
                        cur_out_off = mul_p3_dst_addr_q[4:0] + 5'(i);
                        s2_out_buf_comb[cur_out_off * 8 +: 8] =
                            i8_byte_from_i16(mul_p3_shifted_q[i]);
                        s2_out_be_comb[cur_out_off] = 1'b1;
                    end
                end

                if (!wfifo_full_i) begin
                    mul_p3_pop = 1'b1;
                    if (mul_p3_flush_mid_q) begin
                        wfifo_push_o = 1'b1;
                        out_buf_n = '0;
                        out_be_n  = '0;
                        out_base_n = out_base_q + 32;
                        s2_flush_mid_completed = 1'b1;
                    end else if (mul_p3_flush_done_q) begin
                        if (s2_out_be_comb != 0) begin
                            wfifo_push_o = 1'b1;
                        end
                        out_buf_n = '0;
                        out_be_n  = '0;
                        s2_flush_done_completed = 1'b1;
                    end else begin
                        out_buf_n = s2_out_buf_comb;
                        out_be_n  = s2_out_be_comb;
                    end
                end
            end

            mul_p2_advance_s2 = mul_p2_valid_q && (!mul_p3_valid_q || mul_p3_pop);
            if (mul_p2_advance_s2) begin
                mul_p3_valid_n = 1'b1;
                mul_p3_num_valid_lanes_n = mul_p2_num_valid_lanes_q;
                mul_p3_dst_addr_n = mul_p2_dst_addr_q;
                mul_p3_flush_mid_n = mul_p2_flush_mid_q;
                mul_p3_flush_done_n = mul_p2_flush_done_q;
                for (int i = 0; i < 32; i++) begin
                    if (i < mul_p2_num_valid_lanes_q) begin
                        mul_p3_shifted_n[i] = mul_q7_shifted(mul_p2_product_q[i]);
                    end else begin
                        mul_p3_shifted_n[i] = '0;
                    end
                end
                mul_p2_valid_n = 1'b0;
            end else if (mul_p3_pop) begin
                mul_p3_valid_n = 1'b0;
            end

            mul_p1_advance_s2 = p1_valid_q && !s2_stall &&
                                (!mul_p2_valid_q || mul_p2_advance_s2);
            if (mul_p1_advance_s2) begin
                mul_p2_valid_n = 1'b1;
                mul_p2_num_valid_lanes_n = p1_num_valid_lanes_q;
                mul_p2_dst_addr_n = p1_dst_addr_q;
                mul_p2_flush_mid_n = p1_flush_mid_q;
                mul_p2_flush_done_n = p1_flush_done_q;
                for (int i = 0; i < 32; i++) begin
                    if (i < p1_num_valid_lanes_q) begin
                        mul_p2_product_n[i] = mul_q7_product(
                            select_input_byte(p1_lhs_data_q, {1'b0, p1_src_addr_q[4:0]} + 6'(i)),
                            select_input_byte(p1_rhs_data_q, {1'b0, p1_rhs_addr_q[4:0]} + 6'(i))
                        );
                    end else begin
                        mul_p2_product_n[i] = '0;
                    end
                end
            end
        end else if (p1_valid_q && !s2_stall) begin
            // Process lanes directly into combinational buffer
            if (cfg_mode_i == MODE_ADD_I8) begin
                for (int i = 0; i < 32; i++) begin
                    if (i < p1_num_valid_lanes_q) begin
                        cur_out_off = p1_dst_addr_q[4:0] + 5'(i);
                        s2_out_buf_comb[cur_out_off * 8 +: 8] = add_i8_byte(
                            select_input_byte(p1_lhs_data_q, {1'b0, p1_src_addr_q[4:0]} + 6'(i)),
                            select_input_byte(p1_rhs_data_q, {1'b0, p1_rhs_addr_q[4:0]} + 6'(i))
                        );
                        s2_out_be_comb[cur_out_off] = 1'b1;
                    end
                end
            end else begin
                for (int i = 0; i < LUT_LANES; i++) begin
                    if (i < p1_num_valid_lanes_q) begin
                        lut_val = s2_lut_rdata_saved_valid_q ? s2_lut_rdata_saved_q[i] : lut_rdata_ports[i];

                        if (cfg_mode_i == MODE_8BIT) begin
                            cur_out_off = p1_dst_addr_q[4:0] + 5'(i);
                            s2_out_buf_comb[cur_out_off * 8 +: 8] = lut_val[7:0];
                            s2_out_be_comb[cur_out_off] = 1'b1;
                        end else if (cfg_mode_i == MODE_16BIT) begin
                            cur_out_off = p1_dst_addr_q[4:0] + 5'(i * 2);
                            s2_out_buf_comb[cur_out_off * 8 +: 16] = lut_val[15:0];
                            s2_out_be_comb[cur_out_off +: 2] = 2'b11;
                        end else begin
                            cur_out_off = p1_dst_addr_q[4:0] + 5'(i * 4);
                            s2_out_buf_comb[cur_out_off * 8 +: 32] = lut_val;
                            s2_out_be_comb[cur_out_off +: 4] = 4'b1111;
                        end
                    end
                end
            end

            if (cfg_mode_i == MODE_MUL_Q7) begin
                // MUL_Q7 completion is reported when the registered product
                // stage is packed and accepted by the write FIFO.
            end else if (p1_flush_mid_q) begin
                wfifo_push_o = 1'b1;
                out_buf_n = '0;
                out_be_n  = '0;
                out_base_n = out_base_q + 32;
                s2_flush_mid_completed = 1'b1;
            end else if (p1_flush_done_q) begin
                if (s2_out_be_comb != 0) begin
                    wfifo_push_o = 1'b1;
                end
                out_buf_n = '0;
                out_be_n  = '0;
                s2_flush_done_completed = 1'b1;
            end else begin
                out_buf_n = s2_out_buf_comb;
                out_be_n  = s2_out_be_comb;
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_common_regs
        if (!rst_ni) begin
            state_q    <= ST_IDLE;
            src_addr_q <= '0;
            rhs_addr_q <= '0;
            dst_addr_q <= '0;
            elem_cnt_q <= '0;
            in_buf_q   <= '0;
            rhs_buf_q  <= '0;
        end else begin
            state_q    <= state_n;
            src_addr_q <= src_addr_n;
            rhs_addr_q <= rhs_addr_n;
            dst_addr_q <= dst_addr_n;
            elem_cnt_q <= elem_cnt_n;
            in_buf_q   <= in_buf_n;
            rhs_buf_q  <= rhs_buf_n;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_writeback_regs
        if (!rst_ni) begin
            out_base_q <= '0;
            out_buf_q  <= '0;
            out_be_q   <= '0;
        end else begin
            out_base_q <= out_base_n;
            out_buf_q  <= out_buf_n;
            out_be_q   <= out_be_n;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_dfl_regs
        if (!rst_ni) begin
            dfl_side_q <= '0;
            dfl_sum_q <= '0;
            dfl_weighted_q <= '0;
            dfl_shift_q <= '0;
            dfl_mul_product_q <= '0;
            dfl_round_value_q <= '0;
        end else begin
            dfl_side_q <= dfl_side_n;
            dfl_sum_q <= dfl_sum_n;
            dfl_weighted_q <= dfl_weighted_n;
            dfl_shift_q <= dfl_shift_n;
            dfl_mul_product_q <= dfl_mul_product_n;
            dfl_round_value_q <= dfl_round_value_n;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_class_regs
        if (!rst_ni) begin
            class_group_q <= '0;
        end else begin
            class_group_q <= class_group_n;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_gap_regs
        if (!rst_ni) begin
            gap_row_count_q <= '0;
            for (int i = 0; i < 32; i++) begin
                gap_acc_q[i] <= '0;
                gap_mul_product_q[i] <= '0;
                gap_mul_negative_q[i] <= 1'b0;
                gap_avg_q[i] <= '0;
            end
        end else begin
            gap_row_count_q <= gap_row_count_n;
            for (int i = 0; i < 32; i++) begin
                gap_acc_q[i] <= gap_acc_n[i];
                gap_mul_product_q[i] <= gap_mul_product_n[i];
                gap_mul_negative_q[i] <= gap_mul_negative_n[i];
                gap_avg_q[i] <= gap_avg_n[i];
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_stage1_regs
        if (!rst_ni) begin
            p1_valid_q <= 1'b0;
            p1_num_valid_lanes_q <= '0;
            p1_lhs_data_q <= '0;
            p1_rhs_data_q <= '0;
            p1_src_addr_q <= '0;
            p1_rhs_addr_q <= '0;
            p1_dst_addr_q <= '0;
            p1_flush_mid_q <= 1'b0;
            p1_flush_done_q <= 1'b0;
        end else if (!s2_stall) begin
            p1_valid_q <= p1_valid_n;
            p1_num_valid_lanes_q <= p1_num_valid_lanes_n;
            p1_lhs_data_q <= p1_lhs_data_n;
            p1_rhs_data_q <= p1_rhs_data_n;
            p1_src_addr_q <= p1_src_addr_n;
            p1_rhs_addr_q <= p1_rhs_addr_n;
            p1_dst_addr_q <= p1_dst_addr_n;
            p1_flush_mid_q <= p1_flush_mid_n;
            p1_flush_done_q <= p1_flush_done_n;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin : p_mul_pipeline_regs
        if (!rst_ni) begin
            mul_p2_valid_q <= 1'b0;
            mul_p2_num_valid_lanes_q <= '0;
            mul_p2_dst_addr_q <= '0;
            mul_p2_flush_mid_q <= 1'b0;
            mul_p2_flush_done_q <= 1'b0;
            mul_p3_valid_q <= 1'b0;
            mul_p3_num_valid_lanes_q <= '0;
            mul_p3_dst_addr_q <= '0;
            mul_p3_flush_mid_q <= 1'b0;
            mul_p3_flush_done_q <= 1'b0;
            for (int i = 0; i < 32; i++) begin
                mul_p2_product_q[i] <= '0;
                mul_p3_shifted_q[i] <= '0;
            end
        end else begin
            mul_p2_valid_q <= mul_p2_valid_n;
            mul_p2_num_valid_lanes_q <= mul_p2_num_valid_lanes_n;
            mul_p2_dst_addr_q <= mul_p2_dst_addr_n;
            mul_p2_flush_mid_q <= mul_p2_flush_mid_n;
            mul_p2_flush_done_q <= mul_p2_flush_done_n;
            mul_p3_valid_q <= mul_p3_valid_n;
            mul_p3_num_valid_lanes_q <= mul_p3_num_valid_lanes_n;
            mul_p3_dst_addr_q <= mul_p3_dst_addr_n;
            mul_p3_flush_mid_q <= mul_p3_flush_mid_n;
            mul_p3_flush_done_q <= mul_p3_flush_done_n;
            for (int i = 0; i < 32; i++) begin
                mul_p2_product_q[i] <= mul_p2_product_n[i];
                mul_p3_shifted_q[i] <= mul_p3_shifted_n[i];
            end
        end
    end

endmodule
