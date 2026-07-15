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

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_READ_IN,
        ST_PROCESS,
        ST_WAIT_FLUSH,
        ST_DFL_EXP_REQ,
        ST_DFL_EXP_WAIT,
        ST_DFL_RECIP_WAIT,
        ST_DFL_PUSH,
        ST_CLASS_LUT_REQ,
        ST_CLASS_LUT_WAIT,
        ST_CLASS_PUSH,
        ST_GAP_ACCUM,
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

    assign s2_stall = wfifo_full_i && p1_valid_q &&
                      ((cfg_mode_i == MODE_MUL_Q7) || (cfg_mode_i == MODE_ADD_I8) ||
                       p1_flush_mid_q || (p1_flush_done_q && out_be_q != 0));

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
    logic        class_lut_req;
    logic [7:0]  dfl_exp_idx [LUT_LANES];
    logic [7:0]  dfl_recip_idx;
    logic [7:0]  class_lut_idx [LUT_LANES];
    logic [1:0]  dfl_side_q, dfl_side_n;
    logic [1:0]  class_group_q, class_group_n;
    logic [17:0] dfl_sum_q, dfl_sum_n;
    logic [18:0] dfl_weighted_q, dfl_weighted_n;
    logic [4:0]  dfl_shift_q, dfl_shift_n;
    logic signed [7:0] dfl_s1_max_value;
    logic [17:0] dfl_s1_sum_value;
    logic [18:0] dfl_s1_weighted_value;
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

    assign gap_spatial_count = cfg_src2_ptr_i;
    assign gap_next_elem_cnt = elem_cnt_q + 32'd32;
    assign gap_group_done = ((gap_row_count_q + 32'd1) >= gap_spatial_count);
    assign gap_final_input = (gap_next_elem_cnt >= cfg_length_i);

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

    function automatic logic [7:0] mul_q7_byte(
        input logic [7:0] lhs_u8,
        input logic [7:0] rhs_u8
    );
        logic signed [7:0] lhs_i8;
        logic signed [7:0] rhs_i8;
        logic signed [15:0] product_i16;
        logic signed [15:0] shifted_i16;
        logic signed [8:0] clamped_i9;
        begin
            lhs_i8 = $signed(lhs_u8);
            rhs_i8 = $signed(rhs_u8);
            product_i16 = lhs_i8 * rhs_i8;
            shifted_i16 = product_i16 >>> 7;
            clamped_i9 = clamp_i8(shifted_i16);
            mul_q7_byte = clamped_i9[7:0];
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

    function automatic logic [7:0] avg_i8_byte(
        input logic signed [31:0] sum,
        input logic [31:0] count
    );
        logic signed [31:0] avg;
        logic signed [15:0] avg16;
        logic signed [8:0] clamped_i9;
        begin
            if (count == 32'd0) begin
                avg = '0;
            end else begin
                avg = sum / $signed({1'b0, count[30:0]});
            end
            avg16 = avg[15:0];
            clamped_i9 = clamp_i8(avg16);
            avg_i8_byte = clamped_i9[7:0];
        end
    endfunction

    function automatic logic [7:0] select_dfl_byte(
        input logic [255:0] data,
        input logic [1:0] side,
        input logic [1:0] bin
    );
        logic [5:0] byte_idx;
        begin
            byte_idx = ({4'd0, side} * 6'd4) + {4'd0, bin};
            select_dfl_byte = data[byte_idx * 8 +: 8];
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

    function automatic logic [4:0] msb_pos18(input logic [17:0] value);
        logic [4:0] pos;
        begin
            pos = 5'd0;
            for (int i = 0; i < 18; i++) begin
                if (value[i]) pos = 5'(i);
            end
            msb_pos18 = pos;
        end
    endfunction

    function automatic logic [7:0] recip_index_from_sum(input logic [17:0] sum);
        logic [4:0]  shift;
        logic [25:0] shifted;
        logic [8:0]  norm_q8;
        begin
            shift = msb_pos18(sum);
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

    function automatic logic [15:0] dfl_q8_from_recip(
        input logic [18:0] weighted,
        input logic [31:0] recip_q28,
        input logic [4:0]  sum_shift
    );
        logic [63:0] product;
        logic [63:0] rounded;
        logic [5:0]  total_shift;
        logic [63:0] round_add;
        begin
            product = ({37'd0, weighted} << 8) * {32'd0, recip_q28};
            total_shift = 6'd28 + {1'b0, sum_shift};
            round_add = 64'd1 << (total_shift - 6'd1);
            rounded = (product + round_add) >> total_shift;
            if (rounded > 64'hFFFF) begin
                dfl_q8_from_recip = 16'hFFFF;
            end else begin
                dfl_q8_from_recip = rounded[15:0];
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
                           (class_lut_req && !active_lut_bank_q) ||
                           (s1_sram_req && !active_lut_bank_q),
                           lut_we_i && ((lut_fixed_bank_i && !lut_bank_i) ||
                                        (!lut_fixed_bank_i && !stage_lut_bank_q))}),
                .we_i    ({1'b0,        1'b1}),
                .addr_i  ({dfl_exp_req ? dfl_exp_idx[i] :
                            (class_lut_req ? class_lut_idx[i] : lut_idx_s1[i]), lut_addr_i}),
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
                           (class_lut_req && active_lut_bank_q) ||
                           (s1_sram_req && active_lut_bank_q),
                           lut_we_i && ((lut_fixed_bank_i && lut_bank_i) ||
                                        (!lut_fixed_bank_i && stage_lut_bank_q))}),
                .we_i    ({1'b0,        1'b1}),
                .addr_i  ({dfl_recip_req && i == 0 ? dfl_recip_idx :
                            (class_lut_req ? class_lut_idx[i] : lut_idx_s1[i]), lut_addr_i}),
                .wdata_i ({32'd0,       lut_wdata_i}),
                .be_i    ({4'b1111,     lut_be_i}),
                .rdata_o ({lut_rdata_bank1[i], lut_rdata_dummy_bank1[i]})
            );
        end
    endgenerate

    // removed duplicate wfifo_data_o assignment

    // Stage 1 Logic
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
        dfl_exp_req = 1'b0;
        dfl_recip_req = 1'b0;
        class_lut_req = 1'b0;
        dfl_recip_idx = 8'd0;
        dfl_s1_max_value = '0;
        dfl_s1_sum_value = '0;
        dfl_s1_weighted_value = '0;
        dfl_s1_next_elem_cnt = '0;
        dfl_s1_next_dst_addr = '0;
        dfl_s1_flush_output = 1'b0;
        dfl_s1_final_location = 1'b0;
        class_s1_final_group = 1'b0;
        class_s1_next_elem_cnt = '0;
        class_s1_next_dst_addr = '0;
        class_s1_flush_output = 1'b0;
        gap_row_count_n = gap_row_count_q;
        for (int i = 0; i < 32; i++) begin
            gap_acc_n[i] = gap_acc_q[i];
        end
        for (int i = 0; i < LUT_LANES; i++) begin
            dfl_exp_idx[i] = 8'd0;
            class_lut_idx[i] = 8'd0;
        end

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
            gap_row_count_n = '0;
            for (int i = 0; i < 32; i++) begin
                gap_acc_n[i] = '0;
            end
            if (cfg_length_i == 0) begin
                state_n = ST_DONE;
            end else begin
                state_n = ST_READ_IN;
            end
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    // waiting for start
                end

            ST_READ_IN: begin
                if (cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                    if (!rfifo_empty_i && !rhs_rfifo_empty_i) begin
                        in_buf_n = rfifo_data_i;
                        rhs_buf_n = rhs_rfifo_data_i;
                        rfifo_pop_o = 1'b1;
                        rhs_rfifo_pop_o = 1'b1;
                        state_n = ST_PROCESS;
                    end
                end else if (!rfifo_empty_i) begin
                    in_buf_n = rfifo_data_i;
                    rfifo_pop_o = 1'b1;
                    if (cfg_mode_i == MODE_DFL4_ROW32_Q8) begin
                        state_n = ST_DFL_EXP_REQ;
                    end else if (cfg_mode_i == MODE_CLASS_SIGMOID_ROW32_HIGH16) begin
                        state_n = ST_CLASS_LUT_REQ;
                    end else if (cfg_mode_i == MODE_GLOBAL_AVGPOOL_C32) begin
                        state_n = (gap_spatial_count == 32'd0) ? ST_DONE : ST_GAP_ACCUM;
                    end else begin
                        state_n = ST_PROCESS;
                    end
                end
            end

            ST_GAP_ACCUM: begin
                for (int i = 0; i < 32; i++) begin
                    gap_acc_n[i] = gap_acc_q[i] +
                                   {{24{in_buf_q[i * 8 + 7]}}, in_buf_q[i * 8 +: 8]};
                end
                elem_cnt_n = gap_next_elem_cnt;

                if (gap_group_done) begin
                    gap_row_count_n = '0;
                    for (int i = 0; i < 32; i++) begin
                        gap_acc_n[i] = '0;
                    end
                    if (wfifo_full_i) begin
                        state_n = ST_GAP_PUSH;
                    end else if (gap_final_input) begin
                        state_n = ST_DONE;
                    end else begin
                        state_n = ST_READ_IN;
                    end
                end else begin
                    gap_row_count_n = gap_row_count_q + 32'd1;
                    state_n = ST_READ_IN;
                end
            end

            ST_GAP_PUSH: begin
                if (!wfifo_full_i) begin
                    if (elem_cnt_q >= cfg_length_i) begin
                        state_n = ST_DONE;
                    end else begin
                        state_n = ST_READ_IN;
                    end
                end
            end

            ST_CLASS_LUT_REQ: begin
                for (int i = 0; i < LUT_LANES; i++) begin
                    class_lut_idx[i] = select_input_byte(
                        in_buf_q,
                        {1'b0, 5'd16 + {1'b0, class_group_q, 2'b00} + 5'(i)}
                    );
                end
                class_lut_req = 1'b1;
                state_n = ST_CLASS_LUT_WAIT;
            end

            ST_CLASS_LUT_WAIT: begin
                class_s1_final_group = (class_group_q == 2'd3);
                class_s1_next_elem_cnt = elem_cnt_q + (class_s1_final_group ? 32'd32 : 32'd0);
                class_s1_next_dst_addr = dst_addr_q + (class_s1_final_group ? 32'd16 : 32'd0);
                class_s1_flush_output = class_s1_final_group &&
                                        ((class_s1_next_dst_addr[4:0] == 5'd0) ||
                                         (class_s1_next_elem_cnt >= cfg_length_i));

                if (class_s1_flush_output && wfifo_full_i) begin
                    elem_cnt_n = class_s1_next_elem_cnt;
                    dst_addr_n = class_s1_next_dst_addr;
                    class_group_n = 2'd0;
                    state_n = ST_CLASS_PUSH;
                end else begin
                    if (class_s1_final_group) begin
                        elem_cnt_n = class_s1_next_elem_cnt;
                        dst_addr_n = class_s1_next_dst_addr;
                        class_group_n = 2'd0;
                        if (class_s1_next_elem_cnt >= cfg_length_i) begin
                            state_n = ST_DONE;
                        end else begin
                            state_n = ST_READ_IN;
                        end
                    end else begin
                        class_group_n = class_group_q + 2'd1;
                        state_n = ST_CLASS_LUT_REQ;
                    end
                end
            end

            ST_CLASS_PUSH: begin
                if (!wfifo_full_i) begin
                    if (elem_cnt_q >= cfg_length_i) begin
                        state_n = ST_DONE;
                    end else begin
                        state_n = ST_READ_IN;
                    end
                end
            end

            ST_DFL_EXP_REQ: begin
                dfl_s1_max_value = max4_i8(select_dfl_byte(in_buf_q, dfl_side_q, 2'd0),
                                           select_dfl_byte(in_buf_q, dfl_side_q, 2'd1),
                                           select_dfl_byte(in_buf_q, dfl_side_q, 2'd2),
                                           select_dfl_byte(in_buf_q, dfl_side_q, 2'd3));
                for (int i = 0; i < LUT_LANES; i++) begin
                    dfl_exp_idx[i] = select_dfl_byte(in_buf_q, dfl_side_q, 2'(i)) - dfl_s1_max_value[7:0];
                end
                dfl_exp_req = 1'b1;
                state_n = ST_DFL_EXP_WAIT;
            end

            ST_DFL_EXP_WAIT: begin
                dfl_s1_sum_value = {2'd0, lut_rdata_bank0[0][15:0]} +
                                   {2'd0, lut_rdata_bank0[1][15:0]} +
                                   {2'd0, lut_rdata_bank0[2][15:0]} +
                                   {2'd0, lut_rdata_bank0[3][15:0]};
                dfl_s1_weighted_value = {3'd0, lut_rdata_bank0[1][15:0]} +
                                        ({2'd0, lut_rdata_bank0[2][15:0]} << 1) +
                                        ({2'd0, lut_rdata_bank0[3][15:0]} * 19'd3);
                dfl_sum_n = dfl_s1_sum_value;
                dfl_weighted_n = dfl_s1_weighted_value;
                dfl_shift_n = msb_pos18(dfl_s1_sum_value);
                dfl_recip_idx = recip_index_from_sum(dfl_s1_sum_value);
                dfl_recip_req = 1'b1;
                state_n = ST_DFL_RECIP_WAIT;
            end

            ST_DFL_RECIP_WAIT: begin
                dfl_s1_final_location = (dfl_side_q == 2'd3);
                dfl_s1_next_elem_cnt = elem_cnt_q + (dfl_s1_final_location ? 32'd32 : 32'd0);
                dfl_s1_next_dst_addr = dst_addr_q + (dfl_s1_final_location ? 32'd8 : 32'd0);
                dfl_s1_flush_output = dfl_s1_final_location &&
                                      ((dfl_s1_next_dst_addr[4:0] == 5'd0) ||
                                       (dfl_s1_next_elem_cnt >= cfg_length_i));

                if (dfl_s1_flush_output && wfifo_full_i) begin
                    elem_cnt_n = dfl_s1_next_elem_cnt;
                    dst_addr_n = dfl_s1_next_dst_addr;
                    dfl_side_n = 2'd0;
                    state_n = ST_DFL_PUSH;
                end else begin
                    if (dfl_s1_final_location) begin
                        elem_cnt_n = dfl_s1_next_elem_cnt;
                        dst_addr_n = dfl_s1_next_dst_addr;
                        dfl_side_n = 2'd0;
                        if (dfl_s1_next_elem_cnt >= cfg_length_i) begin
                            state_n = ST_DONE;
                        end else begin
                            state_n = ST_READ_IN;
                        end
                    end else begin
                        dfl_side_n = dfl_side_q + 2'd1;
                        state_n = ST_DFL_EXP_REQ;
                    end
                end
            end

            ST_DFL_PUSH: begin
                if (!wfifo_full_i) begin
                    if (elem_cnt_q >= cfg_length_i) begin
                        state_n = ST_DONE;
                    end else begin
                        state_n = ST_READ_IN;
                    end
                end
            end

            ST_PROCESS: begin
                remaining_elems = cfg_length_i - elem_cnt_q;
                p1_src_addr_n = src_addr_q;
                p1_rhs_addr_n = rhs_addr_q;
                p1_dst_addr_n = dst_addr_q;

                if (remaining_elems == 0) begin
                    state_n = ST_DONE;
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
                        max_lanes_1 = (LUT_LANES < remaining_elems) ? LUT_LANES[5:0] : (remaining_elems > 6'd31 ? 6'd31 : 6'(remaining_elems));
                    end
                    max_lanes_2 = (max_lanes_1 < in_avail) ? max_lanes_1 : in_avail;
                    max_lanes_3 = ((cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) &&
                                   rhs_avail < max_lanes_2) ? rhs_avail : max_lanes_2;
                    max_lanes_4 = (max_lanes_3 < out_avail_elems) ? max_lanes_3 : out_avail_elems;
                    num_valid_lanes = max_lanes_4;

                    if (num_valid_lanes > 0) begin
                        s1_sram_req = (cfg_mode_i != MODE_MUL_Q7 && cfg_mode_i != MODE_ADD_I8);
                        p1_num_valid_lanes_n = num_valid_lanes;
                        p1_valid_n = 1'b1;
                        p1_lhs_data_n = in_buf_q;
                        p1_rhs_data_n = rhs_buf_q;

                        src_addr_n = src_addr_q + 32'(num_valid_lanes);
                        rhs_addr_n = rhs_addr_q + 32'(num_valid_lanes);
                        elem_cnt_n = elem_cnt_q + 32'(num_valid_lanes);

                        if (cfg_mode_i == MODE_8BIT || cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                            dst_addr_n = dst_addr_q + 32'(num_valid_lanes);
                        end else if (cfg_mode_i == MODE_16BIT) begin
                            dst_addr_n = dst_addr_q + 32'(num_valid_lanes * 2);
                        end else begin
                            dst_addr_n = dst_addr_q + 32'(num_valid_lanes * 4);
                        end

                        if (cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                            if (elem_cnt_n == cfg_length_i) begin
                                p1_flush_done_n = 1'b1;
                            end else begin
                                p1_flush_mid_n = 1'b1;
                            end
                            state_n = ST_WAIT_FLUSH;
                        end else if (dst_addr_n[4:0] == 0) begin
                            p1_flush_mid_n = 1'b1;
                            state_n = ST_WAIT_FLUSH;
                        end else if (elem_cnt_n == cfg_length_i) begin
                            p1_flush_done_n = 1'b1;
                            state_n = ST_WAIT_FLUSH;
                        end else if (src_addr_n[4:0] == 0) begin
                            state_n = ST_READ_IN;
                        end
                    end
                end
            end

            ST_WAIT_FLUSH: begin
                if (s2_flush_mid_completed) begin
                    if (cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                        if (((src_addr_q[4:0] == 0) || (rhs_addr_q[4:0] == 0)) && elem_cnt_q < cfg_length_i) begin
                            state_n = ST_READ_IN;
                        end else begin
                            state_n = ST_PROCESS;
                        end
                    end else if (src_addr_q[4:0] == 0 && elem_cnt_q < cfg_length_i) begin
                        state_n = ST_READ_IN;
                    end else begin
                        state_n = ST_PROCESS;
                    end
                end else if (s2_flush_done_completed) begin
                    state_n = ST_DONE;
                end
            end
            
            ST_DONE: begin
                // Waiting for new start
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

    always_comb begin
        logic [31:0] lut_val;
        logic [4:0]  cur_out_off;
        lut_val = '0;
        cur_out_off = '0;
        dfl_s2_out_value = '0;
        dfl_s2_out_off = '0;
        dfl_s2_final_location = 1'b0;
        dfl_s2_next_elem_cnt = '0;
        dfl_s2_next_dst_addr = '0;
        dfl_s2_flush_output = 1'b0;
        class_s2_final_group = 1'b0;
        class_s2_next_elem_cnt = '0;
        class_s2_next_dst_addr = '0;
        class_s2_flush_output = 1'b0;
        
        s2_flush_mid_completed = 1'b0;
        s2_flush_done_completed = 1'b0;
        wfifo_push_o = 1'b0;
        out_buf_n = out_buf_q;
        out_be_n  = out_be_q;
        out_base_n = out_base_q;
        
        s2_out_buf_comb = out_buf_q;
        s2_out_be_comb  = out_be_q;

        if (cfg_start_i) begin
            out_base_n = cfg_dst_ptr_i;
            out_buf_n  = '0;
            out_be_n   = '0;
        end else if (state_q == ST_GAP_ACCUM) begin
            if (gap_group_done) begin
                for (int i = 0; i < 32; i++) begin
                    s2_out_buf_comb[i * 8 +: 8] = avg_i8_byte(
                        gap_acc_q[i] + {{24{in_buf_q[i * 8 + 7]}}, in_buf_q[i * 8 +: 8]},
                        gap_spatial_count
                    );
                    s2_out_be_comb[i] = 1'b1;
                end

                if (!wfifo_full_i) begin
                    wfifo_push_o = 1'b1;
                    out_buf_n = '0;
                    out_be_n = '0;
                end else begin
                    out_buf_n = s2_out_buf_comb;
                    out_be_n = s2_out_be_comb;
                end
            end
        end else if (state_q == ST_GAP_PUSH) begin
            if (!wfifo_full_i) begin
                wfifo_push_o = 1'b1;
                out_buf_n = '0;
                out_be_n = '0;
            end
        end else if (state_q == ST_CLASS_LUT_WAIT) begin
            class_s2_final_group = (class_group_q == 2'd3);
            class_s2_next_elem_cnt = elem_cnt_q + (class_s2_final_group ? 32'd32 : 32'd0);
            class_s2_next_dst_addr = dst_addr_q + (class_s2_final_group ? 32'd16 : 32'd0);
            class_s2_flush_output = class_s2_final_group &&
                                    ((class_s2_next_dst_addr[4:0] == 5'd0) ||
                                     (class_s2_next_elem_cnt >= cfg_length_i));

            s2_out_buf_comb = out_buf_q;
            s2_out_be_comb = out_be_q;
            for (int i = 0; i < LUT_LANES; i++) begin
                cur_out_off = dst_addr_q[4:0] + {1'b0, class_group_q, 2'b00} + 5'(i);
                lut_val = active_lut_bank_q ? lut_rdata_bank1[i] : lut_rdata_bank0[i];
                s2_out_buf_comb[cur_out_off * 8 +: 8] = lut_val[7:0];
                s2_out_be_comb[cur_out_off] = 1'b1;
            end

            if (class_s2_flush_output && !wfifo_full_i) begin
                wfifo_push_o = 1'b1;
                out_buf_n = '0;
                out_be_n = '0;
            end else begin
                out_buf_n = s2_out_buf_comb;
                out_be_n = s2_out_be_comb;
            end
        end else if (state_q == ST_DFL_RECIP_WAIT) begin
            dfl_s2_out_value = dfl_q8_from_recip(dfl_weighted_q, lut_rdata_bank1[0], dfl_shift_q);
            dfl_s2_out_off = dst_addr_q[4:0] + {2'd0, dfl_side_q, 1'b0};
            dfl_s2_final_location = (dfl_side_q == 2'd3);
            dfl_s2_next_elem_cnt = elem_cnt_q + (dfl_s2_final_location ? 32'd32 : 32'd0);
            dfl_s2_next_dst_addr = dst_addr_q + (dfl_s2_final_location ? 32'd8 : 32'd0);
            dfl_s2_flush_output = dfl_s2_final_location &&
                                  ((dfl_s2_next_dst_addr[4:0] == 5'd0) ||
                                   (dfl_s2_next_elem_cnt >= cfg_length_i));

            s2_out_buf_comb = out_buf_q;
            s2_out_be_comb = out_be_q;
            s2_out_buf_comb[dfl_s2_out_off * 8 +: 16] = dfl_s2_out_value;
            s2_out_be_comb[dfl_s2_out_off +: 2] = 2'b11;

            if (dfl_s2_flush_output && !wfifo_full_i) begin
                wfifo_push_o = 1'b1;
                out_buf_n = '0;
                out_be_n = '0;
            end else begin
                out_buf_n = s2_out_buf_comb;
                out_be_n = s2_out_be_comb;
            end
        end else if (state_q == ST_DFL_PUSH) begin
            if (!wfifo_full_i) begin
                wfifo_push_o = 1'b1;
                out_buf_n = '0;
                out_be_n = '0;
            end
        end else if (state_q == ST_CLASS_PUSH) begin
            if (!wfifo_full_i) begin
                wfifo_push_o = 1'b1;
                out_buf_n = '0;
                out_be_n = '0;
            end
        end else if (p1_valid_q && !s2_stall) begin
            // Process lanes directly into combinational buffer
            if (cfg_mode_i == MODE_MUL_Q7 || cfg_mode_i == MODE_ADD_I8) begin
                for (int i = 0; i < 32; i++) begin
                    if (i < p1_num_valid_lanes_q) begin
                        cur_out_off = p1_dst_addr_q[4:0] + 5'(i);
                        if (cfg_mode_i == MODE_MUL_Q7) begin
                            s2_out_buf_comb[cur_out_off * 8 +: 8] = mul_q7_byte(
                                select_input_byte(p1_lhs_data_q, {1'b0, p1_src_addr_q[4:0]} + 6'(i)),
                                select_input_byte(p1_rhs_data_q, {1'b0, p1_rhs_addr_q[4:0]} + 6'(i))
                            );
                        end else begin
                            s2_out_buf_comb[cur_out_off * 8 +: 8] = add_i8_byte(
                                select_input_byte(p1_lhs_data_q, {1'b0, p1_src_addr_q[4:0]} + 6'(i)),
                                select_input_byte(p1_rhs_data_q, {1'b0, p1_rhs_addr_q[4:0]} + 6'(i))
                            );
                        end
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

            if (p1_flush_mid_q) begin
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

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q    <= ST_IDLE;
            src_addr_q <= '0;
            rhs_addr_q <= '0;
            dst_addr_q <= '0;
            elem_cnt_q <= '0;
            in_buf_q   <= '0;
            rhs_buf_q  <= '0;
            out_base_q <= '0;
            out_buf_q  <= '0;
            out_be_q   <= '0;
            dfl_side_q <= '0;
            class_group_q <= '0;
            dfl_sum_q <= '0;
            dfl_weighted_q <= '0;
            dfl_shift_q <= '0;
            gap_row_count_q <= '0;
            for (int i = 0; i < 32; i++) begin
                gap_acc_q[i] <= '0;
            end
            
            p1_valid_q <= 1'b0;
            p1_num_valid_lanes_q <= '0;
            p1_lhs_data_q <= '0;
            p1_rhs_data_q <= '0;
            p1_src_addr_q <= '0;
            p1_rhs_addr_q <= '0;
            p1_dst_addr_q <= '0;
            p1_flush_mid_q <= 1'b0;
            p1_flush_done_q <= 1'b0;
        end else begin
            state_q    <= state_n;
            src_addr_q <= src_addr_n;
            rhs_addr_q <= rhs_addr_n;
            dst_addr_q <= dst_addr_n;
            elem_cnt_q <= elem_cnt_n;
            in_buf_q   <= in_buf_n;
            rhs_buf_q  <= rhs_buf_n;
            
            out_base_q <= out_base_n;
            out_buf_q  <= out_buf_n;
            out_be_q   <= out_be_n;
            dfl_side_q <= dfl_side_n;
            class_group_q <= class_group_n;
            dfl_sum_q <= dfl_sum_n;
            dfl_weighted_q <= dfl_weighted_n;
            dfl_shift_q <= dfl_shift_n;
            gap_row_count_q <= gap_row_count_n;
            for (int i = 0; i < 32; i++) begin
                gap_acc_q[i] <= gap_acc_n[i];
            end

            if (!s2_stall) begin
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
    end

endmodule
