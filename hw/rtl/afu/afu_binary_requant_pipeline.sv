`default_nettype none

module afu_binary_requant_pipeline #(
    parameter int unsigned LANES = 32
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         flush_i,
    input  logic                         in_valid_i,
    output logic                         in_ready_o,
    input  logic [LANES-1:0][7:0]        lhs_i,
    input  logic [LANES-1:0][7:0]        rhs_i,
    input  logic [1:0]                   mode_i,
    input  logic signed [31:0]           lhs_multiplier_i,
    input  logic [6:0]                   lhs_shift_i,
    input  logic signed [31:0]           rhs_multiplier_i,
    input  logic [6:0]                   rhs_shift_i,
    input  logic signed [31:0]           output_multiplier_i,
    input  logic [6:0]                   output_shift_i,
    input  logic signed [31:0]           lhs_zero_point_i,
    input  logic signed [31:0]           rhs_zero_point_i,
    input  logic signed [31:0]           output_zero_point_i,
    input  logic signed [31:0]           clamp_min_i,
    input  logic signed [31:0]           clamp_max_i,
    input  logic [5:0]                   double_round_shift_i,
    output logic                         out_valid_o,
    input  logic                         out_ready_i,
    output logic [LANES-1:0][7:0]        packed_o,
    output logic                         invalid_o
);

    localparam logic [1:0] MODE_ADD = 2'd0;
    localparam logic [1:0] MODE_SUB = 2'd1;
    localparam logic [1:0] MODE_MUL = 2'd2;

    typedef logic signed [8:0]  centered_t;
    typedef logic signed [31:0] scaled_t;
    typedef logic signed [32:0] combined_t;
    typedef logic signed [40:0] input_product_t;
    typedef logic signed [17:0] mul_product_t;
    typedef logic signed [64:0] wide_t;

    logic [9:0] valid_q;
    logic s1_ready, s2_ready, s3_ready, s4_ready, s5_ready;
    logic s6_ready, s7_ready, s8_ready, s9_ready, s10_ready;

    centered_t s1_lhs_q [LANES];
    centered_t s1_rhs_q [LANES];
    logic [1:0] s1_mode_q;
    logic signed [31:0] s1_lhs_multiplier_q, s1_rhs_multiplier_q;
    logic [6:0] s1_lhs_shift_q, s1_rhs_shift_q;
    logic signed [31:0] s1_output_multiplier_q;
    logic [6:0] s1_output_shift_q;
    logic signed [31:0] s1_output_zero_point_q;
    logic signed [31:0] s1_clamp_min_q, s1_clamp_max_q;
    wide_t s1_lhs_round_q, s1_lhs_double_q;
    wide_t s1_rhs_round_q, s1_rhs_double_q;
    wide_t s1_output_round_q, s1_output_double_q;
    logic s1_invalid_q;

    wide_t s2_lhs_product_q [LANES];
    wide_t s2_rhs_product_q [LANES];
    logic [1:0] s2_mode_q;
    logic [6:0] s2_lhs_shift_q, s2_rhs_shift_q;
    logic signed [31:0] s2_output_multiplier_q;
    logic [6:0] s2_output_shift_q;
    logic signed [31:0] s2_output_zero_point_q;
    logic signed [31:0] s2_clamp_min_q, s2_clamp_max_q;
    wide_t s2_lhs_round_q, s2_lhs_double_q;
    wide_t s2_rhs_round_q, s2_rhs_double_q;
    wide_t s2_output_round_q, s2_output_double_q;
    logic s2_invalid_q;

    wide_t s3_lhs_adjusted_q [LANES];
    wide_t s3_rhs_adjusted_q [LANES];
    logic [1:0] s3_mode_q;
    logic [6:0] s3_lhs_shift_q, s3_rhs_shift_q;
    logic signed [31:0] s3_output_multiplier_q;
    logic [6:0] s3_output_shift_q;
    logic signed [31:0] s3_output_zero_point_q;
    logic signed [31:0] s3_clamp_min_q, s3_clamp_max_q;
    wide_t s3_output_round_q, s3_output_double_q;
    logic s3_invalid_q;

    wide_t s4_lhs_shifted_q [LANES];
    wide_t s4_rhs_shifted_q [LANES];
    logic [1:0] s4_mode_q;
    logic [2:0] s4_lhs_shift_high_q, s4_rhs_shift_high_q;
    logic signed [31:0] s4_output_multiplier_q;
    logic [6:0] s4_output_shift_q;
    logic signed [31:0] s4_output_zero_point_q;
    logic signed [31:0] s4_clamp_min_q, s4_clamp_max_q;
    wide_t s4_output_round_q, s4_output_double_q;
    logic s4_invalid_q;

    combined_t s5_combined_q [LANES];
    logic signed [31:0] s5_output_multiplier_q;
    logic [6:0] s5_output_shift_q;
    logic signed [31:0] s5_output_zero_point_q;
    logic signed [31:0] s5_clamp_min_q, s5_clamp_max_q;
    wide_t s5_output_round_q, s5_output_double_q;
    logic s5_invalid_q;

    wide_t s6_product_q [LANES];
    logic [6:0] s6_output_shift_q;
    logic signed [31:0] s6_output_zero_point_q;
    logic signed [31:0] s6_clamp_min_q, s6_clamp_max_q;
    wide_t s6_output_round_q, s6_output_double_q;
    logic s6_invalid_q;

    wide_t s7_adjusted_q [LANES];
    logic [6:0] s7_output_shift_q;
    logic signed [31:0] s7_output_zero_point_q;
    logic signed [31:0] s7_clamp_min_q, s7_clamp_max_q;
    logic s7_invalid_q;

    wide_t s8_shifted_q [LANES];
    logic [2:0] s8_output_shift_high_q;
    logic signed [31:0] s8_output_zero_point_q;
    logic signed [31:0] s8_clamp_min_q, s8_clamp_max_q;
    logic s8_invalid_q;

    wide_t s9_value_q [LANES];
    logic signed [31:0] s9_clamp_min_q, s9_clamp_max_q;
    logic s9_invalid_q;
    logic [LANES-1:0][7:0] s10_packed_q;
    logic s10_invalid_q;

    function automatic wide_t normal_round_offset(input logic [6:0] shift);
        begin
            normal_round_offset = '0;
            if (shift != 0 && shift <= 7'd63)
                normal_round_offset = wide_t'(65'd1) <<< (shift - 1'b1);
        end
    endfunction

    function automatic wide_t double_round_offset(
        input logic [6:0] shift,
        input logic [5:0] double_round_shift
    );
        begin
            double_round_offset = '0;
            if (double_round_shift != 0 && double_round_shift <= 6'd30 &&
                shift > (7'd31 - {1'b0, double_round_shift}))
                double_round_offset = wide_t'(65'd1) <<<
                    (7'd30 - {1'b0, double_round_shift});
        end
    endfunction

    function automatic logic [7:0] clamp_byte(
        input wide_t value,
        input logic signed [31:0] min_value,
        input logic signed [31:0] max_value
    );
        wide_t min_ext;
        wide_t max_ext;
        begin
            min_ext = wide_t'(min_value);
            max_ext = wide_t'(max_value);
            if (value < min_ext) clamp_byte = min_value[7:0];
            else if (value > max_ext) clamp_byte = max_value[7:0];
            else clamp_byte = value[7:0];
        end
    endfunction

    function automatic input_product_t multiply_input_scale(
        input centered_t value,
        input logic signed [31:0] multiplier
    );
        begin
            multiply_input_scale = value * multiplier;
        end
    endfunction

    function automatic mul_product_t multiply_centered(
        input centered_t lhs,
        input centered_t rhs
    );
        begin
            multiply_centered = lhs * rhs;
        end
    endfunction

    function automatic wide_t multiply_output_scale(
        input combined_t value,
        input logic signed [31:0] multiplier
    );
        begin
            multiply_output_scale = value * multiplier;
        end
    endfunction

    assign s10_ready = out_ready_i || !valid_q[9];
    assign s9_ready = s10_ready || !valid_q[8];
    assign s8_ready = s9_ready || !valid_q[7];
    assign s7_ready = s8_ready || !valid_q[6];
    assign s6_ready = s7_ready || !valid_q[5];
    assign s5_ready = s6_ready || !valid_q[4];
    assign s4_ready = s5_ready || !valid_q[3];
    assign s3_ready = s4_ready || !valid_q[2];
    assign s2_ready = s3_ready || !valid_q[1];
    assign s1_ready = s2_ready || !valid_q[0];
    assign in_ready_o = s1_ready;
    assign out_valid_o = valid_q[9];
    assign packed_o = s10_packed_q;
    assign invalid_o = s10_invalid_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_q <= '0;
        end else if (flush_i) begin
            valid_q <= '0;
        end else begin
            if (s10_ready) begin
                valid_q[9] <= valid_q[8];
                if (valid_q[8]) begin
                    s10_invalid_q <= s9_invalid_q || (s9_clamp_min_q > s9_clamp_max_q);
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s10_packed_q[lane] <= clamp_byte(
                            s9_value_q[lane], s9_clamp_min_q, s9_clamp_max_q);
                    end
                end
            end

            if (s9_ready) begin
                valid_q[8] <= valid_q[7];
                if (valid_q[7]) begin
                    s9_clamp_min_q <= s8_clamp_min_q;
                    s9_clamp_max_q <= s8_clamp_max_q;
                    s9_invalid_q <= s8_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s9_value_q[lane] <=
                            (s8_shifted_q[lane] >>> {s8_output_shift_high_q, 4'b0000}) +
                            wide_t'(s8_output_zero_point_q);
                    end
                end
            end

            if (s8_ready) begin
                valid_q[7] <= valid_q[6];
                if (valid_q[6]) begin
                    s8_output_shift_high_q <= s7_output_shift_q[6:4];
                    s8_output_zero_point_q <= s7_output_zero_point_q;
                    s8_clamp_min_q <= s7_clamp_min_q;
                    s8_clamp_max_q <= s7_clamp_max_q;
                    s8_invalid_q <= s7_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s8_shifted_q[lane] <= s7_adjusted_q[lane] >>>
                            s7_output_shift_q[3:0];
                    end
                end
            end

            if (s7_ready) begin
                valid_q[6] <= valid_q[5];
                if (valid_q[5]) begin
                    s7_output_shift_q <= s6_output_shift_q;
                    s7_output_zero_point_q <= s6_output_zero_point_q;
                    s7_clamp_min_q <= s6_clamp_min_q;
                    s7_clamp_max_q <= s6_clamp_max_q;
                    s7_invalid_q <= s6_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s7_adjusted_q[lane] <= s6_product_q[lane] +
                            s6_output_round_q +
                            (s6_product_q[lane] < 0 ?
                                -s6_output_double_q : s6_output_double_q);
                    end
                end
            end

            if (s6_ready) begin
                valid_q[5] <= valid_q[4];
                if (valid_q[4]) begin
                    s6_output_shift_q <= s5_output_shift_q;
                    s6_output_zero_point_q <= s5_output_zero_point_q;
                    s6_clamp_min_q <= s5_clamp_min_q;
                    s6_clamp_max_q <= s5_clamp_max_q;
                    s6_output_round_q <= s5_output_round_q;
                    s6_output_double_q <= s5_output_double_q;
                    s6_invalid_q <= s5_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s6_product_q[lane] <= multiply_output_scale(
                            s5_combined_q[lane], s5_output_multiplier_q);
                    end
                end
            end

            if (s5_ready) begin
                valid_q[4] <= valid_q[3];
                if (valid_q[3]) begin
                    s5_output_multiplier_q <= s4_output_multiplier_q;
                    s5_output_shift_q <= s4_output_shift_q;
                    s5_output_zero_point_q <= s4_output_zero_point_q;
                    s5_clamp_min_q <= s4_clamp_min_q;
                    s5_clamp_max_q <= s4_clamp_max_q;
                    s5_output_round_q <= s4_output_round_q;
                    s5_output_double_q <= s4_output_double_q;
                    s5_invalid_q <= s4_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        scaled_t lhs_scaled;
                        scaled_t rhs_scaled;
                        lhs_scaled = scaled_t'(s4_lhs_shifted_q[lane] >>>
                            {s4_lhs_shift_high_q, 4'b0000});
                        rhs_scaled = scaled_t'(s4_rhs_shifted_q[lane] >>>
                            {s4_rhs_shift_high_q, 4'b0000});
                        unique case (s4_mode_q)
                            MODE_ADD: s5_combined_q[lane] <=
                                combined_t'(lhs_scaled) + combined_t'(rhs_scaled);
                            MODE_SUB: s5_combined_q[lane] <=
                                combined_t'(lhs_scaled) - combined_t'(rhs_scaled);
                            default: s5_combined_q[lane] <= combined_t'(lhs_scaled);
                        endcase
                    end
                end
            end

            if (s4_ready) begin
                valid_q[3] <= valid_q[2];
                if (valid_q[2]) begin
                    s4_mode_q <= s3_mode_q;
                    s4_lhs_shift_high_q <= s3_lhs_shift_q[6:4];
                    s4_rhs_shift_high_q <= s3_rhs_shift_q[6:4];
                    s4_output_multiplier_q <= s3_output_multiplier_q;
                    s4_output_shift_q <= s3_output_shift_q;
                    s4_output_zero_point_q <= s3_output_zero_point_q;
                    s4_clamp_min_q <= s3_clamp_min_q;
                    s4_clamp_max_q <= s3_clamp_max_q;
                    s4_output_round_q <= s3_output_round_q;
                    s4_output_double_q <= s3_output_double_q;
                    s4_invalid_q <= s3_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s4_lhs_shifted_q[lane] <= s3_lhs_adjusted_q[lane] >>>
                            s3_lhs_shift_q[3:0];
                        s4_rhs_shifted_q[lane] <= s3_rhs_adjusted_q[lane] >>>
                            s3_rhs_shift_q[3:0];
                    end
                end
            end

            if (s3_ready) begin
                valid_q[2] <= valid_q[1];
                if (valid_q[1]) begin
                    s3_mode_q <= s2_mode_q;
                    s3_lhs_shift_q <= s2_lhs_shift_q;
                    s3_rhs_shift_q <= s2_rhs_shift_q;
                    s3_output_multiplier_q <= s2_output_multiplier_q;
                    s3_output_shift_q <= s2_output_shift_q;
                    s3_output_zero_point_q <= s2_output_zero_point_q;
                    s3_clamp_min_q <= s2_clamp_min_q;
                    s3_clamp_max_q <= s2_clamp_max_q;
                    s3_output_round_q <= s2_output_round_q;
                    s3_output_double_q <= s2_output_double_q;
                    s3_invalid_q <= s2_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s3_lhs_adjusted_q[lane] <= s2_lhs_product_q[lane] +
                            s2_lhs_round_q + (s2_lhs_product_q[lane] < 0 ?
                                -s2_lhs_double_q : s2_lhs_double_q);
                        s3_rhs_adjusted_q[lane] <= s2_rhs_product_q[lane] +
                            s2_rhs_round_q + (s2_rhs_product_q[lane] < 0 ?
                                -s2_rhs_double_q : s2_rhs_double_q);
                    end
                end
            end

            if (s2_ready) begin
                valid_q[1] <= valid_q[0];
                if (valid_q[0]) begin
                    s2_mode_q <= s1_mode_q;
                    s2_lhs_shift_q <= s1_mode_q == MODE_MUL ? 7'd0 : s1_lhs_shift_q;
                    s2_rhs_shift_q <= s1_mode_q == MODE_MUL ? 7'd0 : s1_rhs_shift_q;
                    s2_output_multiplier_q <= s1_output_multiplier_q;
                    s2_output_shift_q <= s1_output_shift_q;
                    s2_output_zero_point_q <= s1_output_zero_point_q;
                    s2_clamp_min_q <= s1_clamp_min_q;
                    s2_clamp_max_q <= s1_clamp_max_q;
                    s2_lhs_round_q <= s1_mode_q == MODE_MUL ? '0 : s1_lhs_round_q;
                    s2_lhs_double_q <= s1_mode_q == MODE_MUL ? '0 : s1_lhs_double_q;
                    s2_rhs_round_q <= s1_mode_q == MODE_MUL ? '0 : s1_rhs_round_q;
                    s2_rhs_double_q <= s1_mode_q == MODE_MUL ? '0 : s1_rhs_double_q;
                    s2_output_round_q <= s1_output_round_q;
                    s2_output_double_q <= s1_output_double_q;
                    s2_invalid_q <= s1_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        if (s1_mode_q == MODE_MUL) begin
                            s2_lhs_product_q[lane] <= wide_t'(
                                multiply_centered(s1_lhs_q[lane], s1_rhs_q[lane]));
                            s2_rhs_product_q[lane] <= '0;
                        end else begin
                            s2_lhs_product_q[lane] <= wide_t'(multiply_input_scale(
                                s1_lhs_q[lane], s1_lhs_multiplier_q));
                            s2_rhs_product_q[lane] <= wide_t'(multiply_input_scale(
                                s1_rhs_q[lane], s1_rhs_multiplier_q));
                        end
                    end
                end
            end

            if (s1_ready) begin
                valid_q[0] <= in_valid_i;
                if (in_valid_i) begin
                    s1_mode_q <= mode_i;
                    s1_lhs_multiplier_q <= lhs_multiplier_i;
                    s1_rhs_multiplier_q <= rhs_multiplier_i;
                    s1_lhs_shift_q <= lhs_shift_i;
                    s1_rhs_shift_q <= rhs_shift_i;
                    s1_output_multiplier_q <= output_multiplier_i;
                    s1_output_shift_q <= output_shift_i;
                    s1_output_zero_point_q <= output_zero_point_i;
                    s1_clamp_min_q <= clamp_min_i;
                    s1_clamp_max_q <= clamp_max_i;
                    s1_lhs_round_q <= normal_round_offset(lhs_shift_i);
                    s1_lhs_double_q <= double_round_offset(lhs_shift_i, double_round_shift_i);
                    s1_rhs_round_q <= normal_round_offset(rhs_shift_i);
                    s1_rhs_double_q <= double_round_offset(rhs_shift_i, double_round_shift_i);
                    s1_output_round_q <= normal_round_offset(output_shift_i);
                    s1_output_double_q <= double_round_offset(output_shift_i, double_round_shift_i);
                    s1_invalid_q <= mode_i > MODE_MUL || output_multiplier_i <= 0 ||
                        (mode_i != MODE_MUL &&
                            (lhs_multiplier_i <= 0 || rhs_multiplier_i <= 0)) ||
                        lhs_zero_point_i < -32'sd128 || lhs_zero_point_i > 32'sd127 ||
                        rhs_zero_point_i < -32'sd128 || rhs_zero_point_i > 32'sd127 ||
                        output_zero_point_i < -32'sd128 || output_zero_point_i > 32'sd127 ||
                        clamp_min_i < -32'sd128 || clamp_max_i > 32'sd127 ||
                        lhs_shift_i > 7'd63 || rhs_shift_i > 7'd63 ||
                        output_shift_i > 7'd63 || double_round_shift_i > 6'd30 ||
                        clamp_min_i > clamp_max_i;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s1_lhs_q[lane] <= centered_t'($signed(lhs_i[lane])) -
                            centered_t'($signed(lhs_zero_point_i[7:0]));
                        s1_rhs_q[lane] <= centered_t'($signed(rhs_i[lane])) -
                            centered_t'($signed(rhs_zero_point_i[7:0]));
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
