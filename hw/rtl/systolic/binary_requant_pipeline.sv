`default_nettype none

module binary_requant_pipeline #(
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

    typedef logic signed [63:0] value_t;
    typedef logic signed [95:0] product_t;

    logic s1_valid_q, s2_valid_q, s3_valid_q, s4_valid_q, s5_valid_q;
    logic s1_ready, s2_ready, s3_ready, s4_ready, s5_ready;

    value_t s1_lhs_product_q [LANES];
    value_t s1_rhs_product_q [LANES];
    logic [1:0] s1_mode_q;
    logic [6:0] s1_lhs_shift_q, s1_rhs_shift_q;
    logic signed [31:0] s1_output_multiplier_q;
    logic [6:0] s1_output_shift_q;
    logic signed [31:0] s1_output_zero_point_q;
    logic signed [31:0] s1_clamp_min_q, s1_clamp_max_q;
    logic [5:0] s1_double_round_shift_q;
    logic s1_invalid_q;

    value_t s2_value_q [LANES];
    logic signed [31:0] s2_output_multiplier_q;
    logic [6:0] s2_output_shift_q;
    logic signed [31:0] s2_output_zero_point_q;
    logic signed [31:0] s2_clamp_min_q, s2_clamp_max_q;
    logic [5:0] s2_double_round_shift_q;
    logic s2_invalid_q;

    product_t s3_product_q [LANES];
    value_t s3_source_q [LANES];
    logic [6:0] s3_output_shift_q;
    logic signed [31:0] s3_output_zero_point_q;
    logic signed [31:0] s3_clamp_min_q, s3_clamp_max_q;
    logic [5:0] s3_double_round_shift_q;
    logic s3_invalid_q;

    value_t s4_value_q [LANES];
    logic signed [31:0] s4_clamp_min_q, s4_clamp_max_q;
    logic s4_invalid_q;

    logic [LANES-1:0][7:0] s5_packed_q;
    logic s5_invalid_q;

    function automatic value_t scale_product(
        input product_t product,
        input value_t source,
        input logic [6:0] shift,
        input logic [5:0] double_round_shift
    );
        product_t adjusted;
        product_t round_offset;
        product_t double_offset;
        begin
            adjusted = product;
            if (shift != 0) begin
                round_offset = product_t'(96'd1) <<< (shift - 1'b1);
                adjusted = adjusted + round_offset;
                if (double_round_shift != 0 &&
                    shift > (7'd31 - {1'b0, double_round_shift})) begin
                    double_offset = product_t'(96'd1) <<< (7'd30 - {1'b0, double_round_shift});
                    adjusted = adjusted + (source < 0 ? -double_offset : double_offset);
                end
                scale_product = value_t'(adjusted >>> shift);
            end else begin
                scale_product = value_t'(adjusted);
            end
        end
    endfunction

    function automatic logic [7:0] clamp_byte(
        input value_t value,
        input logic signed [31:0] min_value,
        input logic signed [31:0] max_value
    );
        value_t min_ext;
        value_t max_ext;
        begin
            min_ext = value_t'(min_value);
            max_ext = value_t'(max_value);
            if (value < min_ext) clamp_byte = min_value[7:0];
            else if (value > max_ext) clamp_byte = max_value[7:0];
            else clamp_byte = value[7:0];
        end
    endfunction

    assign s5_ready = out_ready_i || !s5_valid_q;
    assign s4_ready = s5_ready || !s4_valid_q;
    assign s3_ready = s4_ready || !s3_valid_q;
    assign s2_ready = s3_ready || !s2_valid_q;
    assign s1_ready = s2_ready || !s1_valid_q;

    assign in_ready_o = s1_ready;
    assign out_valid_o = s5_valid_q;
    assign packed_o = s5_packed_q;
    assign invalid_o = s5_invalid_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            s1_valid_q <= 1'b0;
            s2_valid_q <= 1'b0;
            s3_valid_q <= 1'b0;
            s4_valid_q <= 1'b0;
            s5_valid_q <= 1'b0;
            s1_mode_q <= MODE_ADD;
            s1_lhs_shift_q <= '0;
            s1_rhs_shift_q <= '0;
            s1_output_multiplier_q <= '0;
            s1_output_shift_q <= '0;
            s1_output_zero_point_q <= '0;
            s1_clamp_min_q <= '0;
            s1_clamp_max_q <= '0;
            s1_double_round_shift_q <= '0;
            s1_invalid_q <= 1'b0;
            s2_output_multiplier_q <= '0;
            s2_output_shift_q <= '0;
            s2_output_zero_point_q <= '0;
            s2_clamp_min_q <= '0;
            s2_clamp_max_q <= '0;
            s2_double_round_shift_q <= '0;
            s2_invalid_q <= 1'b0;
            s3_output_shift_q <= '0;
            s3_output_zero_point_q <= '0;
            s3_clamp_min_q <= '0;
            s3_clamp_max_q <= '0;
            s3_double_round_shift_q <= '0;
            s3_invalid_q <= 1'b0;
            s4_clamp_min_q <= '0;
            s4_clamp_max_q <= '0;
            s4_invalid_q <= 1'b0;
            s5_packed_q <= '0;
            s5_invalid_q <= 1'b0;
            for (int unsigned lane = 0; lane < LANES; lane++) begin
                s1_lhs_product_q[lane] <= '0;
                s1_rhs_product_q[lane] <= '0;
                s2_value_q[lane] <= '0;
                s3_product_q[lane] <= '0;
                s3_source_q[lane] <= '0;
                s4_value_q[lane] <= '0;
            end
        end else if (flush_i) begin
            s1_valid_q <= 1'b0;
            s2_valid_q <= 1'b0;
            s3_valid_q <= 1'b0;
            s4_valid_q <= 1'b0;
            s5_valid_q <= 1'b0;
        end else begin
            if (s5_ready) begin
                s5_valid_q <= s4_valid_q;
                if (s4_valid_q) begin
                    s5_invalid_q <= s4_invalid_q || (s4_clamp_min_q > s4_clamp_max_q);
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s5_packed_q[lane] <= clamp_byte(s4_value_q[lane],
                            s4_clamp_min_q, s4_clamp_max_q);
                    end
                end
            end

            if (s4_ready) begin
                s4_valid_q <= s3_valid_q;
                if (s3_valid_q) begin
                    s4_clamp_min_q <= s3_clamp_min_q;
                    s4_clamp_max_q <= s3_clamp_max_q;
                    s4_invalid_q <= s3_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s4_value_q[lane] <= scale_product(
                            s3_product_q[lane], s3_source_q[lane],
                            s3_output_shift_q, s3_double_round_shift_q) +
                            value_t'(s3_output_zero_point_q);
                    end
                end
            end

            if (s3_ready) begin
                s3_valid_q <= s2_valid_q;
                if (s2_valid_q) begin
                    s3_output_shift_q <= s2_output_shift_q;
                    s3_output_zero_point_q <= s2_output_zero_point_q;
                    s3_clamp_min_q <= s2_clamp_min_q;
                    s3_clamp_max_q <= s2_clamp_max_q;
                    s3_double_round_shift_q <= s2_double_round_shift_q;
                    s3_invalid_q <= s2_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        s3_product_q[lane] <= product_t'(s2_value_q[lane]) *
                            product_t'(s2_output_multiplier_q);
                        s3_source_q[lane] <= s2_value_q[lane];
                    end
                end
            end

            if (s2_ready) begin
                s2_valid_q <= s1_valid_q;
                if (s1_valid_q) begin
                    s2_output_multiplier_q <= s1_output_multiplier_q;
                    s2_output_shift_q <= s1_output_shift_q;
                    s2_output_zero_point_q <= s1_output_zero_point_q;
                    s2_clamp_min_q <= s1_clamp_min_q;
                    s2_clamp_max_q <= s1_clamp_max_q;
                    s2_double_round_shift_q <= s1_double_round_shift_q;
                    s2_invalid_q <= s1_invalid_q;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        value_t lhs_scaled;
                        value_t rhs_scaled;
                        lhs_scaled = scale_product(product_t'(s1_lhs_product_q[lane]),
                            s1_lhs_product_q[lane], s1_lhs_shift_q,
                            s1_double_round_shift_q);
                        rhs_scaled = scale_product(product_t'(s1_rhs_product_q[lane]),
                            s1_rhs_product_q[lane], s1_rhs_shift_q,
                            s1_double_round_shift_q);
                        unique case (s1_mode_q)
                            MODE_ADD: s2_value_q[lane] <= lhs_scaled + rhs_scaled;
                            MODE_SUB: s2_value_q[lane] <= lhs_scaled - rhs_scaled;
                            default:  s2_value_q[lane] <= s1_lhs_product_q[lane];
                        endcase
                    end
                end
            end

            if (s1_ready) begin
                s1_valid_q <= in_valid_i;
                if (in_valid_i) begin
                    s1_mode_q <= mode_i;
                    s1_lhs_shift_q <= lhs_shift_i;
                    s1_rhs_shift_q <= rhs_shift_i;
                    s1_output_multiplier_q <= output_multiplier_i;
                    s1_output_shift_q <= output_shift_i;
                    s1_output_zero_point_q <= output_zero_point_i;
                    s1_clamp_min_q <= clamp_min_i;
                    s1_clamp_max_q <= clamp_max_i;
                    s1_double_round_shift_q <= double_round_shift_i;
                    s1_invalid_q <= mode_i > MODE_MUL || output_multiplier_i <= 0 ||
                        (mode_i != MODE_MUL &&
                            (lhs_multiplier_i <= 0 || rhs_multiplier_i <= 0)) ||
                        lhs_zero_point_i < -32'sd128 || lhs_zero_point_i > 32'sd127 ||
                        rhs_zero_point_i < -32'sd128 || rhs_zero_point_i > 32'sd127 ||
                        output_zero_point_i < -32'sd128 || output_zero_point_i > 32'sd127 ||
                        clamp_min_i < -32'sd128 || clamp_max_i > 32'sd127 ||
                        lhs_shift_i > 7'd63 ||
                        rhs_shift_i > 7'd63 || output_shift_i > 7'd63 ||
                        double_round_shift_i > 6'd30 || clamp_min_i > clamp_max_i;
                    for (int unsigned lane = 0; lane < LANES; lane++) begin
                        value_t lhs_centered;
                        value_t rhs_centered;
                        lhs_centered = value_t'($signed(lhs_i[lane])) -
                            value_t'(lhs_zero_point_i);
                        rhs_centered = value_t'($signed(rhs_i[lane])) -
                            value_t'(rhs_zero_point_i);
                        if (mode_i == MODE_MUL) begin
                            s1_lhs_product_q[lane] <= lhs_centered * rhs_centered;
                            s1_rhs_product_q[lane] <= '0;
                        end else begin
                            s1_lhs_product_q[lane] <= lhs_centered *
                                value_t'(lhs_multiplier_i);
                            s1_rhs_product_q[lane] <= rhs_centered *
                                value_t'(rhs_multiplier_i);
                        end
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
