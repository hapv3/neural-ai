`default_nettype none

module requant_pipeline #(
    parameter int unsigned ARRAY_DIM = 32
)(
    input  logic                       clk_i,
    input  logic                       rst_ni,
    input  logic                       in_valid_i,
    output logic                       in_ready_o,
    input  logic [ARRAY_DIM-1:0][31:0] acc_i,
    input  logic [ARRAY_DIM-1:0][31:0] bias_i,
    input  logic [ARRAY_DIM-1:0][31:0] multiplier_i,
    input  logic [ARRAY_DIM-1:0][7:0]  shift_i,
    input  logic [ARRAY_DIM-1:0][31:0] zero_point_i,
    input  logic [31:0]                clamp_min_i,
    input  logic [31:0]                clamp_max_i,
    output logic                       out_valid_o,
    input  logic                       out_ready_i,
    output logic [255:0]               packed_o,
    output logic                       invalid_o
);

    localparam int unsigned BIASED_WIDTH = 34;
    localparam int unsigned PRODUCT_WIDTH = BIASED_WIDTH + 32;
    localparam int unsigned EXT_WIDTH = PRODUCT_WIDTH + 1;

    typedef logic signed [BIASED_WIDTH-1:0] biased_t;
    typedef logic signed [31:0] multiplier_t;
    typedef logic signed [PRODUCT_WIDTH-1:0] product_t;
    typedef logic signed [EXT_WIDTH-1:0] ext_t;

    logic s1_valid_q;
    logic s2_valid_q;
    logic s3_valid_q;
    logic s1_ready;
    logic s2_ready;
    logic s3_ready;

    product_t s1_scaled_q [ARRAY_DIM];
    logic [ARRAY_DIM-1:0][7:0]  s1_shift_q;
    logic [ARRAY_DIM-1:0][31:0] s1_zero_point_q;
    logic [31:0] s1_clamp_min_q;
    logic [31:0] s1_clamp_max_q;
    logic        s1_invalid_q;

    ext_t s2_with_zero_point_q [ARRAY_DIM];
    logic [31:0] s2_clamp_min_q;
    logic [31:0] s2_clamp_max_q;
    logic        s2_invalid_q;

    logic [255:0] s3_packed_q;
    logic         s3_invalid_q;

    assign s3_ready = out_ready_i || !s3_valid_q;
    assign s2_ready = s3_ready || !s2_valid_q;
    assign s1_ready = s2_ready || !s1_valid_q;

    assign in_ready_o  = s1_ready;
    assign out_valid_o = s3_valid_q;
    assign packed_o    = s3_packed_q;
    assign invalid_o   = s3_invalid_q;

    function automatic ext_t round_shift_right(
        input product_t value,
        input logic [7:0] shift
    );
        ext_t value_ext;
        ext_t offset;
        ext_t magnitude;
        ext_t rounded_magnitude;
        begin
            value_ext = ext_t'(value);
            if (shift == 8'd0) begin
                round_shift_right = value_ext;
            end else begin
                offset = ext_t'(1) <<< (shift - 8'd1);
                if (value_ext >= ext_t'(0)) begin
                    round_shift_right = (value_ext + offset) >>> shift;
                end else begin
                    magnitude = -value_ext;
                    rounded_magnitude = (magnitude + offset) >>> shift;
                    round_shift_right = -rounded_magnitude;
                end
            end
        end
    endfunction

    function automatic ext_t sign_extend_i32(input logic signed [31:0] value);
        begin
            sign_extend_i32 = {{(EXT_WIDTH-32){value[31]}}, value};
        end
    endfunction

    function automatic logic signed [31:0] clamp_ext_to_i32(
        input ext_t value,
        input logic signed [31:0] min_value,
        input logic signed [31:0] max_value
    );
        ext_t min_ext;
        ext_t max_ext;
        begin
            min_ext = sign_extend_i32(min_value);
            max_ext = sign_extend_i32(max_value);
            if (value < min_ext) begin
                clamp_ext_to_i32 = min_value;
            end else if (value > max_ext) begin
                clamp_ext_to_i32 = max_value;
            end else begin
                clamp_ext_to_i32 = value[31:0];
            end
        end
    endfunction

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            s1_valid_q <= 1'b0;
            s2_valid_q <= 1'b0;
            s3_valid_q <= 1'b0;
            s1_shift_q <= '0;
            s1_zero_point_q <= '0;
            s1_clamp_min_q <= '0;
            s1_clamp_max_q <= '0;
            s1_invalid_q <= 1'b0;
            s2_clamp_min_q <= '0;
            s2_clamp_max_q <= '0;
            s2_invalid_q <= 1'b0;
            s3_packed_q <= '0;
            s3_invalid_q <= 1'b0;
            for (int unsigned i = 0; i < ARRAY_DIM; i++) begin
                s1_scaled_q[i] <= '0;
                s2_with_zero_point_q[i] <= '0;
            end
        end else begin
            if (s3_ready) begin
                s3_valid_q <= s2_valid_q;
                if (s2_valid_q) begin
                    s3_packed_q <= '0;
                    s3_invalid_q <= s2_invalid_q ||
                                    ($signed(s2_clamp_min_q) > $signed(s2_clamp_max_q));
                    for (int unsigned i = 0; i < ARRAY_DIM; i++) begin
                        logic signed [31:0] clamped;
                        clamped = clamp_ext_to_i32(s2_with_zero_point_q[i],
                                                   $signed(s2_clamp_min_q),
                                                   $signed(s2_clamp_max_q));
                        s3_packed_q[i*8 +: 8] <= clamped[7:0];
                    end
                end
            end

            if (s2_ready) begin
                s2_valid_q <= s1_valid_q;
                if (s1_valid_q) begin
                    s2_clamp_min_q <= s1_clamp_min_q;
                    s2_clamp_max_q <= s1_clamp_max_q;
                    s2_invalid_q <= s1_invalid_q;
                    for (int unsigned i = 0; i < ARRAY_DIM; i++) begin
                        ext_t rounded;
                        rounded = round_shift_right(s1_scaled_q[i],
                                                    (s1_shift_q[i] > 8'd31) ? 8'd31 : s1_shift_q[i]);
                        s2_with_zero_point_q[i] <= rounded + sign_extend_i32($signed(s1_zero_point_q[i]));
                    end
                end
            end

            if (s1_ready) begin
                s1_valid_q <= in_valid_i;
                if (in_valid_i) begin
                    s1_shift_q <= shift_i;
                    s1_zero_point_q <= zero_point_i;
                    s1_clamp_min_q <= clamp_min_i;
                    s1_clamp_max_q <= clamp_max_i;
                    s1_invalid_q <= ($signed(clamp_min_i) > $signed(clamp_max_i));
                    for (int unsigned i = 0; i < ARRAY_DIM; i++) begin
                        biased_t biased;
                        biased = biased_t'($signed(acc_i[i])) + biased_t'($signed(bias_i[i]));
                        s1_scaled_q[i] <= biased * multiplier_t'(multiplier_i[i]);
                        if (shift_i[i] > 8'd31) begin
                            s1_invalid_q <= 1'b1;
                        end
                    end
                end
            end
        end
    end

endmodule

`default_nettype wire
