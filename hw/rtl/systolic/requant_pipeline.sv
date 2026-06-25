`default_nettype none

module requant_pipeline #(
    parameter int unsigned ARRAY_DIM = 32
)(
    input  logic [ARRAY_DIM-1:0][31:0] acc_i,
    input  logic [ARRAY_DIM-1:0][31:0] bias_i,
    input  logic [ARRAY_DIM-1:0][31:0] multiplier_i,
    input  logic [ARRAY_DIM-1:0][7:0]  shift_i,
    input  logic [ARRAY_DIM-1:0][31:0] zero_point_i,
    input  logic [31:0]                clamp_min_i,
    input  logic [31:0]                clamp_max_i,
    output logic [255:0]               packed_o,
    output logic                       invalid_o
);

    function automatic logic signed [63:0] round_shift_right(
        input logic signed [63:0] value,
        input logic [7:0] shift
    );
        logic signed [63:0] offset;
        logic signed [63:0] magnitude;
        logic signed [63:0] rounded_magnitude;
        begin
            if (shift == 8'd0) begin
                round_shift_right = value;
            end else begin
                offset = 64'sd1 <<< (shift - 8'd1);
                if (value >= 64'sd0) begin
                    round_shift_right = (value + offset) >>> shift;
                end else begin
                    magnitude = -value;
                    rounded_magnitude = (magnitude + offset) >>> shift;
                    round_shift_right = -rounded_magnitude;
                end
            end
        end
    endfunction

    function automatic logic signed [31:0] clamp_i64_to_i32(
        input logic signed [63:0] value,
        input logic signed [31:0] min_value,
        input logic signed [31:0] max_value
    );
        begin
            if (value < 64'(min_value)) begin
                clamp_i64_to_i32 = min_value;
            end else if (value > 64'(max_value)) begin
                clamp_i64_to_i32 = max_value;
            end else begin
                clamp_i64_to_i32 = value[31:0];
            end
        end
    endfunction

    always_comb begin
        packed_o = '0;
        invalid_o = ($signed(clamp_min_i) > $signed(clamp_max_i));

        for (int unsigned i = 0; i < ARRAY_DIM; i++) begin
            logic signed [63:0] biased;
            logic signed [63:0] scaled;
            logic signed [63:0] rounded;
            logic signed [63:0] with_zero_point;
            logic signed [31:0] clamped;

            if (shift_i[i] > 8'd31) begin
                invalid_o = 1'b1;
            end

            biased = 64'($signed(acc_i[i])) + 64'($signed(bias_i[i]));
            scaled = biased * 64'($signed(multiplier_i[i]));
            rounded = round_shift_right(scaled, (shift_i[i] > 8'd31) ? 8'd31 : shift_i[i]);
            with_zero_point = rounded + 64'($signed(zero_point_i[i]));
            clamped = clamp_i64_to_i32(with_zero_point, $signed(clamp_min_i), $signed(clamp_max_i));
            packed_o[i*8 +: 8] = clamped[7:0];
        end
    end

endmodule

`default_nettype wire
