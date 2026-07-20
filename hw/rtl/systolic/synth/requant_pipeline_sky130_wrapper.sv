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

    requant_pipeline_mapped i_mapped (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .in_valid_i   (in_valid_i),
        .in_ready_o   (in_ready_o),
        .acc_i        (acc_i),
        .bias_i       (bias_i),
        .multiplier_i (multiplier_i),
        .shift_i      (shift_i),
        .zero_point_i (zero_point_i),
        .clamp_min_i  (clamp_min_i),
        .clamp_max_i  (clamp_max_i),
        .out_valid_o  (out_valid_o),
        .out_ready_i  (out_ready_i),
        .packed_o     (packed_o),
        .invalid_o    (invalid_o)
    );

endmodule

`default_nettype wire
