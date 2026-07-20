module npu_systolic_array #(
    parameter int unsigned ARRAY_DIM = 32
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic                              weight_load_en_i,
    input  logic                              clear_acc_i,
    input  logic                              compute_en_i,
    input  logic                              ofm_ready_i,
    input  logic signed [ARRAY_DIM*8-1:0]     weight_data_i,
    input  logic signed [ARRAY_DIM*8-1:0]     ifm_data_i,
    input  logic signed [ARRAY_DIM*32-1:0]    psum_data_i,
    output logic signed [ARRAY_DIM*32-1:0]    ofm_data_o,
    output logic                              ofm_valid_o
);
    npu_systolic_array_mapped i_mapped_array (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .weight_load_en_i (weight_load_en_i),
        .clear_acc_i      (clear_acc_i),
        .compute_en_i     (compute_en_i),
        .ofm_ready_i      (ofm_ready_i),
        .weight_data_i    (weight_data_i),
        .ifm_data_i       (ifm_data_i),
        .psum_data_i      (psum_data_i),
        .ofm_data_o       (ofm_data_o),
        .ofm_valid_o      (ofm_valid_o)
    );
endmodule
