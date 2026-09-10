`default_nettype none

module systolic_output_postprocess #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned LANES = 32,
    parameter int unsigned BINARY_FIFO_DEPTH = 8
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic flush_i,
    input  logic job_start_i,

    input  logic requant_enable_i,
    input  logic [LANES-1:0][31:0] acc_i,
    input  logic acc_valid_i,
    output logic acc_ready_o,
    input  logic [LANES-1:0][31:0] bias_i,
    input  logic [LANES-1:0][31:0] multiplier_i,
    input  logic [LANES-1:0][7:0] shift_i,
    input  logic [LANES-1:0][31:0] zero_point_i,
    input  logic [31:0] clamp_min_i,
    input  logic [31:0] clamp_max_i,

    input  logic binary_enable_i,
    input  logic binary_active_i,
    input  logic [1:0] binary_mode_i,
    input  logic [31:0] binary_rhs_ptr_i,
    input  logic [31:0] binary_rhs_row_stride_bytes_i,
    input  logic [31:0] binary_rhs_tile_cols_i,
    input  logic [31:0] row_count_i,
    input  logic [31:0] binary_lhs_multiplier_i,
    input  logic [6:0] binary_lhs_shift_i,
    input  logic [31:0] binary_rhs_multiplier_i,
    input  logic [6:0] binary_rhs_shift_i,
    input  logic [31:0] binary_output_multiplier_i,
    input  logic [6:0] binary_output_shift_i,
    input  logic signed [31:0] binary_lhs_zero_point_i,
    input  logic signed [31:0] binary_rhs_zero_point_i,
    input  logic signed [31:0] binary_output_zero_point_i,
    input  logic signed [31:0] binary_clamp_min_i,
    input  logic signed [31:0] binary_clamp_max_i,
    input  logic [5:0] binary_double_round_shift_i,
    input  logic binary_forbidden_i,

    output logic obi_req_o,
    input  logic obi_gnt_i,
    output logic [ADDR_WIDTH-1:0] obi_addr_o,
    output logic obi_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_be_o,
    output logic [DATA_WIDTH-1:0] obi_wdata_o,
    input  logic obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0] obi_rdata_i,

    output logic out_valid_o,
    input  logic out_ready_i,
    output logic [DATA_WIDTH-1:0] packed_o,
    output logic invalid_o,
    output logic requant_config_invalid_o,
    output logic binary_config_invalid_o,
    output logic binary_busy_o,

    output logic debug_requant_out_valid_o,
    output logic debug_requant_out_ready_o
);

    logic requant_out_valid;
    logic requant_out_ready;
    logic [DATA_WIDTH-1:0] requant_packed_data;
    logic requant_invalid;
    logic binary_in_valid;
    logic binary_in_ready;
    logic binary_out_valid;
    logic binary_out_ready;
    logic [DATA_WIDTH-1:0] binary_packed_data;
    logic binary_invalid;
    logic binary_operand_start;
    logic binary_operand_valid;
    logic binary_operand_ready;
    logic [DATA_WIDTH-1:0] binary_operand_data;
    logic binary_operand_done;

    assign binary_operand_start = job_start_i && binary_enable_i &&
                                  !binary_config_invalid_o;
    assign binary_in_valid = binary_active_i && requant_out_valid &&
                             binary_operand_valid;
    assign requant_out_ready = binary_active_i ?
                               (binary_in_ready && binary_operand_valid) :
                               out_ready_i;
    assign binary_operand_ready = binary_active_i && binary_in_ready &&
                                  requant_out_valid;
    assign binary_out_ready = binary_active_i && out_ready_i;
    assign out_valid_o = binary_active_i ? binary_out_valid : requant_out_valid;
    assign packed_o = binary_active_i ? binary_packed_data : requant_packed_data;
    assign invalid_o = binary_active_i ? binary_invalid : requant_invalid;
    assign debug_requant_out_valid_o = requant_out_valid;
    assign debug_requant_out_ready_o = requant_out_ready;

    always_comb begin
        requant_config_invalid_o = ($signed(clamp_min_i) > $signed(clamp_max_i));
        for (int unsigned lane = 0; lane < LANES; lane++) begin
            if (shift_i[lane] > 8'd31) begin
                requant_config_invalid_o = 1'b1;
            end
        end
    end

    always_comb begin
        binary_config_invalid_o = binary_enable_i && (
            !requant_enable_i || binary_mode_i > 2'd2 ||
            $signed(binary_output_multiplier_i) <= 0 ||
            (binary_mode_i != 2'd2 &&
                ($signed(binary_lhs_multiplier_i) <= 0 ||
                 $signed(binary_rhs_multiplier_i) <= 0)) ||
            binary_lhs_shift_i > 7'd63 ||
            binary_rhs_shift_i > 7'd63 ||
            binary_output_shift_i > 7'd63 ||
            binary_double_round_shift_i > 6'd30 ||
            binary_clamp_min_i > binary_clamp_max_i ||
            binary_rhs_ptr_i[4:0] != 5'd0 || binary_forbidden_i);
    end

    requant_pipeline #(
        .ARRAY_DIM(LANES)
    ) i_requant_pipeline (
        .clk_i,
        .rst_ni,
        .in_valid_i   (acc_valid_i),
        .in_ready_o   (acc_ready_o),
        .acc_i,
        .bias_i,
        .multiplier_i,
        .shift_i,
        .zero_point_i,
        .clamp_min_i,
        .clamp_max_i,
        .out_valid_o  (requant_out_valid),
        .out_ready_i  (requant_out_ready),
        .packed_o     (requant_packed_data),
        .invalid_o    (requant_invalid)
    );

    binary_operand_stream #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FIFO_DEPTH(BINARY_FIFO_DEPTH)
    ) i_binary_operand_stream (
        .clk_i,
        .rst_ni,
        .start_i            (binary_operand_start),
        .base_addr_i        (binary_rhs_ptr_i),
        .row_count_i        (row_count_i),
        .row_stride_bytes_i (binary_rhs_row_stride_bytes_i),
        .tile_cols_i        (binary_rhs_tile_cols_i),
        .obi_req_o,
        .obi_gnt_i,
        .obi_addr_o,
        .obi_we_o,
        .obi_be_o,
        .obi_wdata_o,
        .obi_rvalid_i,
        .obi_rdata_i,
        .out_valid_o        (binary_operand_valid),
        .out_ready_i        (binary_operand_ready),
        .out_data_o         (binary_operand_data),
        .busy_o             (binary_busy_o),
        .done_o             (binary_operand_done)
    );

    binary_requant_pipeline #(
        .LANES(LANES)
    ) i_binary_requant_pipeline (
        .clk_i,
        .rst_ni,
        .flush_i              (flush_i),
        .in_valid_i           (binary_in_valid),
        .in_ready_o           (binary_in_ready),
        .lhs_i                (requant_packed_data),
        .rhs_i                (binary_operand_data),
        .mode_i               (binary_mode_i),
        .lhs_multiplier_i     (binary_lhs_multiplier_i),
        .lhs_shift_i          (binary_lhs_shift_i),
        .rhs_multiplier_i     (binary_rhs_multiplier_i),
        .rhs_shift_i          (binary_rhs_shift_i),
        .output_multiplier_i  (binary_output_multiplier_i),
        .output_shift_i       (binary_output_shift_i),
        .lhs_zero_point_i     (binary_lhs_zero_point_i),
        .rhs_zero_point_i     (binary_rhs_zero_point_i),
        .output_zero_point_i  (binary_output_zero_point_i),
        .clamp_min_i          (binary_clamp_min_i),
        .clamp_max_i          (binary_clamp_max_i),
        .double_round_shift_i (binary_double_round_shift_i),
        .out_valid_o          (binary_out_valid),
        .out_ready_i          (binary_out_ready),
        .packed_o             (binary_packed_data),
        .invalid_o            (binary_invalid)
    );

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (binary_operand_done) begin
            assert (!binary_busy_o)
                else $error("binary operand stream done while busy");
        end
    end
`endif

endmodule

`default_nettype wire
