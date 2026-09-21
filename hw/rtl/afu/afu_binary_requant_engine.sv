`default_nettype none

module afu_binary_requant_engine #(
    parameter int unsigned LANES = 32,
    parameter int unsigned META_DEPTH = 16
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         start_i,
    input  logic                         chain_i,
    input  logic [31:0]                  length_i,
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

    input  logic                         lhs_empty_i,
    output logic                         lhs_pop_o,
    input  logic [LANES*8-1:0]           lhs_data_i,
    input  logic                         chain_valid_i,
    output logic                         chain_ready_o,
    input  logic [LANES*8+LANES-1:0]     chain_data_i,
    input  logic                         rhs_empty_i,
    output logic                         rhs_pop_o,
    input  logic [LANES*8-1:0]           rhs_data_i,

    output logic                         out_valid_o,
    input  logic                         out_ready_i,
    output logic [LANES*8+LANES-1:0]     out_data_o,
    output logic                         done_o,
    output logic                         busy_o,
    output logic                         error_o,
    output logic                         input_wait_o,
    output logic                         rhs_wait_o,
    output logic                         output_stall_o
);

    localparam int unsigned PTR_WIDTH = $clog2(META_DEPTH);

    logic active_q;
    logic done_q;
    logic error_q;
    logic [31:0] input_bytes_q;
    logic [31:0] output_bytes_q;

    logic [31:0] meta_be_q [META_DEPTH];
    logic [PTR_WIDTH-1:0] meta_write_q;
    logic [PTR_WIDTH-1:0] meta_read_q;
    logic [PTR_WIDTH:0] meta_count_q;
    logic meta_full;
    logic meta_empty;

    logic input_valid;
    logic input_ready;
    logic input_fire;
    logic [255:0] input_lhs;
    logic [31:0] input_be;
    logic [5:0] input_bytes;
    logic [31:0] remaining_bytes;

    logic pipeline_out_valid;
    logic pipeline_out_ready;
    logic [LANES-1:0][7:0] pipeline_packed;
    logic pipeline_invalid;
    logic output_fire;
    logic [5:0] output_bytes;

    function automatic logic [31:0] low_byte_mask(input logic [5:0] bytes);
        logic [32:0] wide;
        begin
            wide = (33'd1 << bytes) - 1'b1;
            low_byte_mask = bytes >= 6'd32 ? 32'hffff_ffff : wide[31:0];
        end
    endfunction

    function automatic logic [5:0] byte_count(input logic [31:0] mask);
        logic [5:0] count;
        begin
            count = '0;
            for (int unsigned byte_index = 0; byte_index < 32; byte_index++)
                count = count + mask[byte_index];
            byte_count = count;
        end
    endfunction

    assign meta_full = meta_count_q == (PTR_WIDTH+1)'(META_DEPTH);
    assign meta_empty = meta_count_q == 0;
    assign remaining_bytes = length_i - input_bytes_q;
    assign input_bytes = remaining_bytes >= 32 ? 6'd32 : 6'(remaining_bytes);
    assign input_be = chain_i ? chain_data_i[287:256] : low_byte_mask(input_bytes);
    assign input_lhs = chain_i ? chain_data_i[255:0] : lhs_data_i;
    assign input_valid = active_q && input_bytes_q < length_i && !rhs_empty_i &&
                         (chain_i ? chain_valid_i : !lhs_empty_i) && !meta_full;
    assign input_fire = input_valid && input_ready;

    assign lhs_pop_o = input_fire && !chain_i;
    assign rhs_pop_o = input_fire;
    assign chain_ready_o = active_q && chain_i && input_bytes_q < length_i &&
                           !rhs_empty_i && !meta_full && input_ready;

    assign pipeline_out_ready = out_ready_i && !meta_empty;
    assign out_valid_o = pipeline_out_valid && !meta_empty;
    assign out_data_o = {meta_be_q[meta_read_q], pipeline_packed};
    assign output_fire = out_valid_o && out_ready_i;
    assign output_bytes = byte_count(meta_be_q[meta_read_q]);

    assign done_o = done_q;
    assign busy_o = active_q;
    assign error_o = error_q;
    assign input_wait_o = active_q && input_bytes_q < length_i &&
                          !(chain_i ? chain_valid_i : !lhs_empty_i);
    assign rhs_wait_o = active_q && input_bytes_q < length_i && rhs_empty_i;
    assign output_stall_o = pipeline_out_valid && !out_ready_i;

    afu_binary_requant_pipeline #(
        .LANES(LANES)
    ) i_binary_requant_pipeline (
        .clk_i,
        .rst_ni,
        .flush_i              (start_i),
        .in_valid_i           (input_valid),
        .in_ready_o           (input_ready),
        .lhs_i                (input_lhs),
        .rhs_i                (rhs_data_i),
        .mode_i,
        .lhs_multiplier_i,
        .lhs_shift_i,
        .rhs_multiplier_i,
        .rhs_shift_i,
        .output_multiplier_i,
        .output_shift_i,
        .lhs_zero_point_i,
        .rhs_zero_point_i,
        .output_zero_point_i,
        .clamp_min_i,
        .clamp_max_i,
        .double_round_shift_i,
        .out_valid_o          (pipeline_out_valid),
        .out_ready_i          (pipeline_out_ready),
        .packed_o             (pipeline_packed),
        .invalid_o            (pipeline_invalid)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            active_q <= 1'b0;
            done_q <= 1'b0;
            error_q <= 1'b0;
            input_bytes_q <= '0;
            output_bytes_q <= '0;
            meta_write_q <= '0;
            meta_read_q <= '0;
            meta_count_q <= '0;
        end else if (start_i) begin
            active_q <= length_i != 0;
            done_q <= length_i == 0;
            error_q <= length_i == 0;
            input_bytes_q <= '0;
            output_bytes_q <= '0;
            meta_write_q <= '0;
            meta_read_q <= '0;
            meta_count_q <= '0;
        end else begin
            if (input_fire) begin
                meta_be_q[meta_write_q] <= input_be;
                meta_write_q <= meta_write_q + 1'b1;
                input_bytes_q <= input_bytes_q + 32'(byte_count(input_be));
            end
            if (output_fire) begin
                meta_read_q <= meta_read_q + 1'b1;
                output_bytes_q <= output_bytes_q + 32'(output_bytes);
                error_q <= error_q || pipeline_invalid;
                if (output_bytes_q + 32'(output_bytes) >= length_i) begin
                    active_q <= 1'b0;
                    done_q <= 1'b1;
                end
            end
            unique case ({input_fire, output_fire})
                2'b10: meta_count_q <= meta_count_q + 1'b1;
                2'b01: meta_count_q <= meta_count_q - 1'b1;
                default: ;
            endcase
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rst_ni && active_q) begin
            assert (input_bytes_q <= length_i)
                else $error("AFU binary accepted more input bytes than configured");
            assert (output_bytes_q <= length_i)
                else $error("AFU binary produced more output bytes than configured");
        end
    end
`endif

endmodule

`default_nettype wire
