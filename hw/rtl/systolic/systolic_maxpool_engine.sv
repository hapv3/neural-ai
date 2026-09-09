`default_nettype none

module systolic_maxpool_engine #(
    parameter int unsigned LANES = 32,
    parameter int unsigned ELEM_WIDTH = 8,
    parameter int unsigned TAP_COUNT_WIDTH = 8
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic flush_i,

    input  logic [31:0] kernel_vectors_i,
    input  logic [LANES-1:0][ELEM_WIDTH-1:0] in_data_i,
    input  logic in_valid_i,
    output logic in_ready_o,

    output logic [LANES-1:0][ELEM_WIDTH-1:0] out_data_o,
    output logic out_valid_o,
    input  logic out_ready_i
);

    typedef logic [LANES-1:0][ELEM_WIDTH-1:0] row_t;

    row_t acc_q;
    row_t out_q;
    row_t next_acc;
    logic [TAP_COUNT_WIDTH-1:0] tap_count_q;
    logic input_fire;

    always_comb begin
        for (int unsigned lane = 0; lane < LANES; lane++) begin
            next_acc[lane] = (tap_count_q == '0) ? in_data_i[lane] :
                             (($signed(acc_q[lane]) >= $signed(in_data_i[lane])) ?
                              acc_q[lane] : in_data_i[lane]);
        end
    end

    // Preserve the controller's original one-entry output behavior: a new
    // input row is accepted on the cycle after the pending result is consumed.
    assign in_ready_o = !out_valid_o;
    assign input_fire = in_valid_i && in_ready_o;
    assign out_data_o = out_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            acc_q <= '0;
            out_q <= '0;
            out_valid_o <= 1'b0;
            tap_count_q <= '0;
        end else if (flush_i) begin
            acc_q <= '0;
            out_q <= '0;
            out_valid_o <= 1'b0;
            tap_count_q <= '0;
        end else begin
            if (out_valid_o && out_ready_i) begin
                out_valid_o <= 1'b0;
            end

            if (input_fire) begin
                acc_q <= next_acc;
                if (({24'd0, tap_count_q} + 32'd1) == kernel_vectors_i) begin
                    out_q <= next_acc;
                    out_valid_o <= 1'b1;
                    tap_count_q <= '0;
                end else begin
                    tap_count_q <= tap_count_q + 1'b1;
                end
            end
        end
    end

endmodule

`default_nettype wire
