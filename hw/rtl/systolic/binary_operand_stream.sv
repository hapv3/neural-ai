`default_nettype none

module binary_operand_stream #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned FIFO_DEPTH = 8
)(
    input  logic                          clk_i,
    input  logic                          rst_ni,
    input  logic                          start_i,
    input  logic [ADDR_WIDTH-1:0]         base_addr_i,
    input  logic [31:0]                   row_count_i,
    input  logic [31:0]                   row_stride_bytes_i,
    input  logic [31:0]                   tile_cols_i,

    output logic                          obi_req_o,
    input  logic                          obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_addr_o,
    output logic                          obi_we_o,
    output logic [(DATA_WIDTH/8)-1:0]     obi_be_o,
    output logic [DATA_WIDTH-1:0]         obi_wdata_o,
    input  logic                          obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0]         obi_rdata_i,

    output logic                          out_valid_o,
    input  logic                          out_ready_i,
    output logic [DATA_WIDTH-1:0]         out_data_o,
    output logic                          busy_o,
    output logic                          done_o
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned CREDIT_WIDTH = $clog2(FIFO_DEPTH + 1);
    localparam logic [CREDIT_WIDTH-1:0] FIFO_DEPTH_CREDITS = CREDIT_WIDTH'(FIFO_DEPTH);

    logic [ADDR_WIDTH-1:0] request_ptr_q;
    logic [31:0] request_count_q;
    logic [31:0] response_count_q;
    logic [31:0] request_col_q;
    logic [31:0] row_count_q;
    logic [31:0] row_stride_bytes_q;
    logic [31:0] tile_cols_q;
    logic [CREDIT_WIDTH-1:0] reserved_q;
    logic active_q;
    logic done_q;

    logic response_fifo_full;
    logic response_fifo_empty;
    logic response_fifo_push;
    logic response_fifo_pop;
    logic request_fire;
    logic output_fire;

    function automatic logic [ADDR_WIDTH-1:0] next_request_ptr(
        input logic [ADDR_WIDTH-1:0] ptr,
        input logic [31:0] col,
        input logic [31:0] row_stride_bytes,
        input logic [31:0] tile_cols
    );
        logic [31:0] row_span;
        begin
            row_span = (tile_cols - 32'd1) << $clog2(BEAT_BYTES);
            if (row_stride_bytes != 0 && tile_cols != 0 &&
                (col + 32'd1) == tile_cols) begin
                next_request_ptr = ptr + row_stride_bytes - row_span;
            end else begin
                next_request_ptr = ptr + BEAT_BYTES;
            end
        end
    endfunction

    assign out_valid_o = !response_fifo_empty;
    assign response_fifo_pop = out_valid_o && out_ready_i;
    assign output_fire = response_fifo_pop;
    assign response_fifo_push = obi_rvalid_i;

    // reserved_q counts both returned FIFO entries and accepted reads whose
    // response has not arrived yet. A simultaneous output transfer frees a
    // credit soon enough for a new request whose response arrives later.
    assign obi_req_o = active_q && request_count_q < row_count_q &&
        ((reserved_q < FIFO_DEPTH_CREDITS) || output_fire);
    assign obi_addr_o = request_ptr_q;
    assign obi_we_o = 1'b0;
    assign obi_be_o = '1;
    assign obi_wdata_o = '0;
    assign request_fire = obi_req_o && obi_gnt_i;

    assign busy_o = active_q;
    assign done_o = done_q;

    /* verilator lint_off PINCONNECTEMPTY */
    fifo_v3 #(
        .FALL_THROUGH(1'b1),
        .DEPTH       (FIFO_DEPTH),
        .dtype       (logic [DATA_WIDTH-1:0])
    ) i_response_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (start_i),
        .testmode_i (1'b0),
        .full_o     (response_fifo_full),
        .empty_o    (response_fifo_empty),
        .usage_o    (),
        .data_i     (obi_rdata_i),
        .push_i     (response_fifo_push),
        .data_o     (out_data_o),
        .pop_i      (response_fifo_pop)
    );
    /* verilator lint_on PINCONNECTEMPTY */

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            request_ptr_q <= '0;
            request_count_q <= '0;
            response_count_q <= '0;
            request_col_q <= '0;
            row_count_q <= '0;
            row_stride_bytes_q <= '0;
            tile_cols_q <= '0;
            reserved_q <= '0;
            active_q <= 1'b0;
            done_q <= 1'b0;
        end else begin
            done_q <= 1'b0;
            if (start_i) begin
                request_ptr_q <= base_addr_i;
                request_count_q <= '0;
                response_count_q <= '0;
                request_col_q <= '0;
                row_count_q <= row_count_i;
                row_stride_bytes_q <= row_stride_bytes_i;
                tile_cols_q <= tile_cols_i;
                reserved_q <= '0;
                active_q <= row_count_i != 0;
                done_q <= row_count_i == 0;
            end else begin
                if (request_fire) begin
                    request_ptr_q <= next_request_ptr(request_ptr_q, request_col_q,
                        row_stride_bytes_q, tile_cols_q);
                    request_count_q <= request_count_q + 32'd1;
                    if (tile_cols_q != 0 && request_col_q + 32'd1 == tile_cols_q)
                        request_col_q <= '0;
                    else
                        request_col_q <= request_col_q + 32'd1;
                end

                if (response_fifo_push)
                    response_count_q <= response_count_q + 32'd1;

                unique case ({request_fire, output_fire})
                    2'b10: reserved_q <= reserved_q + 1'b1;
                    2'b01: reserved_q <= reserved_q - 1'b1;
                    default: reserved_q <= reserved_q;
                endcase

                if (active_q && output_fire && request_count_q == row_count_q &&
                    reserved_q == 1) begin
                    active_q <= 1'b0;
                    done_q <= 1'b1;
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (!start_i) begin
            assert (!(response_fifo_push && response_fifo_full && !response_fifo_pop))
                else $error("binary operand response FIFO overflow");
            assert (response_count_q <= request_count_q)
                else $error("binary operand response without accepted request");
        end
    end
`endif

endmodule

`default_nettype wire
