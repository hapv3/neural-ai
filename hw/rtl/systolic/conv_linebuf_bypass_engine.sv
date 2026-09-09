`default_nettype none

module conv_linebuf_bypass_engine #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic start_i,
    input  logic last_i,
    input  logic [ADDR_WIDTH-1:0] spatial_addr_i,
    input  logic signed [31:0] base_ih_i,
    input  logic signed [31:0] base_iw_i,
    input  logic [15:0] input_h_i,
    input  logic [15:0] input_w_i,
    input  logic [31:0] channel_addr_offset_i,
    input  logic [5:0] valid_bytes_i,
    input  logic [5:0] lane_base_i,
    input  logic c32_blocked_mode_i,

    output logic obi_req_o,
    input  logic obi_gnt_i,
    output logic [ADDR_WIDTH-1:0] obi_addr_o,
    input  logic obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0] obi_rdata_i,

    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] row_o,
    output logic row_valid_o,
    input  logic row_ready_i,
    output logic [31:0] fetch_beats_o,
    output logic [31:0] emitted_vectors_o,
    output logic [4:0] debug_state_o
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BYTE_SEL_BITS = $clog2(BEAT_BYTES);

    typedef enum logic [2:0] {
        BYPASS_IDLE,
        BYPASS_PREP,
        BYPASS_REQ0,
        BYPASS_WAIT0,
        BYPASS_REQ1,
        BYPASS_WAIT1,
        BYPASS_EMIT
    } bypass_state_e;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;

    bypass_state_e state_q;
    logic [ADDR_WIDTH-1:0] address_q;
    logic [ADDR_WIDTH-1:0] pending_beat_addr_q;
    logic [DATA_WIDTH-1:0] beat0_q;
    logic [5:0] valid_bytes_q;
    logic crosses_beat_q;
    input_row_t row_q;
    logic row_valid_q;
    logic [31:0] fetch_beats_q;
    logic [31:0] emitted_vectors_q;

    logic [ADDR_WIDTH-1:0] candidate_address;
    logic candidate_in_bounds;
    logic candidate_crosses;

    assign candidate_address = spatial_addr_i + channel_addr_offset_i;
    assign candidate_in_bounds = (base_ih_i >= 32'sd0) &&
                                 (base_iw_i >= 32'sd0) &&
                                 (base_ih_i < $signed({16'd0, input_h_i})) &&
                                 (base_iw_i < $signed({16'd0, input_w_i})) &&
                                 (valid_bytes_i != 6'd0);
    assign candidate_crosses =
        ({2'b00, candidate_address[BYTE_SEL_BITS-1:0]} + {1'b0, valid_bytes_i}) >
        (BYTE_SEL_BITS+2)'(BEAT_BYTES);

    assign obi_req_o = (state_q == BYPASS_REQ0) || (state_q == BYPASS_REQ1);
    assign obi_addr_o = pending_beat_addr_q;
    assign row_o = row_q;
    assign row_valid_o = row_valid_q;
    assign fetch_beats_o = fetch_beats_q;
    assign emitted_vectors_o = emitted_vectors_q;

    // Preserve the legacy parent debug-state encoding cycle-for-cycle.
    always_comb begin
        unique case (state_q)
            BYPASS_PREP:  debug_state_o = 5'd9;
            BYPASS_REQ0:  debug_state_o = 5'd10;
            BYPASS_WAIT0: debug_state_o = 5'd11;
            BYPASS_REQ1:  debug_state_o = 5'd12;
            BYPASS_WAIT1: debug_state_o = 5'd13;
            BYPASS_EMIT:  debug_state_o = 5'd8;
            default:      debug_state_o = 5'd0;
        endcase
    end

    function automatic logic [ADDR_WIDTH-1:0] beat_base(
        input logic [ADDR_WIDTH-1:0] address
    );
        beat_base = {address[ADDR_WIDTH-1:BYTE_SEL_BITS], {BYTE_SEL_BITS{1'b0}}};
    endfunction

    function automatic logic [DATA_WIDTH-1:0] merge_beats(
        input logic [DATA_WIDTH-1:0] beat0,
        input logic [DATA_WIDTH-1:0] beat1,
        input logic [BYTE_SEL_BITS-1:0] addr_lsb,
        input logic [5:0] valid_bytes
    );
        logic [DATA_WIDTH-1:0] merged;
        logic [BYTE_SEL_BITS:0] byte_sel;
        begin
            merged = '0;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                byte_sel = {1'b0, addr_lsb} + (BYTE_SEL_BITS+1)'(lane);
                if (lane < valid_bytes) begin
                    if (byte_sel < (BYTE_SEL_BITS+1)'(BEAT_BYTES)) begin
                        merged[(lane << 3) +: 8] =
                            beat0[{byte_sel[BYTE_SEL_BITS-1:0], 3'b000} +: 8];
                    end else begin
                        merged[(lane << 3) +: 8] =
                            beat1[{byte_sel[BYTE_SEL_BITS-1:0], 3'b000} +: 8];
                    end
                end
            end
            merge_beats = merged;
        end
    endfunction

    function automatic input_row_t unpack_row(
        input logic [DATA_WIDTH-1:0] data,
        input logic [5:0] valid_bytes,
        input logic [5:0] lane_base
    );
        input_row_t row;
        logic [6:0] dst_lane;
        begin
            row = '0;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                dst_lane = {1'b0, lane_base} + 7'(lane);
                if ((lane < valid_bytes) && (dst_lane < 7'(ARRAY_DIM))) begin
                    row[dst_lane[4:0]] = data[(lane << 3) +: 8];
                end
            end
            unpack_row = row;
        end
    endfunction

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= BYPASS_IDLE;
            address_q <= '0;
            pending_beat_addr_q <= '0;
            beat0_q <= '0;
            valid_bytes_q <= '0;
            crosses_beat_q <= 1'b0;
            row_q <= '0;
            row_valid_q <= 1'b0;
            fetch_beats_q <= '0;
            emitted_vectors_q <= '0;
        end else begin
            unique case (state_q)
                BYPASS_IDLE: begin
                    row_valid_q <= 1'b0;
                    if (start_i) begin
                        fetch_beats_q <= '0;
                        emitted_vectors_q <= '0;
                        state_q <= BYPASS_PREP;
                    end
                end

                BYPASS_PREP: begin
                    if (!candidate_in_bounds) begin
                        row_q <= '0;
                        row_valid_q <= 1'b1;
                        state_q <= BYPASS_EMIT;
                    end else begin
                        address_q <= candidate_address;
                        valid_bytes_q <= valid_bytes_i;
                        crosses_beat_q <= candidate_crosses;
                        pending_beat_addr_q <= beat_base(candidate_address);
                        state_q <= BYPASS_REQ0;
                    end
                end

                BYPASS_REQ0: begin
                    if (obi_gnt_i) begin
                        state_q <= BYPASS_WAIT0;
                    end
                end

                BYPASS_WAIT0: begin
                    if (obi_rvalid_i) begin
                        beat0_q <= obi_rdata_i;
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        if (crosses_beat_q) begin
                            pending_beat_addr_q <= beat_base(address_q) + ADDR_WIDTH'(BEAT_BYTES);
                            state_q <= BYPASS_REQ1;
                        end else begin
                            row_q <= (c32_blocked_mode_i &&
                                      (address_q[BYTE_SEL_BITS-1:0] == '0) &&
                                      (valid_bytes_q == 6'(BEAT_BYTES))) ?
                                     unpack_row(obi_rdata_i, valid_bytes_q, lane_base_i) :
                                     unpack_row(
                                         merge_beats(
                                             obi_rdata_i, '0,
                                             address_q[BYTE_SEL_BITS-1:0], valid_bytes_q
                                         ),
                                         valid_bytes_q, lane_base_i
                                     );
                            row_valid_q <= 1'b1;
                            state_q <= BYPASS_EMIT;
                        end
                    end
                end

                BYPASS_REQ1: begin
                    if (obi_gnt_i) begin
                        state_q <= BYPASS_WAIT1;
                    end
                end

                BYPASS_WAIT1: begin
                    if (obi_rvalid_i) begin
                        fetch_beats_q <= fetch_beats_q + 32'd1;
                        row_q <= unpack_row(
                            merge_beats(
                                beat0_q, obi_rdata_i,
                                address_q[BYTE_SEL_BITS-1:0], valid_bytes_q
                            ),
                            valid_bytes_q, lane_base_i
                        );
                        row_valid_q <= 1'b1;
                        state_q <= BYPASS_EMIT;
                    end
                end

                BYPASS_EMIT: begin
                    if (row_valid_q && row_ready_i) begin
                        row_valid_q <= 1'b0;
                        emitted_vectors_q <= emitted_vectors_q + 32'd1;
                        state_q <= last_i ? BYPASS_IDLE : BYPASS_PREP;
                    end
                end

                default: state_q <= BYPASS_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
