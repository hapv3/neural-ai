`default_nettype none

// TCDM-to-OBI Bridge
// Converts Spatz VLSU TCDM memory channel (32-bit) to OBI protocol (256-bit)
// for the shared TCDM interconnect.
//
// Protocol difference:
//   - TCDM: 1-phase (valid/ready, response same cycle or next)
//   - OBI:  2-phase (req/gnt for address, rvalid for response)
//
// Width adaptation:
//   - Spatz ELEN = 32-bit (with N_FPU=0, N_IPU=1)
//   - OBI bus = 256-bit (32 bytes)
//   - 32-bit data is placed in the correct lane of 256-bit word

module tcdm_to_obi_bridge #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32  // Spatz ELEN (32-bit for INT-only)
)(
    input  logic clk_i,
    input  logic rst_ni,

    // TCDM Slave Side (from Spatz VLSU)
    input  logic [ADDR_WIDTH-1:0]    tcdm_req_addr_i,
    input  logic                     tcdm_req_write_i,
    input  logic [DATA_WIDTH-1:0]    tcdm_req_data_i,
    input  logic [DATA_WIDTH/8-1:0]  tcdm_req_strb_i,
    input  logic                     tcdm_req_valid_i,
    output logic                     tcdm_req_ready_o,

    output logic [DATA_WIDTH-1:0]    tcdm_rsp_data_o,
    output logic                     tcdm_rsp_valid_o,

    // OBI Master Side (to TCDM Interconnect — 256-bit wide)
    output logic                     obi_req_o,
    input  logic                     obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]    obi_addr_o,
    output logic                     obi_we_o,
    output logic [31:0]              obi_be_o,   // 256/8 = 32 byte-enables
    output logic [255:0]             obi_wdata_o,
    input  logic                     obi_rvalid_i,
    input  logic [255:0]             obi_rdata_i
);

    // =========================================================
    // Width Adaptation: Spatz 32-bit ↔ OBI 256-bit
    // =========================================================
    // Lane index = addr[4:2] selects one of 8 x 32-bit lanes
    // within the 32-byte (256-bit) OBI word.

    localparam int unsigned BYTE_SEL_BITS = 5;  // log2(32) = 5

    logic [2:0] lane_idx;
    assign lane_idx = tcdm_req_addr_i[4:2];

    // Align address to 32-byte boundary for OBI
    assign obi_addr_o = {tcdm_req_addr_i[ADDR_WIDTH-1:BYTE_SEL_BITS], {BYTE_SEL_BITS{1'b0}}};
    assign obi_we_o   = tcdm_req_write_i;

    // Place 32-bit data and strobe in the correct 32-bit lane
    always_comb begin
        obi_wdata_o = '0;
        obi_be_o    = '0;
        obi_wdata_o[lane_idx*32 +: 32] = tcdm_req_data_i;
        obi_be_o[lane_idx*4 +: 4]      = tcdm_req_strb_i;
    end

    // =========================================================
    // FSM: Handle OBI 2-phase protocol
    // =========================================================
    typedef enum logic { IDLE, WAIT_RVALID } state_e;
    state_e state_q, state_d;

    // Latch the lane index when request is granted, so we can
    // extract the correct 32-bit lane when rvalid arrives later.
    logic [2:0] resp_lane_q;

    always_comb begin
        state_d          = state_q;
        obi_req_o        = 1'b0;
        tcdm_req_ready_o = 1'b0;
        tcdm_rsp_valid_o = 1'b0;

        case (state_q)
            IDLE: begin
                if (tcdm_req_valid_i) begin
                    obi_req_o = 1'b1;
                    if (obi_gnt_i) begin
                        tcdm_req_ready_o = 1'b1;  // Consume TCDM request
                        state_d = WAIT_RVALID;
                    end
                end
            end

            WAIT_RVALID: begin
                if (obi_rvalid_i) begin
                    tcdm_rsp_valid_o = 1'b1;
                    state_d = IDLE;
                end
            end

            default: state_d = IDLE;
        endcase
    end

    // Extract the correct 32-bit lane from 256-bit read response
    // Use the latched lane index (from when the request was granted)
    assign tcdm_rsp_data_o = obi_rdata_i[resp_lane_q*32 +: 32];

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q     <= IDLE;
            resp_lane_q <= '0;
        end else begin
            state_q <= state_d;
            // Latch lane index when request is accepted
            if (state_q == IDLE && tcdm_req_valid_i && obi_gnt_i) begin
                resp_lane_q <= lane_idx;
            end
        end
    end

endmodule
