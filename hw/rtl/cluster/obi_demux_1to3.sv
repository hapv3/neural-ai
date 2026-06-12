`default_nettype none

module obi_demux_1to3 #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter logic [ADDR_WIDTH-1:0] M0_BASE = 32'h1000_8000,
    parameter logic [ADDR_WIDTH-1:0] M0_MASK = 32'hFFFF_8000,
    parameter logic [ADDR_WIDTH-1:0] M1_BASE = 32'h1010_0000,
    parameter logic [ADDR_WIDTH-1:0] M1_MASK = 32'hFFF0_0000,
    parameter logic [ADDR_WIDTH-1:0] M2_BASE = 32'h2000_0000,
    parameter logic [ADDR_WIDTH-1:0] M2_MASK = 32'hFFFF_0000
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Slave Port (from Snitch D-Bus)
    input  logic                      slv_req_i,
    output logic                      slv_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     slv_addr_i,
    input  logic                      slv_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] slv_be_i,
    input  logic [DATA_WIDTH-1:0]     slv_wdata_i,
    output logic                      slv_rvalid_o,
    output logic [DATA_WIDTH-1:0]     slv_rdata_o,

    // Master Port 0 (to D-TCM)
    output logic                      m0_req_o,
    input  logic                      m0_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m0_addr_o,
    output logic                      m0_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m0_be_o,
    output logic [DATA_WIDTH-1:0]     m0_wdata_o,
    input  logic                      m0_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m0_rdata_i,

    // Master Port 1 (to Shared Data TCDM)
    output logic                      m1_req_o,
    input  logic                      m1_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m1_addr_o,
    output logic                      m1_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m1_be_o,
    output logic [DATA_WIDTH-1:0]     m1_wdata_o,
    input  logic                      m1_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m1_rdata_i,

    // Master Port 2 (to MMIO)
    output logic                      m2_req_o,
    input  logic                      m2_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m2_addr_o,
    output logic                      m2_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m2_be_o,
    output logic [DATA_WIDTH-1:0]     m2_wdata_o,
    input  logic                      m2_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m2_rdata_i
);

    logic sel_m0, sel_m1, sel_m2;

    assign sel_m0 = ((slv_addr_i & M0_MASK) == (M0_BASE & M0_MASK));
    assign sel_m1 = ((slv_addr_i & M1_MASK) == (M1_BASE & M1_MASK));
    assign sel_m2 = ((slv_addr_i & M2_MASK) == (M2_BASE & M2_MASK));

    // Request Routing
    assign m0_req_o   = slv_req_i & sel_m0;
    assign m1_req_o   = slv_req_i & sel_m1;
    assign m2_req_o   = slv_req_i & sel_m2;

    assign m0_addr_o  = slv_addr_i;
    assign m1_addr_o  = slv_addr_i;
    assign m2_addr_o  = slv_addr_i;

    assign m0_we_o    = slv_we_i;
    assign m1_we_o    = slv_we_i;
    assign m2_we_o    = slv_we_i;

    assign m0_be_o    = slv_be_i;
    assign m1_be_o    = slv_be_i;
    assign m2_be_o    = slv_be_i;

    assign m0_wdata_o = slv_wdata_i;
    assign m1_wdata_o = slv_wdata_i;
    assign m2_wdata_o = slv_wdata_i;

    // Grant Routing
    assign slv_gnt_o  = (sel_m0 & m0_gnt_i) | (sel_m1 & m1_gnt_i) | (sel_m2 & m2_gnt_i);

    // Response Routing
    // OBI rvalid/rdata is decoupled from req. We need a FIFO to track which master
    // was granted to route the responses back correctly.
    // For simplicity, assuming in-order responses across all masters and only one outstanding
    // request to different masters at a time. This is true for a single scalar core.
    
    logic [1:0] out_sel_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            out_sel_q <= 2'b00;
        end else if (slv_req_i && slv_gnt_o) begin
            if (sel_m0) out_sel_q <= 2'b00;
            else if (sel_m1) out_sel_q <= 2'b01;
            else if (sel_m2) out_sel_q <= 2'b10;
        end
    end

    always_comb begin
        slv_rvalid_o = 1'b0;
        slv_rdata_o  = '0;
        case (out_sel_q)
            2'b00: begin
                slv_rvalid_o = m0_rvalid_i;
                slv_rdata_o  = m0_rdata_i;
            end
            2'b01: begin
                slv_rvalid_o = m1_rvalid_i;
                slv_rdata_o  = m1_rdata_i;
            end
            2'b10: begin
                slv_rvalid_o = m2_rvalid_i;
                slv_rdata_o  = m2_rdata_i;
            end
            default: ;
        endcase
    end

endmodule
