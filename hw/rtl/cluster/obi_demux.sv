`default_nettype none

module obi_demux #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter logic [ADDR_WIDTH-1:0] TCDM_BASE = 32'h1000_0000,
    parameter logic [ADDR_WIDTH-1:0] TCDM_MASK = 32'hFFF0_0000,
    parameter logic [ADDR_WIDTH-1:0] REGS_BASE = 32'h2000_0000,
    parameter logic [ADDR_WIDTH-1:0] REGS_MASK = 32'hFFFF_0000
)(
    input  logic clk_i,
    input  logic rst_ni,

    // OBI Slave (connected to Snitch Master)
    input  logic                      slv_req_i,
    output logic                      slv_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     slv_addr_i,
    input  logic                      slv_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] slv_be_i,
    input  logic [DATA_WIDTH-1:0]     slv_wdata_i,
    output logic                      slv_rvalid_o,
    output logic [DATA_WIDTH-1:0]     slv_rdata_o,

    // OBI Master 0 (connected to TCDM)
    output logic                      m0_req_o,
    input  logic                      m0_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m0_addr_o,
    output logic                      m0_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m0_be_o,
    output logic [DATA_WIDTH-1:0]     m0_wdata_o,
    input  logic                      m0_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m0_rdata_i,

    // OBI Master 1 (connected to Control Regs)
    output logic                      m1_req_o,
    input  logic                      m1_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m1_addr_o,
    output logic                      m1_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m1_be_o,
    output logic [DATA_WIDTH-1:0]     m1_wdata_o,
    input  logic                      m1_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m1_rdata_i
);

    logic sel_m0;
    logic sel_m1;

    assign sel_m0 = ((slv_addr_i & TCDM_MASK) == TCDM_BASE);
    assign sel_m1 = ((slv_addr_i & REGS_MASK) == REGS_BASE);

    // Route Request
    assign m0_req_o   = slv_req_i & sel_m0;
    assign m0_addr_o  = slv_addr_i;
    assign m0_we_o    = slv_we_i;
    assign m0_be_o    = slv_be_i;
    assign m0_wdata_o = slv_wdata_i;

    assign m1_req_o   = slv_req_i & sel_m1;
    assign m1_addr_o  = slv_addr_i;
    assign m1_we_o    = slv_we_i;
    assign m1_be_o    = slv_be_i;
    assign m1_wdata_o = slv_wdata_i;

    // Route Grant
    assign slv_gnt_o = (sel_m0 & m0_gnt_i) | (sel_m1 & m1_gnt_i);

    // Track outstanding responses
    // Assumes both endpoints have exactly 1 cycle latency or strictly in-order responses
    // with no overlapping outstanding transactions between different masters.
    // For TCDM and Regs with exactly 1 cycle response, a simple delayed sel signal works.
    
    logic sel_m0_q, sel_m1_q;
    
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            sel_m0_q <= 1'b0;
            sel_m1_q <= 1'b0;
        end else begin
            if (slv_req_i && slv_gnt_o) begin
                sel_m0_q <= sel_m0;
                sel_m1_q <= sel_m1;
            end else begin
                sel_m0_q <= 1'b0;
                sel_m1_q <= 1'b0;
            end
        end
    end

    // Route Response
    assign slv_rvalid_o = (sel_m0_q & m0_rvalid_i) | (sel_m1_q & m1_rvalid_i);
    
    always_comb begin
        if (sel_m0_q) slv_rdata_o = m0_rdata_i;
        else if (sel_m1_q) slv_rdata_o = m1_rdata_i;
        else slv_rdata_o = '0;
    end

endmodule
