`default_nettype none

module obi_arbiter_2to1 #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Master 0 (Priority 0 - Bootloader)
    input  logic                      m0_req_i,
    output logic                      m0_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     m0_addr_i,
    input  logic                      m0_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] m0_be_i,
    input  logic [DATA_WIDTH-1:0]     m0_wdata_i,
    output logic                      m0_rvalid_o,
    output logic [DATA_WIDTH-1:0]     m0_rdata_o,

    // Master 1 (Priority 1 - Snitch I-Bus)
    input  logic                      m1_req_i,
    output logic                      m1_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     m1_addr_i,
    input  logic                      m1_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] m1_be_i,
    input  logic [DATA_WIDTH-1:0]     m1_wdata_i,
    output logic                      m1_rvalid_o,
    output logic [DATA_WIDTH-1:0]     m1_rdata_o,

    // Slave
    output logic                      slv_req_o,
    input  logic                      slv_gnt_i,
    output logic [ADDR_WIDTH-1:0]     slv_addr_o,
    output logic                      slv_we_o,
    output logic [(DATA_WIDTH/8)-1:0] slv_be_o,
    output logic [DATA_WIDTH-1:0]     slv_wdata_o,
    input  logic                      slv_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     slv_rdata_i
);

    logic sel_m0;
    
    // Priority to Master 0
    assign sel_m0 = m0_req_i;

    assign slv_req_o   = m0_req_i | m1_req_i;
    assign slv_addr_o  = sel_m0 ? m0_addr_i  : m1_addr_i;
    assign slv_we_o    = sel_m0 ? m0_we_i    : m1_we_i;
    assign slv_be_o    = sel_m0 ? m0_be_i    : m1_be_i;
    assign slv_wdata_o = sel_m0 ? m0_wdata_i : m1_wdata_i;

    assign m0_gnt_o = slv_gnt_i & sel_m0;
    assign m1_gnt_o = slv_gnt_i & ~sel_m0 & m1_req_i;

    // Track which master was granted for the response
    logic resp_sel_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            resp_sel_q <= 1'b0;
        end else if (slv_req_o && slv_gnt_i) begin
            resp_sel_q <= sel_m0;
        end
    end

    assign m0_rvalid_o = slv_rvalid_i & resp_sel_q;
    assign m1_rvalid_o = slv_rvalid_i & ~resp_sel_q;

    assign m0_rdata_o  = slv_rdata_i;
    assign m1_rdata_o  = slv_rdata_i;

endmodule
