`default_nettype none

module obi_demux_1to5 #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter logic [31:0] M0_BASE = 32'h0000_0000, parameter logic [31:0] M0_MASK = 32'hFFFF_0000,
    parameter logic [31:0] M1_BASE = 32'h0001_0000, parameter logic [31:0] M1_MASK = 32'hFFFF_0000,
    parameter logic [31:0] M2_BASE = 32'h0002_0000, parameter logic [31:0] M2_MASK = 32'hFFFF_0000,
    parameter logic [31:0] M3_BASE = 32'h0003_0000, parameter logic [31:0] M3_MASK = 32'hFFFF_0000,
    parameter logic [31:0] M4_BASE = 32'h0004_0000, parameter logic [31:0] M4_MASK = 32'hFFFF_0000
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Slave Port (from Master)
    input  logic                      slv_req_i,
    output logic                      slv_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     slv_addr_i,
    input  logic                      slv_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] slv_be_i,
    input  logic [DATA_WIDTH-1:0]     slv_wdata_i,
    output logic                      slv_rvalid_o,
    output logic [DATA_WIDTH-1:0]     slv_rdata_o,

    // Master 0
    output logic                      m0_req_o,
    input  logic                      m0_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m0_addr_o,
    output logic                      m0_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m0_be_o,
    output logic [DATA_WIDTH-1:0]     m0_wdata_o,
    input  logic                      m0_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m0_rdata_i,

    // Master 1
    output logic                      m1_req_o,
    input  logic                      m1_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m1_addr_o,
    output logic                      m1_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m1_be_o,
    output logic [DATA_WIDTH-1:0]     m1_wdata_o,
    input  logic                      m1_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m1_rdata_i,

    // Master 2
    output logic                      m2_req_o,
    input  logic                      m2_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m2_addr_o,
    output logic                      m2_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m2_be_o,
    output logic [DATA_WIDTH-1:0]     m2_wdata_o,
    input  logic                      m2_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m2_rdata_i,

    // Master 3
    output logic                      m3_req_o,
    input  logic                      m3_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m3_addr_o,
    output logic                      m3_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m3_be_o,
    output logic [DATA_WIDTH-1:0]     m3_wdata_o,
    input  logic                      m3_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m3_rdata_i,

    // Master 4
    output logic                      m4_req_o,
    input  logic                      m4_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m4_addr_o,
    output logic                      m4_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m4_be_o,
    output logic [DATA_WIDTH-1:0]     m4_wdata_o,
    input  logic                      m4_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m4_rdata_i
);

    logic sel_m0, sel_m1, sel_m2, sel_m3, sel_m4;

    assign sel_m0 = ((slv_addr_i & M0_MASK) == (M0_BASE & M0_MASK));
    assign sel_m1 = ((slv_addr_i & M1_MASK) == (M1_BASE & M1_MASK));
    assign sel_m2 = ((slv_addr_i & M2_MASK) == (M2_BASE & M2_MASK));
    assign sel_m3 = ((slv_addr_i & M3_MASK) == (M3_BASE & M3_MASK));
    assign sel_m4 = ((slv_addr_i & M4_MASK) == (M4_BASE & M4_MASK));

    logic [2:0] sel_q;
    logic [2:0] sel_d;

    assign m0_req_o = slv_req_i & sel_m0;
    assign m1_req_o = slv_req_i & sel_m1;
    assign m2_req_o = slv_req_i & sel_m2;
    assign m3_req_o = slv_req_i & sel_m3;
    assign m4_req_o = slv_req_i & sel_m4;

    assign m0_addr_o = slv_addr_i;
    assign m1_addr_o = slv_addr_i;
    assign m2_addr_o = slv_addr_i;
    assign m3_addr_o = slv_addr_i;
    assign m4_addr_o = slv_addr_i;

    assign m0_we_o = slv_we_i;
    assign m1_we_o = slv_we_i;
    assign m2_we_o = slv_we_i;
    assign m3_we_o = slv_we_i;
    assign m4_we_o = slv_we_i;

    assign m0_be_o = slv_be_i;
    assign m1_be_o = slv_be_i;
    assign m2_be_o = slv_be_i;
    assign m3_be_o = slv_be_i;
    assign m4_be_o = slv_be_i;

    assign m0_wdata_o = slv_wdata_i;
    assign m1_wdata_o = slv_wdata_i;
    assign m2_wdata_o = slv_wdata_i;
    assign m3_wdata_o = slv_wdata_i;
    assign m4_wdata_o = slv_wdata_i;

    always_comb begin
        slv_gnt_o = 1'b0;
        if (sel_m0) slv_gnt_o = m0_gnt_i;
        else if (sel_m1) slv_gnt_o = m1_gnt_i;
        else if (sel_m2) slv_gnt_o = m2_gnt_i;
        else if (sel_m3) slv_gnt_o = m3_gnt_i;
        else if (sel_m4) slv_gnt_o = m4_gnt_i;
    end

    always_comb begin
        sel_d = sel_q;
        if (slv_req_i) begin
            if (sel_m0) sel_d = 3'd0;
            else if (sel_m1) sel_d = 3'd1;
            else if (sel_m2) sel_d = 3'd2;
            else if (sel_m3) sel_d = 3'd3;
            else if (sel_m4) sel_d = 3'd4;
            else sel_d = 3'd5;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            sel_q <= 3'd0;
        end else begin
            if (slv_req_i && slv_gnt_o) begin
                sel_q <= sel_d;
            end
        end
    end

    always_comb begin
        slv_rvalid_o = 1'b0;
        slv_rdata_o = '0;
        case (sel_q)
            3'd0: begin slv_rvalid_o = m0_rvalid_i; slv_rdata_o = m0_rdata_i; end
            3'd1: begin slv_rvalid_o = m1_rvalid_i; slv_rdata_o = m1_rdata_i; end
            3'd2: begin slv_rvalid_o = m2_rvalid_i; slv_rdata_o = m2_rdata_i; end
            3'd3: begin slv_rvalid_o = m3_rvalid_i; slv_rdata_o = m3_rdata_i; end
            3'd4: begin slv_rvalid_o = m4_rvalid_i; slv_rdata_o = m4_rdata_i; end
            default: ;
        endcase
    end

endmodule
