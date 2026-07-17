`default_nettype none

module obi_demux_1to4 #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter logic [ADDR_WIDTH-1:0] M0_BASE = 32'h1000_0000,
    parameter logic [ADDR_WIDTH-1:0] M0_MASK = 32'hFFFF_8000,
    parameter logic [ADDR_WIDTH-1:0] M1_BASE = 32'h1000_8000,
    parameter logic [ADDR_WIDTH-1:0] M1_MASK = 32'hFFFF_8000,
    parameter logic [ADDR_WIDTH-1:0] M2_BASE = 32'h1010_0000,
    parameter logic [ADDR_WIDTH-1:0] M2_MASK = 32'hFFF0_0000,
    parameter logic [ADDR_WIDTH-1:0] M3_BASE = 32'h2000_0000,
    parameter logic [ADDR_WIDTH-1:0] M3_MASK = 32'hFFFF_0000,
    parameter bit M0_REQ_REGISTER = 1'b0
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Slave Port
    input  logic                      slv_req_i,
    output logic                      slv_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     slv_addr_i,
    input  logic                      slv_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] slv_be_i,
    input  logic [DATA_WIDTH-1:0]     slv_wdata_i,
    output logic                      slv_rvalid_o,
    output logic [DATA_WIDTH-1:0]     slv_rdata_o,

    // Master Port 0
    output logic                      m0_req_o,
    input  logic                      m0_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m0_addr_o,
    output logic                      m0_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m0_be_o,
    output logic [DATA_WIDTH-1:0]     m0_wdata_o,
    input  logic                      m0_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m0_rdata_i,

    // Master Port 1
    output logic                      m1_req_o,
    input  logic                      m1_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m1_addr_o,
    output logic                      m1_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m1_be_o,
    output logic [DATA_WIDTH-1:0]     m1_wdata_o,
    input  logic                      m1_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m1_rdata_i,

    // Master Port 2
    output logic                      m2_req_o,
    input  logic                      m2_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m2_addr_o,
    output logic                      m2_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m2_be_o,
    output logic [DATA_WIDTH-1:0]     m2_wdata_o,
    input  logic                      m2_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m2_rdata_i,

    // Master Port 3
    output logic                      m3_req_o,
    input  logic                      m3_gnt_i,
    output logic [ADDR_WIDTH-1:0]     m3_addr_o,
    output logic                      m3_we_o,
    output logic [(DATA_WIDTH/8)-1:0] m3_be_o,
    output logic [DATA_WIDTH-1:0]     m3_wdata_o,
    input  logic                      m3_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     m3_rdata_i
);

    logic sel_m0, sel_m1, sel_m2, sel_m3;
    logic outstanding_q;
    logic selected_rvalid;
    logic accept_req;
    logic m0_req_raw;
    logic m0_gnt_raw;
    logic [ADDR_WIDTH-1:0] m0_addr_raw;
    logic m0_we_raw;
    logic [(DATA_WIDTH/8)-1:0] m0_be_raw;
    logic [DATA_WIDTH-1:0] m0_wdata_raw;
    logic m0_rvalid_raw;
    logic [DATA_WIDTH-1:0] m0_rdata_raw;

    assign sel_m0 = ((slv_addr_i & M0_MASK) == (M0_BASE & M0_MASK));
    assign sel_m1 = ((slv_addr_i & M1_MASK) == (M1_BASE & M1_MASK));
    assign sel_m2 = ((slv_addr_i & M2_MASK) == (M2_BASE & M2_MASK));
    assign sel_m3 = ((slv_addr_i & M3_MASK) == (M3_BASE & M3_MASK));

    // Request Routing
    assign m0_req_raw = slv_req_i & sel_m0 & !outstanding_q;
    assign m1_req_o   = slv_req_i & sel_m1 & !outstanding_q;
    assign m2_req_o   = slv_req_i & sel_m2 & !outstanding_q;
    assign m3_req_o   = slv_req_i & sel_m3 & !outstanding_q;

    assign m0_addr_raw = slv_addr_i;
    assign m1_addr_o  = slv_addr_i;
    assign m2_addr_o  = slv_addr_i;
    assign m3_addr_o  = slv_addr_i;

    assign m0_we_raw = slv_we_i;
    assign m1_we_o    = slv_we_i;
    assign m2_we_o    = slv_we_i;
    assign m3_we_o    = slv_we_i;

    assign m0_be_raw = slv_be_i;
    assign m1_be_o    = slv_be_i;
    assign m2_be_o    = slv_be_i;
    assign m3_be_o    = slv_be_i;

    assign m0_wdata_raw = slv_wdata_i;
    assign m1_wdata_o = slv_wdata_i;
    assign m2_wdata_o = slv_wdata_i;
    assign m3_wdata_o = slv_wdata_i;

    generate
        if (M0_REQ_REGISTER) begin : gen_m0_req_slice
            obi_req_register_slice #(
                .ADDR_WIDTH (ADDR_WIDTH),
                .DATA_WIDTH (DATA_WIDTH)
            ) u_m0_req_slice (
                .clk_i        (clk_i),
                .rst_ni       (rst_ni),

                .slv_req_i    (m0_req_raw),
                .slv_gnt_o    (m0_gnt_raw),
                .slv_addr_i   (m0_addr_raw),
                .slv_we_i     (m0_we_raw),
                .slv_be_i     (m0_be_raw),
                .slv_wdata_i  (m0_wdata_raw),
                .slv_rvalid_o (m0_rvalid_raw),
                .slv_rdata_o  (m0_rdata_raw),

                .mst_req_o    (m0_req_o),
                .mst_gnt_i    (m0_gnt_i),
                .mst_addr_o   (m0_addr_o),
                .mst_we_o     (m0_we_o),
                .mst_be_o     (m0_be_o),
                .mst_wdata_o  (m0_wdata_o),
                .mst_rvalid_i (m0_rvalid_i),
                .mst_rdata_i  (m0_rdata_i)
            );
        end else begin : gen_m0_passthrough
            assign m0_req_o = m0_req_raw;
            assign m0_gnt_raw = m0_gnt_i;
            assign m0_addr_o = m0_addr_raw;
            assign m0_we_o = m0_we_raw;
            assign m0_be_o = m0_be_raw;
            assign m0_wdata_o = m0_wdata_raw;
            assign m0_rvalid_raw = m0_rvalid_i;
            assign m0_rdata_raw = m0_rdata_i;
        end
    endgenerate

    // Grant Routing
    assign slv_gnt_o  = !outstanding_q &&
                        ((sel_m0 & m0_gnt_raw) | (sel_m1 & m1_gnt_i) |
                         (sel_m2 & m2_gnt_i) | (sel_m3 & m3_gnt_i));
    assign accept_req = slv_req_i && slv_gnt_o;



    // Response Routing
    logic [1:0] out_sel_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            out_sel_q <= 2'b00;
        end else if (accept_req) begin
            if (sel_m0) out_sel_q <= 2'b00;
            else if (sel_m1) out_sel_q <= 2'b01;
            else if (sel_m2) out_sel_q <= 2'b10;
            else if (sel_m3) out_sel_q <= 2'b11;
        end
    end

    always_comb begin
        slv_rvalid_o = 1'b0;
        slv_rdata_o  = '0;
        case (out_sel_q)
            2'b00: begin
                slv_rvalid_o = m0_rvalid_raw;
                slv_rdata_o  = m0_rdata_raw;
            end
            2'b01: begin
                slv_rvalid_o = m1_rvalid_i;
                slv_rdata_o  = m1_rdata_i;
            end
            2'b10: begin
                slv_rvalid_o = m2_rvalid_i;
                slv_rdata_o  = m2_rdata_i;
            end
            2'b11: begin
                slv_rvalid_o = m3_rvalid_i;
                slv_rdata_o  = m3_rdata_i;
            end
            default: ;
        endcase
    end

    assign selected_rvalid = slv_rvalid_o;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            outstanding_q <= 1'b0;
        end else begin
            if (accept_req) begin
                outstanding_q <= 1'b1;
            end
            if (selected_rvalid) begin
                outstanding_q <= 1'b0;
            end
        end
    end

endmodule
