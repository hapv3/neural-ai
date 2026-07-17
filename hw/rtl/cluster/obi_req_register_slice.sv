`default_nettype none

module obi_req_register_slice #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      slv_req_i,
    output logic                      slv_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     slv_addr_i,
    input  logic                      slv_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] slv_be_i,
    input  logic [DATA_WIDTH-1:0]     slv_wdata_i,
    output logic                      slv_rvalid_o,
    output logic [DATA_WIDTH-1:0]     slv_rdata_o,

    output logic                      mst_req_o,
    input  logic                      mst_gnt_i,
    output logic [ADDR_WIDTH-1:0]     mst_addr_o,
    output logic                      mst_we_o,
    output logic [(DATA_WIDTH/8)-1:0] mst_be_o,
    output logic [DATA_WIDTH-1:0]     mst_wdata_o,
    input  logic                      mst_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     mst_rdata_i
);

    logic valid_q;
    logic [ADDR_WIDTH-1:0] addr_q;
    logic we_q;
    logic [(DATA_WIDTH/8)-1:0] be_q;
    logic [DATA_WIDTH-1:0] wdata_q;

    assign slv_gnt_o = !valid_q;

    assign mst_req_o = valid_q;
    assign mst_addr_o = addr_q;
    assign mst_we_o = we_q;
    assign mst_be_o = be_q;
    assign mst_wdata_o = wdata_q;

    assign slv_rvalid_o = mst_rvalid_i;
    assign slv_rdata_o = mst_rdata_i;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_q <= 1'b0;
            addr_q  <= '0;
            we_q    <= 1'b0;
            be_q    <= '0;
            wdata_q <= '0;
        end else begin
            if (valid_q && mst_gnt_i) begin
                valid_q <= 1'b0;
            end

            if (!valid_q && slv_req_i) begin
                valid_q <= 1'b1;
                addr_q  <= slv_addr_i;
                we_q    <= slv_we_i;
                be_q    <= slv_be_i;
                wdata_q <= slv_wdata_i;
            end
        end
    end

endmodule
