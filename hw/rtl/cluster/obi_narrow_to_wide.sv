`default_nettype none

module obi_narrow_to_wide #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned M_DATA_WIDTH = 64,
    parameter int unsigned S_DATA_WIDTH = 256
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Master port (Narrow, e.g. 64-bit)
    input  logic                      mst_req_i,
    output logic                      mst_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     mst_addr_i,
    input  logic                      mst_we_i,
    input  logic [(M_DATA_WIDTH/8)-1:0] mst_be_i,
    input  logic [M_DATA_WIDTH-1:0]   mst_wdata_i,
    output logic                      mst_rvalid_o,
    output logic [M_DATA_WIDTH-1:0]   mst_rdata_o,

    // Slave port (Wide, e.g. 256-bit)
    output logic                      slv_req_o,
    input  logic                      slv_gnt_i,
    output logic [ADDR_WIDTH-1:0]     slv_addr_o,
    output logic                      slv_we_o,
    output logic [(S_DATA_WIDTH/8)-1:0] slv_be_o,
    output logic [S_DATA_WIDTH-1:0]   slv_wdata_o,
    input  logic                      slv_rvalid_i,
    input  logic [S_DATA_WIDTH-1:0]   slv_rdata_i
);

    localparam int unsigned RATIO = S_DATA_WIDTH / M_DATA_WIDTH;
    localparam int unsigned OFFSET_BITS = $clog2(RATIO); // e.g. 256/64=4, clog2(4)=2
    localparam int unsigned M_BYTES = M_DATA_WIDTH / 8; // 8 bytes

    // Extract word offset from address (e.g. bits [4:3] for 64-bit master in 256-bit slave)
    logic [OFFSET_BITS-1:0] w_offset;
    assign w_offset = mst_addr_i[$clog2(M_BYTES) +: OFFSET_BITS];

    assign slv_req_o  = mst_req_i;
    assign slv_addr_o = {mst_addr_i[ADDR_WIDTH-1 : $clog2(M_BYTES)+OFFSET_BITS], {($clog2(M_BYTES)+OFFSET_BITS){1'b0}}};
    assign slv_we_o   = mst_we_i;

    always_comb begin
        slv_be_o    = '0;
        slv_wdata_o = '0;
        slv_be_o[w_offset * M_BYTES +: M_BYTES]    = mst_be_i;
        slv_wdata_o[w_offset * M_DATA_WIDTH +: M_DATA_WIDTH] = mst_wdata_i;
    end

    assign mst_gnt_o = slv_gnt_i;
    assign mst_rvalid_o = slv_rvalid_i;

    // To properly route read data, we need to know the offset.
    // For OBI, responses come in order. We can store the offset of requests in a FIFO.
    // For a minimal adapter, if the interconnect guarantees 1 cycle latency or we just pipeline it simply:
    // Let's use a simple shift register / queue for offsets.
    logic [OFFSET_BITS-1:0] offset_q;
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            offset_q <= '0;
        end else if (mst_req_i && mst_gnt_o) begin
            offset_q <= w_offset;
        end
    end

    assign mst_rdata_o = slv_rdata_i[offset_q * M_DATA_WIDTH +: M_DATA_WIDTH];

endmodule
