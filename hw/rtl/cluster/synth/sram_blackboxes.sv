`default_nettype none

(* blackbox *)
module cluster_sram_bank #(
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned SIZE_BYTES = 32768
)(
    input  logic                      clk_i,
    input  logic                      rst_ni,
    input  logic                      req_i,
    input  logic                      we_i,
    input  logic [31:0]               addr_i,
    input  logic [255:0]              wdata_i,
    input  logic [31:0]               be_i,
    output logic                      gnt_o,
    output logic                      rvalid_o,
    output logic [255:0]              rdata_o
);
endmodule

(* blackbox *)
module tc_sram #(
    parameter int unsigned NumWords  = 32'd1024,
    parameter int unsigned DataWidth = 32'd128,
    parameter int unsigned ByteWidth = 32'd8,
    parameter int unsigned NumPorts  = 32'd2,
    parameter int unsigned Latency   = 32'd1,
    parameter              SimInit   = "none",
    parameter bit          PrintSimCfg = 1'b0,
    parameter              ImplKey   = "none",
    parameter int unsigned AddrWidth = (NumWords > 32'd1) ? $clog2(NumWords) : 32'd1,
    parameter int unsigned BeWidth   = (DataWidth + ByteWidth - 32'd1) / ByteWidth
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic [NumPorts-1:0]          req_i,
    input  logic [NumPorts-1:0]          we_i,
    input  logic [17:0]                  addr_i,
    input  logic [511:0]                 wdata_i,
    input  logic [63:0]                  be_i,
    output logic [511:0]                 rdata_o
);
endmodule

(* blackbox *)
module tc_sram_impl #(
    parameter int unsigned NumWords  = 32'd1024,
    parameter int unsigned DataWidth = 32'd128,
    parameter int unsigned ByteWidth = 32'd8,
    parameter int unsigned NumPorts  = 32'd2,
    parameter int unsigned Latency   = 32'd1,
    parameter              SimInit   = "none",
    parameter bit          PrintSimCfg = 1'b0,
    parameter              ImplKey   = "none",
    parameter int unsigned AddrWidth = (NumWords > 32'd1) ? $clog2(NumWords) : 32'd1,
    parameter int unsigned BeWidth   = (DataWidth + ByteWidth - 32'd1) / ByteWidth,
    parameter type         impl_in_t = logic,
    parameter type         impl_out_t = logic,
    parameter impl_out_t   ImplOutSim = 'X
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  impl_in_t                     impl_i,
    output impl_out_t                    impl_o,
    input  logic [NumPorts-1:0]          req_i,
    input  logic [NumPorts-1:0]          we_i,
    input  logic [17:0]                  addr_i,
    input  logic [511:0]                 wdata_i,
    input  logic [63:0]                  be_i,
    output logic [511:0]                 rdata_o
);
endmodule

(* blackbox *)
module systolic_psum_sram #(
    parameter int unsigned DataWidth = 32'd1024,
    parameter int unsigned Depth     = 32'd256,
    parameter int unsigned AddrWidth = (Depth > 32'd1) ? $clog2(Depth) : 32'd1
)(
    input  logic                 clk_i,
    input  logic                 req_i,
    input  logic                 we_i,
    input  logic [AddrWidth-1:0] addr_i,
    input  logic [DataWidth-1:0] wdata_i,
    output logic [DataWidth-1:0] rdata_o
);
endmodule

`default_nettype wire
