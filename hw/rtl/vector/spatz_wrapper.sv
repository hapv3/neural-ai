`default_nettype none

//-------------------------------------------------------------------------
// Wrapper cho lõi Spatz Vector Engine.
// Lõi Spatz sẽ đóng vai trò như một RVV Coprocessor cho lõi Snitch.
// Giao tiếp dữ liệu OBI với TCDM sẽ ở mức 256-bit.
//-------------------------------------------------------------------------
module spatz_wrapper #(
    parameter int unsigned VLEN = 256, // Chiều dài thanh ghi vector (Vector Length)
    parameter int unsigned DLEN = 256  // Độ rộng bus dữ liệu kết nối TCDM (Data Length)
)(
    input  logic         clk_i,
    input  logic         rst_ni,

    //---------------------------------------------------
    // Giao diện Lệnh Coprocessor (Nối với Snitch Core)
    //---------------------------------------------------
    input  logic         coprocessor_req_i,
    input  logic [31:0]  coprocessor_insn_i,
    input  logic [31:0]  coprocessor_rs1_i,
    input  logic [31:0]  coprocessor_rs2_i,
    output logic         coprocessor_gnt_o,
    output logic         coprocessor_valid_o,
    output logic [31:0]  coprocessor_result_o,

    //---------------------------------------------------
    // Giao diện Dữ liệu OBI (Nối với TCDM Interconnect)
    //---------------------------------------------------
    output logic         obi_req_o,
    input  logic         obi_gnt_i,
    output logic [31:0]  obi_addr_o,
    output logic         obi_we_o,
    output logic [31:0]  obi_be_o,     // 256-bit -> 32 bytes enable
    output logic [255:0] obi_wdata_o,
    input  logic         obi_rvalid_i,
    input  logic [255:0] obi_rdata_i
);

    // TODO: Instantiate thực tế lõi Spatz từ repo hw/spatz.
    // Hiện tại tạo cấu trúc Wrapper Stub để chuẩn bị cho Cluster Integration.
    
    /*
    spatz #(
        .VLEN(VLEN),
        .DLEN(DLEN)
    ) i_spatz (
        .clk_i    (clk_i),
        .rst_ni   (rst_ni),
        
        .req_i    (coprocessor_req_i),
        .insn_i   (coprocessor_insn_i),
        .rs1_i    (coprocessor_rs1_i),
        .rs2_i    (coprocessor_rs2_i),
        .gnt_o    (coprocessor_gnt_o),
        .valid_o  (coprocessor_valid_o),
        .result_o (coprocessor_result_o),
        
        // Cần mapping OBI interface của Spatz ra tín hiệu của wrapper này
        ...
    );
    */

    // Dummy tie-offs cho quá trình tổng hợp (Synthesis Stub)
    assign coprocessor_gnt_o    = coprocessor_req_i; // Luôn sẵn sàng nhận lệnh trong stub
    assign coprocessor_valid_o  = 1'b0;
    assign coprocessor_result_o = '0;
    
    assign obi_req_o   = 1'b0;
    assign obi_addr_o  = '0;
    assign obi_we_o    = 1'b0;
    assign obi_be_o    = '0;
    assign obi_wdata_o = '0;

endmodule
