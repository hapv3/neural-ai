`default_nettype none

`include "snitch/typedef.svh"
`include "reqrsp_interface/typedef.svh"
`include "tcdm_interface/typedef.svh"
`include "cv_x_if/typedef.svh"

module snitch_core #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned I_DATA_WIDTH = 256,
    parameter int unsigned D_DATA_WIDTH = 64,
    parameter logic [31:0] BOOT_ADDR  = 32'h0000_1000
)(
    input  logic clk_i,
    input  logic rst_ni,
    
    input  logic [31:0] hart_id_i,

    // Instruction Fetch OBI Master (I-TCM)
    output logic                      obi_i_req_o,
    input  logic                      obi_i_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_i_addr_o,
    output logic                      obi_i_we_o,
    output logic [(I_DATA_WIDTH/8)-1:0] obi_i_be_o,
    output logic [I_DATA_WIDTH-1:0]     obi_i_wdata_o,
    input  logic                      obi_i_rvalid_i,
    input  logic [I_DATA_WIDTH-1:0]     obi_i_rdata_i,

    // Data OBI Master (D-TCM & MMIO)
    output logic                      obi_d_req_o,
    input  logic                      obi_d_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_d_addr_o,
    output logic                      obi_d_we_o,
    output logic [(D_DATA_WIDTH/8)-1:0] obi_d_be_o,
    output logic [D_DATA_WIDTH-1:0]     obi_d_wdata_o,
    input  logic                      obi_d_rvalid_i,
    input  logic [D_DATA_WIDTH-1:0]     obi_d_rdata_i
);

    import snitch_pkg::*;

    localparam type dreq_t = `SNITCH_DATA_REQ_STRUCT(D_DATA_WIDTH, ADDR_WIDTH);
    localparam type drsp_t = `SNITCH_DATA_RSP_STRUCT(D_DATA_WIDTH);
    localparam type ireq_t = `SNITCH_INSTR_REQ_STRUCT(ADDR_WIDTH);
    localparam type irsp_t = `SNITCH_INSTR_RSP_STRUCT;

    // Dummy types for unused accelerator interface
    localparam type acc_req_t      = `SNITCH_ACC_REQ_STRUCT(D_DATA_WIDTH, ADDR_WIDTH);
    localparam type acc_rsp_t      = `SNITCH_ACC_RSP_STRUCT(D_DATA_WIDTH);
    localparam type x_issue_req_t  = `CV_X_IF_ISSUE_REQ_STRUCT(4);
    localparam type x_issue_resp_t = `CV_X_IF_ISSUE_RESP_STRUCT;
    localparam type x_register_t   = `CV_X_IF_REGISTER_STRUCT(4);
    localparam type x_commit_t     = `CV_X_IF_COMMIT_STRUCT(4);
    localparam type x_result_t     = `CV_X_IF_RESULT_STRUCT(4);
    localparam type ptw_req_t      = `SNITCH_PTW_REQ_STRUCT(ADDR_WIDTH);
    localparam type ptw_rsp_t      = `SNITCH_PTW_RSP_STRUCT(ADDR_WIDTH);

    ireq_t         inst_req;
    irsp_t         inst_rsp;
    dreq_t         data_req;
    drsp_t         data_rsp;

    // Tie-off unused signals
    acc_req_t      acc_req;
    acc_rsp_t      acc_rsp;
    x_issue_req_t  x_issue_req;
    x_issue_resp_t x_issue_resp;
    x_register_t   x_register;
    x_commit_t     x_commit;
    x_result_t     x_result;
    ptw_req_t [1:0] ptw_req;
    ptw_rsp_t [1:0] ptw_rsp;

    assign acc_rsp = '0;
    assign x_issue_resp = '0;
    assign x_result = '0;
    assign ptw_rsp = '0;

    snitch #(
        .BootAddr (BOOT_ADDR),
        .AddrWidth(ADDR_WIDTH),
        .DataWidth(D_DATA_WIDTH),
        .NumIntOutstandingLoads(1),
        .NumIntOutstandingMem(1),
        .VMSupport(0),
        .EnableXif(0)
    ) i_snitch (
        .clk_i             (clk_i),
        .rst_i             (~rst_ni),
        .hart_id_i         (hart_id_i),
        .irq_i             ('0),
        .flush_i_valid_o   (),
        .flush_i_ready_i   (1'b1),
        .inst_req_o        (inst_req),
        .inst_rsp_i        (inst_rsp),
        .acc_req_o         (acc_req),
        .acc_rsp_i         (acc_rsp),
        .x_issue_req_o     (x_issue_req),
        .x_issue_resp_i    (x_issue_resp),
        .x_issue_valid_o   (),
        .x_issue_ready_i   (1'b0),
        .x_register_o      (x_register),
        .x_register_valid_o(),
        .x_register_ready_i(1'b0),
        .x_commit_o        (x_commit),
        .x_commit_valid_o  (),
        .x_result_i        (x_result),
        .x_result_valid_i  (1'b0),
        .x_result_ready_o  (),
        .i2f_rdata_o       (),
        .i2f_rvalid_o      (),
        .i2f_rready_i      (1'b0),
        .f2i_wdata_i       ('0),
        .f2i_wvalid_i      (1'b0),
        .f2i_wready_o      (),
        .data_req_o        (data_req),
        .data_rsp_i        (data_rsp),
        .ptw_req_o         (ptw_req),
        .ptw_rsp_i         (ptw_rsp),
        .fpu_rnd_mode_o    (),
        .fpu_fmt_mode_o    (),
        .fpu_status_i      ('0),
        .caq_pvalid_i      (1'b0),
        .core_events_o     (),
        .en_copift_o       (),
        .barrier_o         (),
        .barrier_i         (1'b0)
    );

    // --- Instruction Fetch Interface Adapter ---
    logic [I_DATA_WIDTH-1:0] adapter_rsp_data;

    obi_snitch_if_adapter #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(I_DATA_WIDTH)
    ) u_if_adapter (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .snitch_req_valid_i (inst_req.q_valid),
        .snitch_req_ready_o (inst_rsp.q_ready),
        .snitch_req_addr_i  (inst_req.addr),
        .snitch_rsp_data_o  (adapter_rsp_data),
        .obi_req_o          (obi_i_req_o),
        .obi_gnt_i          (obi_i_gnt_i),
        .obi_addr_o         (obi_i_addr_o),
        .obi_rvalid_i       (obi_i_rvalid_i),
        .obi_rdata_i        (obi_i_rdata_i)
    );

    // Combinationally select the correct 32-bit instruction from the 256-bit line
    assign inst_rsp.data    = adapter_rsp_data >> (inst_req.addr[$clog2(I_DATA_WIDTH/8)-1:2] * 32);
    assign inst_rsp.error   = 1'b0;

    assign obi_i_we_o       = 1'b0;
    assign obi_i_be_o       = '1;
    assign obi_i_wdata_o    = '0;

    // --- Data Interface (1-cycle latency OK for Load/Store, Snitch LSU handles OBI protocol) ---
    assign obi_d_req_o      = data_req.q_valid;
    assign data_rsp.q_ready = obi_d_gnt_i;
    assign obi_d_addr_o     = data_req.q.addr;
    assign obi_d_we_o       = data_req.q.write;
    assign obi_d_be_o       = data_req.q.strb;
    assign obi_d_wdata_o    = data_req.q.data;
    assign data_rsp.p_valid = obi_d_rvalid_i;
    assign data_rsp.p.data  = obi_d_rdata_i;
    assign data_rsp.p.error = 1'b0;

endmodule
