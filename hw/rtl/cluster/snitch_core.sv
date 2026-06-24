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
    input  logic [D_DATA_WIDTH-1:0]     obi_d_rdata_i,

    // Accelerator Offload Interface (Snitch → Spatz)
    output logic                      acc_qvalid_o,
    input  logic                      acc_qready_i,
    output logic [31:0]               acc_qdata_op_o,    // instruction word
    output logic [D_DATA_WIDTH-1:0]   acc_qdata_arga_o,  // rs1
    output logic [D_DATA_WIDTH-1:0]   acc_qdata_argb_o,  // rs2
    output logic [ADDR_WIDTH-1:0]     acc_qdata_argc_o,  // rs3/addr
    output logic [4:0]                acc_qid_o,         // rd
    input  logic                      acc_qaccept_i,
    input  logic                      acc_qwriteback_i,
    input  logic                      acc_qloadstore_i,
    input  logic                      acc_qexception_i,
    input  logic                      acc_qisfloat_i,
    input  logic [1:0]                acc_mem_finished_i,
    input  logic [1:0]                acc_mem_str_finished_i,

    // Accelerator Response Interface (Spatz → Snitch)
    input  logic                      acc_pvalid_i,
    output logic                      acc_pready_o,
    input  logic [4:0]                acc_pid_i,
    input  logic [D_DATA_WIDTH-1:0]   acc_pdata_i,
    input  logic                      acc_perror_i,

    // FPU side-channel
    output logic [2:0]                fpu_rnd_mode_o,
    output logic                      fpu_fmt_mode_o,
    input  logic [4:0]                fpu_status_i
);

    import snitch_pkg::*;

    typedef struct packed {
        logic [ADDR_WIDTH-1:0] addr;
        logic                  write;
        reqrsp_pkg::amo_op_e   amo;
        logic [D_DATA_WIDTH-1:0] data;
        logic [(D_DATA_WIDTH/8)-1:0] strb;
        logic [63:0]           user;
        reqrsp_pkg::size_t     size;
    } data_req_chan_t;

    typedef struct packed {
        data_req_chan_t q;
        logic           q_valid;
        logic           p_ready;
    } data_req_t;

    typedef struct packed {
        logic [D_DATA_WIDTH-1:0] data;
        logic                    error;
    } data_rsp_chan_t;

    typedef struct packed {
        data_rsp_chan_t p;
        logic           p_valid;
        logic           q_ready;
    } data_rsp_t;

    typedef struct packed {
        snitch_pkg::acc_addr_e addr;
        logic [4:0]            id;
        logic [31:0]           data_op;
        logic [D_DATA_WIDTH-1:0] data_arga;
        logic [D_DATA_WIDTH-1:0] data_argb;
        logic [ADDR_WIDTH-1:0] data_argc;
    } acc_issue_req_t;

    typedef struct packed {
        logic accept;
        logic writeback;
        logic loadstore;
        logic exception;
        logic isfloat;
    } acc_issue_rsp_t;

    typedef struct packed {
        logic [4:0]              id;
        logic                    error;
        logic [D_DATA_WIDTH-1:0] data;
    } acc_rsp_t;

    localparam type pa_t = logic [ADDR_WIDTH-1:0];
    typedef struct packed {
        pa_t                  pa;
        snitch_pkg::pte_flags_t flags;
    } l0_pte_t;

    logic [ADDR_WIDTH-1:0] inst_addr;
    logic                  inst_cacheable;
    logic                  inst_valid;
    logic                  inst_ready;
    logic [31:0]           inst_data;
    data_req_t             data_req;
    data_rsp_t             data_rsp;
    acc_issue_req_t        acc_issue_req;
    acc_issue_rsp_t        acc_issue_rsp;
    acc_rsp_t              acc_rsp;
    logic [1:0]            ptw_valid;
    logic [1:0]            ptw_ready;
    snitch_pkg::va_t [1:0] ptw_va;
    pa_t [1:0]             ptw_ppn;
    l0_pte_t [1:0]         ptw_pte;
    logic [1:0]            ptw_is_4mega;

    assign ptw_ready     = '0;
    assign ptw_pte       = '0;
    assign ptw_is_4mega  = '0;

    assign acc_qdata_op_o   = acc_issue_req.data_op;
    assign acc_qdata_arga_o = acc_issue_req.data_arga;
    assign acc_qdata_argb_o = acc_issue_req.data_argb;
    assign acc_qdata_argc_o = acc_issue_req.data_argc;
    assign acc_qid_o        = acc_issue_req.id;

    assign acc_issue_rsp = '{
        accept:    acc_qaccept_i,
        writeback: acc_qwriteback_i,
        loadstore: acc_qloadstore_i,
        exception: acc_qexception_i,
        isfloat:   acc_qisfloat_i
    };

    assign acc_rsp.id    = acc_pid_i;
    assign acc_rsp.data  = acc_pdata_i;
    assign acc_rsp.error = acc_perror_i;

    snitch #(
        .BootAddr              (BOOT_ADDR),
        .AddrWidth             (ADDR_WIDTH),
        .DataWidth             (D_DATA_WIDTH),
        .RVE                   (0),
        .Xdma                  (0),
        .Xssr                  (0),
        .FP_EN                 (0),
        .RVF                   (0),
        .RVD                   (0),
        .XF16                  (0),
        .XF16ALT               (0),
        .XF8                   (0),
        .XF8ALT                (0),
        .XDivSqrt              (0),
        .RVV                   (1),
        .XFVEC                 (0),
        .XFDOTP                (0),
        .XFAUX                 (0),
        .FLEN                  (D_DATA_WIDTH),
        .NumIntOutstandingLoads(1),
        .NumIntOutstandingMem(1),
        .VMSupport             (0),
        .Xipu                  (0),
        .dreq_t                (data_req_t),
        .drsp_t                (data_rsp_t),
        .acc_issue_req_t       (acc_issue_req_t),
        .acc_issue_rsp_t       (acc_issue_rsp_t),
        .acc_rsp_t             (acc_rsp_t),
        .pa_t                  (pa_t),
        .l0_pte_t              (l0_pte_t),
        .NumDTLBEntries        (0),
        .NumITLBEntries        (0)
    ) i_snitch (
        .clk_i                  (clk_i),
        .rst_i                  (~rst_ni),
        .hart_id_i              (hart_id_i),
        .irq_i                  ('0),
        .flush_i_valid_o        (),
        .flush_i_ready_i        (1'b1),
        .inst_addr_o            (inst_addr),
        .inst_cacheable_o       (inst_cacheable),
        .inst_data_i            (inst_data),
        .inst_valid_o           (inst_valid),
        .inst_ready_i           (inst_ready),
        .acc_qreq_o             (acc_issue_req),
        .acc_qrsp_i             (acc_issue_rsp),
        .acc_qvalid_o           (acc_qvalid_o),
        .acc_qready_i           (acc_qready_i),
        .acc_prsp_i             (acc_rsp),
        .acc_pvalid_i           (acc_pvalid_i),
        .acc_pready_o           (acc_pready_o),
        .acc_mem_finished_i     (acc_mem_finished_i),
        .acc_mem_str_finished_i (acc_mem_str_finished_i),
        .data_req_o             (data_req),
        .data_rsp_i             (data_rsp),
        .ptw_valid_o            (ptw_valid),
        .ptw_ready_i            (ptw_ready),
        .ptw_va_o               (ptw_va),
        .ptw_ppn_o              (ptw_ppn),
        .ptw_pte_i              (ptw_pte),
        .ptw_is_4mega_i         (ptw_is_4mega),
        .fpu_rnd_mode_o         (fpu_rnd_mode_o),
        .fpu_fmt_mode_o         (fpu_fmt_mode_o),
        .fpu_status_i           (fpu_status_i),
        .core_events_o          ()
    );

    // --- Instruction Fetch Interface Adapter ---
    logic [I_DATA_WIDTH-1:0] adapter_rsp_data;

    obi_snitch_if_adapter #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(I_DATA_WIDTH)
    ) u_if_adapter (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .snitch_req_valid_i (inst_valid),
        .snitch_req_ready_o (inst_ready),
        .snitch_req_addr_i  (inst_addr),
        .snitch_rsp_data_o  (adapter_rsp_data),
        .obi_req_o          (obi_i_req_o),
        .obi_gnt_i          (obi_i_gnt_i),
        .obi_addr_o         (obi_i_addr_o),
        .obi_rvalid_i       (obi_i_rvalid_i),
        .obi_rdata_i        (obi_i_rdata_i)
    );

    generate
        if (I_DATA_WIDTH == 32) begin : gen_inst_word_32
            assign inst_data = adapter_rsp_data[31:0];
        end else begin : gen_inst_word_wide
            assign inst_data = adapter_rsp_data >> (inst_addr[$clog2(I_DATA_WIDTH/8)-1:2] * 32);
        end
    endgenerate

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
