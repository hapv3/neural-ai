`default_nettype none

import npu_cluster_pkg::*;

module npu_cluster #(
    parameter int unsigned SYSTOLIC_OFM_FIFO_DEPTH = 8,
    parameter int unsigned SYSTOLIC_OTCDM_STALL_PERIOD = 0,
    parameter int unsigned SYSTOLIC_OTCDM_STALL_HOLD = 0
)(
    input  logic clk_i,       // 1 GHz NPU Core Clock
    input  logic rst_ni,      // NPU Core Reset
    input  logic fetch_enable_i,

    //---------------------------------------------------------
    // AXI4 Master Interface (To External Memory via DMA)
    //---------------------------------------------------------
    output logic [AXI_ADDR_WIDTH-1:0]       axi_aw_addr_o,
    output logic [7:0]                      axi_aw_len_o,
    output logic [2:0]                      axi_aw_size_o,
    output logic [1:0]                      axi_aw_burst_o,
    output logic                            axi_aw_valid_o,
    input  logic                            axi_aw_ready_i,

    output logic [AXI_DATA_WIDTH-1:0]       axi_w_data_o,
    output logic [(AXI_DATA_WIDTH/8)-1:0]   axi_w_strb_o,
    output logic                            axi_w_last_o,
    output logic                            axi_w_valid_o,
    input  logic                            axi_w_ready_i,

    input  logic [1:0]                      axi_b_resp_i,
    input  logic                            axi_b_valid_i,
    output logic                            axi_b_ready_o,

    output logic [AXI_ADDR_WIDTH-1:0]       axi_ar_addr_o,
    output logic [7:0]                      axi_ar_len_o,
    output logic [2:0]                      axi_ar_size_o,
    output logic [1:0]                      axi_ar_burst_o,
    output logic                            axi_ar_valid_o,
    input  logic                            axi_ar_ready_i,

    input  logic [AXI_DATA_WIDTH-1:0]       axi_r_data_i,
    input  logic [1:0]                      axi_r_resp_i,
    input  logic                            axi_r_last_i,
    input  logic                            axi_r_valid_i,
    output logic                            axi_r_ready_o,

    //---------------------------------------------------------
    // AXI4 Slave Interface (For Host Firmware Boot)
    //---------------------------------------------------------
    input  logic [AXI_ADDR_WIDTH-1:0]       s_axi_aw_addr_i,
    input  logic [7:0]                      s_axi_aw_len_i,
    input  logic [2:0]                      s_axi_aw_size_i,
    input  logic [1:0]                      s_axi_aw_burst_i,
    input  logic                            s_axi_aw_valid_i,
    output logic                            s_axi_aw_ready_o,

    input  logic [AXI_HOST_DATA_WIDTH-1:0]       s_axi_w_data_i,
    input  logic [(AXI_HOST_DATA_WIDTH/8)-1:0]   s_axi_w_strb_i,
    input  logic                            s_axi_w_last_i,
    input  logic                            s_axi_w_valid_i,
    output logic                            s_axi_w_ready_o,

    output logic [1:0]                      s_axi_b_resp_o,
    output logic                            s_axi_b_valid_o,
    input  logic                            s_axi_b_ready_i,

    input  logic [AXI_ADDR_WIDTH-1:0]       s_axi_ar_addr_i,
    input  logic [7:0]                      s_axi_ar_len_i,
    input  logic [2:0]                      s_axi_ar_size_i,
    input  logic [1:0]                      s_axi_ar_burst_i,
    input  logic                            s_axi_ar_valid_i,
    output logic                            s_axi_ar_ready_o,

    output logic [AXI_HOST_DATA_WIDTH-1:0]       s_axi_r_data_o,
    output logic [1:0]                      s_axi_r_resp_o,
    output logic                            s_axi_r_last_o,
    output logic                            s_axi_r_valid_o,
    input  logic                            s_axi_r_ready_i,

    //---------------------------------------------------------
    // Interrupts
    //---------------------------------------------------------
    output logic [2:0]                      debug_sys_state_o,
    output logic [1:0]                      debug_sys_drain_state_o,
    output logic [4:0]                      debug_linebuf_state_o,
    output logic [1:0]                      debug_linebuf_fetch_main_state_o,
    output logic [2:0]                      debug_linebuf_fetch_background_state_o,
    output logic [2:0]                      debug_linebuf_bypass_state_o,
    output logic                            irq_o
);

    //=========================================================
    // 1. Host AXI to OBI Bootloader Interface
    //=========================================================
    logic                      s_axi_obi_req;
    logic                      s_axi_obi_gnt;
    logic [OBI_ADDR_WIDTH-1:0] s_axi_obi_addr;
    logic                      s_axi_obi_we;
    logic [(ITCM_DATA_WIDTH/8)-1:0] s_axi_obi_be;
    logic [ITCM_DATA_WIDTH-1:0] s_axi_obi_wdata;
    logic                      s_axi_obi_rvalid;
    logic [ITCM_DATA_WIDTH-1:0] s_axi_obi_rdata;

    logic                      host_itcm_req;
    logic                      host_itcm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] host_itcm_addr;
    logic                      host_itcm_we;
    logic [(ITCM_DATA_WIDTH/8)-1:0] host_itcm_be;
    logic [ITCM_DATA_WIDTH-1:0] host_itcm_wdata;
    logic                      host_itcm_rvalid;
    logic [ITCM_DATA_WIDTH-1:0] host_itcm_rdata;

    logic                      pmu_mm_req;
    logic                      pmu_mm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] pmu_mm_addr;
    logic                      pmu_mm_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] pmu_mm_be;
    logic [MMIO_DATA_WIDTH-1:0] pmu_mm_wdata;
    logic                      pmu_mm_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] pmu_mm_rdata;

    logic                      host_cmd_req;
    logic                      host_cmd_gnt;
    logic [OBI_ADDR_WIDTH-1:0] host_cmd_addr;
    logic                      host_cmd_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] host_cmd_be;
    logic [MMIO_DATA_WIDTH-1:0] host_cmd_wdata;
    logic                      host_cmd_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] host_cmd_rdata;

    axi_lite_to_obi #(
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH(AXI_HOST_DATA_WIDTH),
        .OBI_ADDR_WIDTH(OBI_ADDR_WIDTH),
        .OBI_DATA_WIDTH(ITCM_DATA_WIDTH)
    ) u_axi_to_obi (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),

        .s_axi_aw_addr_i  (s_axi_aw_addr_i),
        .s_axi_aw_valid_i (s_axi_aw_valid_i),
        .s_axi_aw_ready_o (s_axi_aw_ready_o),
        .s_axi_w_data_i   (s_axi_w_data_i),
        .s_axi_w_strb_i   (s_axi_w_strb_i),
        .s_axi_w_valid_i  (s_axi_w_valid_i),
        .s_axi_w_ready_o  (s_axi_w_ready_o),
        .s_axi_b_resp_o   (s_axi_b_resp_o),
        .s_axi_b_valid_o  (s_axi_b_valid_o),
        .s_axi_b_ready_i  (s_axi_b_ready_i),
        .s_axi_ar_addr_i  (s_axi_ar_addr_i),
        .s_axi_ar_valid_i (s_axi_ar_valid_i),
        .s_axi_ar_ready_o (s_axi_ar_ready_o),
        .s_axi_r_data_o   (s_axi_r_data_o),
        .s_axi_r_resp_o   (s_axi_r_resp_o),
        .s_axi_r_valid_o  (s_axi_r_valid_o),
        .s_axi_r_ready_i  (s_axi_r_ready_i),

        .obi_req_o        (s_axi_obi_req),
        .obi_gnt_i        (s_axi_obi_gnt),
        .obi_addr_o       (s_axi_obi_addr),
        .obi_we_o         (s_axi_obi_we),
        .obi_be_o         (s_axi_obi_be),
        .obi_wdata_o      (s_axi_obi_wdata),
        .obi_rvalid_i     (s_axi_obi_rvalid),
        .obi_rdata_i      (s_axi_obi_rdata)
    );
    assign s_axi_r_last_o = s_axi_r_valid_o;

    obi_demux_1to4 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(ITCM_DATA_WIDTH),
        .M0_BASE (32'h1000_0000), .M0_MASK (32'hFFFF_8000), // I-TCM boot window
        .M1_BASE (32'h2000_4000), .M1_MASK (32'hFFFF_F000), // Host PMU window
        .M2_BASE (32'h2000_5000), .M2_MASK (32'hFFFF_F000), // Host command-control window
        .M3_BASE (32'hFFFF_1000), .M3_MASK (32'hFFFF_F000), // Unused/error sink
        .M0_REQ_REGISTER(1'b1)
    ) u_host_axi_demux_1to4 (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .slv_req_i   (s_axi_obi_req),
        .slv_gnt_o   (s_axi_obi_gnt),
        .slv_addr_i  (s_axi_obi_addr),
        .slv_we_i    (s_axi_obi_we),
        .slv_be_i    (s_axi_obi_be),
        .slv_wdata_i (s_axi_obi_wdata),
        .slv_rvalid_o(s_axi_obi_rvalid),
        .slv_rdata_o (s_axi_obi_rdata),

        .m0_req_o     (host_itcm_req),
        .m0_gnt_i     (host_itcm_gnt),
        .m0_addr_o    (host_itcm_addr),
        .m0_we_o      (host_itcm_we),
        .m0_be_o      (host_itcm_be),
        .m0_wdata_o   (host_itcm_wdata),
        .m0_rvalid_i  (host_itcm_rvalid),
        .m0_rdata_i   (host_itcm_rdata),

        .m1_req_o     (pmu_mm_req),
        .m1_gnt_i     (pmu_mm_gnt),
        .m1_addr_o    (pmu_mm_addr),
        .m1_we_o      (pmu_mm_we),
        .m1_be_o      (pmu_mm_be),
        .m1_wdata_o   (pmu_mm_wdata),
        .m1_rvalid_i  (pmu_mm_rvalid),
        .m1_rdata_i   (pmu_mm_rdata),

        .m2_req_o     (host_cmd_req),
        .m2_gnt_i     (host_cmd_gnt),
        .m2_addr_o    (host_cmd_addr),
        .m2_we_o      (host_cmd_we),
        .m2_be_o      (host_cmd_be),
        .m2_wdata_o   (host_cmd_wdata),
        .m2_rvalid_i  (host_cmd_rvalid),
        .m2_rdata_i   (host_cmd_rdata),

        .m3_req_o     (),
        .m3_gnt_i     (1'b1),
        .m3_addr_o    (),
        .m3_we_o      (),
        .m3_be_o      (),
        .m3_wdata_o   (),
        .m3_rvalid_i  (1'b1),
        .m3_rdata_i   ('0)
    );

    //=========================================================
    // 2. Snitch Core & I-TCM Arbitration
    //=========================================================
    logic                      snitch_i_req;
    logic                      snitch_i_gnt;
    logic [OBI_ADDR_WIDTH-1:0] snitch_i_addr;
    logic                      snitch_i_we;
    logic [(ITCM_DATA_WIDTH/8)-1:0] snitch_i_be;
    logic [ITCM_DATA_WIDTH-1:0] snitch_i_wdata;
    logic                      snitch_i_rvalid;
    logic [ITCM_DATA_WIDTH-1:0] snitch_i_rdata;

    logic                      snitch_d_req;
    logic                      snitch_d_gnt;
    logic [OBI_ADDR_WIDTH-1:0] snitch_d_addr;
    logic                      snitch_d_we;
    logic [(SNITCH_D_DATA_WIDTH/8)-1:0] snitch_d_be;
    logic [SNITCH_D_DATA_WIDTH-1:0]     snitch_d_wdata;
    logic                      snitch_d_rvalid;
    logic [SNITCH_D_DATA_WIDTH-1:0]     snitch_d_rdata;

    // Accelerator interface wires (Snitch ↔ Spatz)
    logic        acc_qvalid;
    logic        acc_qready;
    logic [31:0] acc_qdata_op;
    logic [SNITCH_D_DATA_WIDTH-1:0] acc_qdata_arga_core;
    logic [SNITCH_D_DATA_WIDTH-1:0] acc_qdata_argb_core;
    logic [63:0] acc_qdata_arga;
    logic [63:0] acc_qdata_argb;
    logic [31:0] acc_qdata_argc;
    logic [4:0]  acc_qid;
    logic        acc_qaccept;
    logic        acc_qwriteback;
    logic        acc_qloadstore;
    logic        acc_qexception;
    logic        acc_qisfloat;
    logic [1:0]  acc_mem_finished;
    logic [1:0]  acc_mem_str_finished;
    logic        acc_pvalid;
    logic        acc_pready;
    logic [4:0]  acc_pid;
    logic [SNITCH_D_DATA_WIDTH-1:0] acc_pdata_core;
    logic [63:0] acc_pdata;
    logic        acc_perror;
    logic [2:0]  fpu_rnd_mode;
    logic        fpu_fmt_mode;
    logic [4:0]  fpu_status;
    snitch_pkg::interrupts_t snitch_irq;
    snitch_pkg::core_events_t snitch_core_events;

    snitch_core #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .I_DATA_WIDTH(ITCM_DATA_WIDTH),
        .D_DATA_WIDTH(SNITCH_D_DATA_WIDTH),
        .BOOT_ADDR (32'h1000_0000) // I-TCM Base Addr
    ) u_snitch_core (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni & fetch_enable_i),
        .hart_id_i        (32'd0),
        .irq_i            (snitch_irq),

        .obi_i_req_o      (snitch_i_req),
        .obi_i_gnt_i      (snitch_i_gnt),
        .obi_i_addr_o     (snitch_i_addr),
        .obi_i_we_o       (snitch_i_we),
        .obi_i_be_o       (snitch_i_be),
        .obi_i_wdata_o    (snitch_i_wdata),
        .obi_i_rvalid_i   (snitch_i_rvalid),
        .obi_i_rdata_i    (snitch_i_rdata),

        .obi_d_req_o      (snitch_d_req),
        .obi_d_gnt_i      (snitch_d_gnt),
        .obi_d_addr_o     (snitch_d_addr),
        .obi_d_we_o       (snitch_d_we),
        .obi_d_be_o       (snitch_d_be),
        .obi_d_wdata_o    (snitch_d_wdata),
        .obi_d_rvalid_i   (snitch_d_rvalid),
        .obi_d_rdata_i    (snitch_d_rdata),

        // Accelerator Offload → Spatz
        .acc_qvalid_o     (acc_qvalid),
        .acc_qready_i     (acc_qready),
        .acc_qdata_op_o   (acc_qdata_op),
        .acc_qdata_arga_o (acc_qdata_arga_core),
        .acc_qdata_argb_o (acc_qdata_argb_core),
        .acc_qdata_argc_o (acc_qdata_argc),
        .acc_qid_o        (acc_qid),
        .acc_qaccept_i    (acc_qaccept),
        .acc_qwriteback_i (acc_qwriteback),
        .acc_qloadstore_i (acc_qloadstore),
        .acc_qexception_i (acc_qexception),
        .acc_qisfloat_i   (acc_qisfloat),
        .acc_mem_finished_i    (acc_mem_finished),
        .acc_mem_str_finished_i(acc_mem_str_finished),
        // Accelerator Response ← Spatz
        .acc_pvalid_i     (acc_pvalid),
        .acc_pready_o     (acc_pready),
        .acc_pid_i        (acc_pid),
        .acc_pdata_i      (acc_pdata_core),
        .acc_perror_i     (acc_perror),
        // FPU side-channel
        .fpu_rnd_mode_o   (fpu_rnd_mode),
        .fpu_fmt_mode_o   (fpu_fmt_mode),
        .fpu_status_i     (fpu_status),
        .core_events_o    (snitch_core_events)
    );

    // Arbiter for I-TCM
    logic                      itcm_req;
    logic                      itcm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] itcm_addr;
    logic                      itcm_we;
    logic [(ITCM_DATA_WIDTH/8)-1:0] itcm_be;
    logic [ITCM_DATA_WIDTH-1:0] itcm_wdata;
    logic                      itcm_rvalid;
    logic [ITCM_DATA_WIDTH-1:0] itcm_rdata;

    obi_arbiter_2to1 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(ITCM_DATA_WIDTH)
    ) u_itcm_arbiter (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        
        .m0_req_i    (host_itcm_req),
        .m0_gnt_o    (host_itcm_gnt),
        .m0_addr_i   (host_itcm_addr),
        .m0_we_i     (host_itcm_we),
        .m0_be_i     (host_itcm_be),
        .m0_wdata_i  (host_itcm_wdata),
        .m0_rvalid_o (host_itcm_rvalid),
        .m0_rdata_o  (host_itcm_rdata),
        
        .m1_req_i    (snitch_i_req),
        .m1_gnt_o    (snitch_i_gnt),
        .m1_addr_i   (snitch_i_addr),
        .m1_we_i     (snitch_i_we),
        .m1_be_i     (snitch_i_be),
        .m1_wdata_i  (snitch_i_wdata),
        .m1_rvalid_o (snitch_i_rvalid),
        .m1_rdata_o  (snitch_i_rdata),
        
        .slv_req_o   (itcm_req),
        .slv_gnt_i   (itcm_gnt),
        .slv_addr_o  (itcm_addr),
        .slv_we_o    (itcm_we),
        .slv_be_o    (itcm_be),
        .slv_wdata_o (itcm_wdata),
        .slv_rvalid_i(itcm_rvalid),
        .slv_rdata_i (itcm_rdata)
    );

    // I-TCM SRAM Bank (32 KB)
    cluster_sram_bank #(
        .DATA_WIDTH(ITCM_DATA_WIDTH),
        .SIZE_BYTES(32768)
    ) u_sram_i_tcm (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .req_i   (itcm_req),
        .we_i    (itcm_we),
        .addr_i  ((itcm_addr & 32'h0000_7FFF) >> 2),
        .wdata_i (itcm_wdata),
        .be_i    (itcm_be),
        .gnt_o   (itcm_gnt),
        .rvalid_o(itcm_rvalid),
        .rdata_o (itcm_rdata)
    );


    //=========================================================
    // 3. Snitch D-Bus Demux (D-TCM, Shared Data TCDM, MMIO)
    //=========================================================
    logic                      dtcm_req;
    logic                      dtcm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] dtcm_addr;
    logic                      dtcm_we;
    logic [(DTCM_DATA_WIDTH/8)-1:0] dtcm_be;
    logic [DTCM_DATA_WIDTH-1:0] dtcm_wdata;
    logic                      dtcm_rvalid;
    logic [DTCM_DATA_WIDTH-1:0] dtcm_rdata;

    logic                      ddata_req;
    logic                      ddata_gnt;
    logic [OBI_ADDR_WIDTH-1:0] ddata_addr;
    logic                      ddata_we;
    logic [(SNITCH_D_DATA_WIDTH/8)-1:0] ddata_be;
    logic [SNITCH_D_DATA_WIDTH-1:0] ddata_wdata;
    logic                      ddata_rvalid;
    logic [SNITCH_D_DATA_WIDTH-1:0] ddata_rdata;

    logic                      ddata_wide_req;
    logic                      ddata_wide_gnt;
    logic [OBI_ADDR_WIDTH-1:0] ddata_wide_addr;
    logic                      ddata_wide_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] ddata_wide_be;
    logic [OBI_DATA_WIDTH-1:0] ddata_wide_wdata;
    logic                      ddata_wide_rvalid;
    logic [OBI_DATA_WIDTH-1:0] ddata_wide_rdata;

    logic                      reg_req;
    logic                      reg_gnt;
    logic [OBI_ADDR_WIDTH-1:0] reg_addr;
    logic                      reg_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] reg_be;
    logic [MMIO_DATA_WIDTH-1:0] reg_wdata;
    logic                      reg_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] reg_rdata;

    logic                      ctrl_req;
    logic                      ctrl_gnt;
    logic [OBI_ADDR_WIDTH-1:0] ctrl_addr;
    logic                      ctrl_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] ctrl_be;
    logic [MMIO_DATA_WIDTH-1:0] ctrl_wdata;
    logic                      ctrl_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] ctrl_rdata;
    logic                      ctrl_unused_req;
    logic                      ctrl_unused_gnt;
    logic                      ctrl_unused_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] ctrl_unused_rdata;
    logic                      systolic_ctrl_req;
    logic                      systolic_ctrl_gnt;
    logic                      systolic_ctrl_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] systolic_ctrl_rdata;
    logic                      ctrl_systolic_sel;

    logic                      idma_mm_req;
    logic                      idma_mm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] idma_mm_addr;
    logic                      idma_mm_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] idma_mm_be;
    logic [MMIO_DATA_WIDTH-1:0] idma_mm_wdata;
    logic                      idma_mm_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] idma_mm_rdata;

    logic                      irq_ctrl_req;
    logic                      irq_ctrl_gnt;
    logic [OBI_ADDR_WIDTH-1:0] irq_ctrl_addr;
    logic                      irq_ctrl_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] irq_ctrl_be;
    logic [MMIO_DATA_WIDTH-1:0] irq_ctrl_wdata;
    logic                      irq_ctrl_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] irq_ctrl_rdata;

    logic                      afu_mm_req;
    logic                      afu_mm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] afu_mm_addr;
    logic                      afu_mm_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] afu_mm_be;
    logic [MMIO_DATA_WIDTH-1:0] afu_mm_wdata;
    logic                      afu_mm_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] afu_mm_rdata;

    logic                      snitch_cmd_req;
    logic                      snitch_cmd_gnt;
    logic [OBI_ADDR_WIDTH-1:0] snitch_cmd_addr;
    logic                      snitch_cmd_we;
    logic [(MMIO_DATA_WIDTH/8)-1:0] snitch_cmd_be;
    logic [MMIO_DATA_WIDTH-1:0] snitch_cmd_wdata;
    logic                      snitch_cmd_rvalid;
    logic [MMIO_DATA_WIDTH-1:0] snitch_cmd_rdata;

    logic                      afu_obi_req;
    logic                      afu_obi_gnt;
    logic [OBI_ADDR_WIDTH-1:0] afu_obi_addr;
    logic                      afu_obi_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] afu_obi_be;
    logic [OBI_DATA_WIDTH-1:0] afu_obi_wdata;
    logic                      afu_obi_rvalid;
    logic [OBI_DATA_WIDTH-1:0] afu_obi_rdata;
    logic                      afu_rhs_obi_req;
    logic                      afu_rhs_obi_gnt;
    logic [OBI_ADDR_WIDTH-1:0] afu_rhs_obi_addr;
    logic                      afu_rhs_obi_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] afu_rhs_obi_be;
    logic [OBI_DATA_WIDTH-1:0] afu_rhs_obi_wdata;
    logic                      afu_rhs_obi_rvalid;
    logic [OBI_DATA_WIDTH-1:0] afu_rhs_obi_rdata;
    logic                      afu_done;
    logic                      afu_perf_start;
    logic                      afu_perf_active;
    logic [4:0]                afu_perf_state;
    logic                      afu_perf_lhs_consume;
    logic                      afu_perf_rhs_consume;
    logic                      afu_perf_result_produce;
    logic                      afu_perf_input_wait;
    logic                      afu_perf_rhs_wait;
    logic                      afu_perf_output_stall;

    obi_demux_1to4 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(SNITCH_D_DATA_WIDTH),
        .M0_BASE (32'h1000_8000), .M0_MASK (32'hFFFF_8000), // D-TCM
        .M1_BASE (32'h1010_0000), .M1_MASK (32'hFFF0_0000), // Shared Data
        .M2_BASE (32'h2000_0000), .M2_MASK (32'hFFFF_0000), // MMIO
        .M3_BASE (32'hFFFF_0000), .M3_MASK (32'hFFFF_0000)  // Unused/error sink
    ) u_dside_demux_1to4 (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .slv_req_i   (snitch_d_req),
        .slv_gnt_o   (snitch_d_gnt),
        .slv_addr_i  (snitch_d_addr),
        .slv_we_i    (snitch_d_we),
        .slv_be_i    (snitch_d_be),
        .slv_wdata_i (snitch_d_wdata),
        .slv_rvalid_o(snitch_d_rvalid),
        .slv_rdata_o (snitch_d_rdata),

        .m0_req_o     (dtcm_req),
        .m0_gnt_i     (dtcm_gnt),
        .m0_addr_o    (dtcm_addr),
        .m0_we_o      (dtcm_we),
        .m0_be_o      (dtcm_be),
        .m0_wdata_o   (dtcm_wdata),
        .m0_rvalid_i  (dtcm_rvalid),
        .m0_rdata_i   (dtcm_rdata),

        .m1_req_o     (ddata_req),
        .m1_gnt_i     (ddata_gnt),
        .m1_addr_o    (ddata_addr),
        .m1_we_o      (ddata_we),
        .m1_be_o      (ddata_be),
        .m1_wdata_o   (ddata_wdata),
        .m1_rvalid_i  (ddata_rvalid),
        .m1_rdata_i   (ddata_rdata),

        .m2_req_o     (reg_req),
        .m2_gnt_i     (reg_gnt),
        .m2_addr_o    (reg_addr),
        .m2_we_o      (reg_we),
        .m2_be_o      (reg_be),
        .m2_wdata_o   (reg_wdata),
        .m2_rvalid_i  (reg_rvalid),
        .m2_rdata_i   (reg_rdata),

        .m3_req_o     (),
        .m3_gnt_i     (1'b1),
        .m3_addr_o    (),
        .m3_we_o      (),
        .m3_be_o      (),
        .m3_wdata_o   (),
        .m3_rvalid_i  (1'b1),
        .m3_rdata_i   ('0)
    );

    obi_narrow_to_wide #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .M_DATA_WIDTH(SNITCH_D_DATA_WIDTH),
        .S_DATA_WIDTH(OBI_DATA_WIDTH)
    ) u_ddata_narrow_to_wide (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .mst_req_i   (ddata_req),
        .mst_gnt_o   (ddata_gnt),
        .mst_addr_i  (ddata_addr),
        .mst_we_i    (ddata_we),
        .mst_be_i    (ddata_be),
        .mst_wdata_i (ddata_wdata),
        .mst_rvalid_o(ddata_rvalid),
        .mst_rdata_o (ddata_rdata),

        .slv_req_o   (ddata_wide_req),
        .slv_gnt_i   (ddata_wide_gnt),
        .slv_addr_o  (ddata_wide_addr),
        .slv_we_o    (ddata_wide_we),
        .slv_be_o    (ddata_wide_be),
        .slv_wdata_o (ddata_wide_wdata),
        .slv_rvalid_i(ddata_wide_rvalid),
        .slv_rdata_i (ddata_wide_rdata)
    );

    // D-TCM SRAM Bank (32 KB — matches link.ld)
    cluster_sram_bank #(
        .DATA_WIDTH(DTCM_DATA_WIDTH),
        .SIZE_BYTES(32768)
    ) u_sram_d_tcm (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .req_i   (dtcm_req),
        .we_i    (dtcm_we),
        .addr_i  ((dtcm_addr - 32'h1000_8000) >> 2),
        .wdata_i (dtcm_wdata),
        .be_i    (dtcm_be),
        .gnt_o   (dtcm_gnt),
        .rvalid_o(dtcm_rvalid),
        .rdata_o (dtcm_rdata)
    );

    //=========================================================
    // 4. Cluster Control Registers (MMIO)
    //=========================================================
    obi_demux_1to5 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(MMIO_DATA_WIDTH),
        .M0_BASE (32'h2000_0000), .M0_MASK (32'hFFFF_F000), // Cluster control
        .M1_BASE (32'h2000_1000), .M1_MASK (32'hFFFF_F000), // iDMA-style control
        .M2_BASE (32'h2000_2000), .M2_MASK (32'hFFFF_F000), // Interrupt controller
        .M3_BASE (32'h2000_3000), .M3_MASK (32'hFFFF_F000), // AFU control + LUT
        .M4_BASE (32'h2000_5000), .M4_MASK (32'hFFFF_F000)  // Command-control bootstrap/status
    ) u_mmio_demux_1to5 (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .slv_req_i   (reg_req),
        .slv_gnt_o   (reg_gnt),
        .slv_addr_i  (reg_addr),
        .slv_we_i    (reg_we),
        .slv_be_i    (reg_be),
        .slv_wdata_i (reg_wdata),
        .slv_rvalid_o(reg_rvalid),
        .slv_rdata_o (reg_rdata),

        .m0_req_o     (ctrl_req),
        .m0_gnt_i     (ctrl_gnt),
        .m0_addr_o    (ctrl_addr),
        .m0_we_o      (ctrl_we),
        .m0_be_o      (ctrl_be),
        .m0_wdata_o   (ctrl_wdata),
        .m0_rvalid_i  (ctrl_rvalid),
        .m0_rdata_i   (ctrl_rdata),

        .m1_req_o     (idma_mm_req),
        .m1_gnt_i     (idma_mm_gnt),
        .m1_addr_o    (idma_mm_addr),
        .m1_we_o      (idma_mm_we),
        .m1_be_o      (idma_mm_be),
        .m1_wdata_o   (idma_mm_wdata),
        .m1_rvalid_i  (idma_mm_rvalid),
        .m1_rdata_i   (idma_mm_rdata),

        .m2_req_o     (irq_ctrl_req),
        .m2_gnt_i     (irq_ctrl_gnt),
        .m2_addr_o    (irq_ctrl_addr),
        .m2_we_o      (irq_ctrl_we),
        .m2_be_o      (irq_ctrl_be),
        .m2_wdata_o   (irq_ctrl_wdata),
        .m2_rvalid_i  (irq_ctrl_rvalid),
        .m2_rdata_i   (irq_ctrl_rdata),

        .m3_req_o     (afu_mm_req),
        .m3_gnt_i     (afu_mm_gnt),
        .m3_addr_o    (afu_mm_addr),
        .m3_we_o      (afu_mm_we),
        .m3_be_o      (afu_mm_be),
        .m3_wdata_o   (afu_mm_wdata),
        .m3_rvalid_i  (afu_mm_rvalid),
        .m3_rdata_i   (afu_mm_rdata),

        .m4_req_o     (snitch_cmd_req),
        .m4_gnt_i     (snitch_cmd_gnt),
        .m4_addr_o    (snitch_cmd_addr),
        .m4_we_o      (snitch_cmd_we),
        .m4_be_o      (snitch_cmd_be),
        .m4_wdata_o   (snitch_cmd_wdata),
        .m4_rvalid_i  (snitch_cmd_rvalid),
        .m4_rdata_i   (snitch_cmd_rdata)
    );

    logic        cfg_dma_done;

    logic        cfg_sys_done;
    logic [31:0] pmu_context_id;
    logic        pmu_context_active;
    logic        pmu_context_begin;
    logic        pmu_context_end;
    logic [3:0]  pmu_phase;

    npu_cmd_ctrl #(
        .ADDR_WIDTH        (OBI_ADDR_WIDTH),
        .DATA_WIDTH        (MMIO_DATA_WIDTH),
        .BASE_ADDR         (32'h2000_5000),
        .DEFAULT_TCDM_BASE (32'h1017_F000),
        .DEFAULT_TCDM_BYTES(32'h0000_1000)
    ) u_cmd_ctrl (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),

        .host_req_i       (host_cmd_req),
        .host_gnt_o       (host_cmd_gnt),
        .host_addr_i      (host_cmd_addr),
        .host_we_i        (host_cmd_we),
        .host_be_i        (host_cmd_be),
        .host_wdata_i     (host_cmd_wdata),
        .host_rvalid_o    (host_cmd_rvalid),
        .host_rdata_o     (host_cmd_rdata),

        .snitch_req_i     (snitch_cmd_req),
        .snitch_gnt_o     (snitch_cmd_gnt),
        .snitch_addr_i    (snitch_cmd_addr),
        .snitch_we_i      (snitch_cmd_we),
        .snitch_be_i      (snitch_cmd_be),
        .snitch_wdata_i   (snitch_cmd_wdata),
        .snitch_rvalid_o  (snitch_cmd_rvalid),
        .snitch_rdata_o   (snitch_cmd_rdata),
        .pmu_context_id_o (pmu_context_id),
        .pmu_context_active_o(pmu_context_active),
        .pmu_context_begin_o(pmu_context_begin),
        .pmu_context_end_o(pmu_context_end),
        .pmu_phase_o      (pmu_phase)
    );

    assign ctrl_systolic_sel = ((ctrl_addr & 32'hFFFF) >= 32'h0100) &&
                               ((ctrl_addr & 32'hFFFF) < 32'h0580);
    assign ctrl_unused_req = ctrl_req && !ctrl_systolic_sel;
    assign systolic_ctrl_req = ctrl_req && ctrl_systolic_sel;
    assign ctrl_unused_gnt = 1'b1;
    assign ctrl_unused_rdata = '0;
    assign ctrl_gnt = ctrl_systolic_sel ? systolic_ctrl_gnt : ctrl_unused_gnt;
    assign ctrl_rvalid = systolic_ctrl_rvalid | ctrl_unused_rvalid;
    assign ctrl_rdata = systolic_ctrl_rvalid ? systolic_ctrl_rdata : ctrl_unused_rdata;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            ctrl_unused_rvalid <= 1'b0;
        end else begin
            ctrl_unused_rvalid <= ctrl_unused_req && ctrl_unused_gnt;
        end
    end

    npu_interrupt_ctrl #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(MMIO_DATA_WIDTH)
    ) u_interrupt_ctrl (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .req_i         (irq_ctrl_req),
        .gnt_o         (irq_ctrl_gnt),
        .addr_i        (irq_ctrl_addr),
        .we_i          (irq_ctrl_we),
        .be_i          (irq_ctrl_be),
        .wdata_i       (irq_ctrl_wdata),
        .rvalid_o      (irq_ctrl_rvalid),
        .rdata_o       (irq_ctrl_rdata),
        .dma_done_i    (cfg_dma_done),
        .sys_done_i    (cfg_sys_done),
        .afu_done_i    (afu_done),
        .spatz_done_i  (acc_pvalid),
        .snitch_irq_o  (snitch_irq),
        .host_irq_o    (irq_o)
    );

    afu #(
        .ADDR_WIDTH     (OBI_ADDR_WIDTH),
        .CFG_DATA_WIDTH (MMIO_DATA_WIDTH),
        .MEM_DATA_WIDTH (OBI_DATA_WIDTH),
        .LUT_LANES      (4)
    ) u_afu (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .obi_s_req_i    (afu_mm_req),
        .obi_s_gnt_o    (afu_mm_gnt),
        .obi_s_addr_i   (afu_mm_addr - 32'h2000_3000),
        .obi_s_we_i     (afu_mm_we),
        .obi_s_be_i     (afu_mm_be),
        .obi_s_wdata_i  (afu_mm_wdata),
        .obi_s_rvalid_o (afu_mm_rvalid),
        .obi_s_rdata_o  (afu_mm_rdata),
        .obi_m_req_o    (afu_obi_req),
        .obi_m_gnt_i    (afu_obi_gnt),
        .obi_m_addr_o   (afu_obi_addr),
        .obi_m_we_o     (afu_obi_we),
        .obi_m_be_o     (afu_obi_be),
        .obi_m_wdata_o  (afu_obi_wdata),
        .obi_m_rvalid_i (afu_obi_rvalid),
        .obi_m_rdata_i  (afu_obi_rdata),
        .obi_rhs_req_o  (afu_rhs_obi_req),
        .obi_rhs_gnt_i  (afu_rhs_obi_gnt),
        .obi_rhs_addr_o (afu_rhs_obi_addr),
        .obi_rhs_we_o   (afu_rhs_obi_we),
        .obi_rhs_be_o   (afu_rhs_obi_be),
        .obi_rhs_wdata_o(afu_rhs_obi_wdata),
        .obi_rhs_rvalid_i(afu_rhs_obi_rvalid),
        .obi_rhs_rdata_i(afu_rhs_obi_rdata),
        .done_o         (afu_done),
        .perf_start_o   (afu_perf_start),
        .perf_active_o  (afu_perf_active),
        .perf_state_o   (afu_perf_state),
        .perf_lhs_consume_o(afu_perf_lhs_consume),
        .perf_rhs_consume_o(afu_perf_rhs_consume),
        .perf_result_produce_o(afu_perf_result_produce),
        .perf_input_wait_o(afu_perf_input_wait),
        .perf_rhs_wait_o(afu_perf_rhs_wait),
        .perf_output_stall_o(afu_perf_output_stall)
    );

    //=========================================================
    // 5. Shared Data TCDM Interconnect (14 Masters)
    //=========================================================
    localparam int unsigned NUM_MASTERS = 14;
    // Master 0: Snitch D-Bus
    // Master 1: Spatz Vector Engine (VLSU port 0)
    // Master 2: PULP iDMA AXI2OBI write port
    // Master 3: Systolic Controller IFM/linebuffer read (I-TCDM)
    // Master 4: Systolic Controller Write Port 0 (O-TCDM)
    // Master 5: Systolic Controller Write Port 1 (O-TCDM)
    // Master 6: Systolic Controller Write Port 2 (O-TCDM)
    // Master 7: Systolic Controller Write Port 3 (O-TCDM)
    // Master 8: Spatz Vector Engine (VLSU port 1)
    // Master 9: PULP iDMA OBI2AXI read port
    // Master 10: AFU LUT processor
    // Master 11: Systolic Controller weight read (I-TCDM)
    // Master 12: AFU RHS read port
    // Master 13: Systolic Controller binary RHS read port

    obi_req_t [NUM_MASTERS-1:0] master_req;
    obi_rsp_t [NUM_MASTERS-1:0] master_rsp;

    obi_req_t [TCDM_NUM_BANKS-1:0] slave_req;
    obi_rsp_t [TCDM_NUM_BANKS-1:0] slave_rsp;

    logic [NUM_MASTERS-1:0]                      mst_req, mst_we, mst_gnt, mst_rvalid;
    logic [NUM_MASTERS-1:0][OBI_ADDR_WIDTH-1:0]  mst_addr;
    logic [NUM_MASTERS-1:0][(OBI_DATA_WIDTH/8)-1:0] mst_be;
    logic [NUM_MASTERS-1:0][OBI_DATA_WIDTH-1:0]  mst_wdata, mst_rdata;

    logic [TCDM_NUM_BANKS-1:0]                   slv_req, slv_we;
    logic [TCDM_NUM_BANKS-1:0][OBI_ADDR_WIDTH-1:0] slv_addr;
    logic [TCDM_NUM_BANKS-1:0][(OBI_DATA_WIDTH/8)-1:0] slv_be;
    logic [TCDM_NUM_BANKS-1:0][OBI_DATA_WIDTH-1:0] slv_wdata, slv_rdata;
    logic [TCDM_NUM_BANKS-1:0]                   tcdm_bank_conflict;

    for (genvar m = 0; m < NUM_MASTERS; m++) begin
        assign mst_req[m]   = master_req[m].req;
        assign mst_we[m]    = master_req[m].we;
        assign mst_addr[m]  = master_req[m].addr;
        assign mst_be[m]    = master_req[m].be;
        assign mst_wdata[m] = master_req[m].wdata;
        
        assign master_rsp[m].gnt    = mst_gnt[m];
        assign master_rsp[m].rvalid = mst_rvalid[m];
        assign master_rsp[m].rdata  = mst_rdata[m];
    end

    for (genvar b = 0; b < TCDM_NUM_BANKS; b++) begin
        assign slave_req[b].req   = slv_req[b];
        assign slave_req[b].we    = slv_we[b];
        // tcdm_interconnect already computes the de-interleaved address:
        //   bank_addr_o = ((addr >> BYTE_SEL) / NUM_BANKS) << BYTE_SEL
        // This gives a byte-address within the bank. cluster_sram_bank
        // indexes with addr_i[ADDR_BITS-1:0], so we just need to convert
        // the byte-address to a word-address by shifting right by BYTE_SEL (5).
        assign slave_req[b].addr  = slv_addr[b] >> 5;
        assign slave_req[b].be    = slv_be[b];
        assign slave_req[b].wdata = slv_wdata[b];
        
        assign slv_rdata[b] = slave_rsp[b].rdata;
    end

    tcdm_interconnect #(
        .NUM_MASTERS(NUM_MASTERS),
        .NUM_BANKS(TCDM_NUM_BANKS),
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .HWPE_MASTER_MASK(14'h3DFA), // M1, M3-M8, M10-M13: Spatz + Systolic + AFU
        .DMA_MASTER_MASK (14'h0204), // M2, M9: iDMA local write/read ports
        .CORE_MASTER_MASK(14'h0001)  // M0: Snitch D-Bus
    ) u_tcdm_interconnect (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .master_req_i     (mst_req),
        .master_gnt_o     (mst_gnt),
        .master_addr_i    (mst_addr),
        .master_we_i      (mst_we),
        .master_be_i      (mst_be),
        .master_wdata_i   (mst_wdata),
        .master_rvalid_o  (mst_rvalid),
        .master_rdata_o   (mst_rdata),
        .bank_req_o       (slv_req),
        .bank_addr_o      (slv_addr),
        .bank_we_o        (slv_we),
        .bank_be_o        (slv_be),
        .bank_wdata_o     (slv_wdata),
        .bank_rdata_i     (slv_rdata),
        .perf_bank_conflict_o(tcdm_bank_conflict)
    );

    // Shared Data TCDM SRAM Banks (16 x 32KB = 512KB)
    for (genvar i = 0; i < TCDM_NUM_BANKS; i++) begin : gen_sram_banks
        cluster_sram_bank #(
            .DATA_WIDTH(OBI_DATA_WIDTH),
            .SIZE_BYTES(TCDM_BANK_SIZE)
        ) u_sram_bank (
            .clk_i   (clk_i),
            .rst_ni  (rst_ni),
            .req_i   (slave_req[i].req),
            .we_i    (slave_req[i].we),
            .addr_i  (slave_req[i].addr),
            .wdata_i (slave_req[i].wdata),
            .be_i    (slave_req[i].be),
            .gnt_o   (slave_rsp[i].gnt),
            .rvalid_o(slave_rsp[i].rvalid),
            .rdata_o (slave_rsp[i].rdata)
        );
    end

    // Master 0: Snitch D-Bus
    assign master_req[0].req   = ddata_wide_req;
    assign master_req[0].we    = ddata_wide_we;
    assign master_req[0].be    = ddata_wide_be;
    assign master_req[0].addr  = ddata_wide_addr;
    assign master_req[0].wdata = ddata_wide_wdata;
    
    assign ddata_wide_gnt    = master_rsp[0].gnt;
    assign ddata_wide_rvalid = master_rsp[0].rvalid;
    assign ddata_wide_rdata  = master_rsp[0].rdata;

    //=========================================================
    // 6a. Spatz Vector Engine + TCDM-to-OBI Bridge (Master 1)
    //=========================================================
    // Spatz issue request matches Snitch's accelerator request channel.
    localparam type spatz_issue_req_t = `SNITCH_ACC_REQ_CHAN_STRUCT(64, OBI_ADDR_WIDTH);
    typedef struct packed {
        logic accept;
        logic writeback;
        logic loadstore;
        logic exception;
        logic isfloat;
    } spatz_issue_rsp_t;
    localparam type spatz_rsp_t = `SNITCH_ACC_RSP_CHAN_STRUCT(64);

    typedef struct packed {
        logic [OBI_ADDR_WIDTH-1:0] addr;
        logic                      write;
        reqrsp_pkg::amo_op_e       amo;
        logic [31:0]               data;
        logic [3:0]                strb;
        logic                      user;
    } spatz_tcdm_req_chan_t;

    typedef struct packed {
        logic [31:0] data;
    } spatz_tcdm_rsp_chan_t;

    // Spatz VLSU TCDM memory signals (2 ports for 2-lane INT-only config)
    localparam int unsigned SPATZ_MEM_PORTS = 2;
    spatz_tcdm_req_chan_t [SPATZ_MEM_PORTS-1:0] spatz_mem_req;
    logic                 [SPATZ_MEM_PORTS-1:0] spatz_mem_req_valid;
    logic                 [SPATZ_MEM_PORTS-1:0] spatz_mem_req_ready;
    spatz_tcdm_rsp_chan_t [SPATZ_MEM_PORTS-1:0] spatz_mem_rsp;
    logic                 [SPATZ_MEM_PORTS-1:0] spatz_mem_rsp_valid;

    // Reconstruct full reqrsp structs for Spatz issue interface
    spatz_issue_req_t spatz_issue_req;
    assign acc_qdata_arga = {{(64-SNITCH_D_DATA_WIDTH){1'b0}}, acc_qdata_arga_core};
    assign acc_qdata_argb = {{(64-SNITCH_D_DATA_WIDTH){1'b0}}, acc_qdata_argb_core};
    assign acc_pdata_core = acc_pdata[SNITCH_D_DATA_WIDTH-1:0];

    assign spatz_issue_req.addr       = snitch_pkg::SPATZ;
    assign spatz_issue_req.data_op    = acc_qdata_op;
    assign spatz_issue_req.data_arga  = acc_qdata_arga;
    assign spatz_issue_req.data_argb  = acc_qdata_argb;
    assign spatz_issue_req.data_argc  = acc_qdata_argc;
    assign spatz_issue_req.id         = acc_qid;

    spatz_issue_rsp_t spatz_issue_rsp;
    assign acc_qaccept    = spatz_issue_rsp.accept;
    assign acc_qwriteback = spatz_issue_rsp.writeback;
    assign acc_qloadstore = spatz_issue_rsp.loadstore;
    assign acc_qexception = spatz_issue_rsp.exception;
    assign acc_qisfloat   = spatz_issue_rsp.isfloat;

    // Spatz response → Snitch
    spatz_rsp_t spatz_rsp;

    // Dummy FP LSU interface (tied off — no FPU)
    typedef struct packed {
        logic [OBI_ADDR_WIDTH-1:0] addr;
        logic                      write;
        reqrsp_pkg::amo_op_e       amo;
        logic [63:0]               data;
        logic [7:0]                strb;
        logic [63:0]               user;
        reqrsp_pkg::size_t         size;
    } spatz_dreq_chan_t;
    typedef struct packed {
        spatz_dreq_chan_t q;
        logic             q_valid;
        logic             p_ready;
    } spatz_dreq_t;
    typedef struct packed {
        logic [63:0] data;
        logic        error;
    } spatz_drsp_chan_t;
    typedef struct packed {
        spatz_drsp_chan_t p;
        logic             p_valid;
        logic             q_ready;
    } spatz_drsp_t;
    spatz_dreq_t fp_lsu_mem_req;
    spatz_drsp_t fp_lsu_mem_rsp;
    assign fp_lsu_mem_rsp = '0;

    spatz #(
        .NrMemPorts         (SPATZ_MEM_PORTS),
        .NumOutstandingLoads(8),
        .RegisterRsp        (0),
        .dreq_t             (spatz_dreq_t),
        .drsp_t             (spatz_drsp_t),
        .spatz_mem_req_t    (spatz_tcdm_req_chan_t),
        .spatz_mem_rsp_t    (spatz_tcdm_rsp_chan_t),
        .spatz_issue_req_t  (spatz_issue_req_t),
        .spatz_issue_rsp_t  (spatz_issue_rsp_t),
        .spatz_rsp_t        (spatz_rsp_t)
    ) u_spatz (
        .clk_i                   (clk_i),
        .rst_ni                  (rst_ni),
        .testmode_i              (1'b0),
        .hart_id_i               (32'd0),
        // Snitch Issue Interface
        .issue_valid_i           (acc_qvalid),
        .issue_ready_o           (acc_qready),
        .issue_req_i             (spatz_issue_req),
        .issue_rsp_o             (spatz_issue_rsp),
        // Snitch Response Interface
        .rsp_valid_o             (acc_pvalid),
        .rsp_ready_i             (acc_pready),
        .rsp_o                   (spatz_rsp),
        // VLSU Memory Port
        .spatz_mem_req_o         (spatz_mem_req),
        .spatz_mem_req_valid_o   (spatz_mem_req_valid),
        .spatz_mem_req_ready_i   (spatz_mem_req_ready),
        .spatz_mem_rsp_i         (spatz_mem_rsp),
        .spatz_mem_rsp_valid_i   (spatz_mem_rsp_valid),
        .spatz_mem_finished_o    (acc_mem_finished),
        .spatz_mem_str_finished_o(acc_mem_str_finished),
        // FP LSU (tied off)
        .fp_lsu_mem_req_o        (fp_lsu_mem_req),
        .fp_lsu_mem_rsp_i        (fp_lsu_mem_rsp),
        // FPU side-channel
        .fpu_rnd_mode_i          (fpnew_pkg::roundmode_e'(fpu_rnd_mode)),
        .fpu_fmt_mode_i          (fpnew_pkg::fmt_mode_t'(fpu_fmt_mode)),
        .fpu_status_o            (fpu_status)
    );

    // Wire Spatz response back to Snitch
    assign acc_pid    = spatz_rsp.id;
    assign acc_pdata  = spatz_rsp.data;
    assign acc_perror = spatz_rsp.error;

    // TCDM-to-OBI Bridges for Spatz VLSU → Masters 1 and 8
    logic [SPATZ_MEM_PORTS-1:0]                     spatz_obi_req;
    logic [SPATZ_MEM_PORTS-1:0]                     spatz_obi_gnt;
    logic [SPATZ_MEM_PORTS-1:0][OBI_ADDR_WIDTH-1:0] spatz_obi_addr;
    logic [SPATZ_MEM_PORTS-1:0]                     spatz_obi_we;
    logic [SPATZ_MEM_PORTS-1:0][31:0]               spatz_obi_be;
    logic [SPATZ_MEM_PORTS-1:0][255:0]              spatz_obi_wdata;
    logic [SPATZ_MEM_PORTS-1:0]                     spatz_obi_rvalid;
    logic [SPATZ_MEM_PORTS-1:0][255:0]              spatz_obi_rdata;

    for (genvar p = 0; p < SPATZ_MEM_PORTS; p++) begin : gen_spatz_tcdm_bridge
        tcdm_to_obi_bridge #(
            .ADDR_WIDTH(OBI_ADDR_WIDTH),
            .DATA_WIDTH(32)  // Spatz ELEN=32 (INT-only)
        ) u_spatz_tcdm_bridge (
            .clk_i             (clk_i),
            .rst_ni            (rst_ni),
            .tcdm_req_addr_i   (spatz_mem_req[p].addr),
            .tcdm_req_write_i  (spatz_mem_req[p].write),
            .tcdm_req_data_i   (spatz_mem_req[p].data),
            .tcdm_req_strb_i   (spatz_mem_req[p].strb),
            .tcdm_req_valid_i  (spatz_mem_req_valid[p]),
            .tcdm_req_ready_o  (spatz_mem_req_ready[p]),
            .tcdm_rsp_data_o   (spatz_mem_rsp[p].data),
            .tcdm_rsp_valid_o  (spatz_mem_rsp_valid[p]),
            .obi_req_o         (spatz_obi_req[p]),
            .obi_gnt_i         (spatz_obi_gnt[p]),
            .obi_addr_o        (spatz_obi_addr[p]),
            .obi_we_o          (spatz_obi_we[p]),
            .obi_be_o          (spatz_obi_be[p]),
            .obi_wdata_o       (spatz_obi_wdata[p]),
            .obi_rvalid_i      (spatz_obi_rvalid[p]),
            .obi_rdata_i       (spatz_obi_rdata[p])
        );
    end

    // Master 1: Spatz VLSU port 0 via TCDM-to-OBI Bridge
    assign master_req[1].req   = spatz_obi_req[0];
    assign master_req[1].we    = spatz_obi_we[0];
    assign master_req[1].be    = spatz_obi_be[0];
    assign master_req[1].addr  = spatz_obi_addr[0];
    assign master_req[1].wdata = spatz_obi_wdata[0];

    assign spatz_obi_gnt[0]    = master_rsp[1].gnt;
    assign spatz_obi_rvalid[0] = master_rsp[1].rvalid;
    assign spatz_obi_rdata[0]  = master_rsp[1].rdata;

    // Master 8: Spatz VLSU port 1 via TCDM-to-OBI Bridge
    assign master_req[8].req   = spatz_obi_req[1];
    assign master_req[8].we    = spatz_obi_we[1];
    assign master_req[8].be    = spatz_obi_be[1];
    assign master_req[8].addr  = spatz_obi_addr[1];
    assign master_req[8].wdata = spatz_obi_wdata[1];

    assign spatz_obi_gnt[1]    = master_rsp[8].gnt;
    assign spatz_obi_rvalid[1] = master_rsp[8].rvalid;
    assign spatz_obi_rdata[1]  = master_rsp[8].rdata;

    // Masters 2/9: PULP iDMA MMIO frontend + AXI/OBI backends
    logic                      idma_obi_read_req;
    logic                      idma_obi_read_gnt;
    logic [OBI_ADDR_WIDTH-1:0] idma_obi_read_addr;
    logic                      idma_obi_read_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] idma_obi_read_be;
    logic [OBI_DATA_WIDTH-1:0] idma_obi_read_wdata;
    logic                      idma_obi_read_rvalid;
    logic [OBI_DATA_WIDTH-1:0] idma_obi_read_rdata;

    logic                      idma_obi_write_req;
    logic                      idma_obi_write_gnt;
    logic [OBI_ADDR_WIDTH-1:0] idma_obi_write_addr;
    logic                      idma_obi_write_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] idma_obi_write_be;
    logic [OBI_DATA_WIDTH-1:0] idma_obi_write_wdata;
    logic                      idma_obi_write_rvalid;
    logic [OBI_DATA_WIDTH-1:0] idma_obi_write_rdata;

    logic idma_irq_a2o_busy;
    logic idma_irq_a2o_start;
    logic idma_irq_a2o_done;
    logic idma_irq_a2o_error;
    logic idma_irq_o2a_busy;
    logic idma_irq_o2a_start;
    logic idma_irq_o2a_done;
    logic idma_irq_o2a_error;
    logic [31:0] idma_a2o_queue_usage;
    logic [31:0] idma_o2a_queue_usage;

    assign cfg_dma_done = idma_irq_a2o_done | idma_irq_o2a_done;

    npu_pulp_idma_ctrl_mm #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .CFG_DATA_WIDTH(MMIO_DATA_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .BASE_ADDR (32'h2000_1000)
    ) u_idma_ctrl_mm (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .req_i              (idma_mm_req),
        .gnt_o              (idma_mm_gnt),
        .addr_i             (idma_mm_addr),
        .we_i               (idma_mm_we),
        .be_i               (idma_mm_be),
        .wdata_i            (idma_mm_wdata),
        .rvalid_o           (idma_mm_rvalid),
        .rdata_o            (idma_mm_rdata),

        .axi_aw_addr_o      (axi_aw_addr_o),
        .axi_aw_len_o       (axi_aw_len_o),
        .axi_aw_size_o      (axi_aw_size_o),
        .axi_aw_burst_o     (axi_aw_burst_o),
        .axi_aw_valid_o     (axi_aw_valid_o),
        .axi_aw_ready_i     (axi_aw_ready_i),
        .axi_w_data_o       (axi_w_data_o),
        .axi_w_strb_o       (axi_w_strb_o),
        .axi_w_last_o       (axi_w_last_o),
        .axi_w_valid_o      (axi_w_valid_o),
        .axi_w_ready_i      (axi_w_ready_i),
        .axi_b_resp_i       (axi_b_resp_i),
        .axi_b_valid_i      (axi_b_valid_i),
        .axi_b_ready_o      (axi_b_ready_o),
        .axi_ar_addr_o      (axi_ar_addr_o),
        .axi_ar_len_o       (axi_ar_len_o),
        .axi_ar_size_o      (axi_ar_size_o),
        .axi_ar_burst_o     (axi_ar_burst_o),
        .axi_ar_valid_o     (axi_ar_valid_o),
        .axi_ar_ready_i     (axi_ar_ready_i),
        .axi_r_data_i       (axi_r_data_i),
        .axi_r_resp_i       (axi_r_resp_i),
        .axi_r_last_i       (axi_r_last_i),
        .axi_r_valid_i      (axi_r_valid_i),
        .axi_r_ready_o      (axi_r_ready_o),

        .obi_read_req_o     (idma_obi_read_req),
        .obi_read_gnt_i     (idma_obi_read_gnt),
        .obi_read_addr_o    (idma_obi_read_addr),
        .obi_read_we_o      (idma_obi_read_we),
        .obi_read_be_o      (idma_obi_read_be),
        .obi_read_wdata_o   (idma_obi_read_wdata),
        .obi_read_rvalid_i  (idma_obi_read_rvalid),
        .obi_read_rdata_i   (idma_obi_read_rdata),

        .obi_write_req_o    (idma_obi_write_req),
        .obi_write_gnt_i    (idma_obi_write_gnt),
        .obi_write_addr_o   (idma_obi_write_addr),
        .obi_write_we_o     (idma_obi_write_we),
        .obi_write_be_o     (idma_obi_write_be),
        .obi_write_wdata_o  (idma_obi_write_wdata),
        .obi_write_rvalid_i (idma_obi_write_rvalid),
        .obi_write_rdata_i  (idma_obi_write_rdata),

        .irq_a2o_busy_o     (idma_irq_a2o_busy),
        .irq_a2o_start_o    (idma_irq_a2o_start),
        .irq_a2o_done_o     (idma_irq_a2o_done),
        .irq_a2o_error_o    (idma_irq_a2o_error),
        .irq_o2a_busy_o     (idma_irq_o2a_busy),
        .irq_o2a_start_o    (idma_irq_o2a_start),
        .irq_o2a_done_o     (idma_irq_o2a_done),
        .irq_o2a_error_o    (idma_irq_o2a_error),
        .perf_a2o_queue_usage_o(idma_a2o_queue_usage),
        .perf_o2a_queue_usage_o(idma_o2a_queue_usage)
    );

    assign master_req[2].req   = idma_obi_write_req;
    assign master_req[2].we    = idma_obi_write_we;
    assign master_req[2].be    = idma_obi_write_be;
    assign master_req[2].addr  = idma_obi_write_addr;
    assign master_req[2].wdata = idma_obi_write_wdata;

    assign idma_obi_write_gnt    = master_rsp[2].gnt;
    assign idma_obi_write_rvalid = master_rsp[2].rvalid;
    assign idma_obi_write_rdata  = master_rsp[2].rdata;

    assign master_req[9].req   = idma_obi_read_req;
    assign master_req[9].we    = idma_obi_read_we;
    assign master_req[9].be    = idma_obi_read_be;
    assign master_req[9].addr  = idma_obi_read_addr;
    assign master_req[9].wdata = idma_obi_read_wdata;

    assign idma_obi_read_gnt    = master_rsp[9].gnt;
    assign idma_obi_read_rvalid = master_rsp[9].rvalid;
    assign idma_obi_read_rdata  = master_rsp[9].rdata;

    assign master_req[10].req   = afu_obi_req;
    assign master_req[10].we    = afu_obi_we;
    assign master_req[10].be    = afu_obi_be;
    assign master_req[10].addr  = afu_obi_addr;
    assign master_req[10].wdata = afu_obi_wdata;

    assign afu_obi_gnt    = master_rsp[10].gnt;
    assign afu_obi_rvalid = master_rsp[10].rvalid;
    assign afu_obi_rdata  = master_rsp[10].rdata;

    assign master_req[12].req   = afu_rhs_obi_req;
    assign master_req[12].we    = afu_rhs_obi_we;
    assign master_req[12].be    = afu_rhs_obi_be;
    assign master_req[12].addr  = afu_rhs_obi_addr;
    assign master_req[12].wdata = afu_rhs_obi_wdata;

    assign afu_rhs_obi_gnt    = master_rsp[12].gnt;
    assign afu_rhs_obi_rvalid = master_rsp[12].rvalid;
    assign afu_rhs_obi_rdata  = master_rsp[12].rdata;

    //=========================================================
    // 7. Systolic Array (Matrix Engine)
    //=========================================================
    // Systolic controller PMU pulses. The array is instantiated inside the
    // controller so cluster top only owns the controller instance.
    logic                      sys_weight_load_en;
    logic                      sys_compute_en;
    logic                      sys_ofm_valid;
    logic                      sys_ofm_ready;
    logic                      sys_perf_start;
    logic                      sys_linebuf_busy;
    logic                      sys_linebuf_prefetch_busy;
    logic                      sys_binary_busy;

    // Systolic Controller OBI signals
    logic                      sys_obi_i_req;
    logic                      sys_obi_i_gnt;
    logic [OBI_ADDR_WIDTH-1:0] sys_obi_i_addr;
    logic                      sys_obi_i_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] sys_obi_i_be;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_i_wdata;
    logic                      sys_obi_i_rvalid;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_i_rdata;
    logic                      sys_obi_w_req;
    logic                      sys_obi_w_gnt;
    logic [OBI_ADDR_WIDTH-1:0] sys_obi_w_addr;
    logic                      sys_obi_w_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] sys_obi_w_be;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_w_wdata;
    logic                      sys_obi_w_rvalid;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_w_rdata;
    logic                      sys_obi_b_req;
    logic                      sys_obi_b_gnt;
    logic [OBI_ADDR_WIDTH-1:0] sys_obi_b_addr;
    logic                      sys_obi_b_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] sys_obi_b_be;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_b_wdata;
    logic                      sys_obi_b_rvalid;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_b_rdata;

    logic [3:0]                      sys_obi_o_req;
    logic [3:0]                      sys_obi_o_tcdm_req;
    logic [3:0]                      sys_obi_o_gnt;
    logic [3:0][OBI_ADDR_WIDTH-1:0]  sys_obi_o_addr;
    logic [3:0]                      sys_obi_o_we;
    logic [3:0][(OBI_DATA_WIDTH/8)-1:0] sys_obi_o_be;
    logic [3:0][OBI_DATA_WIDTH-1:0]  sys_obi_o_wdata;
    logic [3:0]                      sys_obi_o_rvalid;
    logic [3:0][OBI_DATA_WIDTH-1:0]  sys_obi_o_rdata;
    logic                            sys_otcdm_stall_active;
    logic [31:0]                     sys_otcdm_stall_ctr_q;
    logic [2:0]                      sys_debug_state;
    logic [1:0]                      sys_debug_drain_state;
    logic [4:0]                      sys_debug_linebuf_state;
    logic [1:0]                      sys_debug_linebuf_fetch_main_state;
    logic [2:0]                      sys_debug_linebuf_fetch_background_state;
    logic [2:0]                      sys_debug_linebuf_bypass_state;

    assign debug_sys_state_o = sys_debug_state;
    assign debug_sys_drain_state_o = sys_debug_drain_state;
    assign debug_linebuf_state_o = sys_debug_linebuf_state;
    assign debug_linebuf_fetch_main_state_o = sys_debug_linebuf_fetch_main_state;
    assign debug_linebuf_fetch_background_state_o = sys_debug_linebuf_fetch_background_state;
    assign debug_linebuf_bypass_state_o = sys_debug_linebuf_bypass_state;

    systolic_controller #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .CFG_DATA_WIDTH(MMIO_DATA_WIDTH),
        .ARRAY_DIM(32),
        .INPUT_ELEM_WIDTH(8),
        .OFM_ELEM_WIDTH(32),
        .INPUT_FIFO_DEPTH(4),
        .OFM_FIFO_DEPTH(SYSTOLIC_OFM_FIFO_DEPTH)
    ) u_sys_ctrl (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),

        .ctrl_req_i         (systolic_ctrl_req),
        .ctrl_gnt_o         (systolic_ctrl_gnt),
        .ctrl_addr_i        (ctrl_addr),
        .ctrl_we_i          (ctrl_we),
        .ctrl_be_i          (ctrl_be),
        .ctrl_wdata_i       (ctrl_wdata),
        .ctrl_rvalid_o      (systolic_ctrl_rvalid),
        .ctrl_rdata_o       (systolic_ctrl_rdata),
        .cfg_sys_done_o     (cfg_sys_done),

        .obi_i_req_o        (sys_obi_i_req),
        .obi_i_gnt_i        (sys_obi_i_gnt),
        .obi_i_addr_o       (sys_obi_i_addr),
        .obi_i_we_o         (sys_obi_i_we),
        .obi_i_be_o         (sys_obi_i_be),
        .obi_i_wdata_o      (sys_obi_i_wdata),
        .obi_i_rvalid_i     (sys_obi_i_rvalid),
        .obi_i_rdata_i      (sys_obi_i_rdata),

        .obi_w_req_o        (sys_obi_w_req),
        .obi_w_gnt_i        (sys_obi_w_gnt),
        .obi_w_addr_o       (sys_obi_w_addr),
        .obi_w_we_o         (sys_obi_w_we),
        .obi_w_be_o         (sys_obi_w_be),
        .obi_w_wdata_o      (sys_obi_w_wdata),
        .obi_w_rvalid_i     (sys_obi_w_rvalid),
        .obi_w_rdata_i      (sys_obi_w_rdata),

        .obi_b_req_o        (sys_obi_b_req),
        .obi_b_gnt_i        (sys_obi_b_gnt),
        .obi_b_addr_o       (sys_obi_b_addr),
        .obi_b_we_o         (sys_obi_b_we),
        .obi_b_be_o         (sys_obi_b_be),
        .obi_b_wdata_o      (sys_obi_b_wdata),
        .obi_b_rvalid_i     (sys_obi_b_rvalid),
        .obi_b_rdata_i      (sys_obi_b_rdata),

        .obi_o_req_o        (sys_obi_o_req),
        .obi_o_gnt_i        (sys_obi_o_gnt),
        .obi_o_addr_o       (sys_obi_o_addr),
        .obi_o_we_o         (sys_obi_o_we),
        .obi_o_be_o         (sys_obi_o_be),
        .obi_o_wdata_o      (sys_obi_o_wdata),
        .obi_o_rvalid_i     (sys_obi_o_rvalid),
        .obi_o_rdata_i      (sys_obi_o_rdata),

        .perf_weight_load_en_o(sys_weight_load_en),
        .perf_compute_en_o  (sys_compute_en),
        .perf_ofm_valid_o   (sys_ofm_valid),
        .perf_ofm_ready_o   (sys_ofm_ready),
        .perf_start_o       (sys_perf_start),
        .perf_linebuf_busy_o(sys_linebuf_busy),
        .perf_linebuf_prefetch_busy_o(sys_linebuf_prefetch_busy),
        .perf_binary_busy_o (sys_binary_busy),
        .debug_state_o      (sys_debug_state),
        .debug_drain_state_o(sys_debug_drain_state),
        .debug_linebuf_state_o(sys_debug_linebuf_state),
        .debug_linebuf_fetch_main_state_o(sys_debug_linebuf_fetch_main_state),
        .debug_linebuf_fetch_background_state_o(sys_debug_linebuf_fetch_background_state),
        .debug_linebuf_bypass_state_o(sys_debug_linebuf_bypass_state)
    );

    // Master 3: Systolic Controller IFM/linebuffer read port (I-TCDM)
    assign master_req[3].req   = sys_obi_i_req;
    assign master_req[3].we    = sys_obi_i_we;
    assign master_req[3].be    = sys_obi_i_be;
    assign master_req[3].addr  = sys_obi_i_addr;
    assign master_req[3].wdata = sys_obi_i_wdata;

    assign sys_obi_i_gnt    = master_rsp[3].gnt;
    assign sys_obi_i_rvalid = master_rsp[3].rvalid;
    assign sys_obi_i_rdata  = master_rsp[3].rdata;

    // Master 11: Systolic Controller weight read port (I-TCDM)
    assign master_req[11].req   = sys_obi_w_req;
    assign master_req[11].we    = sys_obi_w_we;
    assign master_req[11].be    = sys_obi_w_be;
    assign master_req[11].addr  = sys_obi_w_addr;
    assign master_req[11].wdata = sys_obi_w_wdata;

    assign sys_obi_w_gnt    = master_rsp[11].gnt;
    assign sys_obi_w_rvalid = master_rsp[11].rvalid;
    assign sys_obi_w_rdata  = master_rsp[11].rdata;

    // Master 13: Systolic Controller binary RHS read port (I-TCDM)
    assign master_req[13].req   = sys_obi_b_req;
    assign master_req[13].we    = sys_obi_b_we;
    assign master_req[13].be    = sys_obi_b_be;
    assign master_req[13].addr  = sys_obi_b_addr;
    assign master_req[13].wdata = sys_obi_b_wdata;

    assign sys_obi_b_gnt    = master_rsp[13].gnt;
    assign sys_obi_b_rvalid = master_rsp[13].rvalid;
    assign sys_obi_b_rdata  = master_rsp[13].rdata;

    if (SYSTOLIC_OTCDM_STALL_PERIOD == 0 || SYSTOLIC_OTCDM_STALL_HOLD == 0) begin : gen_no_sys_otcdm_stall
        assign sys_otcdm_stall_active = 1'b0;
        assign sys_otcdm_stall_ctr_q = '0;
    end else begin : gen_sys_otcdm_stall
        always_ff @(posedge clk_i or negedge rst_ni) begin
            if (!rst_ni) begin
                sys_otcdm_stall_ctr_q <= '0;
            end else if (sys_otcdm_stall_ctr_q == (SYSTOLIC_OTCDM_STALL_PERIOD - 1)) begin
                sys_otcdm_stall_ctr_q <= '0;
            end else begin
                sys_otcdm_stall_ctr_q <= sys_otcdm_stall_ctr_q + 32'd1;
            end
        end

        assign sys_otcdm_stall_active = sys_otcdm_stall_ctr_q < SYSTOLIC_OTCDM_STALL_HOLD;
    end

    assign sys_obi_o_tcdm_req = sys_obi_o_req & {4{!sys_otcdm_stall_active}};

    // Masters 4-7: Systolic Controller Write Ports (O-TCDM)
    for (genvar i = 0; i < 4; i++) begin : gen_sys_obi_o
        assign master_req[4+i].req   = sys_obi_o_tcdm_req[i];
        assign master_req[4+i].we    = sys_obi_o_we[i];
        assign master_req[4+i].be    = sys_obi_o_be[i];
        assign master_req[4+i].addr  = sys_obi_o_addr[i];
        assign master_req[4+i].wdata = sys_obi_o_wdata[i];

        assign sys_obi_o_gnt[i]    = master_rsp[4+i].gnt;
        assign sys_obi_o_rvalid[i] = master_rsp[4+i].rvalid;
        assign sys_obi_o_rdata[i]  = master_rsp[4+i].rdata;
    end

    //=========================================================
    // 8. Performance Management Unit
    //=========================================================
    localparam int unsigned PMU_NUM_COUNTERS = 163;
    localparam int unsigned PMU_INC_WIDTH = 32;
    localparam int unsigned PMU_DMA_TAG_DEPTH = 16;
    localparam logic [PMU_NUM_COUNTERS-1:0] PMU_MAX_COUNTER_MASK =
        (PMU_NUM_COUNTERS'(1) << 53) | (PMU_NUM_COUNTERS'(1) << 54) |
        (PMU_NUM_COUNTERS'(1) << 68) | (PMU_NUM_COUNTERS'(1) << 69) |
        (PMU_NUM_COUNTERS'(1) << 72) | (PMU_NUM_COUNTERS'(1) << 73);

    logic [PMU_NUM_COUNTERS-1:0][PMU_INC_WIDTH-1:0] pmu_event_inc;
    logic pmu_filter_enable;
    logic [31:0] pmu_filter_context;
    logic [NUM_MASTERS-1:0] pmu_master_selected;
    logic pmu_current_selected;
    logic pmu_sys_selected;
    logic pmu_afu_selected;
    logic pmu_spatz_selected;
    logic pmu_dma_a2o_selected;
    logic pmu_dma_o2a_selected;
    logic pmu_any_selected_active;
    logic pmu_sys_state_scope;

    logic [31:0] pmu_sys_tag_q;
    logic [31:0] pmu_afu_tag_q;
    logic [31:0] pmu_spatz_tag_q;
    logic pmu_sys_job_active_q;
    logic pmu_sys_tag_valid_q;
    logic pmu_afu_tag_valid_q;
    logic pmu_spatz_tag_valid_q;
    logic afu_done_q;

    logic [31:0] pmu_dma_a2o_tags_q [PMU_DMA_TAG_DEPTH];
    logic [31:0] pmu_dma_o2a_tags_q [PMU_DMA_TAG_DEPTH];
    logic pmu_dma_a2o_tag_valid_q [PMU_DMA_TAG_DEPTH];
    logic pmu_dma_o2a_tag_valid_q [PMU_DMA_TAG_DEPTH];
    logic [3:0] pmu_dma_a2o_wr_q, pmu_dma_a2o_rd_q;
    logic [3:0] pmu_dma_o2a_wr_q, pmu_dma_o2a_rd_q;
    logic [4:0] pmu_dma_a2o_tag_count_q, pmu_dma_o2a_tag_count_q;
    logic [31:0] pmu_dma_a2o_tag, pmu_dma_o2a_tag;

    logic [63:0] pmu_cycle_q;
    logic [63:0] pmu_axi_read_time_q [16];
    logic [63:0] pmu_axi_write_time_q [16];
    logic [3:0] pmu_axi_read_wr_q, pmu_axi_read_rd_q;
    logic [3:0] pmu_axi_write_wr_q, pmu_axi_write_rd_q;
    logic [4:0] pmu_axi_read_outstanding_q, pmu_axi_write_outstanding_q;
    logic [31:0] pmu_axi_read_latency;
    logic [31:0] pmu_axi_write_latency;

    function automatic logic pmu_tag_selected(input logic [31:0] tag, input logic valid);
        pmu_tag_selected = !pmu_filter_enable ||
                           (valid && (tag == pmu_filter_context));
    endfunction

    assign pmu_dma_a2o_tag = pmu_dma_a2o_tags_q[pmu_dma_a2o_rd_q];
    assign pmu_dma_o2a_tag = pmu_dma_o2a_tags_q[pmu_dma_o2a_rd_q];
    assign pmu_current_selected = pmu_tag_selected(pmu_context_id, pmu_context_active);
    assign pmu_sys_selected = pmu_tag_selected(pmu_sys_tag_q, pmu_sys_tag_valid_q);
    assign pmu_afu_selected = pmu_tag_selected(pmu_afu_tag_q, pmu_afu_tag_valid_q);
    assign pmu_spatz_selected = pmu_tag_selected(pmu_spatz_tag_q, pmu_spatz_tag_valid_q);
    assign pmu_dma_a2o_selected = pmu_tag_selected(
        pmu_dma_a2o_tag, (pmu_dma_a2o_tag_count_q != 0) &&
                          pmu_dma_a2o_tag_valid_q[pmu_dma_a2o_rd_q]);
    assign pmu_dma_o2a_selected = pmu_tag_selected(
        pmu_dma_o2a_tag, (pmu_dma_o2a_tag_count_q != 0) &&
                         pmu_dma_o2a_tag_valid_q[pmu_dma_o2a_rd_q]);
    assign pmu_sys_state_scope = pmu_sys_job_active_q && pmu_sys_selected;
    assign pmu_any_selected_active = pmu_current_selected ||
        (idma_irq_a2o_busy && pmu_dma_a2o_selected) ||
        (idma_irq_o2a_busy && pmu_dma_o2a_selected) ||
        (sys_debug_state != 3'd0 && pmu_sys_selected) ||
        (afu_perf_active && pmu_afu_selected) || pmu_spatz_selected;
    assign pmu_axi_read_latency = 32'(pmu_cycle_q - pmu_axi_read_time_q[pmu_axi_read_rd_q]);
    assign pmu_axi_write_latency = 32'(pmu_cycle_q - pmu_axi_write_time_q[pmu_axi_write_rd_q]);

    always_comb begin
        pmu_master_selected = '0;
        pmu_master_selected[0] = pmu_current_selected;
        pmu_master_selected[1] = pmu_spatz_selected;
        pmu_master_selected[8] = pmu_spatz_selected;
        pmu_master_selected[2] = pmu_dma_a2o_selected;
        pmu_master_selected[9] = pmu_dma_o2a_selected;
        pmu_master_selected[3] = pmu_sys_selected;
        pmu_master_selected[4] = pmu_sys_selected;
        pmu_master_selected[5] = pmu_sys_selected;
        pmu_master_selected[6] = pmu_sys_selected;
        pmu_master_selected[7] = pmu_sys_selected;
        pmu_master_selected[11] = pmu_sys_selected;
        pmu_master_selected[13] = pmu_sys_selected;
        pmu_master_selected[10] = pmu_afu_selected;
        pmu_master_selected[12] = pmu_afu_selected;
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            pmu_cycle_q <= '0;
            pmu_sys_tag_q <= '0;
            pmu_sys_job_active_q <= 1'b0;
            pmu_afu_tag_q <= '0;
            pmu_spatz_tag_q <= '0;
            pmu_sys_tag_valid_q <= 1'b0;
            pmu_afu_tag_valid_q <= 1'b0;
            pmu_spatz_tag_valid_q <= 1'b0;
            afu_done_q <= 1'b0;
            pmu_dma_a2o_wr_q <= '0;
            pmu_dma_a2o_rd_q <= '0;
            pmu_dma_o2a_wr_q <= '0;
            pmu_dma_o2a_rd_q <= '0;
            pmu_dma_a2o_tag_count_q <= '0;
            pmu_dma_o2a_tag_count_q <= '0;
            pmu_axi_read_wr_q <= '0;
            pmu_axi_read_rd_q <= '0;
            pmu_axi_write_wr_q <= '0;
            pmu_axi_write_rd_q <= '0;
            pmu_axi_read_outstanding_q <= '0;
            pmu_axi_write_outstanding_q <= '0;
            for (int idx = 0; idx < PMU_DMA_TAG_DEPTH; idx++) begin
                pmu_dma_a2o_tags_q[idx] <= '0;
                pmu_dma_o2a_tags_q[idx] <= '0;
                pmu_dma_a2o_tag_valid_q[idx] <= 1'b0;
                pmu_dma_o2a_tag_valid_q[idx] <= 1'b0;
                pmu_axi_read_time_q[idx] <= '0;
                pmu_axi_write_time_q[idx] <= '0;
            end
        end else begin
            pmu_cycle_q <= pmu_cycle_q + 64'd1;
            afu_done_q <= afu_done;

            if (sys_perf_start) begin
                pmu_sys_tag_q <= pmu_context_id;
                pmu_sys_tag_valid_q <= pmu_context_active;
                pmu_sys_job_active_q <= 1'b1;
            end else if (cfg_sys_done) begin
                pmu_sys_tag_valid_q <= 1'b0;
                pmu_sys_job_active_q <= 1'b0;
            end
            if (afu_perf_start) begin
                pmu_afu_tag_q <= pmu_context_id;
                pmu_afu_tag_valid_q <= pmu_context_active;
            end else if (afu_done && !afu_done_q) begin
                pmu_afu_tag_valid_q <= 1'b0;
            end
            if (acc_qvalid && acc_qready) begin
                pmu_spatz_tag_q <= pmu_context_id;
                pmu_spatz_tag_valid_q <= pmu_context_active;
            end else if (acc_pvalid && acc_pready) begin
                pmu_spatz_tag_valid_q <= 1'b0;
            end

            if (idma_irq_a2o_start) begin
                pmu_dma_a2o_tags_q[pmu_dma_a2o_wr_q] <= pmu_context_id;
                pmu_dma_a2o_tag_valid_q[pmu_dma_a2o_wr_q] <= pmu_context_active;
                pmu_dma_a2o_wr_q <= pmu_dma_a2o_wr_q + 4'd1;
            end
            if (idma_irq_a2o_done) pmu_dma_a2o_rd_q <= pmu_dma_a2o_rd_q + 4'd1;
            unique case ({idma_irq_a2o_start, idma_irq_a2o_done})
                2'b10: pmu_dma_a2o_tag_count_q <= pmu_dma_a2o_tag_count_q + 5'd1;
                2'b01: pmu_dma_a2o_tag_count_q <= pmu_dma_a2o_tag_count_q - 5'd1;
                default: begin end
            endcase
            if (idma_irq_o2a_start) begin
                pmu_dma_o2a_tags_q[pmu_dma_o2a_wr_q] <= pmu_context_id;
                pmu_dma_o2a_tag_valid_q[pmu_dma_o2a_wr_q] <= pmu_context_active;
                pmu_dma_o2a_wr_q <= pmu_dma_o2a_wr_q + 4'd1;
            end
            if (idma_irq_o2a_done) pmu_dma_o2a_rd_q <= pmu_dma_o2a_rd_q + 4'd1;
            unique case ({idma_irq_o2a_start, idma_irq_o2a_done})
                2'b10: pmu_dma_o2a_tag_count_q <= pmu_dma_o2a_tag_count_q + 5'd1;
                2'b01: pmu_dma_o2a_tag_count_q <= pmu_dma_o2a_tag_count_q - 5'd1;
                default: begin end
            endcase

            if (axi_ar_valid_o && axi_ar_ready_i) begin
                pmu_axi_read_time_q[pmu_axi_read_wr_q] <= pmu_cycle_q;
                pmu_axi_read_wr_q <= pmu_axi_read_wr_q + 4'd1;
            end
            if (axi_r_valid_i && axi_r_ready_o && axi_r_last_i)
                pmu_axi_read_rd_q <= pmu_axi_read_rd_q + 4'd1;
            unique case ({axi_ar_valid_o && axi_ar_ready_i,
                          axi_r_valid_i && axi_r_ready_o && axi_r_last_i})
                2'b10: pmu_axi_read_outstanding_q <= pmu_axi_read_outstanding_q + 5'd1;
                2'b01: pmu_axi_read_outstanding_q <= pmu_axi_read_outstanding_q - 5'd1;
                default: begin end
            endcase
            if (axi_aw_valid_o && axi_aw_ready_i) begin
                pmu_axi_write_time_q[pmu_axi_write_wr_q] <= pmu_cycle_q;
                pmu_axi_write_wr_q <= pmu_axi_write_wr_q + 4'd1;
            end
            if (axi_b_valid_i && axi_b_ready_o)
                pmu_axi_write_rd_q <= pmu_axi_write_rd_q + 4'd1;
            unique case ({axi_aw_valid_o && axi_aw_ready_i,
                          axi_b_valid_i && axi_b_ready_o})
                2'b10: pmu_axi_write_outstanding_q <= pmu_axi_write_outstanding_q + 5'd1;
                2'b01: pmu_axi_write_outstanding_q <= pmu_axi_write_outstanding_q - 5'd1;
                default: begin end
            endcase
        end
    end

    always_comb begin
        pmu_event_inc = '0;

        // 0-31 retain the original counter ABI for historical comparisons.
        pmu_event_inc[0] = PMU_INC_WIDTH'(!pmu_filter_enable || pmu_any_selected_active);
        pmu_event_inc[1] = PMU_INC_WIDTH'(snitch_core_events.retired_instr && pmu_current_selected);
        pmu_event_inc[2] = PMU_INC_WIDTH'(snitch_core_events.retired_load && pmu_current_selected);
        pmu_event_inc[3] = PMU_INC_WIDTH'(snitch_core_events.retired_i && pmu_current_selected);
        pmu_event_inc[4] = PMU_INC_WIDTH'(snitch_core_events.retired_acc && pmu_current_selected);
        pmu_event_inc[5] = PMU_INC_WIDTH'(master_req[0].req && pmu_current_selected);
        pmu_event_inc[6] = PMU_INC_WIDTH'(master_req[0].req && !master_rsp[0].gnt && pmu_current_selected);
        pmu_event_inc[7] = PMU_INC_WIDTH'(acc_qvalid && acc_qready && pmu_current_selected);
        pmu_event_inc[8] = PMU_INC_WIDTH'(acc_pvalid && acc_pready && pmu_spatz_selected);
        pmu_event_inc[9] = PMU_INC_WIDTH'(master_req[1].req && pmu_spatz_selected) +
                           PMU_INC_WIDTH'(master_req[8].req && pmu_spatz_selected);
        pmu_event_inc[10] = PMU_INC_WIDTH'(master_req[1].req && !master_rsp[1].gnt && pmu_spatz_selected) +
                            PMU_INC_WIDTH'(master_req[8].req && !master_rsp[8].gnt && pmu_spatz_selected);
        pmu_event_inc[11] = PMU_INC_WIDTH'((idma_irq_a2o_busy && pmu_dma_a2o_selected) ||
                                           (idma_irq_o2a_busy && pmu_dma_o2a_selected));
        pmu_event_inc[12] = PMU_INC_WIDTH'(idma_irq_a2o_start && pmu_current_selected) +
                            PMU_INC_WIDTH'(idma_irq_o2a_start && pmu_current_selected);
        pmu_event_inc[13] = PMU_INC_WIDTH'(idma_irq_a2o_done && pmu_dma_a2o_selected) +
                            PMU_INC_WIDTH'(idma_irq_o2a_done && pmu_dma_o2a_selected);
        pmu_event_inc[14] = PMU_INC_WIDTH'(master_req[2].req && pmu_dma_a2o_selected) +
                            PMU_INC_WIDTH'(master_req[9].req && pmu_dma_o2a_selected);
        pmu_event_inc[15] = PMU_INC_WIDTH'(master_req[2].req && !master_rsp[2].gnt && pmu_dma_a2o_selected) +
                            PMU_INC_WIDTH'(master_req[9].req && !master_rsp[9].gnt && pmu_dma_o2a_selected);
        pmu_event_inc[16] = PMU_INC_WIDTH'(afu_done && pmu_afu_selected);
        pmu_event_inc[17] = PMU_INC_WIDTH'(master_req[10].req && pmu_afu_selected) +
                            PMU_INC_WIDTH'(master_req[12].req && pmu_afu_selected);
        pmu_event_inc[18] = PMU_INC_WIDTH'(master_req[10].req && !master_rsp[10].gnt && pmu_afu_selected) +
                            PMU_INC_WIDTH'(master_req[12].req && !master_rsp[12].gnt && pmu_afu_selected);
        pmu_event_inc[19] = PMU_INC_WIDTH'(sys_compute_en && pmu_sys_selected);
        pmu_event_inc[20] = PMU_INC_WIDTH'(sys_weight_load_en && pmu_sys_selected);
        pmu_event_inc[21] = PMU_INC_WIDTH'(sys_ofm_valid && pmu_sys_selected);
        pmu_event_inc[22] = PMU_INC_WIDTH'(master_req[3].req && pmu_sys_selected) +
                            PMU_INC_WIDTH'(master_req[11].req && pmu_sys_selected) +
                            PMU_INC_WIDTH'(master_req[13].req && pmu_sys_selected);
        pmu_event_inc[23] = PMU_INC_WIDTH'(master_req[3].req && !master_rsp[3].gnt && pmu_sys_selected) +
                            PMU_INC_WIDTH'(master_req[11].req && !master_rsp[11].gnt && pmu_sys_selected) +
                            PMU_INC_WIDTH'(master_req[13].req && !master_rsp[13].gnt && pmu_sys_selected);
        for (int port = 0; port < 4; port++) begin
            pmu_event_inc[24] += PMU_INC_WIDTH'(sys_obi_o_req[port] && pmu_sys_selected);
            pmu_event_inc[25] += PMU_INC_WIDTH'(sys_obi_o_req[port] && !sys_obi_o_gnt[port] && pmu_sys_selected);
        end
        for (int mst = 0; mst < NUM_MASTERS; mst++) begin
            pmu_event_inc[26] += PMU_INC_WIDTH'(master_req[mst].req && pmu_master_selected[mst]);
            pmu_event_inc[27] += PMU_INC_WIDTH'(master_req[mst].req && master_rsp[mst].gnt && pmu_master_selected[mst]);
            pmu_event_inc[28] += PMU_INC_WIDTH'(master_req[mst].req && !master_rsp[mst].gnt && pmu_master_selected[mst]);
            pmu_event_inc[30] += PMU_INC_WIDTH'(master_req[mst].req && !master_req[mst].we && pmu_master_selected[mst]);
            pmu_event_inc[31] += PMU_INC_WIDTH'(master_req[mst].req && master_req[mst].we && pmu_master_selected[mst]);
        end
        for (int bank = 0; bank < TCDM_NUM_BANKS; bank++)
            pmu_event_inc[29] += PMU_INC_WIDTH'(slave_req[bank].req);

        // 32-43: firmware command context and coarse runtime phases.
        pmu_event_inc[32] = PMU_INC_WIDTH'(pmu_context_active && pmu_current_selected);
        pmu_event_inc[33] = PMU_INC_WIDTH'(pmu_context_begin && pmu_tag_selected(pmu_context_id, 1'b1));
        pmu_event_inc[34] = PMU_INC_WIDTH'(pmu_context_end && pmu_tag_selected(pmu_context_id, 1'b1));
        for (int phase = 1; phase <= 9; phase++)
            pmu_event_inc[34+phase] = PMU_INC_WIDTH'((pmu_phase == 4'(phase)) &&
                (!pmu_filter_enable || pmu_current_selected));

        // 44-73: split DMA engines, job queues, AXI handshakes and latency.
        pmu_event_inc[44] = PMU_INC_WIDTH'(idma_irq_a2o_busy && pmu_dma_a2o_selected);
        pmu_event_inc[45] = PMU_INC_WIDTH'(idma_irq_o2a_busy && pmu_dma_o2a_selected);
        pmu_event_inc[46] = PMU_INC_WIDTH'(idma_irq_a2o_busy && idma_irq_o2a_busy &&
                                           pmu_dma_a2o_selected && pmu_dma_o2a_selected);
        pmu_event_inc[47] = PMU_INC_WIDTH'(idma_irq_a2o_start && pmu_current_selected);
        pmu_event_inc[48] = PMU_INC_WIDTH'(idma_irq_a2o_done && pmu_dma_a2o_selected);
        pmu_event_inc[49] = PMU_INC_WIDTH'(idma_irq_o2a_start && pmu_current_selected);
        pmu_event_inc[50] = PMU_INC_WIDTH'(idma_irq_o2a_done && pmu_dma_o2a_selected);
        pmu_event_inc[51] = PMU_INC_WIDTH'(pmu_dma_a2o_selected ? idma_a2o_queue_usage : 0);
        pmu_event_inc[52] = PMU_INC_WIDTH'(pmu_dma_o2a_selected ? idma_o2a_queue_usage : 0);
        pmu_event_inc[53] = pmu_event_inc[51];
        pmu_event_inc[54] = pmu_event_inc[52];
        pmu_event_inc[55] = PMU_INC_WIDTH'(axi_ar_valid_o && axi_ar_ready_i && pmu_dma_a2o_selected);
        pmu_event_inc[56] = PMU_INC_WIDTH'(axi_r_valid_i && axi_r_ready_o && pmu_dma_a2o_selected);
        pmu_event_inc[57] = PMU_INC_WIDTH'(axi_aw_valid_o && axi_aw_ready_i && pmu_dma_o2a_selected);
        pmu_event_inc[58] = PMU_INC_WIDTH'(axi_w_valid_o && axi_w_ready_i && pmu_dma_o2a_selected);
        pmu_event_inc[59] = PMU_INC_WIDTH'(axi_b_valid_i && axi_b_ready_o && pmu_dma_o2a_selected);
        pmu_event_inc[60] = PMU_INC_WIDTH'((axi_r_valid_i && axi_r_ready_o && pmu_dma_a2o_selected) ?
                                           (OBI_DATA_WIDTH / 8) : 0);
        pmu_event_inc[61] = PMU_INC_WIDTH'((axi_w_valid_o && axi_w_ready_i && pmu_dma_o2a_selected) ?
                                           $countones(axi_w_strb_o) : 0);
        pmu_event_inc[62] = PMU_INC_WIDTH'(axi_ar_valid_o && !axi_ar_ready_i && pmu_dma_a2o_selected);
        pmu_event_inc[63] = PMU_INC_WIDTH'(axi_r_valid_i && !axi_r_ready_o && pmu_dma_a2o_selected);
        pmu_event_inc[64] = PMU_INC_WIDTH'(axi_aw_valid_o && !axi_aw_ready_i && pmu_dma_o2a_selected);
        pmu_event_inc[65] = PMU_INC_WIDTH'(axi_w_valid_o && !axi_w_ready_i && pmu_dma_o2a_selected);
        pmu_event_inc[66] = PMU_INC_WIDTH'(pmu_dma_a2o_selected ? pmu_axi_read_outstanding_q : 0);
        pmu_event_inc[67] = PMU_INC_WIDTH'(pmu_dma_o2a_selected ? pmu_axi_write_outstanding_q : 0);
        pmu_event_inc[68] = pmu_event_inc[66];
        pmu_event_inc[69] = pmu_event_inc[67];
        pmu_event_inc[70] = PMU_INC_WIDTH'((axi_r_valid_i && axi_r_ready_o && axi_r_last_i &&
                                            pmu_dma_a2o_selected) ? pmu_axi_read_latency : 0);
        pmu_event_inc[71] = PMU_INC_WIDTH'((axi_b_valid_i && axi_b_ready_o && pmu_dma_o2a_selected) ?
                                           pmu_axi_write_latency : 0);
        pmu_event_inc[72] = pmu_event_inc[70];
        pmu_event_inc[73] = pmu_event_inc[71];

        // 74-95: systolic state, useful work and precise TCDM handshakes.
        pmu_event_inc[74] = PMU_INC_WIDTH'((sys_debug_state != 3'd0) && (sys_debug_state != 3'd4) && pmu_sys_selected);
        pmu_event_inc[75] = PMU_INC_WIDTH'((sys_debug_state == 3'd1) && pmu_sys_selected);
        pmu_event_inc[76] = PMU_INC_WIDTH'((sys_debug_state == 3'd2) && pmu_sys_selected);
        pmu_event_inc[77] = PMU_INC_WIDTH'((sys_debug_state == 3'd3) && pmu_sys_selected);
        pmu_event_inc[78] = PMU_INC_WIDTH'((sys_debug_state == 3'd4) && pmu_sys_selected);
        pmu_event_inc[79] = PMU_INC_WIDTH'(sys_compute_en && pmu_sys_selected);
        pmu_event_inc[80] = PMU_INC_WIDTH'(sys_weight_load_en && pmu_sys_selected);
        pmu_event_inc[81] = PMU_INC_WIDTH'(sys_ofm_valid && sys_ofm_ready && pmu_sys_selected);
        pmu_event_inc[82] = PMU_INC_WIDTH'(sys_ofm_valid && !sys_ofm_ready && pmu_sys_selected);
        pmu_event_inc[83] = PMU_INC_WIDTH'(sys_obi_i_req && sys_obi_i_gnt && pmu_sys_selected);
        pmu_event_inc[84] = PMU_INC_WIDTH'(sys_obi_i_req && !sys_obi_i_gnt && pmu_sys_selected);
        pmu_event_inc[85] = PMU_INC_WIDTH'(sys_obi_w_req && sys_obi_w_gnt && pmu_sys_selected);
        pmu_event_inc[86] = PMU_INC_WIDTH'(sys_obi_w_req && !sys_obi_w_gnt && pmu_sys_selected);
        pmu_event_inc[87] = PMU_INC_WIDTH'(sys_obi_b_req && sys_obi_b_gnt && pmu_sys_selected);
        pmu_event_inc[88] = PMU_INC_WIDTH'(sys_obi_b_req && !sys_obi_b_gnt && pmu_sys_selected);
        for (int port = 0; port < 4; port++) begin
            pmu_event_inc[89] += PMU_INC_WIDTH'(sys_obi_o_req[port] && sys_obi_o_gnt[port] && pmu_sys_selected);
            pmu_event_inc[90] += PMU_INC_WIDTH'(sys_obi_o_req[port] && !sys_obi_o_gnt[port] && pmu_sys_selected);
        end
        pmu_event_inc[91] = PMU_INC_WIDTH'(sys_linebuf_busy && pmu_sys_selected);
        pmu_event_inc[92] = PMU_INC_WIDTH'(sys_linebuf_prefetch_busy && pmu_sys_selected);
        pmu_event_inc[93] = PMU_INC_WIDTH'(sys_binary_busy && pmu_sys_selected);
        pmu_event_inc[94] = PMU_INC_WIDTH'(sys_perf_start && pmu_current_selected);
        pmu_event_inc[95] = PMU_INC_WIDTH'(cfg_sys_done && pmu_sys_selected);

        // 96-111: AFU active/state/stall breakdown and accepted core beats.
        pmu_event_inc[96] = PMU_INC_WIDTH'(afu_perf_active && pmu_afu_selected);
        pmu_event_inc[97] = PMU_INC_WIDTH'(afu_perf_start && pmu_current_selected);
        pmu_event_inc[98] = PMU_INC_WIDTH'(afu_done && !afu_done_q && pmu_afu_selected);
        pmu_event_inc[99] = PMU_INC_WIDTH'((afu_perf_state == 5'd1) && pmu_afu_selected);
        pmu_event_inc[100] = PMU_INC_WIDTH'(((afu_perf_state == 5'd2) || (afu_perf_state == 5'd3)) && pmu_afu_selected);
        pmu_event_inc[101] = PMU_INC_WIDTH'((afu_perf_state >= 5'd4) && (afu_perf_state <= 5'd10) && pmu_afu_selected);
        pmu_event_inc[102] = PMU_INC_WIDTH'((afu_perf_state >= 5'd11) && (afu_perf_state <= 5'd13) && pmu_afu_selected);
        pmu_event_inc[103] = PMU_INC_WIDTH'((afu_perf_state >= 5'd14) && (afu_perf_state <= 5'd19) && pmu_afu_selected);
        pmu_event_inc[104] = PMU_INC_WIDTH'((afu_perf_state == 5'd20) && pmu_afu_selected);
        pmu_event_inc[105] = PMU_INC_WIDTH'(afu_perf_active && (afu_perf_state == 5'd0) && pmu_afu_selected);
        pmu_event_inc[106] = PMU_INC_WIDTH'(afu_perf_lhs_consume && pmu_afu_selected);
        pmu_event_inc[107] = PMU_INC_WIDTH'(afu_perf_rhs_consume && pmu_afu_selected);
        pmu_event_inc[108] = PMU_INC_WIDTH'(afu_perf_result_produce && pmu_afu_selected);
        pmu_event_inc[109] = PMU_INC_WIDTH'(afu_perf_input_wait && pmu_afu_selected);
        pmu_event_inc[110] = PMU_INC_WIDTH'(afu_perf_rhs_wait && pmu_afu_selected);
        pmu_event_inc[111] = PMU_INC_WIDTH'(afu_perf_output_stall && pmu_afu_selected);

        // 112-127: Spatz, exact shared-memory transactions/conflicts and overlap.
        pmu_event_inc[112] = PMU_INC_WIDTH'(pmu_spatz_tag_valid_q && pmu_spatz_selected);
        pmu_event_inc[113] = PMU_INC_WIDTH'(acc_qvalid && acc_qready && pmu_current_selected);
        pmu_event_inc[114] = PMU_INC_WIDTH'(acc_pvalid && acc_pready && pmu_spatz_selected);
        pmu_event_inc[115] = PMU_INC_WIDTH'((master_req[1].req && master_rsp[1].gnt && pmu_spatz_selected) +
                                            (master_req[8].req && master_rsp[8].gnt && pmu_spatz_selected));
        pmu_event_inc[116] = PMU_INC_WIDTH'((master_req[1].req && !master_rsp[1].gnt && pmu_spatz_selected) +
                                            (master_req[8].req && !master_rsp[8].gnt && pmu_spatz_selected));
        for (int mst = 0; mst < NUM_MASTERS; mst++) begin
            pmu_event_inc[117] += PMU_INC_WIDTH'(master_req[mst].req && master_rsp[mst].gnt && pmu_master_selected[mst]);
            pmu_event_inc[118] += PMU_INC_WIDTH'(master_req[mst].req && !master_rsp[mst].gnt && pmu_master_selected[mst]);
            pmu_event_inc[122] += PMU_INC_WIDTH'(master_req[mst].req && master_rsp[mst].gnt &&
                                                 !master_req[mst].we && pmu_master_selected[mst]);
            pmu_event_inc[123] += PMU_INC_WIDTH'(master_req[mst].req && master_rsp[mst].gnt &&
                                                 master_req[mst].we && pmu_master_selected[mst]);
            if (master_req[mst].req && master_rsp[mst].gnt && pmu_master_selected[mst]) begin
                if (master_req[mst].we)
                    pmu_event_inc[125] += PMU_INC_WIDTH'($countones(master_req[mst].be));
                else
                    pmu_event_inc[124] += PMU_INC_WIDTH'($countones(master_req[mst].be));
            end
        end
        pmu_event_inc[119] = pmu_event_inc[117];
        for (int bank = 0; bank < TCDM_NUM_BANKS; bank++) begin
            pmu_event_inc[121] += PMU_INC_WIDTH'(tcdm_bank_conflict[bank] &&
                (!pmu_filter_enable || (|pmu_event_inc[118])));
        end
        pmu_event_inc[120] = PMU_INC_WIDTH'((|tcdm_bank_conflict) &&
                                            (!pmu_filter_enable || (|pmu_event_inc[118])));
        pmu_event_inc[126] = PMU_INC_WIDTH'(((idma_irq_a2o_busy && pmu_dma_a2o_selected) ||
                                             (idma_irq_o2a_busy && pmu_dma_o2a_selected)) &&
                                            (((sys_debug_state != 3'd0) && pmu_sys_selected) ||
                                             (afu_perf_active && pmu_afu_selected) ||
                                             (pmu_spatz_tag_valid_q && pmu_spatz_selected)));
        pmu_event_inc[127] = PMU_INC_WIDTH'(pmu_context_active && pmu_current_selected &&
                                            !(idma_irq_a2o_busy && pmu_dma_a2o_selected) &&
                                            !(idma_irq_o2a_busy && pmu_dma_o2a_selected) &&
                                            !((sys_debug_state != 3'd0) && pmu_sys_selected) &&
                                            !(afu_perf_active && pmu_afu_selected) &&
                                            !(pmu_spatz_tag_valid_q && pmu_spatz_selected));

        // 128-162: complete state occupancy for the systolic output drain and
        // every enumerated linebuffer FSM.  Gate IDLE states with the active
        // systolic job tag so idle counters do not accumulate between jobs.
        for (int state = 0; state < 4; state++)
            pmu_event_inc[128+state] = PMU_INC_WIDTH'(
                pmu_sys_state_scope && (sys_debug_drain_state == 2'(state)));
        for (int state = 0; state < 15; state++)
            pmu_event_inc[132+state] = PMU_INC_WIDTH'(
                pmu_sys_state_scope && (sys_debug_linebuf_state == 5'(state)));
        for (int state = 0; state < 4; state++)
            pmu_event_inc[147+state] = PMU_INC_WIDTH'(
                pmu_sys_state_scope && (sys_debug_linebuf_fetch_main_state == 2'(state)));
        for (int state = 0; state < 5; state++)
            pmu_event_inc[151+state] = PMU_INC_WIDTH'(
                pmu_sys_state_scope && (sys_debug_linebuf_fetch_background_state == 3'(state)));
        for (int state = 0; state < 7; state++)
            pmu_event_inc[156+state] = PMU_INC_WIDTH'(
                pmu_sys_state_scope && (sys_debug_linebuf_bypass_state == 3'(state)));
    end

    npu_pmu #(
        .ADDR_WIDTH   (OBI_ADDR_WIDTH),
        .DATA_WIDTH   (MMIO_DATA_WIDTH),
        .NUM_COUNTERS (PMU_NUM_COUNTERS),
        .INC_WIDTH    (PMU_INC_WIDTH),
        .MAX_COUNTER_MASK(PMU_MAX_COUNTER_MASK)
    ) u_pmu (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .req_i       (pmu_mm_req),
        .gnt_o       (pmu_mm_gnt),
        .addr_i      (pmu_mm_addr),
        .we_i        (pmu_mm_we),
        .be_i        (pmu_mm_be),
        .wdata_i     (pmu_mm_wdata),
        .rvalid_o    (pmu_mm_rvalid),
        .rdata_o     (pmu_mm_rdata),
        .event_inc_i (pmu_event_inc),
        .context_id_i(pmu_context_id),
        .context_active_i(pmu_context_active),
        .phase_i     (pmu_phase),
        .filter_enable_o(pmu_filter_enable),
        .filter_context_o(pmu_filter_context)
    );

endmodule
