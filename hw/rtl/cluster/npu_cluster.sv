`default_nettype none

import npu_cluster_pkg::*;

module npu_cluster (
    input  logic clk_i,       // 1 GHz NPU Core Clock
    input  logic rst_ni,      // NPU Core Reset

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

    input  logic [AXI_DATA_WIDTH-1:0]       s_axi_w_data_i,
    input  logic [(AXI_DATA_WIDTH/8)-1:0]   s_axi_w_strb_i,
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

    output logic [AXI_DATA_WIDTH-1:0]       s_axi_r_data_o,
    output logic [1:0]                      s_axi_r_resp_o,
    output logic                            s_axi_r_last_o,
    output logic                            s_axi_r_valid_o,
    input  logic                            s_axi_r_ready_i,

    //---------------------------------------------------------
    // Interrupts
    //---------------------------------------------------------
    output logic                            irq_o
);

    //=========================================================
    // 1. Host AXI to OBI Bootloader Interface
    //=========================================================
    logic                      s_axi_obi_req;
    logic                      s_axi_obi_gnt;
    logic [OBI_ADDR_WIDTH-1:0] s_axi_obi_addr;
    logic                      s_axi_obi_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] s_axi_obi_be;
    logic [OBI_DATA_WIDTH-1:0] s_axi_obi_wdata;
    logic                      s_axi_obi_rvalid;
    logic [OBI_DATA_WIDTH-1:0] s_axi_obi_rdata;

    axi_lite_to_obi #(
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .OBI_ADDR_WIDTH(OBI_ADDR_WIDTH),
        .OBI_DATA_WIDTH(OBI_DATA_WIDTH)
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

    //=========================================================
    // 2. Snitch Core & I-TCM Arbitration
    //=========================================================
    logic                      snitch_i_req;
    logic                      snitch_i_gnt;
    logic [OBI_ADDR_WIDTH-1:0] snitch_i_addr;
    logic                      snitch_i_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] snitch_i_be;
    logic [OBI_DATA_WIDTH-1:0] snitch_i_wdata;
    logic                      snitch_i_rvalid;
    logic [OBI_DATA_WIDTH-1:0] snitch_i_rdata;

    logic                      snitch_d_req;
    logic                      snitch_d_gnt;
    logic [OBI_ADDR_WIDTH-1:0] snitch_d_addr;
    logic                      snitch_d_we;
    logic [(64/8)-1:0]         snitch_d_be;
    logic [64-1:0]             snitch_d_wdata;
    logic                      snitch_d_rvalid;
    logic [64-1:0]             snitch_d_rdata;

    // Wide signals for D-Bus
    logic                      snitch_d_wide_req;
    logic                      snitch_d_wide_gnt;
    logic [OBI_ADDR_WIDTH-1:0] snitch_d_wide_addr;
    logic                      snitch_d_wide_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] snitch_d_wide_be;
    logic [OBI_DATA_WIDTH-1:0] snitch_d_wide_wdata;
    logic                      snitch_d_wide_rvalid;
    logic [OBI_DATA_WIDTH-1:0] snitch_d_wide_rdata;

    // Accelerator interface wires (Snitch ↔ Spatz)
    logic        acc_qvalid;
    logic        acc_qready;
    logic [31:0] acc_qdata_op;
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
    logic [63:0] acc_pdata;
    logic        acc_perror;
    logic [2:0]  fpu_rnd_mode;
    logic        fpu_fmt_mode;
    logic [4:0]  fpu_status;

    snitch_core #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .I_DATA_WIDTH(OBI_DATA_WIDTH),
        .D_DATA_WIDTH(64),
        .BOOT_ADDR (32'h1000_0000) // I-TCM Base Addr
    ) u_snitch_core (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .hart_id_i        (32'd0),

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
        .acc_qdata_arga_o (acc_qdata_arga),
        .acc_qdata_argb_o (acc_qdata_argb),
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
        .acc_pdata_i      (acc_pdata),
        .acc_perror_i     (acc_perror),
        // FPU side-channel
        .fpu_rnd_mode_o   (fpu_rnd_mode),
        .fpu_fmt_mode_o   (fpu_fmt_mode),
        .fpu_status_i     (fpu_status)
    );

    // Arbiter for I-TCM
    logic                      itcm_req;
    logic                      itcm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] itcm_addr;
    logic                      itcm_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] itcm_be;
    logic [OBI_DATA_WIDTH-1:0] itcm_wdata;
    logic                      itcm_rvalid;
    logic [OBI_DATA_WIDTH-1:0] itcm_rdata;

    obi_arbiter_2to1 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH)
    ) u_itcm_arbiter (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        
        .m0_req_i    (sys_m0_req),
        .m0_gnt_o    (sys_m0_gnt),
        .m0_addr_i   (sys_m0_addr),
        .m0_we_i     (sys_m0_we),
        .m0_be_i     (sys_m0_be),
        .m0_wdata_i  (sys_m0_wdata),
        .m0_rvalid_o (sys_m0_rvalid),
        .m0_rdata_o  (sys_m0_rdata),
        
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
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .SIZE_BYTES(32768)
    ) u_sram_i_tcm (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .req_i   (itcm_req),
        .we_i    (itcm_we),
        .addr_i  ((itcm_addr & 32'h0000_7FFF) >> 5), // Bank offset for 256-bit width
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
    logic [(OBI_DATA_WIDTH/8)-1:0] dtcm_be;
    logic [OBI_DATA_WIDTH-1:0] dtcm_wdata;
    logic                      dtcm_rvalid;
    logic [OBI_DATA_WIDTH-1:0] dtcm_rdata;

    logic                      ddata_req;
    logic                      ddata_gnt;
    logic [OBI_ADDR_WIDTH-1:0] ddata_addr;
    logic                      ddata_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] ddata_be;
    logic [OBI_DATA_WIDTH-1:0] ddata_wdata;
    logic                      ddata_rvalid;
    logic [OBI_DATA_WIDTH-1:0] ddata_rdata;

    logic                      reg_req;
    logic                      reg_gnt;
    logic [OBI_ADDR_WIDTH-1:0] reg_addr;
    logic                      reg_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] reg_be;
    logic [OBI_DATA_WIDTH-1:0] reg_wdata;
    logic                      reg_rvalid;
    logic [OBI_DATA_WIDTH-1:0] reg_rdata;

    logic                      ctrl_req;
    logic                      ctrl_gnt;
    logic [OBI_ADDR_WIDTH-1:0] ctrl_addr;
    logic                      ctrl_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] ctrl_be;
    logic [OBI_DATA_WIDTH-1:0] ctrl_wdata;
    logic                      ctrl_rvalid;
    logic [OBI_DATA_WIDTH-1:0] ctrl_rdata;

    logic                      idma_mm_req;
    logic                      idma_mm_gnt;
    logic [OBI_ADDR_WIDTH-1:0] idma_mm_addr;
    logic                      idma_mm_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] idma_mm_be;
    logic [OBI_DATA_WIDTH-1:0] idma_mm_wdata;
    logic                      idma_mm_rvalid;
    logic [OBI_DATA_WIDTH-1:0] idma_mm_rdata;

    // System Arbiter (AXI Lite vs Snitch D-Bus)
    logic                      sys_req;
    logic                      sys_gnt;
    logic [OBI_ADDR_WIDTH-1:0] sys_addr;
    logic                      sys_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] sys_be;
    logic [OBI_DATA_WIDTH-1:0] sys_wdata;
    logic                      sys_rvalid;
    logic [OBI_DATA_WIDTH-1:0] sys_rdata;

    obi_arbiter_2to1 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH)
    ) u_sys_arbiter (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        
        .m0_req_i    (s_axi_obi_req),
        .m0_gnt_o    (s_axi_obi_gnt),
        .m0_addr_i   (s_axi_obi_addr),
        .m0_we_i     (s_axi_obi_we),
        .m0_be_i     (s_axi_obi_be),
        .m0_wdata_i  (s_axi_obi_wdata),
        .m0_rvalid_o (s_axi_obi_rvalid),
        .m0_rdata_o  (s_axi_obi_rdata),
        
        .m1_req_i    (snitch_d_wide_req),
        .m1_gnt_o    (snitch_d_wide_gnt),
        .m1_addr_i   (snitch_d_wide_addr),
        .m1_we_i     (snitch_d_wide_we),
        .m1_be_i     (snitch_d_wide_be),
        .m1_wdata_i  (snitch_d_wide_wdata),
        .m1_rvalid_o (snitch_d_wide_rvalid),
        .m1_rdata_o  (snitch_d_wide_rdata),
        
        .slv_req_o   (sys_req),
        .slv_gnt_i   (sys_gnt),
        .slv_addr_o  (sys_addr),
        .slv_we_o    (sys_we),
        .slv_be_o    (sys_be),
        .slv_wdata_o (sys_wdata),
        .slv_rvalid_i(sys_rvalid),
        .slv_rdata_i (sys_rdata)
    );

    // System Demux 1-to-4
    logic                      sys_m0_req;
    logic                      sys_m0_gnt;
    logic [OBI_ADDR_WIDTH-1:0] sys_m0_addr;
    logic                      sys_m0_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] sys_m0_be;
    logic [OBI_DATA_WIDTH-1:0] sys_m0_wdata;
    logic                      sys_m0_rvalid;
    logic [OBI_DATA_WIDTH-1:0] sys_m0_rdata;

    // Narrow-to-Wide Adapter for Snitch D-Bus
    obi_narrow_to_wide #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .M_DATA_WIDTH(64),
        .S_DATA_WIDTH(OBI_DATA_WIDTH)
    ) u_d_narrow_to_wide (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .mst_req_i   (snitch_d_req),
        .mst_gnt_o   (snitch_d_gnt),
        .mst_addr_i  (snitch_d_addr),
        .mst_we_i    (snitch_d_we),
        .mst_be_i    (snitch_d_be),
        .mst_wdata_i (snitch_d_wdata),
        .mst_rvalid_o(snitch_d_rvalid),
        .mst_rdata_o (snitch_d_rdata),
        
        .slv_req_o   (snitch_d_wide_req),
        .slv_gnt_i   (snitch_d_wide_gnt),
        .slv_addr_o  (snitch_d_wide_addr),
        .slv_we_o    (snitch_d_wide_we),
        .slv_be_o    (snitch_d_wide_be),
        .slv_wdata_o (snitch_d_wide_wdata),
        .slv_rvalid_i(snitch_d_wide_rvalid),
        .slv_rdata_i (snitch_d_wide_rdata)
    );

    obi_demux_1to4 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .M0_BASE (32'h1000_0000), .M0_MASK (32'hFFFF_8000), // I-TCM
        .M1_BASE (32'h1000_8000), .M1_MASK (32'hFFFF_8000), // D-TCM
        .M2_BASE (32'h1010_0000), .M2_MASK (32'hFFF0_0000), // Shared Data
        .M3_BASE (32'h2000_0000), .M3_MASK (32'hFFFF_0000)  // MMIO
    ) u_sys_demux_1to4 (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .slv_req_i   (sys_req),
        .slv_gnt_o   (sys_gnt),
        .slv_addr_i  (sys_addr),
        .slv_we_i    (sys_we),
        .slv_be_i    (sys_be),
        .slv_wdata_i (sys_wdata),
        .slv_rvalid_o(sys_rvalid),
        .slv_rdata_o (sys_rdata),

        .m0_req_o     (sys_m0_req),
        .m0_gnt_i     (sys_m0_gnt),
        .m0_addr_o    (sys_m0_addr),
        .m0_we_o      (sys_m0_we),
        .m0_be_o      (sys_m0_be),
        .m0_wdata_o   (sys_m0_wdata),
        .m0_rvalid_i  (sys_m0_rvalid),
        .m0_rdata_i   (sys_m0_rdata),

        .m1_req_o     (dtcm_req),
        .m1_gnt_i     (dtcm_gnt),
        .m1_addr_o    (dtcm_addr),
        .m1_we_o      (dtcm_we),
        .m1_be_o      (dtcm_be),
        .m1_wdata_o   (dtcm_wdata),
        .m1_rvalid_i  (dtcm_rvalid),
        .m1_rdata_i   (dtcm_rdata),

        .m2_req_o     (ddata_req),
        .m2_gnt_i     (ddata_gnt),
        .m2_addr_o    (ddata_addr),
        .m2_we_o      (ddata_we),
        .m2_be_o      (ddata_be),
        .m2_wdata_o   (ddata_wdata),
        .m2_rvalid_i  (ddata_rvalid),
        .m2_rdata_i   (ddata_rdata),

        .m3_req_o     (reg_req),
        .m3_gnt_i     (reg_gnt),
        .m3_addr_o    (reg_addr),
        .m3_we_o      (reg_we),
        .m3_be_o      (reg_be),
        .m3_wdata_o   (reg_wdata),
        .m3_rvalid_i  (reg_rvalid),
        .m3_rdata_i   (reg_rdata)
    );

    // D-TCM SRAM Bank (32 KB — matches link.ld)
    cluster_sram_bank #(
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .SIZE_BYTES(32768)
    ) u_sram_d_tcm (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .req_i   (dtcm_req),
        .we_i    (dtcm_we),
        .addr_i  ((dtcm_addr - 32'h1000_8000) >> 5),
        .wdata_i (dtcm_wdata),
        .be_i    (dtcm_be),
        .gnt_o   (dtcm_gnt),
        .rvalid_o(dtcm_rvalid),
        .rdata_o (dtcm_rdata)
    );

    //=========================================================
    // 4. Cluster Control Registers (MMIO)
    //=========================================================
    obi_demux_1to4 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .M0_BASE (32'h2000_0000), .M0_MASK (32'hFFFF_F000), // Cluster control
        .M1_BASE (32'h2000_1000), .M1_MASK (32'hFFFF_F000), // iDMA-style control
        .M2_BASE (32'h2000_2000), .M2_MASK (32'hFFFF_F000), // Reserved
        .M3_BASE (32'h2000_3000), .M3_MASK (32'hFFFF_F000)  // Reserved
    ) u_mmio_demux_1to4 (
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

        .m2_req_o     (),
        .m2_gnt_i     (1'b1),
        .m2_addr_o    (),
        .m2_we_o      (),
        .m2_be_o      (),
        .m2_wdata_o   (),
        .m2_rvalid_i  (1'b1),
        .m2_rdata_i   ('0),

        .m3_req_o     (),
        .m3_gnt_i     (1'b1),
        .m3_addr_o    (),
        .m3_we_o      (),
        .m3_be_o      (),
        .m3_wdata_o   (),
        .m3_rvalid_i  (1'b1),
        .m3_rdata_i   ('0)
    );

    logic        cfg_dma_start;
    logic [31:0] cfg_dma_src_addr;
    logic [31:0] cfg_dma_dst_addr;
    logic [31:0] cfg_dma_length;
    logic        cfg_dma_done;

    logic        cfg_sys_start;
    logic [31:0] cfg_sys_w_ptr;
    logic [31:0] cfg_sys_i_ptr;
    logic [31:0] cfg_sys_o_ptr;
    logic [31:0] cfg_sys_dim_m;
    logic        cfg_sys_done;

    cluster_ctrl_regs #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH)
    ) u_ctrl_regs (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .req_i              (ctrl_req),
        .gnt_o              (ctrl_gnt),
        .addr_i             (ctrl_addr),
        .we_i               (ctrl_we),
        .be_i               (ctrl_be),
        .wdata_i            (ctrl_wdata),
        .rvalid_o           (ctrl_rvalid),
        .rdata_o            (ctrl_rdata),

        .cfg_dma_start_o    (cfg_dma_start),
        .cfg_dma_src_addr_o (cfg_dma_src_addr),
        .cfg_dma_dst_addr_o (cfg_dma_dst_addr),
        .cfg_dma_length_o   (cfg_dma_length),
        .cfg_dma_done_i     (cfg_dma_done),

        .cfg_sys_start_o      (cfg_sys_start),
        .cfg_sys_weight_ptr_o (cfg_sys_w_ptr),
        .cfg_sys_ifm_ptr_o    (cfg_sys_i_ptr),
        .cfg_sys_ofm_ptr_o    (cfg_sys_o_ptr),
        .cfg_sys_dim_m_o      (cfg_sys_dim_m),
        .cfg_sys_done_i       (cfg_sys_done)
    );

    //=========================================================
    // 5. Shared Data TCDM Interconnect (10 Masters)
    //=========================================================
    localparam int unsigned NUM_MASTERS = 10;
    // Master 0: Snitch D-Bus
    // Master 1: Spatz Vector Engine (VLSU port 0)
    // Master 2: PULP iDMA AXI2OBI write port
    // Master 3: Systolic Controller Read (I-TCDM)
    // Master 4: Systolic Controller Write Port 0 (O-TCDM)
    // Master 5: Systolic Controller Write Port 1 (O-TCDM)
    // Master 6: Systolic Controller Write Port 2 (O-TCDM)
    // Master 7: Systolic Controller Write Port 3 (O-TCDM)
    // Master 8: Spatz Vector Engine (VLSU port 1)
    // Master 9: PULP iDMA OBI2AXI read port

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
        .DATA_WIDTH(OBI_DATA_WIDTH)
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
        .bank_rdata_i     (slv_rdata)
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
    assign master_req[0].req   = ddata_req;
    assign master_req[0].we    = ddata_we;
    assign master_req[0].be    = ddata_be;
    assign master_req[0].addr  = ddata_addr;
    assign master_req[0].wdata = ddata_wdata;
    
    assign ddata_gnt    = master_rsp[0].gnt;
    assign ddata_rvalid = master_rsp[0].rvalid;
    assign ddata_rdata  = master_rsp[0].rdata;

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

    assign cfg_dma_done = idma_irq_a2o_done | idma_irq_o2a_done;

    npu_pulp_idma_ctrl_mm #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
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
        .irq_o2a_error_o    (idma_irq_o2a_error)
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

    //=========================================================
    // 7. Systolic Array (Matrix Engine)
    //=========================================================
    // Wires between systolic_controller and npu_systolic_array
    logic                      sys_weight_load_en;
    logic                      sys_clear_acc;
    logic                      sys_compute_en;
    logic signed [31:0][7:0]   sys_weight_data;
    logic signed [31:0][7:0]   sys_ifm_data;
    logic signed [31:0][31:0]  sys_psum_data;
    logic signed [31:0][31:0]  sys_ofm_data;
    logic                      sys_ofm_valid;

    // Systolic Controller OBI signals
    logic                      sys_obi_i_req;
    logic                      sys_obi_i_gnt;
    logic [OBI_ADDR_WIDTH-1:0] sys_obi_i_addr;
    logic                      sys_obi_i_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] sys_obi_i_be;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_i_wdata;
    logic                      sys_obi_i_rvalid;
    logic [OBI_DATA_WIDTH-1:0] sys_obi_i_rdata;

    logic [3:0]                      sys_obi_o_req;
    logic [3:0]                      sys_obi_o_gnt;
    logic [3:0][OBI_ADDR_WIDTH-1:0]  sys_obi_o_addr;
    logic [3:0]                      sys_obi_o_we;
    logic [3:0][(OBI_DATA_WIDTH/8)-1:0] sys_obi_o_be;
    logic [3:0][OBI_DATA_WIDTH-1:0]  sys_obi_o_wdata;
    logic [3:0]                      sys_obi_o_rvalid;
    logic [3:0][OBI_DATA_WIDTH-1:0]  sys_obi_o_rdata;

    systolic_controller #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH)
    ) u_sys_ctrl (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),

        .cfg_sys_start_i    (cfg_sys_start),
        .cfg_sys_weight_ptr_i(cfg_sys_w_ptr),
        .cfg_sys_ifm_ptr_i  (cfg_sys_i_ptr),
        .cfg_sys_ofm_ptr_i  (cfg_sys_o_ptr),
        .cfg_sys_dim_m_i    (cfg_sys_dim_m),
        .cfg_sys_done_o     (cfg_sys_done),

        .obi_i_req_o        (sys_obi_i_req),
        .obi_i_gnt_i        (sys_obi_i_gnt),
        .obi_i_addr_o       (sys_obi_i_addr),
        .obi_i_we_o         (sys_obi_i_we),
        .obi_i_be_o         (sys_obi_i_be),
        .obi_i_wdata_o      (sys_obi_i_wdata),
        .obi_i_rvalid_i     (sys_obi_i_rvalid),
        .obi_i_rdata_i      (sys_obi_i_rdata),

        .obi_o_req_o        (sys_obi_o_req),
        .obi_o_gnt_i        (sys_obi_o_gnt),
        .obi_o_addr_o       (sys_obi_o_addr),
        .obi_o_we_o         (sys_obi_o_we),
        .obi_o_be_o         (sys_obi_o_be),
        .obi_o_wdata_o      (sys_obi_o_wdata),
        .obi_o_rvalid_i     (sys_obi_o_rvalid),
        .obi_o_rdata_i      (sys_obi_o_rdata),

        .weight_load_en_o   (sys_weight_load_en),
        .clear_acc_o        (sys_clear_acc),
        .compute_en_o       (sys_compute_en),
        .weight_data_o      (sys_weight_data),
        .ifm_data_o         (sys_ifm_data),
        .psum_data_o        (sys_psum_data),
        .ofm_data_i         (sys_ofm_data),
        .ofm_valid_i        (sys_ofm_valid)
    );

    npu_systolic_array #(
        .ARRAY_DIM(32)
    ) u_systolic_array (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .weight_load_en_i   (sys_weight_load_en),
        .clear_acc_i        (sys_clear_acc),
        .compute_en_i       (sys_compute_en),
        .weight_data_i      (sys_weight_data),
        .ifm_data_i         (sys_ifm_data),
        .psum_data_i        (sys_psum_data),
        .ofm_data_o         (sys_ofm_data),
        .ofm_valid_o        (sys_ofm_valid)
    );

    // Master 3: Systolic Controller Read Port (I-TCDM)
    assign master_req[3].req   = sys_obi_i_req;
    assign master_req[3].we    = sys_obi_i_we;
    assign master_req[3].be    = sys_obi_i_be;
    assign master_req[3].addr  = sys_obi_i_addr;
    assign master_req[3].wdata = sys_obi_i_wdata;

    assign sys_obi_i_gnt    = master_rsp[3].gnt;
    assign sys_obi_i_rvalid = master_rsp[3].rvalid;
    assign sys_obi_i_rdata  = master_rsp[3].rdata;

    // Masters 4-7: Systolic Controller Write Ports (O-TCDM)
    for (genvar i = 0; i < 4; i++) begin : gen_sys_obi_o
        assign master_req[4+i].req   = sys_obi_o_req[i];
        assign master_req[4+i].we    = sys_obi_o_we[i];
        assign master_req[4+i].be    = sys_obi_o_be[i];
        assign master_req[4+i].addr  = sys_obi_o_addr[i];
        assign master_req[4+i].wdata = sys_obi_o_wdata[i];

        assign sys_obi_o_gnt[i]    = master_rsp[4+i].gnt;
        assign sys_obi_o_rvalid[i] = master_rsp[4+i].rvalid;
        assign sys_obi_o_rdata[i]  = master_rsp[4+i].rdata;
    end

endmodule
