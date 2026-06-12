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
    logic [(OBI_DATA_WIDTH/8)-1:0] snitch_d_be;
    logic [OBI_DATA_WIDTH-1:0] snitch_d_wdata;
    logic                      snitch_d_rvalid;
    logic [OBI_DATA_WIDTH-1:0] snitch_d_rdata;

    snitch_core #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH),
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
        .obi_d_rdata_i    (snitch_d_rdata)
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
        
        .m0_req_i    (s_axi_obi_req),
        .m0_gnt_o    (s_axi_obi_gnt),
        .m0_addr_i   (s_axi_obi_addr),
        .m0_we_i     (s_axi_obi_we),
        .m0_be_i     (s_axi_obi_be),
        .m0_wdata_i  (s_axi_obi_wdata),
        .m0_rvalid_o (s_axi_obi_rvalid),
        .m0_rdata_o  (s_axi_obi_rdata),
        
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
        .addr_i  (itcm_addr >> 5), // Bank offset for 256-bit width
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

    obi_demux_1to3 #(
        .ADDR_WIDTH(OBI_ADDR_WIDTH),
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .M0_BASE (32'h1000_8000), .M0_MASK (32'hFFFF_8000), // D-TCM
        .M1_BASE (32'h1010_0000), .M1_MASK (32'hFFF0_0000), // Shared Data
        .M2_BASE (32'h2000_0000), .M2_MASK (32'hFFFF_0000)  // MMIO
    ) u_obi_demux_dbus (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        
        .slv_req_i    (snitch_d_req),
        .slv_gnt_o    (snitch_d_gnt),
        .slv_addr_i   (snitch_d_addr),
        .slv_we_i     (snitch_d_we),
        .slv_be_i     (snitch_d_be),
        .slv_wdata_i  (snitch_d_wdata),
        .slv_rvalid_o (snitch_d_rvalid),
        .slv_rdata_o  (snitch_d_rdata),

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
        .m2_rdata_i   (reg_rdata)
    );

    // D-TCM SRAM Bank (8 KB)
    cluster_sram_bank #(
        .DATA_WIDTH(OBI_DATA_WIDTH),
        .SIZE_BYTES(8192)
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
        .req_i              (reg_req),
        .gnt_o              (reg_gnt),
        .addr_i             (reg_addr),
        .we_i               (reg_we),
        .be_i               (reg_be),
        .wdata_i            (reg_wdata),
        .rvalid_o           (reg_rvalid),
        .rdata_o            (reg_rdata),

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
    // 5. Shared Data TCDM Interconnect (4 Masters)
    //=========================================================
    localparam int unsigned NUM_MASTERS = 4;
    // Master 0: Snitch D-Bus
    // Master 1: Spatz Vector Engine
    // Master 2: DMA Engine
    // Master 3: Systolic Array (Phase 3B)

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
        // Subtract base 0x1010_0000, then shift by 5 (32 bytes per 256-bit word) 
        // Wait, tcdm_interconnect distributes addresses to banks via low bits.
        // Bank ID = addr[8:5]. Word offset in bank = addr[18:9].
        // The address output from tcdm_interconnect (slv_addr[b]) is the original address!
        // We need to mask the base address:
        assign slave_req[b].addr  = (slv_addr[b] & 32'h000F_FFFF) >> 9;
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

    // Master 1: Spatz Vector Engine (Phase 3B Placeholder)
    assign master_req[1].req   = 1'b0;
    assign master_req[1].we    = 1'b0;
    assign master_req[1].be    = '0;
    assign master_req[1].addr  = '0;
    assign master_req[1].wdata = '0;

    // Master 2: DMA Engine
    logic                      dma_obi_req;
    logic                      dma_obi_gnt;
    logic [OBI_ADDR_WIDTH-1:0] dma_obi_addr;
    logic                      dma_obi_we;
    logic [(OBI_DATA_WIDTH/8)-1:0] dma_obi_be;
    logic [OBI_DATA_WIDTH-1:0] dma_obi_wdata;
    logic                      dma_obi_rvalid;
    logic [OBI_DATA_WIDTH-1:0] dma_obi_rdata;

    dma_engine #(
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .OBI_DATA_WIDTH(OBI_DATA_WIDTH)
    ) u_dma_engine (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .cfg_src_addr_i (cfg_dma_src_addr),
        .cfg_dst_addr_i (cfg_dma_dst_addr),
        .cfg_length_i   (cfg_dma_length),
        .cfg_start_i    (cfg_dma_start),
        .cfg_done_o     (cfg_dma_done),

        .axi_aw_addr_o  (axi_aw_addr_o),
        .axi_aw_len_o   (axi_aw_len_o),
        .axi_aw_size_o  (axi_aw_size_o),
        .axi_aw_burst_o (axi_aw_burst_o),
        .axi_aw_valid_o (axi_aw_valid_o),
        .axi_aw_ready_i (axi_aw_ready_i),
        .axi_w_data_o   (axi_w_data_o),
        .axi_w_strb_o   (axi_w_strb_o),
        .axi_w_last_o   (axi_w_last_o),
        .axi_w_valid_o  (axi_w_valid_o),
        .axi_w_ready_i  (axi_w_ready_i),
        .axi_b_resp_i   (axi_b_resp_i),
        .axi_b_valid_i  (axi_b_valid_i),
        .axi_b_ready_o  (axi_b_ready_o),
        .axi_ar_addr_o  (axi_ar_addr_o),
        .axi_ar_len_o   (axi_ar_len_o),
        .axi_ar_size_o  (axi_ar_size_o),
        .axi_ar_burst_o (axi_ar_burst_o),
        .axi_ar_valid_o (axi_ar_valid_o),
        .axi_ar_ready_i (axi_ar_ready_i),
        .axi_r_data_i   (axi_r_data_i),
        .axi_r_resp_i   (axi_r_resp_i),
        .axi_r_last_i   (axi_r_last_i),
        .axi_r_valid_i  (axi_r_valid_i),
        .axi_r_ready_o  (axi_r_ready_o),

        .obi_req_o      (dma_obi_req),
        .obi_gnt_i      (dma_obi_gnt),
        .obi_addr_o     (dma_obi_addr),
        .obi_we_o       (dma_obi_we),
        .obi_be_o       (dma_obi_be),
        .obi_wdata_o    (dma_obi_wdata),
        .obi_rvalid_i   (dma_obi_rvalid),
        .obi_rdata_i    (dma_obi_rdata)
    );

    assign master_req[2].req   = dma_obi_req;
    assign master_req[2].we    = dma_obi_we;
    assign master_req[2].be    = dma_obi_be;
    assign master_req[2].addr  = dma_obi_addr;
    assign master_req[2].wdata = dma_obi_wdata;
    
    assign dma_obi_gnt    = master_rsp[2].gnt;
    assign dma_obi_rvalid = master_rsp[2].rvalid;
    assign dma_obi_rdata  = master_rsp[2].rdata;

    // Master 3: Systolic Array (Phase 3B Placeholder)
    assign master_req[3].req   = 1'b0;
    assign master_req[3].we    = 1'b0;
    assign master_req[3].be    = '0;
    assign master_req[3].addr  = '0;
    assign master_req[3].wdata = '0;

endmodule
