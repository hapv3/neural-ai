`default_nettype none

import npu_cluster_pkg::*;

module npu_cluster (
    input  logic clk_i,       // 1 GHz NPU Core Clock
    input  logic rst_ni,      // NPU Core Reset
    
    //---------------------------------------------------------
    // APB Clock Domain
    //---------------------------------------------------------
    input  logic apb_clk_i,   // 100 MHz APB Clock
    input  logic apb_rst_ni,  // APB Reset

    //---------------------------------------------------------
    // AXI4 Interface (To External Memory / Manager Core)
    //---------------------------------------------------------
    // Address Write Channel
    output logic [AXI_ADDR_WIDTH-1:0]       axi_aw_addr_o,
    output logic [7:0]                      axi_aw_len_o,
    output logic [2:0]                      axi_aw_size_o,
    output logic [1:0]                      axi_aw_burst_o,
    output logic                            axi_aw_valid_o,
    input  logic                            axi_aw_ready_i,

    // Write Channel
    output logic [AXI_DATA_WIDTH-1:0]       axi_w_data_o,
    output logic [(AXI_DATA_WIDTH/8)-1:0]   axi_w_strb_o,
    output logic                            axi_w_last_o,
    output logic                            axi_w_valid_o,
    input  logic                            axi_w_ready_i,

    // Write Response Channel
    input  logic [1:0]                      axi_b_resp_i,
    input  logic                            axi_b_valid_i,
    output logic                            axi_b_ready_o,

    // Address Read Channel
    output logic [AXI_ADDR_WIDTH-1:0]       axi_ar_addr_o,
    output logic [7:0]                      axi_ar_len_o,
    output logic [2:0]                      axi_ar_size_o,
    output logic [1:0]                      axi_ar_burst_o,
    output logic                            axi_ar_valid_o,
    input  logic                            axi_ar_ready_i,

    // Read Channel
    input  logic [AXI_DATA_WIDTH-1:0]       axi_r_data_i,
    input  logic [1:0]                      axi_r_resp_i,
    input  logic                            axi_r_last_i,
    input  logic                            axi_r_valid_i,
    output logic                            axi_r_ready_o,

    //---------------------------------------------------------
    // C. Control Interface (APB Slave for CSRs)
    //---------------------------------------------------------
    input  logic [31:0]                     apb_paddr_i,
    input  logic                            apb_psel_i,
    input  logic                            apb_penable_i,
    input  logic                            apb_pwrite_i,
    input  logic [31:0]                     apb_pwdata_i,
    output logic                            apb_pready_o,
    output logic [31:0]                     apb_prdata_o,
    output logic                            apb_pslverr_o,

    //---------------------------------------------------------
    // Interrupts
    //---------------------------------------------------------
    output logic                            irq_o
);

    //---------------------------------------------------------
    // 1. TCDM Interconnect (Crossbar)
    //---------------------------------------------------------
    localparam int unsigned NUM_MASTERS = 4;
    // Master 0: Snitch Core D-TCM (MMIO and Local Data)
    // Master 1: Spatz Vector Engine
    // Master 2: DMA Engine
    // Master 3: Snitch Core I-TCDM (Instruction Fetch)

    obi_req_t [NUM_MASTERS-1:0] master_req;
    obi_rsp_t [NUM_MASTERS-1:0] master_rsp;

    obi_req_t [TCDM_NUM_BANKS-1:0] slave_req;
    obi_rsp_t [TCDM_NUM_BANKS-1:0] slave_rsp;

    // Unpack logic arrays for tcdm_interconnect
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
        // Tính toán địa chỉ Word bên trong từng Bank
        // Bỏ qua 5 bit Byte Offset và 4 bit Bank Select
        assign slave_req[b].addr  = slv_addr[b] >> 9;
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

    //---------------------------------------------------------
    // 2. SRAM Banks (16 x 32KB = 512KB)
    //---------------------------------------------------------
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

    //---------------------------------------------------------
    // 3. Cluster Control Registers (APB Slave)
    //---------------------------------------------------------
    logic        cfg_dma_start;
    logic [31:0] cfg_dma_src_addr;
    logic [31:0] cfg_dma_dst_addr;
    logic [31:0] cfg_dma_length;
    logic        cfg_dma_done;

    cluster_ctrl_regs #(
        .APB_ADDR_WIDTH(32),
        .APB_DATA_WIDTH(32)
    ) u_ctrl_regs (
        .clk_i              (apb_clk_i),
        .rst_ni             (apb_rst_ni),
        .paddr_i            (apb_paddr_i),
        .psel_i             (apb_psel_i),
        .penable_i          (apb_penable_i),
        .pwrite_i           (apb_pwrite_i),
        .pwdata_i           (apb_pwdata_i),
        .pready_o           (apb_pready_o),
        .prdata_o           (apb_prdata_o),
        .pslverr_o          (apb_pslverr_o),
        .cfg_dma_start_o    (cfg_dma_start),
        .cfg_dma_src_addr_o (cfg_dma_src_addr),
        .cfg_dma_dst_addr_o (cfg_dma_dst_addr),
        .cfg_dma_length_o   (cfg_dma_length),
        .cfg_dma_done_i     (cfg_dma_done_apb_pulse)
    );

    //---------------------------------------------------------
    // CDC Logic: APB (100MHz) <-> NPU (1GHz) using cdc_2phase
    //---------------------------------------------------------
    // 1. APB -> NPU: Sync cfg_dma_start
    logic dma_start_pulse;
    cdc_2phase #(
        .T(logic)
    ) u_cdc_start (
        .src_rst_ni  (apb_rst_ni),
        .src_clk_i   (apb_clk_i),
        .src_data_i  (1'b0),
        .src_valid_i (cfg_dma_start),
        .src_ready_o (),
        .dst_rst_ni  (rst_ni),
        .dst_clk_i   (clk_i),
        .dst_data_o  (),
        .dst_valid_o (dma_start_pulse),
        .dst_ready_i (1'b1)
    );

    // 2. NPU -> APB: Sync cfg_dma_done
    logic dma_engine_done;
    logic cfg_dma_done_apb_pulse;
    cdc_2phase #(
        .T(logic)
    ) u_cdc_done (
        .src_rst_ni  (rst_ni),
        .src_clk_i   (clk_i),
        .src_data_i  (1'b0),
        .src_valid_i (dma_engine_done),
        .src_ready_o (),
        .dst_rst_ni  (apb_rst_ni),
        .dst_clk_i   (apb_clk_i),
        .dst_data_o  (),
        .dst_valid_o (cfg_dma_done_apb_pulse),
        .dst_ready_i (1'b1)
    );

    //---------------------------------------------------------
    // 4. DMA Engine (AXI to OBI)
    //---------------------------------------------------------
    // Mở cổng tín hiệu OBI riêng lẻ để truyền vào DMA
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
        .cfg_start_i    (dma_start_pulse),
        .cfg_done_o     (dma_engine_done),

        // AXI4
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

        // OBI
        .obi_req_o      (dma_obi_req),
        .obi_gnt_i      (dma_obi_gnt),
        .obi_addr_o     (dma_obi_addr),
        .obi_we_o       (dma_obi_we),
        .obi_be_o       (dma_obi_be),
        .obi_wdata_o    (dma_obi_wdata),
        .obi_rvalid_i   (dma_obi_rvalid),
        .obi_rdata_i    (dma_obi_rdata)
    );

    // Gắn DMA vào Master Port 2
    assign master_req[2].req   = dma_obi_req;
    assign master_req[2].we    = dma_obi_we;
    assign master_req[2].be    = dma_obi_be;
    assign master_req[2].addr  = dma_obi_addr;
    assign master_req[2].wdata = dma_obi_wdata;
    
    assign dma_obi_gnt    = master_rsp[2].gnt;
    assign dma_obi_rvalid = master_rsp[2].rvalid;
    assign dma_obi_rdata  = master_rsp[2].rdata;

    //---------------------------------------------------------
    // 4. Memory-Mapped Registers (MMIO) for Systolic
    //---------------------------------------------------------
    logic sys_weight_load, sys_clear_acc, sys_compute_en;
    
    // Giả lập logic bắt gói tin (Intercept) từ Master 0 (Snitch)
    // Nếu địa chỉ rơi vào vùng MMIO SYSTOLIC_MMR_BASE, Snitch sẽ ghi thanh ghi
    // Tạm thời nối cứng (Hardcode) phục vụ Testbench
    // ...

    //---------------------------------------------------------
    // 5. NPU Systolic Array
    //---------------------------------------------------------
    // Nối tín hiệu giả lập phục vụ test (Có thể đưa ra ngoài qua port tạm thời)
    logic [31:0] debug_ofm_valid;
    
    npu_systolic_array #(
        .ARRAY_DIM(32)
    ) u_systolic_array (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .weight_load_en_i (sys_weight_load),
        .clear_acc_i      (sys_clear_acc),
        .compute_en_i     (sys_compute_en),
        
        // Data inputs từ TCDM: Thiết kế nâng cao sẽ có DMA riêng (local DMA) 
        // hoặc đọc trực tiếp từ bộ nhớ bằng cổng OBI.
        // Tạm thời gắn cố định '0 để tránh lỗi compile, Phase 2 tập trung ghép nối Cluster & AXI.
        .weight_data_i    ('0),
        .ifm_data_i       ('0),
        .psum_data_i      ('0),
        .ofm_data_o       (),
        .ofm_valid_o      (debug_ofm_valid[0])
    );

    //---------------------------------------------------------
    // 6. Spatz Vector Engine
    //---------------------------------------------------------
    logic                      spatz_obi_req;
    logic                      spatz_obi_gnt;
    logic [31:0]               spatz_obi_addr;
    logic                      spatz_obi_we;
    logic [31:0]               spatz_obi_be;
    logic [255:0]              spatz_obi_wdata;
    logic                      spatz_obi_rvalid;
    logic [255:0]              spatz_obi_rdata;

    spatz_wrapper u_spatz (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .coprocessor_req_i    (1'b0),
        .coprocessor_insn_i   ('0),
        .coprocessor_rs1_i    ('0),
        .coprocessor_rs2_i    ('0),
        .coprocessor_gnt_o    (),
        .coprocessor_valid_o  (),
        .coprocessor_result_o (),
        .obi_req_o            (spatz_obi_req),
        .obi_gnt_i            (spatz_obi_gnt),
        .obi_addr_o           (spatz_obi_addr),
        .obi_we_o             (spatz_obi_we),
        .obi_be_o             (spatz_obi_be),
        .obi_wdata_o          (spatz_obi_wdata),
        .obi_rvalid_i         (spatz_obi_rvalid),
        .obi_rdata_i          (spatz_obi_rdata)
    );

    assign master_req[1].req   = spatz_obi_req;
    assign master_req[1].we    = spatz_obi_we;
    assign master_req[1].be    = spatz_obi_be;
    assign master_req[1].addr  = spatz_obi_addr;
    assign master_req[1].wdata = spatz_obi_wdata;
    
    assign spatz_obi_gnt    = master_rsp[1].gnt;
    assign spatz_obi_rvalid = master_rsp[1].rvalid;
    assign spatz_obi_rdata  = master_rsp[1].rdata;

    //---------------------------------------------------------
    // 7. Snitch Core (Placeholder)
    //---------------------------------------------------------
    // Gắn giá trị mặc định cho các Master không dùng để tránh lỏng dây (Floating)
    assign master_req[0] = '0; // Snitch D
    assign master_req[1] = '0; // Spatz
    assign master_req[3] = '0; // Snitch I

endmodule
