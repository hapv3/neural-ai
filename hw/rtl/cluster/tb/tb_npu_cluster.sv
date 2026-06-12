`default_nettype none

`include "axi/typedef.svh"

module tb_npu_cluster (
    input  logic        clk_i,       // 1 GHz NPU Clock
    input  logic        rst_ni,      // NPU Reset
    
    input  logic        apb_clk_i,   // 100 MHz APB Clock
    input  logic        apb_rst_ni,  // APB Reset

    // Expose APB ports for Cocotb to drive MMIO
    input  logic [31:0] apb_paddr_i,
    input  logic        apb_psel_i,
    input  logic        apb_penable_i,
    input  logic        apb_pwrite_i,
    input  logic [31:0] apb_pwdata_i,
    output logic        apb_pready_o,
    output logic [31:0] apb_prdata_o,
    output logic        apb_pslverr_o,

    // Backdoor port for Python to initialize axi_sim_mem
    input  logic        backdoor_we_i,
    input  logic [31:0] backdoor_addr_i,
    input  logic [7:0]  backdoor_data_i
);

    // AXI Bus between NPU Cluster DMA and axi_sim_mem
    localparam int unsigned AXI_ADDR_WIDTH = 32;
    localparam int unsigned AXI_DATA_WIDTH = 256;
    localparam int unsigned AXI_ID_WIDTH   = 4;
    localparam int unsigned AXI_USER_WIDTH = 1;

    typedef logic [AXI_ADDR_WIDTH-1:0]   axi_addr_t;
    typedef logic [AXI_DATA_WIDTH-1:0]   axi_data_t;
    typedef logic [AXI_ID_WIDTH-1:0]     axi_id_t;
    typedef logic [AXI_DATA_WIDTH/8-1:0] axi_strb_t;
    typedef logic [AXI_USER_WIDTH-1:0]   axi_user_t;

    `AXI_TYPEDEF_ALL(axi, axi_addr_t, axi_id_t, axi_data_t, axi_strb_t, axi_user_t)

    axi_req_t  axi_req;
    axi_resp_t axi_rsp;

    // Instance of NPU Cluster
    npu_cluster u_npu_cluster (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .apb_clk_i      (apb_clk_i),
        .apb_rst_ni     (apb_rst_ni),

        // AXI Master
        .axi_aw_addr_o  (axi_req.aw.addr),
        .axi_aw_len_o   (axi_req.aw.len),
        .axi_aw_size_o  (axi_req.aw.size),
        .axi_aw_burst_o (axi_req.aw.burst),
        .axi_aw_valid_o (axi_req.aw_valid),
        .axi_aw_ready_i (axi_rsp.aw_ready),

        .axi_w_data_o   (axi_req.w.data),
        .axi_w_strb_o   (axi_req.w.strb),
        .axi_w_last_o   (axi_req.w.last),
        .axi_w_valid_o  (axi_req.w_valid),
        .axi_w_ready_i  (axi_rsp.w_ready),

        .axi_b_resp_i   (axi_rsp.b.resp),
        .axi_b_valid_i  (axi_rsp.b_valid),
        .axi_b_ready_o  (axi_req.b_ready),

        .axi_ar_addr_o  (axi_req.ar.addr),
        .axi_ar_len_o   (axi_req.ar.len),
        .axi_ar_size_o  (axi_req.ar.size),
        .axi_ar_burst_o (axi_req.ar.burst),
        .axi_ar_valid_o (axi_req.ar_valid),
        .axi_ar_ready_i (axi_rsp.ar_ready),

        .axi_r_data_i   (axi_rsp.r.data),
        .axi_r_resp_i   (axi_rsp.r.resp),
        .axi_r_last_i   (axi_rsp.r.last),
        .axi_r_valid_i  (axi_rsp.r_valid),
        .axi_r_ready_o  (axi_req.r_ready),

        // APB Slave
        .apb_paddr_i    (apb_paddr_i),
        .apb_psel_i     (apb_psel_i),
        .apb_penable_i  (apb_penable_i),
        .apb_pwrite_i   (apb_pwrite_i),
        .apb_pwdata_i   (apb_pwdata_i),
        .apb_pready_o   (apb_pready_o),
        .apb_prdata_o   (apb_prdata_o),
        .apb_pslverr_o  (apb_pslverr_o),

        // Interrupts (Not used in this test)
        .irq_o          ()
    );

    // Tie off unused AXI struct fields from DMA Engine
    assign axi_req.aw.id     = '0;
    assign axi_req.aw.lock   = 1'b0;
    assign axi_req.aw.cache  = '0;
    assign axi_req.aw.prot   = '0;
    assign axi_req.aw.qos    = '0;
    assign axi_req.aw.region = '0;
    assign axi_req.aw.atop   = '0;
    assign axi_req.aw.user   = '0;

    assign axi_req.w.user    = '0;

    assign axi_req.ar.id     = '0;
    assign axi_req.ar.lock   = 1'b0;
    assign axi_req.ar.cache  = '0;
    assign axi_req.ar.prot   = '0;
    assign axi_req.ar.qos    = '0;
    assign axi_req.ar.region = '0;
    assign axi_req.ar.user   = '0;

    // Instance of PULP AXI Simulation Memory
    axi_sim_mem #(
        .AddrWidth          (AXI_ADDR_WIDTH),
        .DataWidth          (AXI_DATA_WIDTH),
        .IdWidth            (AXI_ID_WIDTH),
        .UserWidth          (AXI_USER_WIDTH),
        .NumPorts           (1),
        .axi_req_t          (axi_req_t),
        .axi_rsp_t          (axi_resp_t),
        .WarnUninitialized  (1'b0),
        .UninitializedData  ("zeros"),
        .ApplDelay          (0),
        .AcqDelay           (0)
    ) u_axi_sim_mem (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .axi_req_i          (axi_req),
        .axi_rsp_o          (axi_rsp),
        .mon_w_valid_o      (),
        .mon_w_addr_o       (),
        .mon_w_data_o       (),
        .mon_w_id_o         (),
        .mon_w_user_o       (),
        .mon_w_beat_count_o (),
        .mon_w_last_o       (),
        .mon_r_valid_o      (),
        .mon_r_addr_o       (),
        .mon_r_data_o       (),
        .mon_r_id_o         (),
        .mon_r_user_o       (),
        .mon_r_beat_count_o (),
        .mon_r_last_o       ()
    );

    // Backdoor write logic for Python/Cocotb to initialize memory
    always_ff @(posedge clk_i) begin
        if (backdoor_we_i) begin
            u_axi_sim_mem.mem[backdoor_addr_i] = backdoor_data_i;
        end
    end

endmodule
