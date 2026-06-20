`default_nettype none

module npu_pulp_idma_ctrl_mm #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter logic [ADDR_WIDTH-1:0] BASE_ADDR = 32'h2000_1000
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      req_i,
    output logic                      gnt_o,
    input  logic [ADDR_WIDTH-1:0]     addr_i,
    input  logic                      we_i,
    input  logic [(DATA_WIDTH/8)-1:0] be_i,
    input  logic [DATA_WIDTH-1:0]     wdata_i,
    output logic                      rvalid_o,
    output logic [DATA_WIDTH-1:0]     rdata_o,

    output logic [31:0]               axi_aw_addr_o,
    output logic [7:0]                axi_aw_len_o,
    output logic [2:0]                axi_aw_size_o,
    output logic [1:0]                axi_aw_burst_o,
    output logic                      axi_aw_valid_o,
    input  logic                      axi_aw_ready_i,
    output logic [DATA_WIDTH-1:0]     axi_w_data_o,
    output logic [(DATA_WIDTH/8)-1:0] axi_w_strb_o,
    output logic                      axi_w_last_o,
    output logic                      axi_w_valid_o,
    input  logic                      axi_w_ready_i,
    input  logic [1:0]                axi_b_resp_i,
    input  logic                      axi_b_valid_i,
    output logic                      axi_b_ready_o,
    output logic [31:0]               axi_ar_addr_o,
    output logic [7:0]                axi_ar_len_o,
    output logic [2:0]                axi_ar_size_o,
    output logic [1:0]                axi_ar_burst_o,
    output logic                      axi_ar_valid_o,
    input  logic                      axi_ar_ready_i,
    input  logic [DATA_WIDTH-1:0]     axi_r_data_i,
    input  logic [1:0]                axi_r_resp_i,
    input  logic                      axi_r_last_i,
    input  logic                      axi_r_valid_i,
    output logic                      axi_r_ready_o,

    output logic                      obi_read_req_o,
    input  logic                      obi_read_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_read_addr_o,
    output logic                      obi_read_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_read_be_o,
    output logic [DATA_WIDTH-1:0]     obi_read_wdata_o,
    input  logic                      obi_read_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_read_rdata_i,

    output logic                      obi_write_req_o,
    input  logic                      obi_write_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_write_addr_o,
    output logic                      obi_write_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_write_be_o,
    output logic [DATA_WIDTH-1:0]     obi_write_wdata_o,
    input  logic                      obi_write_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_write_rdata_i,

    output logic                      irq_a2o_busy_o,
    output logic                      irq_a2o_start_o,
    output logic                      irq_a2o_done_o,
    output logic                      irq_a2o_error_o,
    output logic                      irq_o2a_busy_o,
    output logic                      irq_o2a_start_o,
    output logic                      irq_o2a_done_o,
    output logic                      irq_o2a_error_o
);

    localparam int unsigned WORDS_PER_BEAT = DATA_WIDTH / 32;

    magia_tile_pkg::core_obi_data_req_t mmio_req;
    magia_tile_pkg::core_obi_data_rsp_t mmio_rsp;

    magia_tile_pkg::idma_fe_reg_req_t idma_fe_reg_axi2obi_req;
    magia_tile_pkg::idma_fe_reg_rsp_t idma_fe_reg_axi2obi_rsp;
    magia_tile_pkg::idma_fe_reg_req_t idma_fe_reg_obi2axi_req;
    magia_tile_pkg::idma_fe_reg_rsp_t idma_fe_reg_obi2axi_rsp;

    magia_tile_pkg::idma_axi_req_t axi_read_req;
    magia_tile_pkg::idma_axi_rsp_t axi_read_rsp;
    magia_tile_pkg::idma_axi_req_t axi_write_req;
    magia_tile_pkg::idma_axi_rsp_t axi_write_rsp;

    magia_tile_pkg::idma_obi_req_t obi_read_req;
    magia_tile_pkg::idma_obi_rsp_t obi_read_rsp;
    magia_tile_pkg::idma_obi_req_t obi_write_req;
    magia_tile_pkg::idma_obi_rsp_t obi_write_rsp;

    logic [$clog2(WORDS_PER_BEAT)-1:0] mmio_word_idx;
    logic [31:0]                       mmio_word_addr;
    logic [31:0]                       mmio_wdata_word;
    logic [3:0]                        mmio_be_word;

    always_comb begin
        mmio_word_idx = addr_i[4:2];
        for (int i = 0; i < WORDS_PER_BEAT; i++) begin
            if (|be_i[i*4 +: 4]) begin
                mmio_word_idx = i[$clog2(WORDS_PER_BEAT)-1:0];
            end
        end

        mmio_word_addr = (addr_i & 32'hFFFF_FFE0) + (32'(mmio_word_idx) << 2);
        mmio_wdata_word = wdata_i[mmio_word_idx*32 +: 32];
        mmio_be_word = we_i ? be_i[mmio_word_idx*4 +: 4] : 4'hf;
    end

    always_comb begin
        mmio_req = '0;
        mmio_req.req = req_i;
        mmio_req.a.addr = magia_tile_pkg::IDMA_CTRL_ADDR_START + (mmio_word_addr - BASE_ADDR);
        mmio_req.a.we = we_i;
        mmio_req.a.be = mmio_be_word;
        mmio_req.a.wdata = mmio_wdata_word;
        mmio_req.a.aid = '0;
        mmio_req.a.a_optional = '0;
    end

    assign gnt_o = mmio_rsp.gnt;
    assign rvalid_o = mmio_rsp.rvalid;

    always_comb begin
        rdata_o = '0;
        rdata_o[mmio_word_idx*32 +: 32] = mmio_rsp.r.rdata;
    end

    idma_obi_ctrl_decoder i_idma_obi_ctrl_decoder (
        .obi_req_i           (mmio_req),
        .obi_rsp_o           (mmio_rsp),
        .idma_axi2obi_req_o  (idma_fe_reg_axi2obi_req),
        .idma_axi2obi_rsp_i  (idma_fe_reg_axi2obi_rsp),
        .idma_obi2axi_req_o  (idma_fe_reg_obi2axi_req),
        .idma_obi2axi_rsp_i  (idma_fe_reg_obi2axi_rsp)
    );

    idma_axi_obi_transfer_ch #(
        .CHANNEL_T         (magia_tile_pkg::AXI2OBI),
        .ERROR_CAP         (idma_pkg::NO_ERROR_HANDLING),
        .idma_fe_reg_req_t (magia_tile_pkg::idma_fe_reg_req_t),
        .idma_fe_reg_rsp_t (magia_tile_pkg::idma_fe_reg_rsp_t),
        .axi_req_t         (magia_tile_pkg::idma_axi_req_t),
        .axi_rsp_t         (magia_tile_pkg::idma_axi_rsp_t),
        .obi_req_t         (magia_tile_pkg::idma_obi_req_t),
        .obi_rsp_t         (magia_tile_pkg::idma_obi_rsp_t)
    ) i_l2_to_l1_ch (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .testmode_i       (1'b0),
        .clear_i          (1'b0),
        .cfg_req_i        (idma_fe_reg_axi2obi_req),
        .cfg_rsp_o        (idma_fe_reg_axi2obi_rsp),
        .axi_req_o        (axi_read_req),
        .axi_rsp_i        (axi_read_rsp),
        .obi_req_o        (obi_write_req),
        .obi_rsp_i        (obi_write_rsp),
        .transfer_busy_o  (irq_a2o_busy_o),
        .transfer_start_o (irq_a2o_start_o),
        .transfer_done_o  (irq_a2o_done_o),
        .transfer_error_o (irq_a2o_error_o)
    );

    idma_axi_obi_transfer_ch #(
        .CHANNEL_T         (magia_tile_pkg::OBI2AXI),
        .ERROR_CAP         (idma_pkg::NO_ERROR_HANDLING),
        .idma_fe_reg_req_t (magia_tile_pkg::idma_fe_reg_req_t),
        .idma_fe_reg_rsp_t (magia_tile_pkg::idma_fe_reg_rsp_t),
        .axi_req_t         (magia_tile_pkg::idma_axi_req_t),
        .axi_rsp_t         (magia_tile_pkg::idma_axi_rsp_t),
        .obi_req_t         (magia_tile_pkg::idma_obi_req_t),
        .obi_rsp_t         (magia_tile_pkg::idma_obi_rsp_t)
    ) i_l1_to_l2_ch (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .testmode_i       (1'b0),
        .clear_i          (1'b0),
        .cfg_req_i        (idma_fe_reg_obi2axi_req),
        .cfg_rsp_o        (idma_fe_reg_obi2axi_rsp),
        .axi_req_o        (axi_write_req),
        .axi_rsp_i        (axi_write_rsp),
        .obi_req_o        (obi_read_req),
        .obi_rsp_i        (obi_read_rsp),
        .transfer_busy_o  (irq_o2a_busy_o),
        .transfer_start_o (irq_o2a_start_o),
        .transfer_done_o  (irq_o2a_done_o),
        .transfer_error_o (irq_o2a_error_o)
    );

    assign axi_ar_addr_o = axi_read_req.ar.addr;
    assign axi_ar_len_o = axi_read_req.ar.len;
    assign axi_ar_size_o = axi_read_req.ar.size;
    assign axi_ar_burst_o = axi_read_req.ar.burst;
    assign axi_ar_valid_o = axi_read_req.ar_valid;
    assign axi_r_ready_o = axi_read_req.r_ready;

    assign axi_aw_addr_o = axi_write_req.aw.addr;
    assign axi_aw_len_o = axi_write_req.aw.len;
    assign axi_aw_size_o = axi_write_req.aw.size;
    assign axi_aw_burst_o = axi_write_req.aw.burst;
    assign axi_aw_valid_o = axi_write_req.aw_valid;
    assign axi_w_data_o = axi_write_req.w.data;
    assign axi_w_strb_o = axi_write_req.w.strb;
    assign axi_w_last_o = axi_write_req.w.last;
    assign axi_w_valid_o = axi_write_req.w_valid;
    assign axi_b_ready_o = axi_write_req.b_ready;

    always_comb begin
        axi_read_rsp = '0;
        axi_read_rsp.ar_ready = axi_ar_ready_i;
        axi_read_rsp.r_valid = axi_r_valid_i;
        axi_read_rsp.r.data = axi_r_data_i;
        axi_read_rsp.r.resp = axi_pkg::resp_t'(axi_r_resp_i);
        axi_read_rsp.r.last = axi_r_last_i;

        axi_write_rsp = '0;
        axi_write_rsp.aw_ready = axi_aw_ready_i;
        axi_write_rsp.w_ready = axi_w_ready_i;
        axi_write_rsp.b_valid = axi_b_valid_i;
        axi_write_rsp.b.resp = axi_pkg::resp_t'(axi_b_resp_i);
    end

    assign obi_read_req_o = obi_read_req.req;
    assign obi_read_addr_o = obi_read_req.a.addr;
    assign obi_read_we_o = obi_read_req.a.we;
    assign obi_read_be_o = obi_read_req.a.be;
    assign obi_read_wdata_o = obi_read_req.a.wdata;

    assign obi_write_req_o = obi_write_req.req;
    assign obi_write_addr_o = obi_write_req.a.addr;
    assign obi_write_we_o = obi_write_req.a.we;
    assign obi_write_be_o = obi_write_req.a.be;
    assign obi_write_wdata_o = obi_write_req.a.wdata;

    always_comb begin
        obi_read_rsp = '0;
        obi_read_rsp.gnt = obi_read_gnt_i;
        obi_read_rsp.rvalid = obi_read_rvalid_i;
        obi_read_rsp.r.rdata = obi_read_rdata_i;

        obi_write_rsp = '0;
        obi_write_rsp.gnt = obi_write_gnt_i;
        obi_write_rsp.rvalid = obi_write_rvalid_i;
        obi_write_rsp.r.rdata = obi_write_rdata_i;
    end

endmodule
