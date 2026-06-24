`default_nettype none

module npu_pulp_idma_ctrl_mm #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned CFG_DATA_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter logic [ADDR_WIDTH-1:0] BASE_ADDR = 32'h2000_1000
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      req_i,
    output logic                      gnt_o,
    input  logic [ADDR_WIDTH-1:0]     addr_i,
    input  logic                      we_i,
    input  logic [(CFG_DATA_WIDTH/8)-1:0] be_i,
    input  logic [CFG_DATA_WIDTH-1:0]     wdata_i,
    output logic                      rvalid_o,
    output logic [CFG_DATA_WIDTH-1:0]     rdata_o,

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

    localparam int unsigned STRB_WIDTH = DATA_WIDTH / 8;
    localparam int unsigned NUM_DIMS = 3;
    localparam logic [11:0] DIR_OFFSET = 12'h200;
    localparam logic [NUM_DIMS-1:0][31:0] REP_WIDTHS = '{default: 32'd32};

    typedef struct packed {
        logic [31:0] addr;
        logic        write;
        logic [31:0] wdata;
        logic [3:0]  wstrb;
        logic        valid;
    } reg_req_t;

    typedef struct packed {
        logic [31:0] rdata;
        logic        error;
        logic        ready;
    } reg_rsp_t;

    typedef struct packed {
        idma_pkg::protocol_e        src_protocol;
        idma_pkg::protocol_e        dst_protocol;
        logic                       axi_id;
        idma_pkg::axi_options_t     src;
        idma_pkg::axi_options_t     dst;
        idma_pkg::backend_options_t beo;
        logic                       last;
    } idma_options_t;

    typedef struct packed {
        logic [31:0]   length;
        logic [31:0]   src_addr;
        logic [31:0]   dst_addr;
        logic          user;
        idma_options_t opt;
    } idma_req_t;

    typedef struct packed {
        axi_pkg::resp_t      cause;
        idma_pkg::err_type_t err_type;
        logic [31:0]         burst_addr;
    } idma_err_payload_t;

    typedef struct packed {
        logic              last;
        logic              error;
        idma_err_payload_t pld;
    } idma_rsp_t;

    typedef struct packed {
        logic [31:0] reps;
        logic [31:0] src_strides;
        logic [31:0] dst_strides;
    } idma_d_req_t;

    typedef struct packed {
        idma_req_t burst_req;
        idma_d_req_t [NUM_DIMS-2:0] d_req;
    } idma_nd_req_t;

    reg_req_t a2o_reg_req;
    reg_rsp_t a2o_reg_rsp;
    reg_req_t [0:0] a2o_reg_req_arr;
    reg_rsp_t [0:0] a2o_reg_rsp_arr;

    reg_req_t o2a_reg_req;
    reg_rsp_t o2a_reg_rsp;
    reg_req_t [0:0] o2a_reg_req_arr;
    reg_rsp_t [0:0] o2a_reg_rsp_arr;

    idma_nd_req_t a2o_front_req;
    logic         a2o_front_valid;
    logic         a2o_front_ready;
    idma_nd_req_t a2o_fe_req;
    logic         a2o_fe_valid;
    logic         a2o_fe_ready;
    idma_req_t    a2o_be_req;
    logic         a2o_be_req_valid;
    logic         a2o_be_req_ready;
    idma_rsp_t    a2o_be_rsp;
    logic         a2o_be_rsp_valid;
    logic         a2o_be_rsp_ready;
    logic         a2o_fe_rsp_valid;
    logic         a2o_me_busy;
    logic [31:0]  a2o_next_id;
    logic [31:0]  a2o_done_id;
    idma_pkg::idma_busy_t a2o_busy;
    idma_pkg::idma_busy_t [0:0] a2o_busy_arr;
    logic [0:0][31:0] a2o_done_id_arr;
    logic [0:0]       a2o_me_busy_arr;

    idma_nd_req_t o2a_front_req;
    logic         o2a_front_valid;
    logic         o2a_front_ready;
    idma_nd_req_t o2a_fe_req;
    logic         o2a_fe_valid;
    logic         o2a_fe_ready;
    idma_req_t    o2a_be_req;
    logic         o2a_be_req_valid;
    logic         o2a_be_req_ready;
    idma_rsp_t    o2a_be_rsp;
    logic         o2a_be_rsp_valid;
    logic         o2a_be_rsp_ready;
    logic         o2a_fe_rsp_valid;
    logic         o2a_me_busy;
    logic [31:0]  o2a_next_id;
    logic [31:0]  o2a_done_id;
    idma_pkg::idma_busy_t o2a_busy;
    idma_pkg::idma_busy_t [0:0] o2a_busy_arr;
    logic [0:0][31:0] o2a_done_id_arr;
    logic [0:0]       o2a_me_busy_arr;

    logic [31:0]                       mmio_word_addr;
    logic [31:0]                       mmio_local_addr;
    logic                              mmio_dir;
    logic [31:0]                       mmio_rdata_word;

    function automatic logic decode_dir(input logic [31:0] exact_addr);
        decode_dir = (exact_addr >= (BASE_ADDR + DIR_OFFSET));
    endfunction

    function automatic logic [31:0] decode_local_addr(input logic [31:0] exact_addr);
        logic direction;
        begin
            direction = decode_dir(exact_addr);
            decode_local_addr = direction
                ? exact_addr - BASE_ADDR - DIR_OFFSET
                : exact_addr - BASE_ADDR;
        end
    endfunction

    always_comb begin
        mmio_word_addr = addr_i & 32'hFFFF_FFFC;
        mmio_local_addr = decode_local_addr(mmio_word_addr);
        mmio_dir = decode_dir(mmio_word_addr);
    end

    always_comb begin
        a2o_reg_req = '0;
        a2o_reg_req.addr = mmio_local_addr;
        a2o_reg_req.write = we_i;
        a2o_reg_req.wdata = wdata_i[31:0];
        a2o_reg_req.wstrb = we_i ? be_i[3:0] : 4'hf;
        a2o_reg_req.valid = req_i && !mmio_dir;

        o2a_reg_req = '0;
        o2a_reg_req.addr = mmio_local_addr;
        o2a_reg_req.write = we_i;
        o2a_reg_req.wdata = wdata_i[31:0];
        o2a_reg_req.wstrb = we_i ? be_i[3:0] : 4'hf;
        o2a_reg_req.valid = req_i && mmio_dir;

        a2o_reg_req_arr[0] = a2o_reg_req;
        o2a_reg_req_arr[0] = o2a_reg_req;
        a2o_reg_rsp = a2o_reg_rsp_arr[0];
        o2a_reg_rsp = o2a_reg_rsp_arr[0];
    end

    assign gnt_o = req_i ? (mmio_dir ? o2a_reg_rsp.ready : a2o_reg_rsp.ready) : 1'b1;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            rvalid_o <= 1'b0;
            rdata_o <= '0;
        end else begin
            rvalid_o <= req_i && gnt_o;
            rdata_o <= '0;
            if (req_i && gnt_o) begin
                rdata_o[31:0] <= mmio_dir ? o2a_reg_rsp.rdata : a2o_reg_rsp.rdata;
            end
        end
    end

    assign a2o_busy_arr[0] = a2o_busy;
    assign a2o_done_id_arr[0] = a2o_done_id;
    assign a2o_me_busy_arr[0] = a2o_me_busy;
    assign o2a_busy_arr[0] = o2a_busy;
    assign o2a_done_id_arr[0] = o2a_done_id;
    assign o2a_me_busy_arr[0] = o2a_me_busy;

    assign irq_a2o_busy_o = (|a2o_busy) || a2o_me_busy || a2o_fe_valid || a2o_be_req_valid;
    assign irq_a2o_start_o = a2o_front_valid && a2o_front_ready;
    assign irq_a2o_done_o = a2o_fe_rsp_valid;
    assign irq_a2o_error_o = a2o_fe_rsp_valid && a2o_be_rsp.error;
    assign irq_o2a_busy_o = (|o2a_busy) || o2a_me_busy || o2a_fe_valid || o2a_be_req_valid;
    assign irq_o2a_start_o = o2a_front_valid && o2a_front_ready;
    assign irq_o2a_done_o = o2a_fe_rsp_valid;
    assign irq_o2a_error_o = o2a_fe_rsp_valid && o2a_be_rsp.error;

    idma_reg32_3d #(
        .NumRegs        (1),
        .NumStreams     (1),
        .IdCounterWidth (32),
        .reg_req_t      (reg_req_t),
        .reg_rsp_t      (reg_rsp_t),
        .dma_req_t      (idma_nd_req_t),
        .cnt_width_t    (logic [31:0]),
        .stream_t       (logic)
    ) i_a2o_frontend (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .dma_ctrl_req_i (a2o_reg_req_arr),
        .dma_ctrl_rsp_o (a2o_reg_rsp_arr),
        .dma_req_o      (a2o_front_req),
        .req_valid_o    (a2o_front_valid),
        .req_ready_i    (a2o_front_ready),
        .next_id_i      (a2o_next_id),
        .stream_idx_o   (),
        .done_id_i      (a2o_done_id_arr),
        .busy_i         (a2o_busy_arr),
        .midend_busy_i  (a2o_me_busy_arr)
    );

    idma_transfer_id_gen #(
        .IdWidth (32)
    ) i_a2o_id_gen (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .issue_i     (a2o_front_valid && a2o_front_ready),
        .retire_i    (a2o_fe_rsp_valid),
        .next_o      (a2o_next_id),
        .completed_o (a2o_done_id)
    );

    assign a2o_front_ready = !a2o_fe_valid || a2o_fe_ready;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            a2o_fe_valid <= 1'b0;
            a2o_fe_req <= '0;
        end else begin
            if (a2o_front_valid && a2o_front_ready) begin
                a2o_fe_valid <= 1'b1;
                a2o_fe_req <= a2o_front_req;
            end else if (a2o_fe_valid && a2o_fe_ready) begin
                a2o_fe_valid <= 1'b0;
            end
        end
    end

    idma_nd_midend #(
        .NumDim        (NUM_DIMS),
        .addr_t        (logic [31:0]),
        .idma_req_t    (idma_req_t),
        .idma_rsp_t    (idma_rsp_t),
        .idma_nd_req_t (idma_nd_req_t),
        .RepWidths     (REP_WIDTHS)
    ) i_a2o_midend (
        .clk_i             (clk_i),
        .rst_ni            (rst_ni),
        .nd_req_i          (a2o_fe_req),
        .nd_req_valid_i    (a2o_fe_valid),
        .nd_req_ready_o    (a2o_fe_ready),
        .nd_rsp_o          (),
        .nd_rsp_valid_o    (a2o_fe_rsp_valid),
        .nd_rsp_ready_i    (1'b1),
        .burst_req_o       (a2o_be_req),
        .burst_req_valid_o (a2o_be_req_valid),
        .burst_req_ready_i (a2o_be_req_ready),
        .burst_rsp_i       (a2o_be_rsp),
        .burst_rsp_valid_i (a2o_be_rsp_valid),
        .burst_rsp_ready_o (a2o_be_rsp_ready),
        .busy_o            (a2o_me_busy)
    );

    logic a2o_rsp_error;
    logic a2o_rsp_last;
    axi_pkg::resp_t a2o_rsp_cause;
    idma_pkg::err_type_t a2o_rsp_err_type;
    logic [31:0] a2o_rsp_burst_addr;

    always_comb begin
        a2o_be_rsp = '0;
        a2o_be_rsp.last = a2o_rsp_last;
        a2o_be_rsp.error = a2o_rsp_error;
        a2o_be_rsp.pld.cause = a2o_rsp_cause;
        a2o_be_rsp.pld.err_type = a2o_rsp_err_type;
        a2o_be_rsp.pld.burst_addr = a2o_rsp_burst_addr;
    end

    idma_backend_synth_r_axi_w_obi #(
        .DataWidth           (DATA_WIDTH),
        .AddrWidth           (ADDR_WIDTH),
        .UserWidth           (1),
        .AxiIdWidth          (1),
        .NumAxInFlight       (16),
        .BufferDepth         (3),
        .TFLenWidth          (32),
        .MemSysDepth         (0),
        .CombinedShifter     (1'b0),
        .MaskInvalidData     (1'b1),
        .RAWCouplingAvail    (1'b0),
        .HardwareLegalizer   (1'b1),
        .RejectZeroTransfers (1'b1),
        .ErrorHandling       (1'b0)
    ) i_l2_to_l1_backend (
        .clk_i                  (clk_i),
        .rst_ni                 (rst_ni),
        .test_i                 (1'b0),
        .req_valid_i            (a2o_be_req_valid),
        .req_ready_o            (a2o_be_req_ready),
        .req_length_i           (a2o_be_req.length),
        .req_src_addr_i         (a2o_be_req.src_addr),
        .req_dst_addr_i         (a2o_be_req.dst_addr),
        .req_src_protocol_i     (a2o_be_req.opt.src_protocol),
        .req_dst_protocol_i     (a2o_be_req.opt.dst_protocol),
        .req_axi_id_i           (1'b0),
        .req_src_burst_i        (a2o_be_req.opt.src.burst),
        .req_src_cache_i        (a2o_be_req.opt.src.cache),
        .req_src_lock_i         (a2o_be_req.opt.src.lock),
        .req_src_prot_i         (a2o_be_req.opt.src.prot),
        .req_src_qos_i          (a2o_be_req.opt.src.qos),
        .req_src_region_i       (a2o_be_req.opt.src.region),
        .req_dst_burst_i        (a2o_be_req.opt.dst.burst),
        .req_dst_cache_i        (a2o_be_req.opt.dst.cache),
        .req_dst_lock_i         (a2o_be_req.opt.dst.lock),
        .req_dst_prot_i         (a2o_be_req.opt.dst.prot),
        .req_dst_qos_i          (a2o_be_req.opt.dst.qos),
        .req_dst_region_i       (a2o_be_req.opt.dst.region),
        .req_decouple_aw_i      (a2o_be_req.opt.beo.decouple_aw),
        .req_decouple_rw_i      (a2o_be_req.opt.beo.decouple_rw),
        .req_src_max_llen_i     (a2o_be_req.opt.beo.src_max_llen),
        .req_dst_max_llen_i     (a2o_be_req.opt.beo.dst_max_llen),
        .req_src_reduce_len_i   (a2o_be_req.opt.beo.src_reduce_len),
        .req_dst_reduce_len_i   (a2o_be_req.opt.beo.dst_reduce_len),
        .req_last_i             (a2o_be_req.opt.last),
        .rsp_valid_o            (a2o_be_rsp_valid),
        .rsp_ready_i            (a2o_be_rsp_ready),
        .rsp_cause_o            (a2o_rsp_cause),
        .rsp_err_type_o         (a2o_rsp_err_type),
        .rsp_burst_addr_o       (a2o_rsp_burst_addr),
        .rsp_error_o            (a2o_rsp_error),
        .rsp_last_o             (a2o_rsp_last),
        .eh_req_valid_i         (1'b0),
        .eh_req_ready_o         (),
        .eh_req_i               ('0),
        .axi_ar_id_o            (),
        .axi_ar_addr_o          (axi_ar_addr_o),
        .axi_ar_len_o           (axi_ar_len_o),
        .axi_ar_size_o          (axi_ar_size_o),
        .axi_ar_burst_o         (axi_ar_burst_o),
        .axi_ar_lock_o          (),
        .axi_ar_cache_o         (),
        .axi_ar_prot_o          (),
        .axi_ar_qos_o           (),
        .axi_ar_region_o        (),
        .axi_ar_user_o          (),
        .axi_ar_valid_o         (axi_ar_valid_o),
        .axi_ar_ready_i         (axi_ar_ready_i),
        .axi_r_id_i             (1'b0),
        .axi_r_data_i           (axi_r_data_i),
        .axi_r_resp_i           (axi_pkg::resp_t'(axi_r_resp_i)),
        .axi_r_last_i           (axi_r_last_i),
        .axi_r_user_i           (1'b0),
        .axi_r_valid_i          (axi_r_valid_i),
        .axi_r_ready_o          (axi_r_ready_o),
        .obi_write_req_a_req_o  (obi_write_req_o),
        .obi_write_req_a_addr_o (obi_write_addr_o),
        .obi_write_req_a_we_o   (obi_write_we_o),
        .obi_write_req_a_be_o   (obi_write_be_o),
        .obi_write_req_a_wdata_o(obi_write_wdata_o),
        .obi_write_req_a_aid_o  (),
        .obi_write_req_r_ready_o(),
        .obi_write_rsp_a_gnt_i  (obi_write_gnt_i),
        .obi_write_rsp_r_valid_i(obi_write_rvalid_i),
        .obi_write_rsp_r_rdata_i(obi_write_rdata_i),
        .idma_busy_o            (a2o_busy)
    );

    idma_reg32_3d #(
        .NumRegs        (1),
        .NumStreams     (1),
        .IdCounterWidth (32),
        .reg_req_t      (reg_req_t),
        .reg_rsp_t      (reg_rsp_t),
        .dma_req_t      (idma_nd_req_t),
        .cnt_width_t    (logic [31:0]),
        .stream_t       (logic)
    ) i_o2a_frontend (
        .clk_i          (clk_i),
        .rst_ni         (rst_ni),
        .dma_ctrl_req_i (o2a_reg_req_arr),
        .dma_ctrl_rsp_o (o2a_reg_rsp_arr),
        .dma_req_o      (o2a_front_req),
        .req_valid_o    (o2a_front_valid),
        .req_ready_i    (o2a_front_ready),
        .next_id_i      (o2a_next_id),
        .stream_idx_o   (),
        .done_id_i      (o2a_done_id_arr),
        .busy_i         (o2a_busy_arr),
        .midend_busy_i  (o2a_me_busy_arr)
    );

    idma_transfer_id_gen #(
        .IdWidth (32)
    ) i_o2a_id_gen (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .issue_i     (o2a_front_valid && o2a_front_ready),
        .retire_i    (o2a_fe_rsp_valid),
        .next_o      (o2a_next_id),
        .completed_o (o2a_done_id)
    );

    assign o2a_front_ready = !o2a_fe_valid || o2a_fe_ready;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            o2a_fe_valid <= 1'b0;
            o2a_fe_req <= '0;
        end else begin
            if (o2a_front_valid && o2a_front_ready) begin
                o2a_fe_valid <= 1'b1;
                o2a_fe_req <= o2a_front_req;
            end else if (o2a_fe_valid && o2a_fe_ready) begin
                o2a_fe_valid <= 1'b0;
            end
        end
    end

    idma_nd_midend #(
        .NumDim        (NUM_DIMS),
        .addr_t        (logic [31:0]),
        .idma_req_t    (idma_req_t),
        .idma_rsp_t    (idma_rsp_t),
        .idma_nd_req_t (idma_nd_req_t),
        .RepWidths     (REP_WIDTHS)
    ) i_o2a_midend (
        .clk_i             (clk_i),
        .rst_ni            (rst_ni),
        .nd_req_i          (o2a_fe_req),
        .nd_req_valid_i    (o2a_fe_valid),
        .nd_req_ready_o    (o2a_fe_ready),
        .nd_rsp_o          (),
        .nd_rsp_valid_o    (o2a_fe_rsp_valid),
        .nd_rsp_ready_i    (1'b1),
        .burst_req_o       (o2a_be_req),
        .burst_req_valid_o (o2a_be_req_valid),
        .burst_req_ready_i (o2a_be_req_ready),
        .burst_rsp_i       (o2a_be_rsp),
        .burst_rsp_valid_i (o2a_be_rsp_valid),
        .burst_rsp_ready_o (o2a_be_rsp_ready),
        .busy_o            (o2a_me_busy)
    );

    logic o2a_rsp_error;
    logic o2a_rsp_last;
    axi_pkg::resp_t o2a_rsp_cause;
    idma_pkg::err_type_t o2a_rsp_err_type;
    logic [31:0] o2a_rsp_burst_addr;

    always_comb begin
        o2a_be_rsp = '0;
        o2a_be_rsp.last = o2a_rsp_last;
        o2a_be_rsp.error = o2a_rsp_error;
        o2a_be_rsp.pld.cause = o2a_rsp_cause;
        o2a_be_rsp.pld.err_type = o2a_rsp_err_type;
        o2a_be_rsp.pld.burst_addr = o2a_rsp_burst_addr;
    end

    idma_backend_synth_r_obi_w_axi #(
        .DataWidth           (DATA_WIDTH),
        .AddrWidth           (ADDR_WIDTH),
        .UserWidth           (1),
        .AxiIdWidth          (1),
        .NumAxInFlight       (16),
        .BufferDepth         (3),
        .TFLenWidth          (32),
        .MemSysDepth         (0),
        .CombinedShifter     (1'b0),
        .MaskInvalidData     (1'b1),
        .RAWCouplingAvail    (1'b0),
        .HardwareLegalizer   (1'b1),
        .RejectZeroTransfers (1'b1),
        .ErrorHandling       (1'b0)
    ) i_l1_to_l2_backend (
        .clk_i                 (clk_i),
        .rst_ni                (rst_ni),
        .test_i                (1'b0),
        .req_valid_i           (o2a_be_req_valid),
        .req_ready_o           (o2a_be_req_ready),
        .req_length_i          (o2a_be_req.length),
        .req_src_addr_i        (o2a_be_req.src_addr),
        .req_dst_addr_i        (o2a_be_req.dst_addr),
        .req_src_protocol_i    (o2a_be_req.opt.src_protocol),
        .req_dst_protocol_i    (o2a_be_req.opt.dst_protocol),
        .req_axi_id_i          (1'b0),
        .req_src_burst_i       (o2a_be_req.opt.src.burst),
        .req_src_cache_i       (o2a_be_req.opt.src.cache),
        .req_src_lock_i        (o2a_be_req.opt.src.lock),
        .req_src_prot_i        (o2a_be_req.opt.src.prot),
        .req_src_qos_i         (o2a_be_req.opt.src.qos),
        .req_src_region_i      (o2a_be_req.opt.src.region),
        .req_dst_burst_i       (o2a_be_req.opt.dst.burst),
        .req_dst_cache_i       (o2a_be_req.opt.dst.cache),
        .req_dst_lock_i        (o2a_be_req.opt.dst.lock),
        .req_dst_prot_i        (o2a_be_req.opt.dst.prot),
        .req_dst_qos_i         (o2a_be_req.opt.dst.qos),
        .req_dst_region_i      (o2a_be_req.opt.dst.region),
        .req_decouple_aw_i     (o2a_be_req.opt.beo.decouple_aw),
        .req_decouple_rw_i     (o2a_be_req.opt.beo.decouple_rw),
        .req_src_max_llen_i    (o2a_be_req.opt.beo.src_max_llen),
        .req_dst_max_llen_i    (o2a_be_req.opt.beo.dst_max_llen),
        .req_src_reduce_len_i  (o2a_be_req.opt.beo.src_reduce_len),
        .req_dst_reduce_len_i  (o2a_be_req.opt.beo.dst_reduce_len),
        .req_last_i            (o2a_be_req.opt.last),
        .rsp_valid_o           (o2a_be_rsp_valid),
        .rsp_ready_i           (o2a_be_rsp_ready),
        .rsp_cause_o           (o2a_rsp_cause),
        .rsp_err_type_o        (o2a_rsp_err_type),
        .rsp_burst_addr_o      (o2a_rsp_burst_addr),
        .rsp_error_o           (o2a_rsp_error),
        .rsp_last_o            (o2a_rsp_last),
        .eh_req_valid_i        (1'b0),
        .eh_req_ready_o        (),
        .eh_req_i              ('0),
        .obi_read_req_a_req_o  (obi_read_req_o),
        .obi_read_req_a_addr_o (obi_read_addr_o),
        .obi_read_req_a_we_o   (obi_read_we_o),
        .obi_read_req_a_be_o   (obi_read_be_o),
        .obi_read_req_a_wdata_o(obi_read_wdata_o),
        .obi_read_req_r_ready_o(),
        .obi_read_rsp_a_gnt_i  (obi_read_gnt_i),
        .obi_read_rsp_r_valid_i(obi_read_rvalid_i),
        .obi_read_rsp_r_rdata_i(obi_read_rdata_i),
        .obi_read_rsp_r_rid_i  (1'b0),
        .obi_read_rsp_r_err_i  (1'b0),
        .axi_aw_id_o           (),
        .axi_aw_addr_o         (axi_aw_addr_o),
        .axi_aw_len_o          (axi_aw_len_o),
        .axi_aw_size_o         (axi_aw_size_o),
        .axi_aw_burst_o        (axi_aw_burst_o),
        .axi_aw_lock_o         (),
        .axi_aw_cache_o        (),
        .axi_aw_prot_o         (),
        .axi_aw_qos_o          (),
        .axi_aw_region_o       (),
        .axi_aw_atop_o         (),
        .axi_aw_user_o         (),
        .axi_aw_valid_o        (axi_aw_valid_o),
        .axi_aw_ready_i        (axi_aw_ready_i),
        .axi_w_data_o          (axi_w_data_o),
        .axi_w_strb_o          (axi_w_strb_o),
        .axi_w_last_o          (axi_w_last_o),
        .axi_w_user_o          (),
        .axi_w_valid_o         (axi_w_valid_o),
        .axi_w_ready_i         (axi_w_ready_i),
        .axi_b_id_i            (1'b0),
        .axi_b_resp_i          (axi_pkg::resp_t'(axi_b_resp_i)),
        .axi_b_user_i          (1'b0),
        .axi_b_valid_i         (axi_b_valid_i),
        .axi_b_ready_o         (axi_b_ready_o),
        .idma_busy_o           (o2a_busy)
    );

endmodule
