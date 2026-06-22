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

    localparam int unsigned STRB_WIDTH = DATA_WIDTH / 8;
    localparam int unsigned WORDS_PER_BEAT = DATA_WIDTH / 32;
    localparam logic [11:0] DIR_OFFSET       = 12'h200;
    localparam logic [11:0] REG_CONF         = 12'h000;
    localparam logic [11:0] REG_STATUS_0     = 12'h004;
    localparam logic [11:0] REG_NEXT_ID_0    = 12'h044;
    localparam logic [11:0] REG_DONE_ID_0    = 12'h084;
    localparam logic [11:0] REG_DST_ADDR_LOW = 12'h0D0;
    localparam logic [11:0] REG_SRC_ADDR_LOW = 12'h0D8;
    localparam logic [11:0] REG_LENGTH_LOW   = 12'h0E0;
    localparam logic [11:0] REG_DST_STRIDE_2 = 12'h0E8;
    localparam logic [11:0] REG_SRC_STRIDE_2 = 12'h0F0;
    localparam logic [11:0] REG_REPS_2       = 12'h0F8;
    localparam logic [11:0] REG_DST_STRIDE_3 = 12'h100;
    localparam logic [11:0] REG_SRC_STRIDE_3 = 12'h108;
    localparam logic [11:0] REG_REPS_3       = 12'h110;

    typedef enum logic {
        DIR_AXI2OBI = 1'b0,
        DIR_OBI2AXI = 1'b1
    } dir_e;

    logic [1:0][31:0] conf_q;
    logic [1:0][31:0] src_addr_q;
    logic [1:0][31:0] dst_addr_q;
    logic [1:0][31:0] length_q;
    logic [1:0][31:0] dst_stride_2_q;
    logic [1:0][31:0] src_stride_2_q;
    logic [1:0][31:0] reps_2_q;
    logic [1:0][31:0] dst_stride_3_q;
    logic [1:0][31:0] src_stride_3_q;
    logic [1:0][31:0] reps_3_q;
    logic [1:0][31:0] next_id_q;
    logic [1:0][31:0] done_id_q;
    logic [1:0][31:0] active_id_q;
    logic [1:0]       req_pending_q;
    logic [31:0]      r_addr_q;

    logic a2o_req_ready;
    logic a2o_rsp_valid;
    logic a2o_rsp_error;
    logic a2o_rsp_last;
    logic o2a_req_ready;
    logic o2a_rsp_valid;
    logic o2a_rsp_error;
    logic o2a_rsp_last;
    idma_pkg::idma_busy_t a2o_busy;
    idma_pkg::idma_busy_t o2a_busy;

    logic a2o_done;
    logic o2a_done;
    logic a2o_accept;
    logic o2a_accept;
    logic start_req;
    logic start_dir;

    assign gnt_o = 1'b1;
    assign a2o_accept = req_pending_q[DIR_AXI2OBI] && a2o_req_ready;
    assign o2a_accept = req_pending_q[DIR_OBI2AXI] && o2a_req_ready;
    assign a2o_done = a2o_rsp_valid && a2o_rsp_last;
    assign o2a_done = o2a_rsp_valid && o2a_rsp_last;

    assign irq_a2o_busy_o  = req_pending_q[DIR_AXI2OBI] || (|a2o_busy);
    assign irq_a2o_start_o = a2o_accept;
    assign irq_a2o_done_o  = a2o_done;
    assign irq_a2o_error_o = a2o_rsp_valid && a2o_rsp_error;
    assign irq_o2a_busy_o  = req_pending_q[DIR_OBI2AXI] || (|o2a_busy);
    assign irq_o2a_start_o = o2a_accept;
    assign irq_o2a_done_o  = o2a_done;
    assign irq_o2a_error_o = o2a_rsp_valid && o2a_rsp_error;

    function automatic logic decode_dir(input logic [31:0] exact_addr);
        decode_dir = (exact_addr >= (BASE_ADDR + DIR_OFFSET));
    endfunction

    function automatic logic [11:0] decode_offset(input logic [31:0] exact_addr);
        logic direction;
        begin
            direction = decode_dir(exact_addr);
            decode_offset = direction
                ? exact_addr[11:0] - BASE_ADDR[11:0] - DIR_OFFSET
                : exact_addr[11:0] - BASE_ADDR[11:0];
        end
    endfunction

    function automatic logic can_start_addr(input logic [31:0] exact_addr);
        logic direction;
        begin
            direction = decode_dir(exact_addr);
            can_start_addr = (decode_offset(exact_addr) == REG_NEXT_ID_0) &&
                             !req_pending_q[direction] &&
                             (length_q[direction] != 32'd0);
        end
    endfunction

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            conf_q         <= '0;
            src_addr_q     <= '0;
            dst_addr_q     <= '0;
            length_q       <= '0;
            dst_stride_2_q <= '0;
            src_stride_2_q <= '0;
            reps_2_q       <= {32'd1, 32'd1};
            dst_stride_3_q <= '0;
            src_stride_3_q <= '0;
            reps_3_q       <= {32'd1, 32'd1};
            next_id_q      <= {32'd1, 32'd1};
            done_id_q      <= '0;
            active_id_q    <= '0;
            req_pending_q  <= '0;
            r_addr_q       <= '0;
            rvalid_o       <= 1'b0;
        end else begin
            if (a2o_accept) begin
                req_pending_q[DIR_AXI2OBI] <= 1'b0;
                next_id_q[DIR_AXI2OBI] <= next_id_q[DIR_AXI2OBI] + 32'd1;
            end
            if (o2a_accept) begin
                req_pending_q[DIR_OBI2AXI] <= 1'b0;
                next_id_q[DIR_OBI2AXI] <= next_id_q[DIR_OBI2AXI] + 32'd1;
            end

            if (a2o_done) begin
                done_id_q[DIR_AXI2OBI] <= active_id_q[DIR_AXI2OBI];
            end
            if (o2a_done) begin
                done_id_q[DIR_OBI2AXI] <= active_id_q[DIR_OBI2AXI];
            end

            if (start_req) begin
                req_pending_q[start_dir] <= 1'b1;
                active_id_q[start_dir] <= next_id_q[start_dir];
                done_id_q[start_dir] <= 32'd0;
            end

            if (req_i && gnt_o) begin
                if (we_i) begin
                    for (int i = 0; i < WORDS_PER_BEAT; i++) begin
                        if (be_i[i*4 +: 4] != 4'b0000) begin
                            logic [31:0] exact_addr;
                            logic        direction;
                            logic [11:0] offset;
                            logic [31:0] wdata_word;

                            exact_addr = (addr_i & 32'hFFFF_FFE0) + (i * 4);
                            direction = decode_dir(exact_addr);
                            offset = decode_offset(exact_addr);
                            wdata_word = wdata_i[i*32 +: 32];

                            case (offset)
                                REG_CONF:         conf_q[direction]         <= wdata_word;
                                REG_DST_ADDR_LOW: dst_addr_q[direction]     <= wdata_word;
                                REG_SRC_ADDR_LOW: src_addr_q[direction]     <= wdata_word;
                                REG_LENGTH_LOW:   length_q[direction]       <= wdata_word;
                                REG_DST_STRIDE_2: dst_stride_2_q[direction] <= wdata_word;
                                REG_SRC_STRIDE_2: src_stride_2_q[direction] <= wdata_word;
                                REG_REPS_2:       reps_2_q[direction]       <= wdata_word;
                                REG_DST_STRIDE_3: dst_stride_3_q[direction] <= wdata_word;
                                REG_SRC_STRIDE_3: src_stride_3_q[direction] <= wdata_word;
                                REG_REPS_3:       reps_3_q[direction]       <= wdata_word;
                                default: ;
                            endcase
                        end
                    end
                end else begin
                    r_addr_q <= addr_i & 32'hFFFF_FFE0;
                end
                rvalid_o <= 1'b1;
            end else begin
                rvalid_o <= 1'b0;
            end
        end
    end

    always_comb begin
        start_req = 1'b0;
        start_dir = DIR_AXI2OBI;

        if (req_i && gnt_o && !we_i) begin
            for (int i = 0; i < WORDS_PER_BEAT; i++) begin
                if (can_start_addr((addr_i & 32'hFFFF_FFE0) + (i * 4))) begin
                    start_req = 1'b1;
                    start_dir = decode_dir((addr_i & 32'hFFFF_FFE0) + (i * 4));
                end
            end
        end
    end

    always_comb begin
        rdata_o = '0;
        if (rvalid_o) begin
            for (int i = 0; i < WORDS_PER_BEAT; i++) begin
                logic [31:0] exact_addr;
                logic        direction;
                logic [11:0] offset;
                logic [31:0] rdata_word;

                exact_addr = (r_addr_q & 32'hFFFF_FFE0) + (i * 4);
                direction = decode_dir(exact_addr);
                offset = decode_offset(exact_addr);
                rdata_word = '0;

                case (offset)
                    REG_CONF:         rdata_word = conf_q[direction];
                    REG_STATUS_0:     rdata_word = {31'd0, direction ? irq_o2a_busy_o : irq_a2o_busy_o};
                    REG_NEXT_ID_0:    rdata_word = next_id_q[direction];
                    REG_DONE_ID_0:    rdata_word = done_id_q[direction];
                    REG_DST_ADDR_LOW: rdata_word = dst_addr_q[direction];
                    REG_SRC_ADDR_LOW: rdata_word = src_addr_q[direction];
                    REG_LENGTH_LOW:   rdata_word = length_q[direction];
                    REG_DST_STRIDE_2: rdata_word = dst_stride_2_q[direction];
                    REG_SRC_STRIDE_2: rdata_word = src_stride_2_q[direction];
                    REG_REPS_2:       rdata_word = reps_2_q[direction];
                    REG_DST_STRIDE_3: rdata_word = dst_stride_3_q[direction];
                    REG_SRC_STRIDE_3: rdata_word = src_stride_3_q[direction];
                    REG_REPS_3:       rdata_word = reps_3_q[direction];
                    default:          rdata_word = 32'h0;
                endcase

                rdata_o[i*32 +: 32] = rdata_word;
            end
        end
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
    ) i_l2_to_l1 (
        .clk_i                  (clk_i),
        .rst_ni                 (rst_ni),
        .test_i                 (1'b0),
        .req_valid_i            (req_pending_q[DIR_AXI2OBI]),
        .req_ready_o            (a2o_req_ready),
        .req_length_i           (length_q[DIR_AXI2OBI]),
        .req_src_addr_i         (src_addr_q[DIR_AXI2OBI]),
        .req_dst_addr_i         (dst_addr_q[DIR_AXI2OBI]),
        .req_src_protocol_i     (idma_pkg::AXI),
        .req_dst_protocol_i     (idma_pkg::OBI),
        .req_axi_id_i           (1'b0),
        .req_src_burst_i        (axi_pkg::BURST_INCR),
        .req_src_cache_i        (axi_pkg::CACHE_MODIFIABLE),
        .req_src_lock_i         (1'b0),
        .req_src_prot_i         ('0),
        .req_src_qos_i          ('0),
        .req_src_region_i       ('0),
        .req_dst_burst_i        (axi_pkg::BURST_INCR),
        .req_dst_cache_i        (axi_pkg::CACHE_MODIFIABLE),
        .req_dst_lock_i         (1'b0),
        .req_dst_prot_i         ('0),
        .req_dst_qos_i          ('0),
        .req_dst_region_i       ('0),
        .req_decouple_aw_i      (1'b0),
        .req_decouple_rw_i      (1'b0),
        .req_src_max_llen_i     (3'd4),
        .req_dst_max_llen_i     (3'd4),
        .req_src_reduce_len_i   (1'b0),
        .req_dst_reduce_len_i   (1'b0),
        .req_last_i             (1'b1),
        .rsp_valid_o            (a2o_rsp_valid),
        .rsp_ready_i            (1'b1),
        .rsp_cause_o            (),
        .rsp_err_type_o         (),
        .rsp_burst_addr_o       (),
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
    ) i_l1_to_l2 (
        .clk_i                 (clk_i),
        .rst_ni                (rst_ni),
        .test_i                (1'b0),
        .req_valid_i           (req_pending_q[DIR_OBI2AXI]),
        .req_ready_o           (o2a_req_ready),
        .req_length_i          (length_q[DIR_OBI2AXI]),
        .req_src_addr_i        (src_addr_q[DIR_OBI2AXI]),
        .req_dst_addr_i        (dst_addr_q[DIR_OBI2AXI]),
        .req_src_protocol_i    (idma_pkg::OBI),
        .req_dst_protocol_i    (idma_pkg::AXI),
        .req_axi_id_i          (1'b0),
        .req_src_burst_i       (axi_pkg::BURST_INCR),
        .req_src_cache_i       (axi_pkg::CACHE_MODIFIABLE),
        .req_src_lock_i        (1'b0),
        .req_src_prot_i        ('0),
        .req_src_qos_i         ('0),
        .req_src_region_i      ('0),
        .req_dst_burst_i       (axi_pkg::BURST_INCR),
        .req_dst_cache_i       (axi_pkg::CACHE_MODIFIABLE),
        .req_dst_lock_i        (1'b0),
        .req_dst_prot_i        ('0),
        .req_dst_qos_i         ('0),
        .req_dst_region_i      ('0),
        .req_decouple_aw_i     (1'b0),
        .req_decouple_rw_i     (1'b0),
        .req_src_max_llen_i    (3'd4),
        .req_dst_max_llen_i    (3'd4),
        .req_src_reduce_len_i  (1'b0),
        .req_dst_reduce_len_i  (1'b0),
        .req_last_i            (1'b1),
        .rsp_valid_o           (o2a_rsp_valid),
        .rsp_ready_i           (1'b1),
        .rsp_cause_o           (),
        .rsp_err_type_o        (),
        .rsp_burst_addr_o      (),
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
