`default_nettype none

module systolic_controller #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned CFG_DATA_WIDTH = 32,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned OFM_ELEM_WIDTH = 32,
    parameter int unsigned INPUT_FIFO_DEPTH = 4,
    parameter int unsigned OFM_FIFO_DEPTH = 128
)(
    input  logic clk_i,
    input  logic rst_ni,

    // MMIO slave for integrated systolic/requant register block.
    input  logic                          ctrl_req_i,
    output logic                          ctrl_gnt_o,
    input  logic [ADDR_WIDTH-1:0]         ctrl_addr_i,
    input  logic                          ctrl_we_i,
    input  logic [(CFG_DATA_WIDTH/8)-1:0] ctrl_be_i,
    input  logic [CFG_DATA_WIDTH-1:0]     ctrl_wdata_i,
    output logic                          ctrl_rvalid_o,
    output logic [CFG_DATA_WIDTH-1:0]     ctrl_rdata_o,

    // Completion pulse to cluster interrupt controller.
    output logic                      cfg_sys_done_o,

    // 1x OBI Master for I-TCDM (Read Weights & IFM)
    output logic                      obi_i_req_o,
    input  logic                      obi_i_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_i_addr_o,
    output logic                      obi_i_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_i_be_o,
    output logic [DATA_WIDTH-1:0]     obi_i_wdata_o,
    input  logic                      obi_i_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_i_rdata_i,


    // 4x OBI Masters for O-TCDM (Write OFM)
    output logic [3:0]                obi_o_req_o,
    input  logic [3:0]                obi_o_gnt_i,
    output logic [3:0][ADDR_WIDTH-1:0]obi_o_addr_o,
    output logic [3:0]                obi_o_we_o,
    output logic [3:0][(DATA_WIDTH/8)-1:0] obi_o_be_o,
    output logic [3:0][DATA_WIDTH-1:0]obi_o_wdata_o,
    input  logic [3:0]                obi_o_rvalid_i,
    input  logic [3:0][DATA_WIDTH-1:0]obi_o_rdata_i,

    // Performance/debug pulses exported to PMU. The array is local.
    output logic                      perf_weight_load_en_o,
    output logic                      perf_compute_en_o,
    output logic                      perf_ofm_valid_o,
    output logic                      perf_ofm_ready_o
);

    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        COMPUTE,
        WAIT_DRAIN,
        ACCUM_READ,
        ACCUM_WRITE,
        ACCUM_REQUANT,
        DONE
    } state_e;

    state_e state_q, state_d;

    logic [31:0] w_ptr_q, w_ptr_d;
    logic [31:0] i_ptr_q, i_ptr_d;
    logic [31:0] o_ptr_q, o_ptr_d;
    logic [31:0] a_ptr_q, a_ptr_d;
    logic [31:0] req_cnt_q, req_cnt_d; // Counter for requests
    logic [31:0] rsp_cnt_q, rsp_cnt_d; // Counter for responses
    logic [31:0] drain_cnt_q, drain_cnt_d; // Counter for valid outputs

    localparam int unsigned OFM_BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned OFM_ROW_BYTES = (ARRAY_DIM * OFM_ELEM_WIDTH) / 8;
    localparam int unsigned REQUANT_ROW_BYTES = ARRAY_DIM;
    localparam int unsigned OFM_ELEMS_PER_OBI = DATA_WIDTH / OFM_ELEM_WIDTH;
    localparam int unsigned OFM_ROW_BEATS = OFM_ROW_BYTES / OFM_BEAT_BYTES;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0]   ofm_row_t;

    localparam int unsigned OFM_FIFO_ADDR_DEPTH = (OFM_FIFO_DEPTH > 1) ? $clog2(OFM_FIFO_DEPTH) : 1;

    input_row_t    weight_fifo_data;
    input_row_t    weight_fifo_out;
    logic          weight_fifo_push;
    logic          weight_fifo_pop;
    logic          weight_fifo_full;
    logic          weight_fifo_empty;

    input_row_t    ifm_fifo_data;
    input_row_t    ifm_fifo_out;
    logic          ifm_fifo_push;
    logic          ifm_fifo_pop;
    logic          ifm_fifo_full;
    logic          ifm_fifo_empty;

    ofm_row_t      ofm_fifo_data;
    ofm_row_t      ofm_fifo_out;
    ofm_row_t      accum_row_q, accum_row_d;
    ofm_row_t      accum_sum;
    ofm_row_t      requant_acc;
    logic          ofm_fifo_push;
    logic          ofm_fifo_pop;
    logic          ofm_fifo_full;
    logic          ofm_fifo_empty;
    logic [OFM_FIFO_ADDR_DEPTH-1:0] ofm_fifo_usage;
    logic          array_pipe_ready;

    logic          fifo_flush;
    logic          weight_load_en;
    logic          clear_acc;
    logic          compute_en;
    input_row_t    weight_data;
    input_row_t    ifm_data;
    ofm_row_t      psum_data;
    ofm_row_t      ofm_data;
    logic          ofm_valid;
    logic          ofm_ready;
    logic          requant_in_valid;
    logic          requant_in_ready;
    logic          requant_out_valid;
    logic          requant_out_ready;
    logic [255:0]  requant_packed_data;
    logic          requant_invalid;
    logic          requant_config_invalid;
    logic [$clog2(OFM_ROW_BEATS+1)-1:0] accum_beat_q, accum_beat_d;
    logic          accum_requant_sent_q, accum_requant_sent_d;
    logic          cfg_sys_start_i;
    logic [31:0]   cfg_sys_weight_ptr_i;
    logic [31:0]   cfg_sys_ifm_ptr_i;
    logic [31:0]   cfg_sys_ofm_ptr_i;
    logic [31:0]   cfg_sys_psum_ptr_i;
    logic [31:0]   cfg_sys_dim_m_i;
    logic          cfg_sys_accum_en_i;
    logic          cfg_requant_en_i;
    logic [ARRAY_DIM-1:0][31:0] cfg_requant_bias_i;
    logic [ARRAY_DIM-1:0][31:0] cfg_requant_multiplier_i;
    logic [ARRAY_DIM-1:0][7:0] cfg_requant_shift_i;
    logic [ARRAY_DIM-1:0][31:0] cfg_requant_zero_point_i;
    logic [31:0]   cfg_requant_clamp_min_i;
    logic [31:0]   cfg_requant_clamp_max_i;
    logic          cfg_linebuf_en_i;
    logic [31:0]   cfg_linebuf_input_base_i;
    logic [15:0]   cfg_linebuf_input_h_i;
    logic [15:0]   cfg_linebuf_input_w_i;
    logic [15:0]   cfg_linebuf_input_c_i;
    logic [15:0]   cfg_linebuf_output_w_i;
    logic [15:0]   cfg_linebuf_stride_h_i;
    logic [15:0]   cfg_linebuf_stride_w_i;
    logic [15:0]   cfg_linebuf_pad_h_i;
    logic [15:0]   cfg_linebuf_pad_w_i;
    logic [15:0]   cfg_linebuf_tile_oh_base_i;
    logic [15:0]   cfg_linebuf_tile_ow_base_i;
    logic [31:0]   cfg_linebuf_lane_valid_i;
    logic [ARRAY_DIM-1:0][7:0] cfg_linebuf_lane_kh_i;
    logic [ARRAY_DIM-1:0][7:0] cfg_linebuf_lane_kw_i;
    logic [ARRAY_DIM-1:0][15:0] cfg_linebuf_lane_ic_i;

    logic          linebuf_start;
    logic          linebuf_obi_req;
    logic [ADDR_WIDTH-1:0] linebuf_obi_addr;
    input_row_t    linebuf_row_data;
    logic          linebuf_row_valid;
    logic          linebuf_row_ready;
    logic          linebuf_done;
    logic          linebuf_busy;
    logic [31:0]   linebuf_cache_hits;
    logic [31:0]   linebuf_cache_misses;

    assign fifo_flush = (state_q == IDLE) && cfg_sys_start_i;

    assign weight_fifo_data = obi_i_rdata_i;
    assign ifm_fifo_data    = obi_i_rdata_i;
    assign ofm_fifo_data    = ofm_data;
    assign ofm_ready = !ofm_fifo_full;
    assign array_pipe_ready = !ofm_valid || ofm_ready;
    assign psum_data = '0;
    assign perf_weight_load_en_o = weight_load_en;
    assign perf_compute_en_o = compute_en;
    assign perf_ofm_valid_o = ofm_valid;
    assign perf_ofm_ready_o = ofm_ready;

    always_comb begin
        for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
            accum_sum[ch] = ofm_fifo_out[ch] + accum_row_q[ch];
        end
    end

    assign requant_acc = (state_q == ACCUM_REQUANT) ? accum_sum : ofm_fifo_out;

    always_comb begin
        requant_config_invalid = ($signed(cfg_requant_clamp_min_i) > $signed(cfg_requant_clamp_max_i));
        for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
            if (cfg_requant_shift_i[ch] > 8'd31) begin
                requant_config_invalid = 1'b1;
            end
        end
    end

    requant_pipeline #(
        .ARRAY_DIM(ARRAY_DIM)
    ) i_requant_pipeline (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .in_valid_i    (requant_in_valid),
        .in_ready_o    (requant_in_ready),
        .acc_i         (requant_acc),
        .bias_i        (cfg_requant_bias_i),
        .multiplier_i  (cfg_requant_multiplier_i),
        .shift_i       (cfg_requant_shift_i),
        .zero_point_i  (cfg_requant_zero_point_i),
        .clamp_min_i   (cfg_requant_clamp_min_i),
        .clamp_max_i   (cfg_requant_clamp_max_i),
        .out_valid_o   (requant_out_valid),
        .out_ready_i   (requant_out_ready),
        .packed_o      (requant_packed_data),
        .invalid_o     (requant_invalid)
    );

    npu_systolic_array #(
        .ARRAY_DIM(ARRAY_DIM)
    ) i_systolic_array (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .weight_load_en_i (weight_load_en),
        .clear_acc_i      (clear_acc),
        .compute_en_i     (compute_en),
        .ofm_ready_i      (ofm_ready),
        .weight_data_i    (weight_data),
        .ifm_data_i       (ifm_data),
        .psum_data_i      (psum_data),
        .ofm_data_o       (ofm_data),
        .ofm_valid_o      (ofm_valid)
    );

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (INPUT_FIFO_DEPTH),
        .dtype        (input_row_t)
    ) i_weight_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (fifo_flush),
        .testmode_i (1'b0),
        .full_o     (weight_fifo_full),
        .empty_o    (weight_fifo_empty),
        .usage_o    (),
        .data_i     (weight_fifo_data),
        .push_i     (weight_fifo_push),
        .data_o     (weight_fifo_out),
        .pop_i      (weight_fifo_pop)
    );

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (INPUT_FIFO_DEPTH),
        .dtype        (input_row_t)
    ) i_ifm_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (fifo_flush),
        .testmode_i (1'b0),
        .full_o     (ifm_fifo_full),
        .empty_o    (ifm_fifo_empty),
        .usage_o    (),
        .data_i     (ifm_fifo_data),
        .push_i     (ifm_fifo_push),
        .data_o     (ifm_fifo_out),
        .pop_i      (ifm_fifo_pop)
    );

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (OFM_FIFO_DEPTH),
        .dtype        (ofm_row_t)
    ) i_ofm_fifo (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),
        .flush_i    (fifo_flush),
        .testmode_i (1'b0),
        .full_o     (ofm_fifo_full),
        .empty_o    (ofm_fifo_empty),
        .usage_o    (ofm_fifo_usage),
        .data_i     (ofm_fifo_data),
        .push_i     (ofm_fifo_push),
        .data_o     (ofm_fifo_out),
        .pop_i      (ofm_fifo_pop)
    );

    systolic_ctrl_regs #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(CFG_DATA_WIDTH)
    ) i_systolic_ctrl_regs (
        .clk_i              (clk_i),
        .rst_ni             (rst_ni),
        .req_i              (ctrl_req_i),
        .gnt_o              (ctrl_gnt_o),
        .addr_i             (ctrl_addr_i),
        .we_i               (ctrl_we_i),
        .be_i               (ctrl_be_i),
        .wdata_i            (ctrl_wdata_i),
        .rvalid_o           (ctrl_rvalid_o),
        .rdata_o            (ctrl_rdata_o),
        .cfg_sys_start_o    (cfg_sys_start_i),
        .cfg_sys_weight_ptr_o(cfg_sys_weight_ptr_i),
        .cfg_sys_ifm_ptr_o  (cfg_sys_ifm_ptr_i),
        .cfg_sys_ofm_ptr_o  (cfg_sys_ofm_ptr_i),
        .cfg_sys_psum_ptr_o (cfg_sys_psum_ptr_i),
        .cfg_sys_dim_m_o    (cfg_sys_dim_m_i),
        .cfg_sys_accum_en_o (cfg_sys_accum_en_i),
        .cfg_requant_en_o   (cfg_requant_en_i),
        .cfg_requant_bias_o (cfg_requant_bias_i),
        .cfg_requant_multiplier_o(cfg_requant_multiplier_i),
        .cfg_requant_shift_o(cfg_requant_shift_i),
        .cfg_requant_zero_point_o(cfg_requant_zero_point_i),
        .cfg_requant_clamp_min_o(cfg_requant_clamp_min_i),
        .cfg_requant_clamp_max_o(cfg_requant_clamp_max_i),
        .cfg_linebuf_en_o   (cfg_linebuf_en_i),
        .cfg_linebuf_input_base_o(cfg_linebuf_input_base_i),
        .cfg_linebuf_input_h_o(cfg_linebuf_input_h_i),
        .cfg_linebuf_input_w_o(cfg_linebuf_input_w_i),
        .cfg_linebuf_input_c_o(cfg_linebuf_input_c_i),
        .cfg_linebuf_output_w_o(cfg_linebuf_output_w_i),
        .cfg_linebuf_stride_h_o(cfg_linebuf_stride_h_i),
        .cfg_linebuf_stride_w_o(cfg_linebuf_stride_w_i),
        .cfg_linebuf_pad_h_o(cfg_linebuf_pad_h_i),
        .cfg_linebuf_pad_w_o(cfg_linebuf_pad_w_i),
        .cfg_linebuf_tile_oh_base_o(cfg_linebuf_tile_oh_base_i),
        .cfg_linebuf_tile_ow_base_o(cfg_linebuf_tile_ow_base_i),
        .cfg_linebuf_lane_valid_o(cfg_linebuf_lane_valid_i),
        .cfg_linebuf_lane_kh_o(cfg_linebuf_lane_kh_i),
        .cfg_linebuf_lane_kw_o(cfg_linebuf_lane_kw_i),
        .cfg_linebuf_lane_ic_o(cfg_linebuf_lane_ic_i),
        .cfg_sys_done_i     (cfg_sys_done_o)
    );

    conv_linebuf_packer #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH),
        .CACHE_ENTRIES    (128)
    ) i_conv_linebuf_packer (
        .clk_i                (clk_i),
        .rst_ni               (rst_ni),
        .start_i              (linebuf_start),
        .dim_m_i              (cfg_sys_dim_m_i),
        .cfg_input_base_i     (cfg_linebuf_input_base_i),
        .cfg_input_h_i        (cfg_linebuf_input_h_i),
        .cfg_input_w_i        (cfg_linebuf_input_w_i),
        .cfg_input_c_i        (cfg_linebuf_input_c_i),
        .cfg_output_w_i       (cfg_linebuf_output_w_i),
        .cfg_stride_h_i       (cfg_linebuf_stride_h_i),
        .cfg_stride_w_i       (cfg_linebuf_stride_w_i),
        .cfg_pad_h_i          (cfg_linebuf_pad_h_i),
        .cfg_pad_w_i          (cfg_linebuf_pad_w_i),
        .cfg_tile_oh_base_i   (cfg_linebuf_tile_oh_base_i),
        .cfg_tile_ow_base_i   (cfg_linebuf_tile_ow_base_i),
        .cfg_lane_valid_i     (cfg_linebuf_lane_valid_i),
        .cfg_lane_kh_i        (cfg_linebuf_lane_kh_i),
        .cfg_lane_kw_i        (cfg_linebuf_lane_kw_i),
        .cfg_lane_ic_i        (cfg_linebuf_lane_ic_i),
        .obi_req_o            (linebuf_obi_req),
        .obi_gnt_i            (obi_i_gnt_i),
        .obi_addr_o           (linebuf_obi_addr),
        .obi_rvalid_i         (obi_i_rvalid_i),
        .obi_rdata_i          (obi_i_rdata_i),
        .row_data_o           (linebuf_row_data),
        .row_valid_o          (linebuf_row_valid),
        .row_ready_i          (linebuf_row_ready),
        .busy_o               (linebuf_busy),
        .done_o               (linebuf_done),
        .cache_hits_o         (linebuf_cache_hits),
        .cache_misses_o       (linebuf_cache_misses)
    );

    // Tie off unused
    assign psum_data = '0;
    assign obi_i_we_o = 1'b0;
    assign obi_i_be_o = '1;
    assign obi_i_wdata_o = '0;

    // 4 OBI write ports are always write-only
    for (genvar i = 0; i < 4; i++) begin : gen_obi_o
        assign obi_o_we_o[i] = 1'b1;
        assign obi_o_be_o[i] = '1;
    end

    // FSM
    always_comb begin
        state_d = state_q;
        w_ptr_d = w_ptr_q;
        i_ptr_d = i_ptr_q;
        o_ptr_d = o_ptr_q;
        a_ptr_d = a_ptr_q;
        req_cnt_d   = req_cnt_q;
        rsp_cnt_d   = rsp_cnt_q;
        drain_cnt_d = drain_cnt_q;
        accum_row_d = accum_row_q;
        accum_beat_d = accum_beat_q;
        accum_requant_sent_d = accum_requant_sent_q;
        linebuf_start = 1'b0;
        linebuf_row_ready = 1'b0;

        cfg_sys_done_o = 1'b0;

        obi_i_req_o = 1'b0;
        obi_i_addr_o = '0;

        weight_load_en = 1'b0;
        compute_en     = 1'b0;
        clear_acc      = 1'b0;
        weight_data    = '0;  // Only driven during LOAD_WEIGHTS
        ifm_data       = '0;  // Only driven during COMPUTE
        weight_fifo_push = 1'b0;
        weight_fifo_pop  = 1'b0;
        ifm_fifo_push    = 1'b0;
        ifm_fifo_pop     = 1'b0;
        ofm_fifo_pop     = 1'b0;
        requant_in_valid = 1'b0;
        requant_out_ready = 1'b0;
        ofm_fifo_push    = ((state_q == COMPUTE) || (state_q == WAIT_DRAIN) ||
                            (state_q == ACCUM_READ) || (state_q == ACCUM_WRITE) ||
                            (state_q == ACCUM_REQUANT)) &&
                            ofm_valid && ofm_ready;

        // OBI Output ports (Write)
        obi_o_req_o = '0;
        obi_o_addr_o[0] = o_ptr_q + 0;
        obi_o_addr_o[1] = o_ptr_q + OFM_BEAT_BYTES;
        obi_o_addr_o[2] = o_ptr_q + (2 * OFM_BEAT_BYTES);
        obi_o_addr_o[3] = o_ptr_q + (3 * OFM_BEAT_BYTES);
        
        obi_o_wdata_o[0] = (state_q == ACCUM_WRITE) ? accum_sum[OFM_ELEMS_PER_OBI-1:0] :
                                                     ofm_fifo_out[OFM_ELEMS_PER_OBI-1:0];
        obi_o_wdata_o[1] = (state_q == ACCUM_WRITE) ? accum_sum[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI] :
                                                     ofm_fifo_out[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
        obi_o_wdata_o[2] = (state_q == ACCUM_WRITE) ? accum_sum[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)] :
                                                     ofm_fifo_out[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
        obi_o_wdata_o[3] = (state_q == ACCUM_WRITE) ? accum_sum[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)] :
                                                     ofm_fifo_out[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];

        // Handle OFM writes through a configurable FIFO.  The FIFO absorbs
        // systolic output rows while O-TCDM write grants are backpressured.
        if (!cfg_sys_accum_en_i && cfg_requant_en_i) begin
            if (requant_out_valid) begin
                obi_o_req_o[0] = 1'b1;
                obi_o_wdata_o[0] = requant_packed_data;
                if (obi_o_gnt_i[0] || requant_invalid) begin
                    requant_out_ready = 1'b1;
                    o_ptr_d = o_ptr_q + REQUANT_ROW_BYTES;
                    drain_cnt_d = drain_cnt_q - 1;
                end
                if (requant_invalid) begin
                    obi_o_req_o[0] = 1'b0;
                end
            end
            if (!ofm_fifo_empty && requant_in_ready && !requant_config_invalid) begin
                requant_in_valid = 1'b1;
                ofm_fifo_pop = 1'b1;
            end
        end else if (!cfg_sys_accum_en_i && !ofm_fifo_empty) begin
            obi_o_req_o = 4'b1111;
            if (obi_o_gnt_i == 4'b1111) begin
                ofm_fifo_pop = 1'b1;
                o_ptr_d = o_ptr_q + OFM_ROW_BYTES;
                drain_cnt_d = drain_cnt_q - 1;
            end
        end

        case (state_q)
            IDLE: begin
                if (cfg_sys_start_i) begin
                    if (cfg_requant_en_i && requant_config_invalid) begin
                        w_ptr_d = cfg_sys_weight_ptr_i;
                        i_ptr_d = cfg_sys_ifm_ptr_i;
                        o_ptr_d = cfg_sys_ofm_ptr_i;
                        a_ptr_d = cfg_sys_psum_ptr_i;
                        req_cnt_d = '0;
                        rsp_cnt_d = '0;
                        drain_cnt_d = '0;
                        accum_row_d = '0;
                        accum_beat_d = '0;
                        accum_requant_sent_d = 1'b0;
                        state_d = DONE;
                    end else begin
                        w_ptr_d = cfg_sys_weight_ptr_i + ((ARRAY_DIM - 1) * 32);
                        i_ptr_d = cfg_sys_ifm_ptr_i;
                        o_ptr_d = cfg_sys_ofm_ptr_i;
                        a_ptr_d = cfg_sys_psum_ptr_i;
                        req_cnt_d   = ARRAY_DIM; // 32 rows of weights
                        rsp_cnt_d   = ARRAY_DIM;
                        drain_cnt_d = cfg_sys_dim_m_i;
                        accum_row_d = '0;
                        accum_beat_d = '0;
                        accum_requant_sent_d = 1'b0;
                        state_d = LOAD_WEIGHTS;
                    end
                end
            end

            LOAD_WEIGHTS: begin
                if (req_cnt_q > 0) begin
                    obi_i_req_o = !weight_fifo_full;
                    obi_i_addr_o = w_ptr_q;
                    if (obi_i_req_o && obi_i_gnt_i) begin
                        w_ptr_d = w_ptr_q - 32;
                        req_cnt_d   = req_cnt_q - 1;
                    end
                end
                weight_fifo_push = obi_i_rvalid_i && !weight_fifo_full;
                if (!weight_fifo_empty) begin
                    weight_load_en = 1'b1;
                    weight_fifo_pop  = 1'b1;
                    weight_data    = weight_fifo_out;
                    rsp_cnt_d = rsp_cnt_q - 1;
                end
                if (req_cnt_q == 0 && rsp_cnt_q == 1 && weight_fifo_pop) begin
                    req_cnt_d = cfg_sys_dim_m_i;
                    rsp_cnt_d = cfg_sys_dim_m_i;
                    if (cfg_linebuf_en_i) begin
                        linebuf_start = 1'b1;
                    end
                    state_d = COMPUTE;
                end
            end

            COMPUTE: begin
                if (cfg_linebuf_en_i) begin
                    obi_i_req_o = linebuf_obi_req;
                    obi_i_addr_o = linebuf_obi_addr;
                    if (linebuf_row_valid && array_pipe_ready) begin
                        compute_en = 1'b1;
                        clear_acc = 1'b0;
                        ifm_data = linebuf_row_data;
                        linebuf_row_ready = 1'b1;
                        req_cnt_d = req_cnt_q - 1;
                    end
                    if (req_cnt_q == 1 && linebuf_row_valid && array_pipe_ready) begin
                        rsp_cnt_d = '0;
                        state_d = WAIT_DRAIN;
                    end
                end else begin
                    if (req_cnt_q > 0) begin
                        obi_i_req_o = !ifm_fifo_full && array_pipe_ready;
                        obi_i_addr_o = i_ptr_q;
                        if (obi_i_req_o && obi_i_gnt_i) begin
                            i_ptr_d = i_ptr_q + 32;
                            req_cnt_d   = req_cnt_q - 1;
                        end
                    end
                    ifm_fifo_push = obi_i_rvalid_i && !ifm_fifo_full;
                    if (!ifm_fifo_empty && array_pipe_ready) begin
                        compute_en = 1'b1;
                        clear_acc  = 1'b0;
                        ifm_fifo_pop = 1'b1;
                        ifm_data   = ifm_fifo_out;
                        rsp_cnt_d = rsp_cnt_q - 1;
                    end
                    if (req_cnt_q == 0 && rsp_cnt_q == 1 && ifm_fifo_pop) begin
                        state_d = WAIT_DRAIN;
                    end
                end
            end

            WAIT_DRAIN: begin
                if (cfg_sys_accum_en_i) begin
                    if (drain_cnt_q == 0 && ofm_fifo_empty) begin
                        state_d = DONE;
                    end else if (!ofm_fifo_empty) begin
                        req_cnt_d = OFM_ROW_BEATS;
                        rsp_cnt_d = OFM_ROW_BEATS;
                        accum_row_d = '0;
                        accum_beat_d = '0;
                        state_d = ACCUM_READ;
                    end
                end else begin
                    if (drain_cnt_q == 0 && ofm_fifo_empty) begin
                        state_d = DONE;
                    end
                end
            end

            ACCUM_READ: begin
                if (req_cnt_q > 0) begin
                    obi_i_req_o = 1'b1;
                    obi_i_addr_o = a_ptr_q;
                    if (obi_i_gnt_i) begin
                        a_ptr_d = a_ptr_q + OFM_BEAT_BYTES;
                        req_cnt_d = req_cnt_q - 1;
                    end
                end
                if (obi_i_rvalid_i) begin
                    for (int unsigned elem = 0; elem < OFM_ELEMS_PER_OBI; elem++) begin
                        accum_row_d[(accum_beat_q * OFM_ELEMS_PER_OBI) + elem] =
                            obi_i_rdata_i[elem * OFM_ELEM_WIDTH +: OFM_ELEM_WIDTH];
                    end
                    accum_beat_d = accum_beat_q + 1;
                    rsp_cnt_d = rsp_cnt_q - 1;
                    if (rsp_cnt_q == 1) begin
                        state_d = cfg_requant_en_i ? ACCUM_REQUANT : ACCUM_WRITE;
                    end
                end
            end

            ACCUM_WRITE: begin
                obi_o_req_o = 4'b1111;
                if (obi_o_gnt_i == 4'b1111) begin
                    ofm_fifo_pop = 1'b1;
                    o_ptr_d = o_ptr_q + OFM_ROW_BYTES;
                    drain_cnt_d = drain_cnt_q - 1;
                    state_d = WAIT_DRAIN;
                end
            end

            ACCUM_REQUANT: begin
                if (!accum_requant_sent_q && requant_in_ready && !requant_config_invalid) begin
                    requant_in_valid = 1'b1;
                    ofm_fifo_pop = 1'b1;
                    accum_requant_sent_d = 1'b1;
                end

                if (requant_out_valid) begin
                    obi_o_req_o[0] = 1'b1;
                    obi_o_wdata_o[0] = requant_packed_data;
                    if (obi_o_gnt_i[0] || requant_invalid) begin
                        requant_out_ready = 1'b1;
                        o_ptr_d = o_ptr_q + REQUANT_ROW_BYTES;
                        drain_cnt_d = drain_cnt_q - 1;
                        accum_requant_sent_d = 1'b0;
                        state_d = WAIT_DRAIN;
                    end
                    if (requant_invalid) begin
                        obi_o_req_o[0] = 1'b0;
                    end
                end
            end

            DONE: begin
                cfg_sys_done_o = 1'b1;
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q         <= IDLE;
            w_ptr_q         <= '0;
            i_ptr_q         <= '0;
            o_ptr_q         <= '0;
            a_ptr_q         <= '0;
            req_cnt_q       <= '0;
            rsp_cnt_q       <= '0;
            drain_cnt_q     <= '0;
            accum_row_q     <= '0;
            accum_beat_q    <= '0;
            accum_requant_sent_q <= 1'b0;
        end else begin
            state_q     <= state_d;
            w_ptr_q     <= w_ptr_d;
            i_ptr_q     <= i_ptr_d;
            o_ptr_q     <= o_ptr_d;
            a_ptr_q     <= a_ptr_d;
            req_cnt_q   <= req_cnt_d;
            rsp_cnt_q   <= rsp_cnt_d;
            drain_cnt_q <= drain_cnt_d;
            accum_row_q <= accum_row_d;
            accum_beat_q <= accum_beat_d;
            accum_requant_sent_q <= accum_requant_sent_d;
        end
    end

endmodule
