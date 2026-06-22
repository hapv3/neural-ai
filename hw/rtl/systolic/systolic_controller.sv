`default_nettype none

module systolic_controller #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned OFM_ELEM_WIDTH = 32,
    parameter int unsigned INPUT_FIFO_DEPTH = 4,
    parameter int unsigned OFM_FIFO_DEPTH = 128
)(
    input  logic clk_i,
    input  logic rst_ni,

    // Configuration from MMIO
    input  logic                      cfg_sys_start_i,
    input  logic [31:0]               cfg_sys_weight_ptr_i,
    input  logic [31:0]               cfg_sys_ifm_ptr_i,
    input  logic [31:0]               cfg_sys_ofm_ptr_i,
    input  logic [31:0]               cfg_sys_dim_m_i, // Number of IFM rows (skewed)
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

    // Systolic Array Interface
    output logic                      weight_load_en_o,
    output logic                      clear_acc_o,
    output logic                      compute_en_o,
    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] weight_data_o,
    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] ifm_data_o,
    output logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0]   psum_data_o,
    input  logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0]   ofm_data_i,
    input  logic                      ofm_valid_i
);

    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        COMPUTE,
        WAIT_DRAIN,
        DONE
    } state_e;

    state_e state_q, state_d;

    // The Systolic Array has a 32-cycle latency.
    localparam int unsigned SYS_LATENCY = 32;

    logic [31:0] w_ptr_q, w_ptr_d;
    logic [31:0] i_ptr_q, i_ptr_d;
    logic [31:0] o_ptr_q, o_ptr_d;
    logic [31:0] req_cnt_q, req_cnt_d; // Counter for requests
    logic [31:0] rsp_cnt_q, rsp_cnt_d; // Counter for responses
    logic [31:0] drain_cnt_q, drain_cnt_d; // Counter for valid outputs

    localparam int unsigned OFM_BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned OFM_ROW_BYTES = (ARRAY_DIM * OFM_ELEM_WIDTH) / 8;
    localparam int unsigned OFM_ELEMS_PER_OBI = DATA_WIDTH / OFM_ELEM_WIDTH;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;
    typedef logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0]   ofm_row_t;

    localparam int unsigned OFM_FIFO_ADDR_DEPTH = (OFM_FIFO_DEPTH > 1) ? $clog2(OFM_FIFO_DEPTH) : 1;
    localparam int unsigned OFM_FIFO_RESERVE = SYS_LATENCY + INPUT_FIFO_DEPTH + 8;
    localparam logic [OFM_FIFO_ADDR_DEPTH-1:0] OFM_FIFO_STOP_LEVEL =
        (OFM_FIFO_DEPTH > OFM_FIFO_RESERVE) ? OFM_FIFO_ADDR_DEPTH'(OFM_FIFO_DEPTH - OFM_FIFO_RESERVE) :
                                              OFM_FIFO_ADDR_DEPTH'(1);

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
    logic          ofm_fifo_push;
    logic          ofm_fifo_pop;
    logic          ofm_fifo_full;
    logic          ofm_fifo_empty;
    logic [OFM_FIFO_ADDR_DEPTH-1:0] ofm_fifo_usage;
    logic          ofm_fifo_almost_full;

    logic          fifo_flush;

    assign fifo_flush = (state_q == IDLE) && cfg_sys_start_i;

    assign weight_fifo_data = obi_i_rdata_i;
    assign ifm_fifo_data    = obi_i_rdata_i;
    assign ofm_fifo_data    = ofm_data_i;
    assign ofm_fifo_almost_full = ofm_fifo_full || (ofm_fifo_usage >= OFM_FIFO_STOP_LEVEL);

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

    // Tie off unused
    assign psum_data_o = '0;
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
        req_cnt_d   = req_cnt_q;
        rsp_cnt_d   = rsp_cnt_q;
        drain_cnt_d = drain_cnt_q;

        cfg_sys_done_o = 1'b0;

        obi_i_req_o = 1'b0;
        obi_i_addr_o = '0;

        weight_load_en_o = 1'b0;
        compute_en_o     = 1'b0;
        clear_acc_o      = 1'b0;
        weight_data_o    = '0;  // Only driven during LOAD_WEIGHTS
        ifm_data_o       = '0;  // Only driven during COMPUTE
        weight_fifo_push = 1'b0;
        weight_fifo_pop  = 1'b0;
        ifm_fifo_push    = 1'b0;
        ifm_fifo_pop     = 1'b0;
        ofm_fifo_pop     = 1'b0;
        ofm_fifo_push    = ((state_q == COMPUTE) || (state_q == WAIT_DRAIN)) &&
                            ofm_valid_i && !ofm_fifo_full;

        // OBI Output ports (Write)
        obi_o_req_o = '0;
        obi_o_addr_o[0] = o_ptr_q + 0;
        obi_o_addr_o[1] = o_ptr_q + OFM_BEAT_BYTES;
        obi_o_addr_o[2] = o_ptr_q + (2 * OFM_BEAT_BYTES);
        obi_o_addr_o[3] = o_ptr_q + (3 * OFM_BEAT_BYTES);
        
        obi_o_wdata_o[0] = ofm_fifo_out[OFM_ELEMS_PER_OBI-1:0];
        obi_o_wdata_o[1] = ofm_fifo_out[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
        obi_o_wdata_o[2] = ofm_fifo_out[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
        obi_o_wdata_o[3] = ofm_fifo_out[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];

        // Handle OFM writes through a configurable FIFO.  The FIFO absorbs
        // systolic output rows while O-TCDM write grants are backpressured.
        if (!ofm_fifo_empty) begin
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
                    w_ptr_d = cfg_sys_weight_ptr_i + ((ARRAY_DIM - 1) * 32);
                    i_ptr_d = cfg_sys_ifm_ptr_i;
                    o_ptr_d = cfg_sys_ofm_ptr_i;
                    req_cnt_d   = ARRAY_DIM; // 32 rows of weights
                    rsp_cnt_d   = ARRAY_DIM;
                    drain_cnt_d = cfg_sys_dim_m_i;
                    state_d = LOAD_WEIGHTS;
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
                    weight_load_en_o = 1'b1;
                    weight_fifo_pop  = 1'b1;
                    weight_data_o    = weight_fifo_out;
                    rsp_cnt_d = rsp_cnt_q - 1;
                end
                if (req_cnt_q == 0 && rsp_cnt_q == 1 && weight_fifo_pop) begin
                    req_cnt_d = cfg_sys_dim_m_i;
                    rsp_cnt_d = cfg_sys_dim_m_i;
                    state_d = COMPUTE;
                end
            end

            COMPUTE: begin
                if (req_cnt_q > 0) begin
                    obi_i_req_o = !ifm_fifo_full && !ofm_fifo_almost_full;
                    obi_i_addr_o = i_ptr_q;
                    if (obi_i_req_o && obi_i_gnt_i) begin
                        i_ptr_d = i_ptr_q + 32;
                        req_cnt_d   = req_cnt_q - 1;
                    end
                end
                ifm_fifo_push = obi_i_rvalid_i && !ifm_fifo_full;
                if (!ifm_fifo_empty && !ofm_fifo_almost_full) begin
                    compute_en_o = 1'b1;
                    clear_acc_o  = 1'b0;
                    ifm_fifo_pop = 1'b1;
                    ifm_data_o   = ifm_fifo_out;
                    rsp_cnt_d = rsp_cnt_q - 1;
                end
                if (req_cnt_q == 0 && rsp_cnt_q == 1 && ifm_fifo_pop) begin
                    state_d = WAIT_DRAIN;
                end
            end

            WAIT_DRAIN: begin
                // Wait until all outputs are drained to memory
                if (drain_cnt_q == 0 && ofm_fifo_empty) begin
                    state_d = DONE;
                end
            end

            DONE: begin
                cfg_sys_done_o = 1'b1;
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
            req_cnt_q       <= '0;
            rsp_cnt_q       <= '0;
            drain_cnt_q     <= '0;
        end else begin
            state_q     <= state_d;
            w_ptr_q     <= w_ptr_d;
            i_ptr_q     <= i_ptr_d;
            o_ptr_q     <= o_ptr_d;
            req_cnt_q   <= req_cnt_d;
            rsp_cnt_q   <= rsp_cnt_d;
            drain_cnt_q <= drain_cnt_d;
        end
    end

endmodule
