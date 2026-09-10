`default_nettype none

module systolic_weight_engine #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned FIFO_DEPTH = 4,
    parameter int unsigned DW_MAX_TAPS = 25
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic job_start_i,
    input  logic load_service_i,
    input  logic preload_service_i,
    input  logic preload_allow_i,
    input  logic preload_consume_i,
    input  logic depthwise_group_start_i,
    input  logic depthwise_mode_i,
    input  logic array_pipe_ready_i,
    input  logic [ADDR_WIDTH-1:0] weight_base_ptr_i,
    input  logic [ADDR_WIDTH-1:0] depthwise_group_weight_ptr_i,
    input  logic [31:0] depthwise_tap_count_i,
    input  logic [31:0] depthwise_tap_index_i,
    input  logic [31:0] next_tile_index_i,

    output logic                     obi_req_o,
    input  logic                     obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]    obi_addr_o,
    output logic                     obi_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_be_o,
    output logic [DATA_WIDTH-1:0]    obi_wdata_o,
    input  logic                     obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0]    obi_rdata_i,

    output logic                                      weight_load_en_o,
    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] weight_data_o,
    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] depthwise_weight_o,
    output logic                                      load_done_o,
    output logic                                      preload_done_o
);

    localparam int unsigned WEIGHT_TILE_SHIFT = $clog2(ARRAY_DIM) + 5;
    localparam int unsigned WEIGHT_TILE_LAST_ROW_BYTES = (ARRAY_DIM - 1) << 5;

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;

    logic [ADDR_WIDTH-1:0] ptr_q, ptr_d;
    logic [31:0] req_cnt_q, req_cnt_d;
    logic [31:0] rsp_cnt_q, rsp_cnt_d;
    logic [31:0] obi_rsp_cnt_q, obi_rsp_cnt_d;
    logic        preload_active_q, preload_active_d;
    logic        preload_done_q, preload_done_d;

    input_row_t depthwise_weight_q [DW_MAX_TAPS];
    input_row_t depthwise_weight_d [DW_MAX_TAPS];
    input_row_t fifo_out;
    logic       fifo_push;
    logic       fifo_pop;
    logic       fifo_full;
    logic       fifo_empty;

    assign obi_we_o = 1'b0;
    assign obi_be_o = '1;
    assign obi_wdata_o = '0;
    assign preload_done_o = preload_done_q;

    always_comb begin
        if (depthwise_tap_index_i < DW_MAX_TAPS) begin
            depthwise_weight_o = depthwise_weight_q[depthwise_tap_index_i];
        end else begin
            depthwise_weight_o = '0;
        end
    end

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (FIFO_DEPTH),
        .dtype        (input_row_t)
    ) i_weight_fifo (
        .clk_i,
        .rst_ni,
        .flush_i    (job_start_i),
        .testmode_i (1'b0),
        .full_o     (fifo_full),
        .empty_o    (fifo_empty),
        .usage_o    (),
        .data_i     (obi_rdata_i),
        .push_i     (fifo_push),
        .data_o     (fifo_out),
        .pop_i      (fifo_pop)
    );

    always_comb begin
        ptr_d = ptr_q;
        req_cnt_d = req_cnt_q;
        rsp_cnt_d = rsp_cnt_q;
        obi_rsp_cnt_d = obi_rsp_cnt_q;
        preload_active_d = preload_active_q;
        preload_done_d = preload_done_q;
        for (int unsigned tap = 0; tap < DW_MAX_TAPS; tap++) begin
            depthwise_weight_d[tap] = depthwise_weight_q[tap];
        end

        obi_req_o = 1'b0;
        obi_addr_o = '0;
        fifo_push = 1'b0;
        fifo_pop = 1'b0;
        weight_load_en_o = 1'b0;
        weight_data_o = '0;
        load_done_o = 1'b0;

        if (load_service_i) begin
            if (depthwise_mode_i) begin
                if (req_cnt_q != 0) begin
                    obi_req_o = 1'b1;
                    obi_addr_o = ptr_q;
                    if (obi_gnt_i) begin
                        ptr_d = ptr_q + ADDR_WIDTH'(32);
                        req_cnt_d = req_cnt_q - 1'b1;
                    end
                end
                if (obi_rvalid_i) begin
                    if ((depthwise_tap_count_i - rsp_cnt_q) < DW_MAX_TAPS) begin
                        depthwise_weight_d[depthwise_tap_count_i - rsp_cnt_q] = obi_rdata_i;
                    end
                    rsp_cnt_d = rsp_cnt_q - 1'b1;
                end
                if ((req_cnt_q == 0) && (rsp_cnt_q == 1) && obi_rvalid_i) begin
                    load_done_o = 1'b1;
                end
            end else if (!preload_done_q) begin
                if (req_cnt_q != 0) begin
                    obi_req_o = !fifo_full;
                    obi_addr_o = ptr_q;
                    if (obi_req_o && obi_gnt_i) begin
                        ptr_d = ptr_q - ADDR_WIDTH'(32);
                        req_cnt_d = req_cnt_q - 1'b1;
                    end
                end
                fifo_push = obi_rvalid_i && !fifo_full;
                if (!fifo_empty) begin
                    weight_load_en_o = 1'b1;
                    weight_data_o = fifo_out;
                    fifo_pop = 1'b1;
                    rsp_cnt_d = rsp_cnt_q - 1'b1;
                end
                if ((req_cnt_q == 0) && (rsp_cnt_q == 1) && fifo_pop) begin
                    load_done_o = 1'b1;
                end
            end
        end else if (preload_service_i) begin
            if (preload_allow_i && !preload_active_q && !preload_done_q) begin
                preload_active_d = 1'b1;
                req_cnt_d = ARRAY_DIM;
                rsp_cnt_d = ARRAY_DIM;
                obi_rsp_cnt_d = ARRAY_DIM;
                ptr_d = weight_base_ptr_i +
                        ADDR_WIDTH'(next_tile_index_i << WEIGHT_TILE_SHIFT) +
                        ADDR_WIDTH'(WEIGHT_TILE_LAST_ROW_BYTES);
            end

            if (preload_active_q) begin
                if (req_cnt_q != 0) begin
                    obi_req_o = !fifo_full;
                    obi_addr_o = ptr_q;
                    if (obi_req_o && obi_gnt_i) begin
                        ptr_d = ptr_q - ADDR_WIDTH'(32);
                        req_cnt_d = req_cnt_q - 1'b1;
                    end
                end
                fifo_push = (obi_rsp_cnt_q != 0) && obi_rvalid_i && !fifo_full;
                if (fifo_push) begin
                    obi_rsp_cnt_d = obi_rsp_cnt_q - 1'b1;
                end
                if (!fifo_empty && array_pipe_ready_i) begin
                    weight_load_en_o = 1'b1;
                    weight_data_o = fifo_out;
                    fifo_pop = 1'b1;
                    rsp_cnt_d = rsp_cnt_q - 1'b1;
                end
                if ((req_cnt_q == 0) && (rsp_cnt_q == 1) && fifo_pop) begin
                    preload_active_d = 1'b0;
                    preload_done_d = 1'b1;
                end
            end
        end

        if (preload_consume_i) begin
            preload_done_d = 1'b0;
        end

        if (depthwise_group_start_i) begin
            ptr_d = depthwise_group_weight_ptr_i;
            req_cnt_d = depthwise_tap_count_i;
            rsp_cnt_d = depthwise_tap_count_i;
            obi_rsp_cnt_d = '0;
        end

        if (job_start_i) begin
            ptr_d = depthwise_mode_i ? weight_base_ptr_i :
                    weight_base_ptr_i + ADDR_WIDTH'(WEIGHT_TILE_LAST_ROW_BYTES);
            req_cnt_d = depthwise_mode_i ? depthwise_tap_count_i : ARRAY_DIM;
            rsp_cnt_d = depthwise_mode_i ? depthwise_tap_count_i : ARRAY_DIM;
            obi_rsp_cnt_d = '0;
            preload_active_d = 1'b0;
            preload_done_d = 1'b0;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            ptr_q <= '0;
            req_cnt_q <= '0;
            rsp_cnt_q <= '0;
            obi_rsp_cnt_q <= '0;
            preload_active_q <= 1'b0;
            preload_done_q <= 1'b0;
            for (int unsigned tap = 0; tap < DW_MAX_TAPS; tap++) begin
                depthwise_weight_q[tap] <= '0;
            end
        end else begin
            ptr_q <= ptr_d;
            req_cnt_q <= req_cnt_d;
            rsp_cnt_q <= rsp_cnt_d;
            obi_rsp_cnt_q <= obi_rsp_cnt_d;
            preload_active_q <= preload_active_d;
            preload_done_q <= preload_done_d;
            for (int unsigned tap = 0; tap < DW_MAX_TAPS; tap++) begin
                depthwise_weight_q[tap] <= depthwise_weight_d[tap];
            end
        end
    end

endmodule

`default_nettype wire
