`default_nettype none

module systolic_input_engine #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned FIFO_DEPTH = 4,
    parameter int unsigned MAX_INPUT_W = 640
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic job_start_i,
    input  logic feed_start_i,
    input  logic feed_service_i,
    input  logic drain_service_i,
    input  logic linebuf_start_i,
    input  logic linebuf_next_tile_i,
    input  logic preload_service_i,
    input  logic preload_has_next_i,
    input  logic preload_hold_i,
    input  logic linebuf_enable_i,
    input  logic side_stream_mode_i,
    input  logic array_pipe_ready_i,
    input  logic side_ready_i,
    input  logic [ADDR_WIDTH-1:0] ifm_base_ptr_i,
    input  logic [31:0] row_count_i,

    input  logic [31:0] cfg_spatial_m_i,
    input  logic [31:0] cfg_k_tiles_i,
    input  logic [31:0] cfg_origin_base_i,
    input  logic [31:0] cfg_row_stride_bytes_i,
    input  logic [31:0] cfg_pixel_stride_bytes_i,
    input  logic [31:0] cfg_ow_step_bytes_i,
    input  logic [31:0] cfg_oh_step_bytes_i,
    input  logic [15:0] cfg_input_h_i,
    input  logic [15:0] cfg_input_w_i,
    input  logic [15:0] cfg_input_c_i,
    input  logic [15:0] cfg_output_w_i,
    input  logic [15:0] cfg_kernel_h_i,
    input  logic [15:0] cfg_kernel_w_i,
    input  logic [15:0] cfg_stride_h_i,
    input  logic [15:0] cfg_stride_w_i,
    input  logic [15:0] cfg_pad_h_i,
    input  logic [15:0] cfg_pad_w_i,
    input  logic [15:0] cfg_c_base_i,
    input  logic [5:0]  cfg_lane_base_i,
    input  logic        cfg_coalesce_i,
    input  logic        cfg_kgen_i,
    input  logic        cfg_pool_i,
    input  logic        cfg_c32_fast_i,
    input  logic        cfg_depthwise_i,
    input  logic [5:0]  cfg_block_valid_bytes_i,
    input  logic [31:0] cfg_channel_addr_offset_i,
    input  logic [31:0] cfg_coalesce_k_bytes_i,
    input  logic [7:0]  cfg_k_seed_kh_i,
    input  logic [7:0]  cfg_k_seed_kw_i,
    input  logic [15:0] cfg_k_seed_ic_i,

    output logic                       obi_req_o,
    input  logic                       obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]      obi_addr_o,
    output logic                       obi_we_o,
    output logic [(DATA_WIDTH/8)-1:0]  obi_be_o,
    output logic [DATA_WIDTH-1:0]      obi_wdata_o,
    input  logic                       obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0]      obi_rdata_i,

    output logic                                      compute_en_o,
    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] compute_data_o,
    output logic                                      feed_done_o,
    output logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] side_data_o,
    output logic                                      side_valid_o,
    output logic                                      linebuf_row_ready_o,
    output logic                                      linebuf_busy_o,
    output logic                                      linebuf_done_o,
    output logic                                      prefetch_busy_o,
    output logic [31:0]                               request_count_o,
    output logic [31:0]                               response_count_o,
    output logic [31:0]                               emitted_vectors_o,
    output logic [31:0]                               fetch_beats_o,
    output logic [31:0]                               bypass_vectors_o,
    output logic [4:0]                                debug_state_o
);

    typedef logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0] input_row_t;

    logic [ADDR_WIDTH-1:0] direct_ptr_q, direct_ptr_d;
    logic [31:0] request_count_q, request_count_d;
    logic [31:0] response_count_q, response_count_d;
    logic        prefetch_req_q, prefetch_req_d;

    input_row_t direct_fifo_out;
    logic       direct_fifo_push;
    logic       direct_fifo_pop;
    logic       direct_fifo_full;
    logic       direct_fifo_empty;

    logic                   linebuf_obi_req;
    logic [ADDR_WIDTH-1:0]  linebuf_obi_addr;
    input_row_t             linebuf_row_data;
    logic                   linebuf_row_valid;
    logic                   linebuf_row_ready;
    logic                   linebuf_prefetch;

    assign obi_we_o = 1'b0;
    assign obi_be_o = '1;
    assign obi_wdata_o = '0;
    assign side_data_o = linebuf_row_data;
    assign side_valid_o = linebuf_row_valid;
    assign linebuf_row_ready_o = linebuf_row_ready;
    assign request_count_o = request_count_q;
    assign response_count_o = response_count_q;

    fifo_v3 #(
        .FALL_THROUGH (1'b1),
        .DEPTH        (FIFO_DEPTH),
        .dtype        (input_row_t)
    ) i_direct_fifo (
        .clk_i,
        .rst_ni,
        .flush_i    (job_start_i),
        .testmode_i (1'b0),
        .full_o     (direct_fifo_full),
        .empty_o    (direct_fifo_empty),
        .usage_o    (),
        .data_i     (obi_rdata_i),
        .push_i     (direct_fifo_push),
        .data_o     (direct_fifo_out),
        .pop_i      (direct_fifo_pop)
    );

    conv_linebuf_stream_packer #(
        .ADDR_WIDTH       (ADDR_WIDTH),
        .DATA_WIDTH       (DATA_WIDTH),
        .ARRAY_DIM        (ARRAY_DIM),
        .INPUT_ELEM_WIDTH (INPUT_ELEM_WIDTH),
        .MAX_INPUT_W      (MAX_INPUT_W)
    ) i_linebuf (
        .clk_i,
        .rst_ni,
        .start_i                 (linebuf_start_i),
        .next_tile_i             (linebuf_next_tile_i),
        .prefetch_i              (linebuf_prefetch),
        .dim_m_i                 (cfg_spatial_m_i),
        .cfg_k_tiles_i           (cfg_k_tiles_i),
        .cfg_origin_base_i       (cfg_origin_base_i),
        .cfg_row_stride_bytes_i  (cfg_row_stride_bytes_i),
        .cfg_pixel_stride_bytes_i(cfg_pixel_stride_bytes_i),
        .cfg_ow_step_bytes_i     (cfg_ow_step_bytes_i),
        .cfg_oh_step_bytes_i     (cfg_oh_step_bytes_i),
        .cfg_input_h_i           (cfg_input_h_i),
        .cfg_input_w_i           (cfg_input_w_i),
        .cfg_input_c_i           (cfg_input_c_i),
        .cfg_output_w_i          (cfg_output_w_i),
        .cfg_kernel_h_i          (cfg_kernel_h_i),
        .cfg_kernel_w_i          (cfg_kernel_w_i),
        .cfg_stride_h_i          (cfg_stride_h_i),
        .cfg_stride_w_i          (cfg_stride_w_i),
        .cfg_pad_h_i             (cfg_pad_h_i),
        .cfg_pad_w_i             (cfg_pad_w_i),
        .cfg_c_base_i            (cfg_c_base_i),
        .cfg_lane_base_i         (cfg_lane_base_i),
        .cfg_coalesce_i          (cfg_coalesce_i),
        .cfg_kgen_i              (cfg_kgen_i),
        .cfg_pool_i              (cfg_pool_i),
        .cfg_c32_fast_i          (cfg_c32_fast_i),
        .cfg_depthwise_i         (cfg_depthwise_i),
        .cfg_block_valid_bytes_i (cfg_block_valid_bytes_i),
        .cfg_channel_addr_offset_i(cfg_channel_addr_offset_i),
        .cfg_coalesce_k_bytes_i  (cfg_coalesce_k_bytes_i),
        .cfg_k_seed_kh_i         (cfg_k_seed_kh_i),
        .cfg_k_seed_kw_i         (cfg_k_seed_kw_i),
        .cfg_k_seed_ic_i         (cfg_k_seed_ic_i),
        .obi_req_o               (linebuf_obi_req),
        .obi_gnt_i               (obi_gnt_i),
        .obi_addr_o              (linebuf_obi_addr),
        .obi_rvalid_i            (obi_rvalid_i),
        .obi_rdata_i             (obi_rdata_i),
        .row_data_o              (linebuf_row_data),
        .row_valid_o             (linebuf_row_valid),
        .row_ready_i             (linebuf_row_ready),
        .busy_o                  (linebuf_busy_o),
        .done_o                  (linebuf_done_o),
        .prefetch_busy_o         (prefetch_busy_o),
        .emitted_vectors_o,
        .fetch_beats_o,
        .bypass_vectors_o,
        .debug_state_o
    );

    always_comb begin
        direct_ptr_d = direct_ptr_q;
        request_count_d = request_count_q;
        response_count_d = response_count_q;
        prefetch_req_d = preload_service_i ? prefetch_req_q : 1'b0;

        obi_req_o = 1'b0;
        obi_addr_o = '0;
        direct_fifo_push = 1'b0;
        direct_fifo_pop = 1'b0;
        linebuf_row_ready = 1'b0;
        linebuf_prefetch = prefetch_req_q && preload_service_i;
        compute_en_o = 1'b0;
        compute_data_o = '0;
        feed_done_o = 1'b0;

        if (feed_service_i) begin
            if (linebuf_enable_i) begin
                obi_req_o = linebuf_obi_req;
                obi_addr_o = linebuf_obi_addr;
                if (side_stream_mode_i) begin
                    linebuf_row_ready = linebuf_row_valid && side_ready_i;
                end else if (linebuf_row_valid && array_pipe_ready_i) begin
                    compute_en_o = 1'b1;
                    compute_data_o = linebuf_row_data;
                    linebuf_row_ready = 1'b1;
                    request_count_d = request_count_q - 1'b1;
                    if (request_count_q == 1) begin
                        response_count_d = '0;
                        feed_done_o = 1'b1;
                    end
                end
            end else begin
                if (request_count_q != 0) begin
                    obi_req_o = !direct_fifo_full && array_pipe_ready_i;
                    obi_addr_o = direct_ptr_q;
                    if (obi_req_o && obi_gnt_i) begin
                        direct_ptr_d = direct_ptr_q + ADDR_WIDTH'(32);
                        request_count_d = request_count_q - 1'b1;
                    end
                end
                direct_fifo_push = obi_rvalid_i && !direct_fifo_full;
                if (!direct_fifo_empty && array_pipe_ready_i) begin
                    compute_en_o = 1'b1;
                    compute_data_o = direct_fifo_out;
                    direct_fifo_pop = 1'b1;
                    response_count_d = response_count_q - 1'b1;
                end
                if ((request_count_q == 0) && (response_count_q == 1) &&
                    direct_fifo_pop) begin
                    feed_done_o = 1'b1;
                end
            end
        end

        if (drain_service_i && linebuf_enable_i) begin
            linebuf_row_ready = 1'b1;
        end

        if (preload_service_i) begin
            if (preload_has_next_i) begin
                prefetch_req_d = prefetch_busy_o || preload_hold_i;
            end else begin
                prefetch_req_d = 1'b0;
            end
            if (prefetch_req_q) begin
                obi_req_o = linebuf_obi_req;
                obi_addr_o = linebuf_obi_addr;
            end
        end

        if (feed_start_i) begin
            direct_ptr_d = ifm_base_ptr_i;
            request_count_d = row_count_i;
            response_count_d = row_count_i;
        end

        if (job_start_i) begin
            prefetch_req_d = 1'b0;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            direct_ptr_q <= '0;
            request_count_q <= '0;
            response_count_q <= '0;
            prefetch_req_q <= 1'b0;
        end else begin
            direct_ptr_q <= direct_ptr_d;
            request_count_q <= request_count_d;
            response_count_q <= response_count_d;
            prefetch_req_q <= prefetch_req_d;
        end
    end

endmodule

`default_nettype wire
