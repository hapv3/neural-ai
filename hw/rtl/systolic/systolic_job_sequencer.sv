`default_nettype none

module systolic_job_sequencer #(
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned DW_MAX_TAPS = 25,
    parameter int unsigned DW_TAP_COUNT_W = $clog2(DW_MAX_TAPS + 1)
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic        start_i,
    input  logic        linebuf_enable_i,
    input  logic        pool_mode_i,
    input  logic        depthwise_mode_i,
    input  logic        kgen_multi_i,
    input  logic [31:0] tile_index_i,
    input  logic        has_next_tile_i,
    input  logic        psum_overlap_active_i,
    input  logic        requant_enable_i,
    input  logic        requant_config_invalid_i,
    input  logic        binary_config_invalid_i,
    input  logic        binary_enable_i,

    input  logic        weight_load_done_i,
    input  logic        weight_preload_done_i,
    input  logic        input_feed_done_i,
    input  logic        array_pipe_ready_i,
    input  logic        linebuf_row_valid_i,
    input  logic        linebuf_busy_i,
    input  logic        linebuf_prefetch_busy_i,
    input  logic [31:0] drain_remaining_i,
    input  logic        ofm_empty_i,
    input  logic        binary_operand_busy_i,
    input  logic        depthwise_input_ready_i,
    input  logic        depthwise_output_valid_i,
    input  logic        quantized_output_valid_i,
    input  logic        pool_input_ready_i,
    input  logic        pool_output_valid_i,

    input  logic [31:0] weight_base_ptr_i,
    input  logic [31:0] output_base_ptr_i,
    input  logic [15:0] input_c_i,
    input  logic [15:0] input_h_i,
    input  logic [31:0] input_row_stride_bytes_i,
    input  logic [31:0] spatial_row_count_i,
    input  logic [31:0] kernel_vectors_i,

    output logic        job_start_o,
    output logic        done_o,
    output logic [2:0]  state_o,
    output logic        load_service_o,
    output logic        compute_service_o,
    output logic        drain_service_o,
    output logic        drain_active_o,
    output logic        weight_preload_allow_o,
    output logic        input_preload_hold_o,
    output logic        use_next_tile_config_o,

    output logic        input_feed_start_o,
    output logic        input_side_ready_o,
    output logic        weight_preload_consume_o,
    output logic        weight_depthwise_group_start_o,
    output logic [31:0] weight_depthwise_group_ptr_o,
    output logic        drain_tile_advance_o,
    output logic        drain_tile_advance_overlap_o,
    output logic        drain_tile_start_o,
    output logic        drain_tile_start_add_rows_o,
    output logic        drain_depthwise_group_start_o,
    output logic [31:0] drain_depthwise_group_output_ptr_o,
    output logic        linebuf_start_o,
    output logic        linebuf_next_tile_o,
    output logic        k_tile_advance_o,
    output logic        depthwise_input_valid_o,

    output logic [DW_TAP_COUNT_W-1:0] depthwise_tap_index_o,
    output logic        depthwise_tap_is_last_o,
    output logic [31:0] depthwise_group_index_o,
    output logic [31:0] depthwise_group_input_offset_o,
    output logic [5:0]  depthwise_group_valid_bytes_o
);

    typedef enum logic [2:0] {
        IDLE,
        LOAD_WEIGHTS,
        COMPUTE,
        WAIT_DRAIN,
        DONE
    } state_e;

    localparam int unsigned ARRAY_FLUSH_CYCLES = (2 * ARRAY_DIM) - 1;
    localparam int unsigned ARRAY_FLUSH_COUNT_W = $clog2(ARRAY_FLUSH_CYCLES + 1);

    state_e state_q, state_d;
    logic [ARRAY_FLUSH_COUNT_W-1:0] array_flush_count_q, array_flush_count_d;
    logic [DW_TAP_COUNT_W-1:0] depthwise_tap_index_q, depthwise_tap_index_d;
    logic [31:0] depthwise_group_index_q, depthwise_group_index_d;
    logic [31:0] depthwise_group_input_offset_q, depthwise_group_input_offset_d;
    logic [31:0] depthwise_group_output_offset_q, depthwise_group_output_offset_d;
    logic [31:0] depthwise_group_weight_offset_q, depthwise_group_weight_offset_d;

    logic [31:0] depthwise_group_count;
    logic [31:0] depthwise_group_input_bytes;
    logic [31:0] depthwise_group_output_bytes;
    logic [31:0] depthwise_group_weight_bytes;
    logic        depthwise_last_group;
    logic        psum_overlap_next_safe;

    function automatic logic [5:0] depthwise_group_valid_bytes(
        input logic [15:0] input_c,
        input logic [31:0] group_index
    );
        logic [31:0] group_base;
        logic [31:0] remaining;
        begin
            group_base = group_index << 5;
            if (group_base >= {16'd0, input_c}) begin
                depthwise_group_valid_bytes = 6'd0;
            end else begin
                remaining = {16'd0, input_c} - group_base;
                depthwise_group_valid_bytes = (remaining >= 32'd32) ? 6'd32 : remaining[5:0];
            end
        end
    endfunction

    assign job_start_o = (state_q == IDLE) && start_i;
    assign state_o = state_q;
    assign load_service_o = state_q == LOAD_WEIGHTS;
    assign compute_service_o = state_q == COMPUTE;
    assign drain_service_o = state_q == WAIT_DRAIN;
    assign drain_active_o = !pool_mode_i && !depthwise_mode_i &&
                            ((state_q == LOAD_WEIGHTS) ||
                             (state_q == COMPUTE) ||
                             (state_q == WAIT_DRAIN));
    assign weight_preload_allow_o = has_next_tile_i &&
                                    (array_flush_count_q == '0) &&
                                    !linebuf_prefetch_busy_i;
    assign input_preload_hold_o = (array_flush_count_q != '0) ||
                                  (drain_remaining_i != 32'd0) ||
                                  !ofm_empty_i;
    assign use_next_tile_config_o = (state_q == WAIT_DRAIN) && has_next_tile_i;

    assign depthwise_group_count = ({16'd0, input_c_i} + 32'd31) >> 5;
    assign depthwise_group_input_bytes = {16'd0, input_h_i} * input_row_stride_bytes_i;
    assign depthwise_group_output_bytes = spatial_row_count_i << 5;
    assign depthwise_group_weight_bytes = kernel_vectors_i << 5;
    assign depthwise_last_group = (depthwise_group_index_q + 32'd1) >= depthwise_group_count;
    assign depthwise_tap_is_last_o =
        ((32'(depthwise_tap_index_q) + 32'd1) == kernel_vectors_i);
    assign depthwise_tap_index_o = depthwise_tap_index_q;
    assign depthwise_group_index_o = depthwise_group_index_q;
    assign depthwise_group_input_offset_o = depthwise_group_input_offset_q;
    assign depthwise_group_valid_bytes_o =
        depthwise_group_valid_bytes(input_c_i, depthwise_group_index_q);

    assign psum_overlap_next_safe = psum_overlap_active_i && has_next_tile_i &&
                                    weight_preload_done_i &&
                                    !linebuf_prefetch_busy_i &&
                                    (array_flush_count_q == '0);

    always_comb begin
        state_d = state_q;
        array_flush_count_d = array_flush_count_q;
        depthwise_tap_index_d = depthwise_tap_index_q;
        depthwise_group_index_d = depthwise_group_index_q;
        depthwise_group_input_offset_d = depthwise_group_input_offset_q;
        depthwise_group_output_offset_d = depthwise_group_output_offset_q;
        depthwise_group_weight_offset_d = depthwise_group_weight_offset_q;

        done_o = 1'b0;
        input_feed_start_o = 1'b0;
        input_side_ready_o = 1'b0;
        weight_preload_consume_o = 1'b0;
        weight_depthwise_group_start_o = 1'b0;
        weight_depthwise_group_ptr_o = weight_base_ptr_i;
        drain_tile_advance_o = 1'b0;
        drain_tile_advance_overlap_o = 1'b0;
        drain_tile_start_o = 1'b0;
        drain_tile_start_add_rows_o = 1'b0;
        drain_depthwise_group_start_o = 1'b0;
        drain_depthwise_group_output_ptr_o = output_base_ptr_i;
        linebuf_start_o = 1'b0;
        linebuf_next_tile_o = 1'b0;
        k_tile_advance_o = 1'b0;
        depthwise_input_valid_o = 1'b0;

        case (state_q)
            IDLE: begin
                if (start_i) begin
                    array_flush_count_d = '0;
                    depthwise_tap_index_d = '0;
                    depthwise_group_index_d = '0;
                    depthwise_group_input_offset_d = '0;
                    depthwise_group_output_offset_d = '0;
                    depthwise_group_weight_offset_d = '0;
                    state_d = LOAD_WEIGHTS;

                    if (pool_mode_i) begin
                        linebuf_start_o = 1'b1;
                        state_d = COMPUTE;
                    end

                    if (depthwise_mode_i) begin
                        state_d = LOAD_WEIGHTS;
                    end

                    if ((requant_enable_i && requant_config_invalid_i) ||
                        binary_config_invalid_i) begin
                        state_d = DONE;
                    end
                end
            end

            LOAD_WEIGHTS: begin
                if (depthwise_mode_i) begin
                    if (weight_load_done_i) begin
                        linebuf_start_o = 1'b1;
                        depthwise_tap_index_d = '0;
                        state_d = COMPUTE;
                    end
                end else if (weight_preload_done_i) begin
                    input_feed_start_o = 1'b1;
                    drain_tile_start_o = 1'b1;
                    drain_tile_start_add_rows_o = psum_overlap_active_i &&
                                                  (tile_index_i != 32'd0);
                    array_flush_count_d = '0;
                    weight_preload_consume_o = 1'b1;
                    if (linebuf_enable_i) begin
                        linebuf_next_tile_o = 1'b1;
                    end
                    state_d = COMPUTE;
                end else if (weight_load_done_i) begin
                    input_feed_start_o = 1'b1;
                    drain_tile_start_o = 1'b1;
                    drain_tile_start_add_rows_o = psum_overlap_active_i &&
                                                  (tile_index_i != 32'd0);
                    if (linebuf_enable_i) begin
                        if (kgen_multi_i && (tile_index_i != 32'd0)) begin
                            linebuf_next_tile_o = 1'b1;
                        end else begin
                            linebuf_start_o = 1'b1;
                        end
                    end
                    state_d = COMPUTE;
                end
            end

            COMPUTE: begin
                if (depthwise_mode_i) begin
                    if (linebuf_row_valid_i && !requant_config_invalid_i &&
                        depthwise_input_ready_i) begin
                        depthwise_input_valid_o = 1'b1;
                        input_side_ready_o = 1'b1;
                        if (depthwise_tap_is_last_o) begin
                            depthwise_tap_index_d = '0;
                        end else begin
                            depthwise_tap_index_d = depthwise_tap_index_q + 1'b1;
                        end
                    end

                    if ((drain_remaining_i == 32'd0) && !linebuf_busy_i &&
                        !depthwise_output_valid_i && !quantized_output_valid_i) begin
                        if (!depthwise_last_group) begin
                            depthwise_group_index_d = depthwise_group_index_q + 32'd1;
                            depthwise_group_input_offset_d = depthwise_group_input_offset_q +
                                                             depthwise_group_input_bytes;
                            depthwise_group_output_offset_d = depthwise_group_output_offset_q +
                                                              depthwise_group_output_bytes;
                            depthwise_group_weight_offset_d = depthwise_group_weight_offset_q +
                                                              depthwise_group_weight_bytes;
                            weight_depthwise_group_start_o = 1'b1;
                            weight_depthwise_group_ptr_o = weight_base_ptr_i +
                                depthwise_group_weight_offset_q + depthwise_group_weight_bytes;
                            drain_depthwise_group_start_o = 1'b1;
                            drain_depthwise_group_output_ptr_o = output_base_ptr_i +
                                depthwise_group_output_offset_q + depthwise_group_output_bytes;
                            depthwise_tap_index_d = '0;
                            state_d = LOAD_WEIGHTS;
                        end else begin
                            state_d = DONE;
                        end
                    end
                end else if (pool_mode_i) begin
                    input_side_ready_o = linebuf_row_valid_i && pool_input_ready_i;

                    if ((drain_remaining_i == 32'd0) && !pool_output_valid_i &&
                        !linebuf_busy_i) begin
                        state_d = DONE;
                    end
                end else if (input_feed_done_i) begin
                    array_flush_count_d = ARRAY_FLUSH_COUNT_W'(ARRAY_FLUSH_CYCLES);
                    state_d = WAIT_DRAIN;
                end
            end

            WAIT_DRAIN: begin
                if ((array_flush_count_q != '0) && array_pipe_ready_i) begin
                    array_flush_count_d = array_flush_count_q - 1'b1;
                end

                if (psum_overlap_next_safe) begin
                    k_tile_advance_o = 1'b1;
                    drain_tile_advance_o = 1'b1;
                    drain_tile_advance_overlap_o = 1'b1;
                    input_feed_start_o = 1'b1;
                    weight_preload_consume_o = 1'b1;
                    linebuf_next_tile_o = 1'b1;
                    state_d = COMPUTE;
                end else if ((drain_remaining_i == 32'd0) && ofm_empty_i &&
                             (!binary_enable_i || !binary_operand_busy_i)) begin
                    if (has_next_tile_i && weight_preload_done_i &&
                        !linebuf_prefetch_busy_i) begin
                        k_tile_advance_o = 1'b1;
                        drain_tile_advance_o = 1'b1;
                        state_d = LOAD_WEIGHTS;
                    end else if (has_next_tile_i) begin
                        state_d = WAIT_DRAIN;
                    end else if (!linebuf_enable_i || !linebuf_busy_i) begin
                        state_d = DONE;
                    end
                end
            end

            DONE: begin
                done_o = 1'b1;
                state_d = IDLE;
            end

            default: begin
                state_d = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
            array_flush_count_q <= '0;
            depthwise_tap_index_q <= '0;
            depthwise_group_index_q <= '0;
            depthwise_group_input_offset_q <= '0;
            depthwise_group_output_offset_q <= '0;
            depthwise_group_weight_offset_q <= '0;
        end else begin
            state_q <= state_d;
            array_flush_count_q <= array_flush_count_d;
            depthwise_tap_index_q <= depthwise_tap_index_d;
            depthwise_group_index_q <= depthwise_group_index_d;
            depthwise_group_input_offset_q <= depthwise_group_input_offset_d;
            depthwise_group_output_offset_q <= depthwise_group_output_offset_d;
            depthwise_group_weight_offset_q <= depthwise_group_weight_offset_d;
        end
    end

endmodule

`default_nettype wire
