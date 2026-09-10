`default_nettype none

module systolic_output_drain #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned OFM_ELEM_WIDTH = 32,
    parameter int unsigned INPUT_FIFO_DEPTH = 4,
    parameter int unsigned OFM_FIFO_DEPTH = 128
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic job_start_i,
    input  logic tile_advance_i,
    input  logic tile_advance_overlap_i,
    input  logic tile_start_i,
    input  logic tile_start_add_rows_i,
    input  logic depthwise_group_start_i,
    input  logic [31:0] depthwise_group_output_ptr_i,

    input  logic drain_active_i,
    input  logic compute_phase_i,
    input  logic pool_mode_i,
    input  logic depthwise_mode_i,
    input  logic external_accum_enable_i,
    input  logic accum_active_i,
    input  logic requant_active_i,
    input  logic psum_buf_active_i,
    input  logic psum_buf_needs_external_i,
    input  logic psum_buf_final_tile_i,
    input  logic [31:0] k_tile_idx_i,

    input  logic [31:0] ofm_base_ptr_i,
    input  logic [31:0] psum_base_ptr_i,
    input  logic [31:0] row_count_i,
    input  logic [31:0] spatial_row_count_i,
    input  logic [31:0] ofm_row_stride_bytes_i,
    input  logic [31:0] ofm_tile_cols_i,
    input  logic [31:0] psum_row_stride_bytes_i,

    input  logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0] result_i,
    input  logic result_valid_i,
    output logic result_ready_o,
    input  logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0] depthwise_result_i,
    input  logic depthwise_result_valid_i,
    output logic depthwise_result_ready_o,
    input  logic [DATA_WIDTH-1:0] pool_result_i,
    input  logic pool_result_valid_i,
    output logic pool_result_ready_o,

    input  logic requant_enable_i,
    input  logic [ARRAY_DIM-1:0][31:0] requant_bias_i,
    input  logic [ARRAY_DIM-1:0][31:0] requant_multiplier_i,
    input  logic [ARRAY_DIM-1:0][7:0] requant_shift_i,
    input  logic [ARRAY_DIM-1:0][31:0] requant_zero_point_i,
    input  logic [31:0] requant_clamp_min_i,
    input  logic [31:0] requant_clamp_max_i,

    input  logic binary_enable_i,
    input  logic [1:0] binary_mode_i,
    input  logic [31:0] binary_rhs_ptr_i,
    input  logic [31:0] binary_rhs_row_stride_bytes_i,
    input  logic [31:0] binary_rhs_tile_cols_i,
    input  logic [31:0] binary_lhs_multiplier_i,
    input  logic [6:0] binary_lhs_shift_i,
    input  logic [31:0] binary_rhs_multiplier_i,
    input  logic [6:0] binary_rhs_shift_i,
    input  logic [31:0] binary_output_multiplier_i,
    input  logic [6:0] binary_output_shift_i,
    input  logic signed [31:0] binary_lhs_zero_point_i,
    input  logic signed [31:0] binary_rhs_zero_point_i,
    input  logic signed [31:0] binary_output_zero_point_i,
    input  logic signed [31:0] binary_clamp_min_i,
    input  logic signed [31:0] binary_clamp_max_i,
    input  logic [5:0] binary_double_round_shift_i,

    output logic obi_b_req_o,
    input  logic obi_b_gnt_i,
    output logic [ADDR_WIDTH-1:0] obi_b_addr_o,
    output logic obi_b_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_b_be_o,
    output logic [DATA_WIDTH-1:0] obi_b_wdata_o,
    input  logic obi_b_rvalid_i,
    input  logic [DATA_WIDTH-1:0] obi_b_rdata_i,

    output logic [3:0] obi_o_req_o,
    input  logic [3:0] obi_o_gnt_i,
    output logic [3:0][ADDR_WIDTH-1:0] obi_o_addr_o,
    output logic [3:0] obi_o_we_o,
    output logic [3:0][(DATA_WIDTH/8)-1:0] obi_o_be_o,
    output logic [3:0][DATA_WIDTH-1:0] obi_o_wdata_o,
    input  logic [3:0] obi_o_rvalid_i,
    input  logic [3:0][DATA_WIDTH-1:0] obi_o_rdata_i,

    output logic [31:0] remaining_rows_o,
    output logic ofm_empty_o,
    output logic psum_empty_o,
    output logic psum_buf_drain_entry_o,
    output logic binary_busy_o,
    output logic postprocess_out_valid_o,
    output logic requant_config_invalid_o,
    output logic binary_config_invalid_o,
    output logic debug_requant_out_valid_o,
    output logic debug_requant_out_ready_o,
    output logic [1:0] debug_state_o
);

    localparam int unsigned OFM_BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned OFM_ROW_BYTES = (ARRAY_DIM * OFM_ELEM_WIDTH) / 8;
    localparam int unsigned REQUANT_ROW_BYTES = ARRAY_DIM;
    localparam int unsigned OFM_ELEMS_PER_OBI = DATA_WIDTH / OFM_ELEM_WIDTH;
    localparam int unsigned PSUM_BUF_M = 256;
    localparam int unsigned PSUM_BUF_ADDR_WIDTH = $clog2(PSUM_BUF_M);

    typedef logic [ARRAY_DIM-1:0][OFM_ELEM_WIDTH-1:0] ofm_row_t;

    typedef enum logic [1:0] {
        DRAIN_IDLE,
        DRAIN_ACCUM_READ,
        DRAIN_ACCUM_WRITE,
        DRAIN_ACCUM_REQUANT
    } drain_state_e;

    typedef struct packed {
        ofm_row_t    row;
        logic [31:0] row_idx;
        logic [31:0] k_tile_idx;
        logic        psum_buf_active;
        logic        needs_external_psum;
        logic        final_tile;
    } ofm_fifo_entry_t;

    drain_state_e drain_state_q, drain_state_d;
    logic [31:0] o_ptr_q, o_ptr_d;
    logic [31:0] a_ptr_q, a_ptr_d;
    logic [31:0] o_col_q, o_col_d;
    logic [31:0] a_col_q, a_col_d;
    logic [31:0] drain_cnt_q, drain_cnt_d;
    logic [31:0] ofm_push_row_idx_q, ofm_push_row_idx_d;
    logic accum_requant_sent_q, accum_requant_sent_d;
    ofm_row_t psum_read_row_q, psum_read_row_d;
    logic psum_read_active_q, psum_read_active_d;
    logic [3:0] psum_read_resp_mask_q, psum_read_resp_mask_d;
    logic [31:0] psum_prefetch_rows_q, psum_prefetch_rows_d;

    ofm_fifo_entry_t ofm_fifo_data;
    ofm_fifo_entry_t ofm_fifo_out;
    logic ofm_fifo_push;
    logic ofm_fifo_pop;
    logic ofm_fifo_full;
    logic ofm_fifo_empty;
    ofm_row_t psum_fifo_data;
    ofm_row_t psum_fifo_out;
    logic psum_fifo_push;
    logic psum_fifo_pop;
    logic psum_fifo_full;
    logic psum_fifo_empty;

    logic psum_buf_sel_q, psum_buf_sel_d;
    logic psum_buf_we;
    logic [PSUM_BUF_ADDR_WIDTH-1:0] psum_buf_addr;
    ofm_row_t psum_buf_wdata;
    ofm_row_t psum_buf_rdata;
    ofm_row_t psum_buf_old;
    ofm_row_t psum_buf_sum;
    ofm_row_t accum_sum;
`ifdef SYNTHESIS_SRAM_BLACKBOX
    ofm_row_t psum_buf_sram_rdata [2];
`else
    ofm_row_t psum_buf_q [2][PSUM_BUF_M];
`endif

    logic psum_buf_drain_entry;
    logic requant_in_valid;
    logic requant_in_ready;
    ofm_row_t requant_acc;
    logic quantized_out_valid;
    logic quantized_out_ready;
    logic [DATA_WIDTH-1:0] quantized_packed_data;
    logic quantized_invalid;

    function automatic logic [31:0] next_strided_ptr(
        input logic [31:0] ptr,
        input logic [31:0] col,
        input logic [31:0] row_bytes,
        input logic [31:0] row_stride_bytes,
        input logic [31:0] tile_cols
    );
        logic [31:0] row_gap;
        logic [31:0] row_span;
        begin
            unique case (row_bytes)
                32'd32:  row_span = (tile_cols - 32'd1) << 5;
                32'd128: row_span = (tile_cols - 32'd1) << 7;
                default: row_span = '0;
            endcase
            row_gap = row_stride_bytes - row_span;
            if ((row_stride_bytes != 32'd0) && (tile_cols != 32'd0) &&
                ((col + 32'd1) == tile_cols)) begin
                next_strided_ptr = ptr + row_gap;
            end else begin
                next_strided_ptr = ptr + row_bytes;
            end
        end
    endfunction

    function automatic logic [31:0] next_strided_col(
        input logic [31:0] col,
        input logic [31:0] tile_cols
    );
        begin
            if ((tile_cols != 32'd0) && ((col + 32'd1) == tile_cols)) begin
                next_strided_col = 32'd0;
            end else begin
                next_strided_col = col + 32'd1;
            end
        end
    endfunction

    assign result_ready_o = !ofm_fifo_full;
    assign remaining_rows_o = drain_cnt_q;
    assign ofm_empty_o = ofm_fifo_empty;
    assign psum_empty_o = psum_fifo_empty;
    assign psum_buf_drain_entry_o = psum_buf_drain_entry;
    assign postprocess_out_valid_o = quantized_out_valid;
    assign debug_state_o = drain_state_q;
    assign psum_buf_drain_entry = !ofm_fifo_empty && ofm_fifo_out.psum_buf_active;

    always_comb begin
        for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
            accum_sum[lane] = ofm_fifo_out.row[lane] + psum_fifo_out[lane];
            psum_buf_sum[lane] = ofm_fifo_out.row[lane] + psum_buf_old[lane];
        end
    end

    assign requant_acc = depthwise_mode_i ? depthwise_result_i :
                         (psum_buf_drain_entry && requant_active_i) ? psum_buf_sum :
                         ((accum_active_i && requant_active_i) ? accum_sum : ofm_fifo_out.row);

`ifdef SYNTHESIS_SRAM_BLACKBOX
    for (genvar psum_bank = 0; psum_bank < 2; psum_bank++) begin : gen_psum_buf_sram
        localparam logic PSUM_BANK_SEL = (psum_bank != 0);

        systolic_psum_sram #(
            .DataWidth(ARRAY_DIM * OFM_ELEM_WIDTH),
            .Depth(PSUM_BUF_M),
            .AddrWidth(PSUM_BUF_ADDR_WIDTH)
        ) i_psum_buf_sram (
            .clk_i,
            .req_i(1'b1),
            .we_i(psum_buf_we && (psum_buf_sel_q == PSUM_BANK_SEL)),
            .addr_i(psum_buf_addr),
            .wdata_i(psum_buf_wdata),
            .rdata_o(psum_buf_sram_rdata[psum_bank])
        );
    end
`endif

    systolic_output_postprocess #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .LANES(ARRAY_DIM),
        .BINARY_FIFO_DEPTH(INPUT_FIFO_DEPTH * 2)
    ) i_postprocess (
        .clk_i,
        .rst_ni,
        .flush_i                         (job_start_i),
        .job_start_i,
        .requant_enable_i,
        .acc_i                           (requant_acc),
        .acc_valid_i                     (requant_in_valid),
        .acc_ready_o                     (requant_in_ready),
        .bias_i                          (requant_bias_i),
        .multiplier_i                    (requant_multiplier_i),
        .shift_i                         (requant_shift_i),
        .zero_point_i                    (requant_zero_point_i),
        .clamp_min_i                     (requant_clamp_min_i),
        .clamp_max_i                     (requant_clamp_max_i),
        .binary_enable_i,
        .binary_active_i                 (binary_enable_i && requant_active_i),
        .binary_mode_i,
        .binary_rhs_ptr_i,
        .binary_rhs_row_stride_bytes_i,
        .binary_rhs_tile_cols_i,
        .row_count_i,
        .binary_lhs_multiplier_i,
        .binary_lhs_shift_i,
        .binary_rhs_multiplier_i,
        .binary_rhs_shift_i,
        .binary_output_multiplier_i,
        .binary_output_shift_i,
        .binary_lhs_zero_point_i,
        .binary_rhs_zero_point_i,
        .binary_output_zero_point_i,
        .binary_clamp_min_i,
        .binary_clamp_max_i,
        .binary_double_round_shift_i,
        .binary_forbidden_i              (pool_mode_i || depthwise_mode_i),
        .obi_req_o                       (obi_b_req_o),
        .obi_gnt_i                       (obi_b_gnt_i),
        .obi_addr_o                      (obi_b_addr_o),
        .obi_we_o                        (obi_b_we_o),
        .obi_be_o                        (obi_b_be_o),
        .obi_wdata_o                     (obi_b_wdata_o),
        .obi_rvalid_i                    (obi_b_rvalid_i),
        .obi_rdata_i                     (obi_b_rdata_i),
        .out_valid_o                     (quantized_out_valid),
        .out_ready_i                     (quantized_out_ready),
        .packed_o                        (quantized_packed_data),
        .invalid_o                       (quantized_invalid),
        .requant_config_invalid_o,
        .binary_config_invalid_o,
        .binary_busy_o,
        .debug_requant_out_valid_o,
        .debug_requant_out_ready_o
    );

    fifo_v3 #(
        .FALL_THROUGH(1'b1),
        .DEPTH(OFM_FIFO_DEPTH),
        .dtype(ofm_fifo_entry_t)
    ) i_ofm_fifo (
        .clk_i,
        .rst_ni,
        .flush_i(job_start_i),
        .testmode_i(1'b0),
        .full_o(ofm_fifo_full),
        .empty_o(ofm_fifo_empty),
        .usage_o(),
        .data_i(ofm_fifo_data),
        .push_i(ofm_fifo_push),
        .data_o(ofm_fifo_out),
        .pop_i(ofm_fifo_pop)
    );

    fifo_v3 #(
        .FALL_THROUGH(1'b1),
        .DEPTH(OFM_FIFO_DEPTH),
        .dtype(ofm_row_t)
    ) i_psum_fifo (
        .clk_i,
        .rst_ni,
        .flush_i(job_start_i),
        .testmode_i(1'b0),
        .full_o(psum_fifo_full),
        .empty_o(psum_fifo_empty),
        .usage_o(),
        .data_i(psum_fifo_data),
        .push_i(psum_fifo_push),
        .data_o(psum_fifo_out),
        .pop_i(psum_fifo_pop)
    );

    task automatic reset_engine(input logic [31:0] psum_prefetch_rows);
        begin
            drain_state_d = DRAIN_IDLE;
            accum_requant_sent_d = 1'b0;
            psum_read_row_d = '0;
            psum_read_active_d = 1'b0;
            psum_read_resp_mask_d = '0;
            psum_prefetch_rows_d = psum_prefetch_rows;
        end
    endtask

    task automatic capture_psum_read_responses();
        begin
            drain_state_d = DRAIN_ACCUM_READ;
            for (int unsigned port = 0; port < 4; port++) begin
                if (obi_o_rvalid_i[port]) begin
                    for (int unsigned elem = 0; elem < OFM_ELEMS_PER_OBI; elem++) begin
                        psum_read_row_d[(port * OFM_ELEMS_PER_OBI) + elem] =
                            obi_o_rdata_i[port][elem * OFM_ELEM_WIDTH +: OFM_ELEM_WIDTH];
                    end
                    psum_read_resp_mask_d[port] = 1'b1;
                end
            end
            if ((psum_read_resp_mask_q | obi_o_rvalid_i) == 4'b1111) begin
                psum_fifo_data = psum_read_row_d;
                psum_fifo_push = 1'b1;
                psum_read_active_d = 1'b0;
                psum_read_resp_mask_d = '0;
            end
        end
    endtask

    task automatic issue_psum_prefetch_read(input logic use_psum_buffer_gate);
        begin
            if ((!use_psum_buffer_gate || psum_buf_needs_external_i) &&
                (psum_prefetch_rows_q != 0) && !psum_read_active_q && !psum_fifo_full &&
                !(|obi_o_req_o)) begin
                drain_state_d = DRAIN_ACCUM_READ;
                obi_o_req_o = 4'b1111;
                obi_o_we_o = '0;
                obi_o_addr_o[0] = a_ptr_q;
                obi_o_addr_o[1] = a_ptr_q + OFM_BEAT_BYTES;
                obi_o_addr_o[2] = a_ptr_q + (2 * OFM_BEAT_BYTES);
                obi_o_addr_o[3] = a_ptr_q + (3 * OFM_BEAT_BYTES);
                if (obi_o_gnt_i == 4'b1111) begin
                    a_ptr_d = next_strided_ptr(a_ptr_q, a_col_q, 32'(OFM_ROW_BYTES),
                                               psum_row_stride_bytes_i, ofm_tile_cols_i);
                    a_col_d = next_strided_col(a_col_q, ofm_tile_cols_i);
                    psum_prefetch_rows_d = psum_prefetch_rows_q - 1'b1;
                    psum_read_active_d = 1'b1;
                    psum_read_resp_mask_d = '0;
                    psum_read_row_d = '0;
                end
            end
        end
    endtask

    task automatic write_quantized_output(input logic [DATA_WIDTH-1:0] packed_data);
        begin
            obi_o_req_o[0] = 1'b1;
            obi_o_wdata_o[0] = packed_data;
            if (obi_o_gnt_i[0] || quantized_invalid) begin
                quantized_out_ready = 1'b1;
                o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                           ofm_row_stride_bytes_i, ofm_tile_cols_i);
                o_col_d = next_strided_col(o_col_q, ofm_tile_cols_i);
                drain_cnt_d = drain_cnt_q - 1'b1;
                accum_requant_sent_d = 1'b0;
            end
            if (quantized_invalid) begin
                obi_o_req_o[0] = 1'b0;
            end
        end
    endtask

    task automatic service_normal_drain();
        begin
            if (psum_buf_active_i || psum_buf_drain_entry) begin
                drain_state_d = DRAIN_IDLE;
                if (psum_read_active_q) capture_psum_read_responses();

                if (accum_requant_sent_q && quantized_out_valid) begin
                    drain_state_d = DRAIN_ACCUM_REQUANT;
                    write_quantized_output(quantized_packed_data);
                end

                if (!ofm_fifo_empty &&
                    (!ofm_fifo_out.needs_external_psum || !psum_fifo_empty)) begin
                    drain_state_d = DRAIN_ACCUM_WRITE;
                    if (ofm_fifo_out.final_tile && requant_active_i) begin
                        drain_state_d = DRAIN_ACCUM_REQUANT;
                        if (!accum_requant_sent_q && requant_in_ready &&
                            !requant_config_invalid_o) begin
                            requant_in_valid = 1'b1;
                            ofm_fifo_pop = 1'b1;
                            psum_fifo_pop = ofm_fifo_out.needs_external_psum;
                            accum_requant_sent_d = 1'b1;
                        end
                        if (quantized_out_valid)
                            write_quantized_output(quantized_packed_data);
                    end else if (ofm_fifo_out.final_tile) begin
                        obi_o_we_o = '1;
                        obi_o_wdata_o[0] = psum_buf_sum[OFM_ELEMS_PER_OBI-1:0];
                        obi_o_wdata_o[1] = psum_buf_sum[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
                        obi_o_wdata_o[2] = psum_buf_sum[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
                        obi_o_wdata_o[3] = psum_buf_sum[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];
                        obi_o_req_o = 4'b1111;
                        if (obi_o_gnt_i == 4'b1111) begin
                            ofm_fifo_pop = 1'b1;
                            psum_fifo_pop = ofm_fifo_out.needs_external_psum;
                            o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(OFM_ROW_BYTES),
                                                       ofm_row_stride_bytes_i, ofm_tile_cols_i);
                            o_col_d = next_strided_col(o_col_q, ofm_tile_cols_i);
                            drain_cnt_d = drain_cnt_q - 1'b1;
                        end
                    end else begin
                        psum_buf_we = 1'b1;
                        psum_buf_wdata = psum_buf_sum;
                        ofm_fifo_pop = 1'b1;
                        psum_fifo_pop = ofm_fifo_out.needs_external_psum;
                        drain_cnt_d = drain_cnt_q - 1'b1;
                    end
                end
                issue_psum_prefetch_read(1'b1);
            end else if (!accum_active_i && requant_active_i) begin
                if (quantized_out_valid)
                    write_quantized_output(quantized_packed_data);
                if (!ofm_fifo_empty && requant_in_ready && !requant_config_invalid_o) begin
                    requant_in_valid = 1'b1;
                    ofm_fifo_pop = 1'b1;
                end
            end else if (!accum_active_i && !ofm_fifo_empty) begin
                obi_o_req_o = 4'b1111;
                if (obi_o_gnt_i == 4'b1111) begin
                    ofm_fifo_pop = 1'b1;
                    o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(OFM_ROW_BYTES),
                                               ofm_row_stride_bytes_i, ofm_tile_cols_i);
                    o_col_d = next_strided_col(o_col_q, ofm_tile_cols_i);
                    drain_cnt_d = drain_cnt_q - 1'b1;
                end
            end else if (accum_active_i) begin
                drain_state_d = DRAIN_IDLE;
                if (psum_read_active_q) capture_psum_read_responses();

                if (!requant_active_i && !ofm_fifo_empty && !psum_fifo_empty) begin
                    drain_state_d = DRAIN_ACCUM_WRITE;
                    obi_o_we_o = '1;
                    obi_o_wdata_o[0] = accum_sum[OFM_ELEMS_PER_OBI-1:0];
                    obi_o_wdata_o[1] = accum_sum[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
                    obi_o_wdata_o[2] = accum_sum[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
                    obi_o_wdata_o[3] = accum_sum[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];
                    obi_o_req_o = 4'b1111;
                    if (obi_o_gnt_i == 4'b1111) begin
                        ofm_fifo_pop = 1'b1;
                        psum_fifo_pop = 1'b1;
                        o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(OFM_ROW_BYTES),
                                                   ofm_row_stride_bytes_i, ofm_tile_cols_i);
                        o_col_d = next_strided_col(o_col_q, ofm_tile_cols_i);
                        drain_cnt_d = drain_cnt_q - 1'b1;
                    end
                end else if (requant_active_i) begin
                    if (!accum_requant_sent_q && !ofm_fifo_empty && !psum_fifo_empty &&
                        requant_in_ready && !requant_config_invalid_o) begin
                        drain_state_d = DRAIN_ACCUM_REQUANT;
                        requant_in_valid = 1'b1;
                        ofm_fifo_pop = 1'b1;
                        psum_fifo_pop = 1'b1;
                        accum_requant_sent_d = 1'b1;
                    end
                    if (quantized_out_valid) begin
                        drain_state_d = DRAIN_ACCUM_REQUANT;
                        write_quantized_output(quantized_packed_data);
                    end
                end
                issue_psum_prefetch_read(1'b0);
            end
        end
    endtask

    /* verilator lint_off MULTIDRIVEN */
    always_comb begin
        drain_state_d = drain_state_q;
        o_ptr_d = o_ptr_q;
        a_ptr_d = a_ptr_q;
        o_col_d = o_col_q;
        a_col_d = a_col_q;
        drain_cnt_d = drain_cnt_q;
        ofm_push_row_idx_d = ofm_push_row_idx_q;
        accum_requant_sent_d = accum_requant_sent_q;
        psum_read_row_d = psum_read_row_q;
        psum_read_active_d = psum_read_active_q;
        psum_read_resp_mask_d = psum_read_resp_mask_q;
        psum_prefetch_rows_d = psum_prefetch_rows_q;
        psum_buf_sel_d = psum_buf_sel_q;

        psum_buf_we = 1'b0;
        psum_buf_addr = '0;
        psum_buf_wdata = '0;
        psum_buf_old = '0;
        if (psum_buf_drain_entry) begin
            psum_buf_addr = PSUM_BUF_ADDR_WIDTH'(ofm_fifo_out.row_idx);
        end else if ((drain_cnt_q != 32'd0) && (drain_cnt_q <= row_count_i)) begin
            psum_buf_addr = PSUM_BUF_ADDR_WIDTH'(row_count_i - drain_cnt_q);
        end
`ifdef SYNTHESIS_SRAM_BLACKBOX
        psum_buf_rdata = psum_buf_sram_rdata[psum_buf_sel_q];
`else
        psum_buf_rdata = psum_buf_q[psum_buf_sel_q][psum_buf_addr];
`endif
        if (psum_buf_drain_entry && ofm_fifo_out.needs_external_psum) begin
            psum_buf_old = psum_fifo_out;
        end else if (psum_buf_drain_entry &&
                     ((ofm_fifo_out.k_tile_idx != 32'd0) || external_accum_enable_i)) begin
            psum_buf_old = psum_buf_rdata;
        end

        ofm_fifo_data = '0;
        ofm_fifo_data.row = result_i;
        ofm_fifo_data.row_idx = ofm_push_row_idx_q;
        ofm_fifo_data.k_tile_idx = k_tile_idx_i;
        ofm_fifo_data.psum_buf_active = psum_buf_active_i;
        ofm_fifo_data.needs_external_psum = psum_buf_needs_external_i;
        ofm_fifo_data.final_tile = psum_buf_final_tile_i;
        ofm_fifo_push = drain_active_i && result_valid_i && result_ready_o;
        ofm_fifo_pop = 1'b0;
        psum_fifo_data = psum_read_row_q;
        psum_fifo_push = 1'b0;
        psum_fifo_pop = 1'b0;
        requant_in_valid = 1'b0;
        quantized_out_ready = 1'b0;
        depthwise_result_ready_o = 1'b0;
        pool_result_ready_o = 1'b0;
        if (ofm_fifo_push)
            ofm_push_row_idx_d = ofm_push_row_idx_q + 32'd1;

        obi_o_req_o = '0;
        obi_o_we_o = '1;
        obi_o_be_o = '1;
        obi_o_addr_o[0] = o_ptr_q;
        obi_o_addr_o[1] = o_ptr_q + OFM_BEAT_BYTES;
        obi_o_addr_o[2] = o_ptr_q + (2 * OFM_BEAT_BYTES);
        obi_o_addr_o[3] = o_ptr_q + (3 * OFM_BEAT_BYTES);
        obi_o_wdata_o[0] = ofm_fifo_out.row[OFM_ELEMS_PER_OBI-1:0];
        obi_o_wdata_o[1] = ofm_fifo_out.row[(2*OFM_ELEMS_PER_OBI)-1:OFM_ELEMS_PER_OBI];
        obi_o_wdata_o[2] = ofm_fifo_out.row[(3*OFM_ELEMS_PER_OBI)-1:(2*OFM_ELEMS_PER_OBI)];
        obi_o_wdata_o[3] = ofm_fifo_out.row[(4*OFM_ELEMS_PER_OBI)-1:(3*OFM_ELEMS_PER_OBI)];

        if (drain_active_i)
            service_normal_drain();

        if (compute_phase_i && depthwise_mode_i) begin
            if (quantized_out_valid)
                write_quantized_output(quantized_packed_data);
            if (depthwise_result_valid_i && !requant_config_invalid_o) begin
                requant_in_valid = requant_enable_i;
                depthwise_result_ready_o = requant_enable_i && requant_in_ready;
            end
        end else if (compute_phase_i && pool_mode_i) begin
            if (pool_result_valid_i) begin
                obi_o_req_o[0] = 1'b1;
                obi_o_we_o[0] = 1'b1;
                obi_o_be_o[0] = '1;
                obi_o_addr_o[0] = o_ptr_q;
                obi_o_wdata_o[0] = pool_result_i;
                if (obi_o_gnt_i[0]) begin
                    pool_result_ready_o = 1'b1;
                    o_ptr_d = next_strided_ptr(o_ptr_q, o_col_q, 32'(REQUANT_ROW_BYTES),
                                               ofm_row_stride_bytes_i, ofm_tile_cols_i);
                    o_col_d = next_strided_col(o_col_q, ofm_tile_cols_i);
                    drain_cnt_d = drain_cnt_q - 1'b1;
                end
            end
        end

        // Lifecycle events override work serviced in the same cycle, matching
        // their former placement after the drain service in the parent FSM.
        if (job_start_i) begin
            o_ptr_d = ofm_base_ptr_i;
            a_ptr_d = psum_base_ptr_i;
            o_col_d = '0;
            a_col_d = '0;
            ofm_push_row_idx_d = '0;
            drain_cnt_d = (pool_mode_i || depthwise_mode_i) ? spatial_row_count_i : row_count_i;
            reset_engine(external_accum_enable_i ? row_count_i : 32'd0);
            if (psum_buf_active_i)
                psum_buf_sel_d = ~psum_buf_sel_q;
            if ((requant_enable_i && requant_config_invalid_o) || binary_config_invalid_o) begin
                drain_cnt_d = '0;
                psum_prefetch_rows_d = '0;
            end
        end else begin
            if (tile_advance_i) begin
                o_ptr_d = ofm_base_ptr_i;
                a_ptr_d = psum_base_ptr_i;
                o_col_d = '0;
                a_col_d = '0;
                if (tile_advance_overlap_i) begin
                    drain_cnt_d = drain_cnt_q + row_count_i;
                    ofm_push_row_idx_d = '0;
                end
            end
            if (tile_start_i) begin
                reset_engine((accum_active_i &&
                              (!psum_buf_active_i || psum_buf_needs_external_i)) ?
                             row_count_i : 32'd0);
                drain_cnt_d = tile_start_add_rows_i ?
                              (drain_cnt_q + row_count_i) : row_count_i;
                ofm_push_row_idx_d = '0;
            end
            if (depthwise_group_start_i) begin
                o_ptr_d = depthwise_group_output_ptr_i;
                o_col_d = '0;
                drain_cnt_d = spatial_row_count_i;
            end
        end
    end
    /* verilator lint_on MULTIDRIVEN */

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            drain_state_q <= DRAIN_IDLE;
            o_ptr_q <= '0;
            a_ptr_q <= '0;
            o_col_q <= '0;
            a_col_q <= '0;
            drain_cnt_q <= '0;
            ofm_push_row_idx_q <= '0;
            accum_requant_sent_q <= 1'b0;
            psum_read_row_q <= '0;
            psum_read_active_q <= 1'b0;
            psum_read_resp_mask_q <= '0;
            psum_prefetch_rows_q <= '0;
            psum_buf_sel_q <= 1'b0;
        end else begin
            drain_state_q <= drain_state_d;
            o_ptr_q <= o_ptr_d;
            a_ptr_q <= a_ptr_d;
            o_col_q <= o_col_d;
            a_col_q <= a_col_d;
            drain_cnt_q <= drain_cnt_d;
            ofm_push_row_idx_q <= ofm_push_row_idx_d;
            accum_requant_sent_q <= accum_requant_sent_d;
            psum_read_row_q <= psum_read_row_d;
            psum_read_active_q <= psum_read_active_d;
            psum_read_resp_mask_q <= psum_read_resp_mask_d;
            psum_prefetch_rows_q <= psum_prefetch_rows_d;
            psum_buf_sel_q <= psum_buf_sel_d;
`ifndef SYNTHESIS_SRAM_BLACKBOX
            if (psum_buf_we)
                psum_buf_q[psum_buf_sel_q][psum_buf_addr] <= psum_buf_wdata;
`endif
        end
    end

endmodule

`default_nettype wire
