// Copyright (c) 2026
// AFU Core - Shift-and-OR parallel processing with SRAM LUT Pipeline

module afu_core #(
    parameter int unsigned LUT_LANES = 4
)(
    input  logic         clk_i,
    input  logic         rst_ni,

    // CSRs
    input  logic [31:0]  cfg_src_ptr_i,
    input  logic [31:0]  cfg_dst_ptr_i,
    input  logic [31:0]  cfg_length_i,
    input  logic [1:0]   cfg_mode_i,
    input  logic         cfg_start_i,

    // LUT write interface
    input  logic         lut_we_i,
    input  logic [7:0]   lut_addr_i,
    input  logic [31:0]  lut_wdata_i,
    input  logic [3:0]   lut_be_i,

    // Read FIFO
    input  logic         rfifo_empty_i,
    output logic         rfifo_pop_o,
    input  logic [255:0] rfifo_data_i,

    // Write FIFO
    input  logic         wfifo_full_i,
    output logic         wfifo_push_o,
    output logic [287:0] wfifo_data_o
);

    localparam logic [1:0] MODE_8BIT  = 2'd0;
    localparam logic [1:0] MODE_16BIT = 2'd1;
    localparam logic [1:0] MODE_32BIT = 2'd2;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_READ_IN,
        ST_PROCESS,
        ST_WAIT_FLUSH,
        ST_DONE
    } state_e;

    state_e state_q, state_n;

    logic [31:0] src_addr_q, src_addr_n;
    logic [31:0] dst_addr_q, dst_addr_n;
    logic [31:0] elem_cnt_q, elem_cnt_n;
    logic [31:0] out_base_q, out_base_n;

    logic [255:0] in_buf_q,  in_buf_n;
    logic [255:0] out_buf_q, out_buf_n;
    logic [31:0]  out_be_q,  out_be_n;

    // Pipeline Registers
    logic       p1_valid_q, p1_valid_n;
    logic [2:0] p1_num_valid_lanes_q, p1_num_valid_lanes_n;
    logic [31:0] p1_dst_addr_q, p1_dst_addr_n;
    logic       p1_flush_mid_q, p1_flush_mid_n;
    logic       p1_flush_done_q, p1_flush_done_n;

    logic s2_stall;
    logic s2_flush_mid_completed;
    logic s2_flush_done_completed;

    assign s2_stall = wfifo_full_i && p1_valid_q && (p1_flush_mid_q || (p1_flush_done_q && out_be_q != 0));

    // SRAM LUT Instances
    logic s1_sram_req;
    logic [7:0] lut_idx_s1 [LUT_LANES];
    logic [31:0] lut_rdata_ports [LUT_LANES];
    logic [31:0] lut_rdata_dummy [LUT_LANES];

    // Read byte extraction for S1
    logic [255:0] shift_in;
    logic [4:0] in_off_s1;
    assign in_off_s1 = src_addr_q[4:0];
    assign shift_in  = in_buf_q >> {in_off_s1, 3'd0};

    generate
        for (genvar i = 0; i < LUT_LANES; i++) begin : gen_lut_sram
            assign lut_idx_s1[i] = shift_in[i*8 +: 8];

            tc_sram #(
                .NumWords  (256),
                .DataWidth (32),
                .NumPorts  (2),
                .Latency   (1)
            ) i_lut_sram (
                .clk_i   (clk_i),
                .rst_ni  (rst_ni),
                .req_i   ({s1_sram_req, lut_we_i}), // Port 1=Read, Port 0=Write
                .we_i    ({1'b0,        1'b1}),
                .addr_i  ({lut_idx_s1[i], lut_addr_i}),
                .wdata_i ({32'd0,       lut_wdata_i}),
                .be_i    ({4'b1111,     lut_be_i}),
                .rdata_o ({lut_rdata_ports[i], lut_rdata_dummy[i]})
            );
        end
    endgenerate

    // removed duplicate wfifo_data_o assignment

    // Stage 1 Logic
    always_comb begin
        state_n = state_q;
        src_addr_n = src_addr_q;
        dst_addr_n = dst_addr_q;
        elem_cnt_n = elem_cnt_q;
        in_buf_n   = in_buf_q;
        rfifo_pop_o = 1'b0;
        s1_sram_req = 1'b0;

        p1_valid_n = 1'b0;
        p1_flush_mid_n = 1'b0;
        p1_flush_done_n = 1'b0;
        p1_num_valid_lanes_n = '0;
        p1_dst_addr_n = dst_addr_q;

        if (cfg_start_i) begin
            elem_cnt_n = '0;
            src_addr_n = cfg_src_ptr_i;
            dst_addr_n = cfg_dst_ptr_i;
            if (cfg_length_i == 0) begin
                state_n = ST_DONE;
            end else begin
                state_n = ST_READ_IN;
            end
        end else begin
            unique case (state_q)
                ST_IDLE: begin
                    // waiting for start
                end

            ST_READ_IN: begin
                if (!rfifo_empty_i) begin
                    in_buf_n = rfifo_data_i;
                    rfifo_pop_o = 1'b1;
                    state_n = ST_PROCESS;
                end
            end

            ST_PROCESS: begin
                logic [31:0] remaining_elems;
                logic [5:0]  in_avail;
                logic [5:0]  out_avail_bytes;
                logic [5:0]  out_avail_elems;
                logic [5:0]  max_lanes_1, max_lanes_2, max_lanes_3;
                logic [2:0]  num_valid_lanes;

                remaining_elems = cfg_length_i - elem_cnt_q;
                p1_dst_addr_n = dst_addr_q;

                if (remaining_elems == 0) begin
                    state_n = ST_DONE;
                end else if (!s2_stall) begin
                    in_avail = 6'd32 - {1'b0, src_addr_q[4:0]};
                    out_avail_bytes = 6'd32 - {1'b0, dst_addr_q[4:0]};

                    unique case (cfg_mode_i)
                        MODE_8BIT:  out_avail_elems = out_avail_bytes;
                        MODE_16BIT: out_avail_elems = {1'b0, out_avail_bytes[5:1]};
                        MODE_32BIT: out_avail_elems = {2'b0, out_avail_bytes[5:2]};
                        default:    out_avail_elems = out_avail_bytes;
                    endcase

                    max_lanes_1 = (LUT_LANES < remaining_elems) ? LUT_LANES[5:0] : (remaining_elems > 6'd31 ? 6'd31 : 6'(remaining_elems));
                    max_lanes_2 = (max_lanes_1 < in_avail) ? max_lanes_1 : in_avail;
                    max_lanes_3 = (max_lanes_2 < out_avail_elems) ? max_lanes_2 : out_avail_elems;
                    num_valid_lanes = max_lanes_3[2:0];

                    if (num_valid_lanes > 0) begin
                        s1_sram_req = 1'b1;
                        p1_num_valid_lanes_n = num_valid_lanes;
                        p1_valid_n = 1'b1;

                        src_addr_n = src_addr_q + 32'(num_valid_lanes);
                        elem_cnt_n = elem_cnt_q + 32'(num_valid_lanes);

                        if (cfg_mode_i == MODE_8BIT) begin
                            dst_addr_n = dst_addr_q + 32'(num_valid_lanes);
                        end else if (cfg_mode_i == MODE_16BIT) begin
                            dst_addr_n = dst_addr_q + 32'(num_valid_lanes * 2);
                        end else begin
                            dst_addr_n = dst_addr_q + 32'(num_valid_lanes * 4);
                        end

                        if (dst_addr_n[4:0] == 0) begin
                            p1_flush_mid_n = 1'b1;
                            state_n = ST_WAIT_FLUSH;
                        end else if (elem_cnt_n == cfg_length_i) begin
                            p1_flush_done_n = 1'b1;
                            state_n = ST_WAIT_FLUSH;
                        end else if (src_addr_n[4:0] == 0) begin
                            state_n = ST_READ_IN;
                        end
                    end
                end
            end

            ST_WAIT_FLUSH: begin
                if (s2_flush_mid_completed) begin
                    if (src_addr_q[4:0] == 0 && elem_cnt_q < cfg_length_i) begin
                        state_n = ST_READ_IN;
                    end else begin
                        state_n = ST_PROCESS;
                    end
                end else if (s2_flush_done_completed) begin
                    state_n = ST_DONE;
                end
            end
            
            ST_DONE: begin
                // Waiting for new start
            end

                default: ;
            endcase
        end
    end

    // Stage 2 Logic
    logic [255:0] s2_out_buf_comb;
    logic [31:0]  s2_out_be_comb;

    assign wfifo_data_o = {s2_out_be_comb, s2_out_buf_comb};
    
    // Pipeline hazard fix: if S2 stalls, SRAM output will change in the next cycle.
    // We must save lut_rdata_ports when S2 stalls.
    logic [31:0] s2_lut_rdata_saved_q [LUT_LANES];
    logic        s2_lut_rdata_saved_valid_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            s2_lut_rdata_saved_valid_q <= 1'b0;
            for (int i=0; i<LUT_LANES; i++) s2_lut_rdata_saved_q[i] <= '0;
        end else begin
            if (p1_valid_q && s2_stall && !s2_lut_rdata_saved_valid_q) begin
                s2_lut_rdata_saved_q <= lut_rdata_ports;
                s2_lut_rdata_saved_valid_q <= 1'b1;
            end else if (!s2_stall) begin
                s2_lut_rdata_saved_valid_q <= 1'b0;
            end
        end
    end

    always_comb begin
        logic [31:0] lut_val;
        logic [4:0]  cur_out_off;
        lut_val = '0;
        cur_out_off = '0;
        
        s2_flush_mid_completed = 1'b0;
        s2_flush_done_completed = 1'b0;
        wfifo_push_o = 1'b0;
        out_buf_n = out_buf_q;
        out_be_n  = out_be_q;
        out_base_n = out_base_q;
        
        s2_out_buf_comb = out_buf_q;
        s2_out_be_comb  = out_be_q;

        if (cfg_start_i) begin
            out_base_n = cfg_dst_ptr_i;
            out_buf_n  = '0;
            out_be_n   = '0;
        end else if (p1_valid_q && !s2_stall) begin
            // Process lanes directly into combinational buffer
            for (int i = 0; i < LUT_LANES; i++) begin
                if (i < p1_num_valid_lanes_q) begin
                    lut_val = s2_lut_rdata_saved_valid_q ? s2_lut_rdata_saved_q[i] : lut_rdata_ports[i];

                    if (cfg_mode_i == MODE_8BIT) begin
                        cur_out_off = p1_dst_addr_q[4:0] + 5'(i);
                        s2_out_buf_comb[cur_out_off * 8 +: 8] = lut_val[7:0];
                        s2_out_be_comb[cur_out_off] = 1'b1;
                    end else if (cfg_mode_i == MODE_16BIT) begin
                        cur_out_off = p1_dst_addr_q[4:0] + 5'(i * 2);
                        s2_out_buf_comb[cur_out_off * 8 +: 16] = lut_val[15:0];
                        s2_out_be_comb[cur_out_off +: 2] = 2'b11;
                    end else begin
                        cur_out_off = p1_dst_addr_q[4:0] + 5'(i * 4);
                        s2_out_buf_comb[cur_out_off * 8 +: 32] = lut_val;
                        s2_out_be_comb[cur_out_off +: 4] = 4'b1111;
                    end
                end
            end

            if (p1_flush_mid_q) begin
                wfifo_push_o = 1'b1;
                out_buf_n = '0;
                out_be_n  = '0;
                out_base_n = out_base_q + 32;
                s2_flush_mid_completed = 1'b1;
            end else if (p1_flush_done_q) begin
                if (s2_out_be_comb != 0) begin
                    wfifo_push_o = 1'b1;
                end
                out_buf_n = '0;
                out_be_n  = '0;
                s2_flush_done_completed = 1'b1;
            end else begin
                out_buf_n = s2_out_buf_comb;
                out_be_n  = s2_out_be_comb;
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q    <= ST_IDLE;
            src_addr_q <= '0;
            dst_addr_q <= '0;
            elem_cnt_q <= '0;
            in_buf_q   <= '0;
            out_base_q <= '0;
            out_buf_q  <= '0;
            out_be_q   <= '0;
            
            p1_valid_q <= 1'b0;
            p1_num_valid_lanes_q <= '0;
            p1_dst_addr_q <= '0;
            p1_flush_mid_q <= 1'b0;
            p1_flush_done_q <= 1'b0;
        end else begin
            state_q    <= state_n;
            src_addr_q <= src_addr_n;
            dst_addr_q <= dst_addr_n;
            elem_cnt_q <= elem_cnt_n;
            in_buf_q   <= in_buf_n;
            
            out_base_q <= out_base_n;
            out_buf_q  <= out_buf_n;
            out_be_q   <= out_be_n;

            if (!s2_stall) begin
                p1_valid_q <= p1_valid_n;
                p1_num_valid_lanes_q <= p1_num_valid_lanes_n;
                p1_dst_addr_q <= p1_dst_addr_n;
                p1_flush_mid_q <= p1_flush_mid_n;
                p1_flush_done_q <= p1_flush_done_n;
            end
        end
    end

endmodule
