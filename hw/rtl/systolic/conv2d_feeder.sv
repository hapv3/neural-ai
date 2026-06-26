`default_nettype none

module conv2d_feeder #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned K_TILE     = 32
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic [31:0] cfg_input_ptr_i,
    input  logic [31:0] cfg_output_ptr_i,
    input  logic [31:0] cfg_rows_i,
    input  logic [31:0] cfg_output_w_i,
    input  logic [31:0] cfg_input_h_i,
    input  logic [31:0] cfg_input_w_i,
    input  logic [31:0] cfg_input_c_i,
    input  logic [15:0] cfg_kernel_h_i,
    input  logic [15:0] cfg_kernel_w_i,
    input  logic [15:0] cfg_stride_h_i,
    input  logic [15:0] cfg_stride_w_i,
    input  logic [15:0] cfg_pad_h_i,
    input  logic [15:0] cfg_pad_w_i,
    input  logic [15:0] cfg_dilation_h_i,
    input  logic [15:0] cfg_dilation_w_i,
    input  logic [31:0] cfg_k_base_i,
    input  logic        cfg_stream_en_i,
    input  logic        start_i,
    output logic        done_o,
    output logic        busy_o,

    output logic                      stream_ifm_valid_o,
    input  logic                      stream_ifm_ready_i,
    output logic [K_TILE-1:0][7:0]    stream_ifm_data_o,

    output logic                      obi_req_o,
    input  logic                      obi_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_addr_o,
    output logic                      obi_we_o,
    output logic [(DATA_WIDTH/8)-1:0] obi_be_o,
    output logic [DATA_WIDTH-1:0]     obi_wdata_o,
    input  logic                      obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0]     obi_rdata_i
);

    typedef enum logic [2:0] {
        IDLE,
        WAIT_STREAM_READY,
        PREP_LANE,
        READ_REQ,
        READ_WAIT,
        WRITE_ROW,
        DONE
    } state_e;

    state_e state_q;
    localparam int unsigned LANE_WIDTH = $clog2(K_TILE);
    localparam logic [LANE_WIDTH-1:0] LANE_LAST = K_TILE - 1;

    logic [31:0] row_q;
    logic [LANE_WIDTH-1:0] lane_q;
    logic [DATA_WIDTH-1:0] row_buf_q;
    logic [4:0] read_byte_offset_q;
    logic [31:0] read_addr;
    logic [31:0] output_row_addr;
    logic [31:0] oh;
    logic [31:0] ow;
    logic [31:0] k_index;
    logic [31:0] k_total;
    logic [31:0] spatial_index;
    logic [31:0] ic;
    logic [31:0] kh;
    logic [31:0] kw;
    logic signed [32:0] ih;
    logic signed [32:0] iw;
    logic lane_zero;
    logic [7:0] read_byte;

    assign done_o = (state_q == DONE);
    assign busy_o = (state_q != IDLE) && (state_q != DONE);
    assign output_row_addr = cfg_output_ptr_i + (row_q * 32'd32);

    always_comb begin
        oh = 32'd0;
        ow = 32'd0;
        if (cfg_output_w_i != 32'd0) begin
            oh = row_q / cfg_output_w_i;
            ow = row_q - ((row_q / cfg_output_w_i) * cfg_output_w_i);
        end

        k_index = cfg_k_base_i + 32'(lane_q);
        k_total = 32'(cfg_kernel_h_i) * 32'(cfg_kernel_w_i) * cfg_input_c_i;
        spatial_index = 32'd0;
        ic = 32'd0;
        kh = 32'd0;
        kw = 32'd0;
        if (cfg_input_c_i != 32'd0) begin
            spatial_index = k_index / cfg_input_c_i;
            ic = k_index - ((k_index / cfg_input_c_i) * cfg_input_c_i);
        end
        if (cfg_kernel_w_i != 16'd0) begin
            kh = spatial_index / 32'(cfg_kernel_w_i);
            kw = spatial_index - ((spatial_index / 32'(cfg_kernel_w_i)) * 32'(cfg_kernel_w_i));
        end

        ih = $signed({1'b0, oh * 32'(cfg_stride_h_i)}) +
             $signed({1'b0, kh * 32'(cfg_dilation_h_i)}) -
             $signed({17'd0, cfg_pad_h_i});
        iw = $signed({1'b0, ow * 32'(cfg_stride_w_i)}) +
             $signed({1'b0, kw * 32'(cfg_dilation_w_i)}) -
             $signed({17'd0, cfg_pad_w_i});

        lane_zero = (cfg_output_w_i == 32'd0) ||
                    (cfg_input_c_i == 32'd0) ||
                    (cfg_kernel_w_i == 16'd0) ||
                    (k_index >= k_total) ||
                    (ih < 33'sd0) ||
                    (iw < 33'sd0) ||
                    (ih >= $signed({1'b0, cfg_input_h_i})) ||
                    (iw >= $signed({1'b0, cfg_input_w_i}));

        read_addr = cfg_input_ptr_i +
                    (((32'(ih[31:0]) * cfg_input_w_i) + 32'(iw[31:0])) * cfg_input_c_i) +
                    ic;
        read_byte = obi_rdata_i[read_byte_offset_q * 8 +: 8];
    end

    always_comb begin
        obi_req_o = 1'b0;
        obi_addr_o = '0;
        obi_we_o = 1'b0;
        obi_be_o = '1;
        obi_wdata_o = row_buf_q;
        stream_ifm_valid_o = 1'b0;
        stream_ifm_data_o = row_buf_q;

        unique case (state_q)
            READ_REQ: begin
                obi_req_o = 1'b1;
                obi_addr_o = read_addr;
                obi_we_o = 1'b0;
            end
            WRITE_ROW: begin
                if (cfg_stream_en_i) begin
                    stream_ifm_valid_o = 1'b1;
                end else begin
                    obi_req_o = 1'b1;
                    obi_addr_o = output_row_addr;
                    obi_we_o = 1'b1;
                end
            end
            default: begin
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= IDLE;
            row_q <= '0;
            lane_q <= '0;
            row_buf_q <= '0;
            read_byte_offset_q <= '0;
        end else begin
            unique case (state_q)
                IDLE: begin
                    if (start_i) begin
                        row_q <= '0;
                        lane_q <= '0;
                        row_buf_q <= '0;
                        read_byte_offset_q <= '0;
                        if (cfg_rows_i == 32'd0) begin
                            state_q <= DONE;
                        end else if (cfg_stream_en_i) begin
                            state_q <= WAIT_STREAM_READY;
                        end else begin
                            state_q <= PREP_LANE;
                        end
                    end
                end

                WAIT_STREAM_READY: begin
                    if (stream_ifm_ready_i) begin
                        state_q <= PREP_LANE;
                    end
                end

                PREP_LANE: begin
                    if (lane_zero) begin
                        row_buf_q[lane_q * 8 +: 8] <= 8'd0;
                        if (lane_q == LANE_LAST) begin
                            state_q <= WRITE_ROW;
                        end else begin
                            lane_q <= lane_q + 1'b1;
                        end
                    end else begin
                        state_q <= READ_REQ;
                    end
                end

                READ_REQ: begin
                    if (obi_gnt_i) begin
                        read_byte_offset_q <= read_addr[4:0];
                        state_q <= READ_WAIT;
                    end
                end

                READ_WAIT: begin
                    if (obi_rvalid_i) begin
                        row_buf_q[lane_q * 8 +: 8] <= read_byte;
                        if (lane_q == LANE_LAST) begin
                            state_q <= WRITE_ROW;
                        end else begin
                            lane_q <= lane_q + 1'b1;
                            state_q <= PREP_LANE;
                        end
                    end
                end

                WRITE_ROW: begin
                    if ((cfg_stream_en_i && stream_ifm_ready_i) ||
                        (!cfg_stream_en_i && obi_gnt_i)) begin
                        if (row_q == (cfg_rows_i - 32'd1)) begin
                            state_q <= DONE;
                        end else begin
                            row_q <= row_q + 32'd1;
                            lane_q <= '0;
                            row_buf_q <= '0;
                            state_q <= PREP_LANE;
                        end
                    end
                end

                DONE: begin
                    state_q <= IDLE;
                end

                default: begin
                    state_q <= IDLE;
                end
            endcase
        end
    end

endmodule
