`default_nettype none

module npu_idma_ctrl_mm #(
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

    output logic                      cfg_dma_start_o,
    output logic [31:0]               cfg_dma_src_addr_o,
    output logic [31:0]               cfg_dma_dst_addr_o,
    output logic [31:0]               cfg_dma_length_o,
    input  logic                      cfg_dma_done_i
);

    localparam logic [11:0] DIR_OFFSET         = 12'h200;
    localparam logic [11:0] REG_CONF           = 12'h000;
    localparam logic [11:0] REG_STATUS_0       = 12'h004;
    localparam logic [11:0] REG_NEXT_ID_0      = 12'h044;
    localparam logic [11:0] REG_DONE_ID_0      = 12'h084;
    localparam logic [11:0] REG_DST_ADDR_LOW   = 12'h0D0;
    localparam logic [11:0] REG_SRC_ADDR_LOW   = 12'h0D8;
    localparam logic [11:0] REG_LENGTH_LOW     = 12'h0E0;
    localparam logic [11:0] REG_DST_STRIDE_2   = 12'h0E8;
    localparam logic [11:0] REG_SRC_STRIDE_2   = 12'h0F0;
    localparam logic [11:0] REG_REPS_2         = 12'h0F8;
    localparam logic [11:0] REG_DST_STRIDE_3   = 12'h100;
    localparam logic [11:0] REG_SRC_STRIDE_3   = 12'h108;
    localparam logic [11:0] REG_REPS_3         = 12'h110;

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
    logic [1:0][31:0] done_id_q;

    logic        busy_q;
    logic        active_dir_q;
    logic [31:0] r_addr_q;
    logic        start_req;
    logic        start_dir;

    assign gnt_o = 1'b1;
    assign cfg_dma_start_o = start_req;
    assign cfg_dma_src_addr_o = src_addr_q[start_dir];
    assign cfg_dma_dst_addr_o = dst_addr_q[start_dir];
    assign cfg_dma_length_o = length_q[start_dir];

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

    function automatic logic is_next_id_addr(input logic [31:0] exact_addr);
        is_next_id_addr = (decode_offset(exact_addr) == REG_NEXT_ID_0);
    endfunction

    function automatic logic can_start_addr(input logic [31:0] exact_addr);
        can_start_addr = is_next_id_addr(exact_addr) &&
                         !busy_q &&
                         (length_q[decode_dir(exact_addr)] != 32'd0);
    endfunction

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            conf_q            <= '0;
            src_addr_q        <= '0;
            dst_addr_q        <= '0;
            length_q          <= '0;
            dst_stride_2_q    <= '0;
            src_stride_2_q    <= '0;
            reps_2_q          <= {32'd1, 32'd1};
            dst_stride_3_q    <= '0;
            src_stride_3_q    <= '0;
            reps_3_q          <= {32'd1, 32'd1};
            done_id_q         <= '0;
            busy_q            <= 1'b0;
            active_dir_q      <= 1'b0;
            r_addr_q          <= '0;
            rvalid_o          <= 1'b0;
        end else begin
            if (cfg_dma_done_i && busy_q) begin
                busy_q <= 1'b0;
                done_id_q[active_dir_q] <= 32'd1;
            end

            if (start_req) begin
                busy_q <= 1'b1;
                active_dir_q <= start_dir;
                done_id_q[start_dir] <= 32'd0;
            end

            if (req_i && gnt_o) begin
                if (we_i) begin
                    for (int i = 0; i < DATA_WIDTH / 32; i++) begin
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
            for (int i = 0; i < DATA_WIDTH / 32; i++) begin
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
            for (int i = 0; i < DATA_WIDTH / 32; i++) begin
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
                    REG_STATUS_0:     rdata_word = {31'd0, busy_q && (active_dir_q == direction)};
                    REG_NEXT_ID_0:    rdata_word = 32'd1;
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

endmodule
