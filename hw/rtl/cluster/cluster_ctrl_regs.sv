`default_nettype none

module cluster_ctrl_regs #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32
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

    localparam int unsigned DATA_BYTES = DATA_WIDTH / 8;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_START = 32'h0000;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_SRC   = 32'h0020;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_DST   = 32'h0040;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_LEN   = 32'h0060;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_DONE  = 32'h0080;

    logic [31:0] r_dma_src;
    logic [31:0] r_dma_dst;
    logic [31:0] r_dma_len;
    logic        r_dma_start;
    logic        r_dma_done;
    logic [ADDR_WIDTH-1:0] r_addr_q;

    assign gnt_o = 1'b1;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            r_dma_src   <= '0;
            r_dma_dst   <= '0;
            r_dma_len   <= '0;
            r_dma_start <= 1'b0;
            r_dma_done  <= 1'b0;
            r_addr_q    <= '0;
            rvalid_o    <= 1'b0;
        end else begin
            r_dma_start <= 1'b0;
            if (cfg_dma_done_i) begin
                r_dma_done <= 1'b1;
            end

            if (req_i && gnt_o) begin
                if (we_i) begin
                    for (int i = 0; i < DATA_WIDTH/32; i++) begin
                        if (be_i[i*4 +: 4] != 4'b0000) begin
                            logic [31:0] exact_addr;
                            logic [31:0] local_addr;
                            logic [31:0] wdata_word;

                            exact_addr = (addr_i & ~(32'(DATA_BYTES - 1))) + (i * 4);
                            local_addr = exact_addr & 32'hFFFF;
                            wdata_word = wdata_i[i*32 +: 32];

                            unique case (local_addr)
                                REG_DMA_SRC:   r_dma_src   <= wdata_word;
                                REG_DMA_DST:   r_dma_dst   <= wdata_word;
                                REG_DMA_LEN:   r_dma_len   <= wdata_word;
                                REG_DMA_START: r_dma_start <= wdata_word[0];
                                REG_DMA_DONE:  r_dma_done  <= 1'b0;
                                default: begin
                                end
                            endcase
                        end
                    end
                end else begin
                    r_addr_q <= addr_i & ~(32'(DATA_BYTES - 1));
                end
                rvalid_o <= 1'b1;
            end else begin
                rvalid_o <= 1'b0;
            end
        end
    end

    always_comb begin
        rdata_o = '0;
        if (rvalid_o) begin
            for (int i = 0; i < DATA_WIDTH/32; i++) begin
                logic [31:0] rdata_word;
                logic [31:0] exact_addr;

                exact_addr = (r_addr_q & 32'hFFFF) + (i * 4);
                rdata_word = '0;
                unique case (exact_addr)
                    REG_DMA_SRC:   rdata_word = r_dma_src;
                    REG_DMA_DST:   rdata_word = r_dma_dst;
                    REG_DMA_LEN:   rdata_word = r_dma_len;
                    REG_DMA_START: rdata_word = {31'd0, r_dma_start};
                    REG_DMA_DONE:  rdata_word = {31'd0, r_dma_done};
                    default: begin
                    end
                endcase
                rdata_o[i*32 +: 32] = rdata_word;
            end
        end
    end

    assign cfg_dma_start_o    = r_dma_start;
    assign cfg_dma_src_addr_o = r_dma_src;
    assign cfg_dma_dst_addr_o = r_dma_dst;
    assign cfg_dma_length_o   = r_dma_len;

endmodule
