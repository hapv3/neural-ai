`default_nettype none

module cluster_ctrl_regs #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256
)(
    input  logic clk_i,
    input  logic rst_ni,

    // OBI Slave Interface
    input  logic                      req_i,
    output logic                      gnt_o,
    input  logic [ADDR_WIDTH-1:0]     addr_i,
    input  logic                      we_i,
    input  logic [(DATA_WIDTH/8)-1:0] be_i,
    input  logic [DATA_WIDTH-1:0]     wdata_i,
    output logic                      rvalid_o,
    output logic [DATA_WIDTH-1:0]     rdata_o,

    // Hardware Outputs to DMA Engine
    output logic                      cfg_dma_start_o,
    output logic [31:0]               cfg_dma_src_addr_o,
    output logic [31:0]               cfg_dma_dst_addr_o,
    output logic [31:0]               cfg_dma_length_o,
    input  logic                      cfg_dma_done_i,

    // Hardware Outputs to Systolic Controller
    output logic                      cfg_sys_start_o,
    output logic [31:0]               cfg_sys_weight_ptr_o,
    output logic [31:0]               cfg_sys_ifm_ptr_o,
    output logic [31:0]               cfg_sys_ofm_ptr_o,
    output logic [31:0]               cfg_sys_dim_m_o,
    output logic                      cfg_requant_en_o,
    output logic [31:0][31:0]         cfg_requant_bias_o,
    output logic [31:0][31:0]         cfg_requant_multiplier_o,
    output logic [31:0][7:0]          cfg_requant_shift_o,
    output logic [31:0][31:0]         cfg_requant_zero_point_o,
    output logic [31:0]               cfg_requant_clamp_min_o,
    output logic [31:0]               cfg_requant_clamp_max_o,
    input  logic                      cfg_sys_done_i
);

    // Register Offsets
    localparam int unsigned DATA_BYTES = DATA_WIDTH / 8;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_START = 32'h0000;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_SRC   = 32'h0020;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_DST   = 32'h0040;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_LEN   = 32'h0060;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_DONE  = 32'h0080;

    localparam logic [ADDR_WIDTH-1:0] REG_SYS_W_PTR = 32'h0100;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_I_PTR = 32'h0104;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_O_PTR = 32'h0108;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_DIM_M = 32'h010C;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_START = 32'h0110;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_DONE  = 32'h0114;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_CTRL   = 32'h0120;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_CMIN   = 32'h0124;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_CMAX   = 32'h0128;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_BIAS_BASE = 32'h0200;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_MULT_BASE = 32'h0280;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_SHIFT_BASE = 32'h0300;
    localparam logic [ADDR_WIDTH-1:0] REG_RQ_ZP_BASE = 32'h0380;

    // Internal Registers
    logic [31:0] r_dma_src;
    logic [31:0] r_dma_dst;
    logic [31:0] r_dma_len;
    logic        r_dma_start;
    logic        r_dma_done;

    logic [31:0] r_sys_w_ptr;
    logic [31:0] r_sys_i_ptr;
    logic [31:0] r_sys_o_ptr;
    logic [31:0] r_sys_dim_m;
    logic        r_sys_start;
    logic        r_sys_done;
    logic        r_requant_en;
    logic [31:0][31:0] r_requant_bias;
    logic [31:0][31:0] r_requant_multiplier;
    logic [31:0][7:0]  r_requant_shift;
    logic [31:0][31:0] r_requant_zero_point;
    logic [31:0]       r_requant_clamp_min;
    logic [31:0]       r_requant_clamp_max;

    // OBI Grant is always ready
    assign gnt_o = 1'b1;

    logic [ADDR_WIDTH-1:0] r_addr_q;
    logic                  r_read_req_q;

    // OBI Write & Read-Address capture Logic
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            r_dma_src   <= '0;
            r_dma_dst   <= '0;
            r_dma_len   <= '0;
            r_dma_start <= 1'b0;
            r_dma_done  <= 1'b0;

            r_sys_w_ptr <= '0;
            r_sys_i_ptr <= '0;
            r_sys_o_ptr <= '0;
            r_sys_dim_m <= '0;
            r_sys_start <= 1'b0;
            r_sys_done  <= 1'b0;
            r_requant_en <= 1'b0;
            r_requant_clamp_min <= 32'hFFFF_FF80;
            r_requant_clamp_max <= 32'h0000_007F;
            for (int unsigned ch = 0; ch < 32; ch++) begin
                r_requant_bias[ch] <= '0;
                r_requant_multiplier[ch] <= 32'd1;
                r_requant_shift[ch] <= '0;
                r_requant_zero_point[ch] <= '0;
            end

            r_addr_q     <= '0;
            r_read_req_q <= 1'b0;
            rvalid_o     <= 1'b0;
        end else begin
            // Start signals are self-clearing pulses
            r_dma_start <= 1'b0;
            r_sys_start <= 1'b0;

            // Latch done signals
            if (cfg_dma_done_i) r_dma_done <= 1'b1;
            if (cfg_sys_done_i) r_sys_done <= 1'b1;

            // Handle OBI Request
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
                            
                            
                            case (local_addr)
                                REG_DMA_SRC:   r_dma_src   <= wdata_word;
                                REG_DMA_DST:   r_dma_dst   <= wdata_word;
                                REG_DMA_LEN:   r_dma_len   <= wdata_word;
                                REG_DMA_START: begin
                                    r_dma_start <= wdata_word[0];
                                end
                                REG_DMA_DONE:  r_dma_done  <= 1'b0;

                                REG_SYS_W_PTR: r_sys_w_ptr <= wdata_word;
                                REG_SYS_I_PTR: r_sys_i_ptr <= wdata_word;
                                REG_SYS_O_PTR: r_sys_o_ptr <= wdata_word;
                                REG_SYS_DIM_M: r_sys_dim_m <= wdata_word;
                                REG_SYS_START: r_sys_start <= wdata_word[0];
                                REG_SYS_DONE:  r_sys_done  <= 1'b0;
                                REG_RQ_CTRL:   r_requant_en <= wdata_word[0];
                                REG_RQ_CMIN:   r_requant_clamp_min <= wdata_word;
                                REG_RQ_CMAX:   r_requant_clamp_max <= wdata_word;
                                default: begin
                                    if ((local_addr & 32'hFF80) == REG_RQ_BIAS_BASE) begin
                                        r_requant_bias[(local_addr - REG_RQ_BIAS_BASE) >> 2] <= wdata_word;
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_MULT_BASE) begin
                                        r_requant_multiplier[(local_addr - REG_RQ_MULT_BASE) >> 2] <= wdata_word;
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_SHIFT_BASE) begin
                                        r_requant_shift[(local_addr - REG_RQ_SHIFT_BASE) >> 2] <= wdata_word[7:0];
                                    end else if ((local_addr & 32'hFF80) == REG_RQ_ZP_BASE) begin
                                        r_requant_zero_point[(local_addr - REG_RQ_ZP_BASE) >> 2] <= wdata_word;
                                    end
                                end
                            endcase
                        end
                    end
                end else begin
                    // Capture read request
                    r_addr_q <= addr_i & ~(32'(DATA_BYTES - 1));
                end
                rvalid_o <= 1'b1; // OBI requires rvalid for both reads and writes
            end else begin
                rvalid_o <= 1'b0;
            end
        end
    end

    // OBI Read Data Logic
    always_comb begin
        rdata_o = '0;
        if (rvalid_o) begin
            for (int i = 0; i < DATA_WIDTH/32; i++) begin
                logic [31:0] rdata_word;
                logic [31:0] exact_addr;
                
                exact_addr = (r_addr_q & 32'hFFFF) + (i * 4);
                rdata_word = '0;
                case (exact_addr)
                    REG_DMA_SRC:   rdata_word = r_dma_src;
                    REG_DMA_DST:   rdata_word = r_dma_dst;
                    REG_DMA_LEN:   rdata_word = r_dma_len;
                    REG_DMA_START: rdata_word = {31'd0, r_dma_start};
                    REG_DMA_DONE:  rdata_word = {31'd0, r_dma_done};

                    REG_SYS_W_PTR: rdata_word = r_sys_w_ptr;
                    REG_SYS_I_PTR: rdata_word = r_sys_i_ptr;
                    REG_SYS_O_PTR: rdata_word = r_sys_o_ptr;
                    REG_SYS_DIM_M: rdata_word = r_sys_dim_m;
                    REG_SYS_START: rdata_word = {31'd0, r_sys_start};
                    REG_SYS_DONE:  rdata_word = {31'd0, r_sys_done};
                    REG_RQ_CTRL:   rdata_word = {31'd0, r_requant_en};
                    REG_RQ_CMIN:   rdata_word = r_requant_clamp_min;
                    REG_RQ_CMAX:   rdata_word = r_requant_clamp_max;
                    default: begin
                        if ((exact_addr & 32'hFF80) == REG_RQ_BIAS_BASE) begin
                            rdata_word = r_requant_bias[(exact_addr - REG_RQ_BIAS_BASE) >> 2];
                        end else if ((exact_addr & 32'hFF80) == REG_RQ_MULT_BASE) begin
                            rdata_word = r_requant_multiplier[(exact_addr - REG_RQ_MULT_BASE) >> 2];
                        end else if ((exact_addr & 32'hFF80) == REG_RQ_SHIFT_BASE) begin
                            rdata_word = {24'd0, r_requant_shift[(exact_addr - REG_RQ_SHIFT_BASE) >> 2]};
                        end else if ((exact_addr & 32'hFF80) == REG_RQ_ZP_BASE) begin
                            rdata_word = r_requant_zero_point[(exact_addr - REG_RQ_ZP_BASE) >> 2];
                        end else begin
                            rdata_word = 32'h0;
                        end
                    end
                endcase
                rdata_o[i*32 +: 32] = rdata_word;
            end
        end
    end

    // Drive outputs
    assign cfg_dma_start_o    = r_dma_start;
    assign cfg_dma_src_addr_o = r_dma_src;
    assign cfg_dma_dst_addr_o = r_dma_dst;
    assign cfg_dma_length_o   = r_dma_len;

    assign cfg_sys_start_o      = r_sys_start;
    assign cfg_sys_weight_ptr_o = r_sys_w_ptr;
    assign cfg_sys_ifm_ptr_o    = r_sys_i_ptr;
    assign cfg_sys_ofm_ptr_o    = r_sys_o_ptr;
    assign cfg_sys_dim_m_o      = r_sys_dim_m;
    assign cfg_requant_en_o     = r_requant_en;
    assign cfg_requant_bias_o   = r_requant_bias;
    assign cfg_requant_multiplier_o = r_requant_multiplier;
    assign cfg_requant_shift_o  = r_requant_shift;
    assign cfg_requant_zero_point_o = r_requant_zero_point;
    assign cfg_requant_clamp_min_o = r_requant_clamp_min;
    assign cfg_requant_clamp_max_o = r_requant_clamp_max;

endmodule
