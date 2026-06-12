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
    input  logic                      cfg_sys_done_i
);

    // Register Offsets (32-byte aligned because OBI DATA_WIDTH is 256-bit = 32 bytes)
    // This allows the software to just access these addresses and data will always be at wdata_i[31:0]
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_SRC    = 32'h00;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_DST    = 32'h20;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_LEN    = 32'h40;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_START  = 32'h60;
    localparam logic [ADDR_WIDTH-1:0] REG_DMA_DONE   = 32'h80;

    localparam logic [ADDR_WIDTH-1:0] REG_SYS_W_PTR  = 32'hA0;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_I_PTR  = 32'hC0;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_O_PTR  = 32'hE0;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_DIM_M  = 32'h100;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_START  = 32'h120;
    localparam logic [ADDR_WIDTH-1:0] REG_SYS_DONE   = 32'h140;

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
                    // Note: addr_i from obi_demux is still absolute, we mask it to get offset
                    case (addr_i & 32'hFFFF)
                        REG_DMA_SRC:   r_dma_src   <= wdata_i[31:0];
                        REG_DMA_DST:   r_dma_dst   <= wdata_i[31:0];
                        REG_DMA_LEN:   r_dma_len   <= wdata_i[31:0];
                        REG_DMA_START: r_dma_start <= wdata_i[0];
                        REG_DMA_DONE:  r_dma_done  <= 1'b0; // Clear on write

                        REG_SYS_W_PTR: r_sys_w_ptr <= wdata_i[31:0];
                        REG_SYS_I_PTR: r_sys_i_ptr <= wdata_i[31:0];
                        REG_SYS_O_PTR: r_sys_o_ptr <= wdata_i[31:0];
                        REG_SYS_DIM_M: r_sys_dim_m <= wdata_i[31:0];
                        REG_SYS_START: r_sys_start <= wdata_i[0];
                        REG_SYS_DONE:  r_sys_done  <= 1'b0; // Clear on write
                        default: ;
                    endcase
                    rvalid_o <= 1'b0; // Writes don't return rvalid
                end else begin
                    // Capture read request
                    r_addr_q <= addr_i;
                    rvalid_o <= 1'b1; // 1-cycle read latency
                end
            end else begin
                rvalid_o <= 1'b0;
            end
        end
    end

    // OBI Read Data Logic
    always_comb begin
        rdata_o = '0;
        if (rvalid_o) begin
            case (r_addr_q & 32'hFFFF)
                REG_DMA_SRC:   rdata_o[31:0] = r_dma_src;
                REG_DMA_DST:   rdata_o[31:0] = r_dma_dst;
                REG_DMA_LEN:   rdata_o[31:0] = r_dma_len;
                REG_DMA_START: rdata_o[31:0] = {31'd0, r_dma_start};
                REG_DMA_DONE:  rdata_o[31:0] = {31'd0, r_dma_done};

                REG_SYS_W_PTR: rdata_o[31:0] = r_sys_w_ptr;
                REG_SYS_I_PTR: rdata_o[31:0] = r_sys_i_ptr;
                REG_SYS_O_PTR: rdata_o[31:0] = r_sys_o_ptr;
                REG_SYS_DIM_M: rdata_o[31:0] = r_sys_dim_m;
                REG_SYS_START: rdata_o[31:0] = {31'd0, r_sys_start};
                REG_SYS_DONE:  rdata_o[31:0] = {31'd0, r_sys_done};
                default:       rdata_o[31:0] = 32'hDEADBEEF;
            endcase
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

endmodule
