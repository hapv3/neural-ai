`default_nettype none

module cluster_ctrl_regs #(
    parameter int unsigned APB_ADDR_WIDTH = 32,
    parameter int unsigned APB_DATA_WIDTH = 32
)(
    input  logic                      clk_i,
    input  logic                      rst_ni,

    // APB Slave Interface
    input  logic [APB_ADDR_WIDTH-1:0] paddr_i,
    input  logic                      psel_i,
    input  logic                      penable_i,
    input  logic                      pwrite_i,
    input  logic [APB_DATA_WIDTH-1:0] pwdata_i,
    output logic                      pready_o,
    output logic [APB_DATA_WIDTH-1:0] prdata_o,
    output logic                      pslverr_o,

    // Hardware Outputs to DMA Engine
    output logic                      cfg_dma_start_o,
    output logic [31:0]               cfg_dma_src_addr_o,
    output logic [31:0]               cfg_dma_dst_addr_o,
    output logic [31:0]               cfg_dma_length_o,
    input  logic                      cfg_dma_done_i
);

    // Register Offsets
    localparam int unsigned REG_DMA_SRC    = 32'h00;
    localparam int unsigned REG_DMA_DST    = 32'h04;
    localparam int unsigned REG_DMA_LEN    = 32'h08;
    localparam int unsigned REG_DMA_START  = 32'h0C;
    localparam int unsigned REG_DMA_DONE   = 32'h10;

    // Internal Registers
    logic [31:0] r_dma_src;
    logic [31:0] r_dma_dst;
    logic [31:0] r_dma_len;
    logic        r_dma_start;
    logic        r_dma_done;

    // APB Write Logic
    logic apb_write;
    assign apb_write = psel_i & penable_i & pwrite_i;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            r_dma_src   <= '0;
            r_dma_dst   <= '0;
            r_dma_len   <= '0;
            r_dma_start <= 1'b0;
            r_dma_done  <= 1'b0;
        end else begin
            // Start is a self-clearing pulse
            r_dma_start <= 1'b0;

            // Latch done signal
            if (cfg_dma_done_i) begin
                r_dma_done <= 1'b1;
            end

            if (apb_write) begin
                case (paddr_i) // Match full 32-bit address
                    REG_DMA_SRC:   r_dma_src   <= pwdata_i;
                    REG_DMA_DST:   r_dma_dst   <= pwdata_i;
                    REG_DMA_LEN:   r_dma_len   <= pwdata_i;
                    REG_DMA_START: r_dma_start <= pwdata_i[0];
                    REG_DMA_DONE:  r_dma_done  <= 1'b0; // Write anything to clear
                    default: ; // Ignore other writes
                endcase
            end
        end
    end

    // APB Read Logic
    logic apb_read;
    assign apb_read = psel_i & penable_i & ~pwrite_i;

    always_comb begin
        prdata_o = '0;
        if (apb_read) begin
            case (paddr_i)
                REG_DMA_SRC:   prdata_o = r_dma_src;
                REG_DMA_DST:   prdata_o = r_dma_dst;
                REG_DMA_LEN:   prdata_o = r_dma_len;
                REG_DMA_START: prdata_o = {31'd0, r_dma_start};
                REG_DMA_DONE:  prdata_o = {31'd0, r_dma_done};
                default:       prdata_o = 32'hDEADBEEF;
            endcase
        end
    end

    // APB Response
    assign pready_o  = 1'b1; // Zero wait states
    assign pslverr_o = 1'b0; // No error generated

    // Drive outputs
    assign cfg_dma_start_o    = r_dma_start;
    assign cfg_dma_src_addr_o = r_dma_src;
    assign cfg_dma_dst_addr_o = r_dma_dst;
    assign cfg_dma_length_o   = r_dma_len;

endmodule
