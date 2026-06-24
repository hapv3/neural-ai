`default_nettype none

module npu_interrupt_ctrl #(
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

    input  logic dma_done_i,
    input  logic sys_done_i,
    input  logic afu_done_i,
    input  logic spatz_done_i,

    output snitch_pkg::interrupts_t   snitch_irq_o,
    output logic                      host_irq_o
);

    localparam logic [ADDR_WIDTH-1:0] REG_INT_ENABLE   = 32'h0000;
    localparam logic [ADDR_WIDTH-1:0] REG_INT_PENDING  = 32'h0004;
    localparam logic [ADDR_WIDTH-1:0] REG_INT_CLEAR    = 32'h0008;
    localparam logic [ADDR_WIDTH-1:0] REG_EXT_ENABLE   = 32'h000C;
    localparam logic [ADDR_WIDTH-1:0] REG_EXT_PENDING  = 32'h0010;
    localparam logic [ADDR_WIDTH-1:0] REG_EXT_CLEAR    = 32'h0014;
    localparam logic [ADDR_WIDTH-1:0] REG_HOST_NOTIFY  = 32'h0018;
    localparam logic [ADDR_WIDTH-1:0] REG_HOST_STATUS  = 32'h001C;

    localparam logic [31:0] IRQ_DMA    = 32'h0000_0001;
    localparam logic [31:0] IRQ_SYS    = 32'h0000_0002;
    localparam logic [31:0] IRQ_AFU    = 32'h0000_0004;
    localparam logic [31:0] IRQ_SPATZ  = 32'h0000_0008;
    localparam logic [31:0] IRQ_HOST   = 32'h0000_0001;

    logic [31:0] int_enable_q;
    logic [31:0] int_pending_q;
    logic [31:0] ext_enable_q;
    logic [31:0] ext_pending_q;
    logic [31:0] host_status_q;
    logic [31:0] event_q;
    logic [31:0] event_d;
    logic [31:0] event_rise;
    logic [31:0] pending_next;
    logic [31:0] local_addr;
    logic [31:0] r_addr_q;

    assign gnt_o = 1'b1;
    assign local_addr = addr_i & 32'h0000_0FFF;

    always_comb begin
        event_d = '0;
        event_d[0] = dma_done_i;
        event_d[1] = sys_done_i;
        event_d[2] = afu_done_i;
        event_d[3] = spatz_done_i;
    end

    assign event_rise = event_d & ~event_q;
    assign pending_next = int_pending_q | event_rise;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            int_enable_q  <= '0;
            int_pending_q <= '0;
            ext_enable_q  <= IRQ_HOST;
            ext_pending_q <= '0;
            host_status_q <= '0;
            event_q       <= '0;
            r_addr_q      <= '0;
            rvalid_o      <= 1'b0;
        end else begin
            event_q <= event_d;
            int_pending_q <= pending_next;

            if (req_i && gnt_o) begin
                r_addr_q <= local_addr;
                if (we_i) begin
                    case (local_addr)
                        REG_INT_ENABLE:  int_enable_q  <= wdata_i[31:0];
                        REG_INT_CLEAR:   int_pending_q <= pending_next & ~wdata_i[31:0];
                        REG_EXT_ENABLE:  ext_enable_q  <= wdata_i[31:0];
                        REG_EXT_CLEAR:   ext_pending_q <= ext_pending_q & ~wdata_i[31:0];
                        REG_HOST_NOTIFY: begin
                            ext_pending_q <= ext_pending_q | IRQ_HOST;
                            host_status_q <= wdata_i[31:0];
                        end
                        REG_HOST_STATUS: host_status_q <= wdata_i[31:0];
                        default: ;
                    endcase
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
            case (r_addr_q)
                REG_INT_ENABLE:  rdata_o[31:0] = int_enable_q;
                REG_INT_PENDING: rdata_o[31:0] = int_pending_q;
                REG_EXT_ENABLE:  rdata_o[31:0] = ext_enable_q;
                REG_EXT_PENDING: rdata_o[31:0] = ext_pending_q;
                REG_HOST_STATUS: rdata_o[31:0] = host_status_q;
                default:         rdata_o[31:0] = 32'h0;
            endcase
        end
    end

    always_comb begin
        snitch_irq_o = '0;
        snitch_irq_o.mcip = |(int_pending_q & int_enable_q);
    end

    assign host_irq_o = |(ext_pending_q & ext_enable_q);

endmodule
