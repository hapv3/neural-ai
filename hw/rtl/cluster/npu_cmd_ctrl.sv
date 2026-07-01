`default_nettype none

module npu_cmd_ctrl #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32,
    parameter logic [31:0] BASE_ADDR = 32'h2000_5000,
    parameter logic [31:0] DEFAULT_TCDM_BASE = 32'h1017_F000,
    parameter logic [31:0] DEFAULT_TCDM_BYTES = 32'h0000_1000
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic                      host_req_i,
    output logic                      host_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     host_addr_i,
    input  logic                      host_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] host_be_i,
    input  logic [DATA_WIDTH-1:0]     host_wdata_i,
    output logic                      host_rvalid_o,
    output logic [DATA_WIDTH-1:0]     host_rdata_o,

    input  logic                      snitch_req_i,
    output logic                      snitch_gnt_o,
    input  logic [ADDR_WIDTH-1:0]     snitch_addr_i,
    input  logic                      snitch_we_i,
    input  logic [(DATA_WIDTH/8)-1:0] snitch_be_i,
    input  logic [DATA_WIDTH-1:0]     snitch_wdata_i,
    output logic                      snitch_rvalid_o,
    output logic [DATA_WIDTH-1:0]     snitch_rdata_o
);

    localparam int unsigned DATA_BYTES = DATA_WIDTH / 8;
    localparam logic [ADDR_WIDTH-1:0] REG_L2_BASE     = 32'h0000;
    localparam logic [ADDR_WIDTH-1:0] REG_TOTAL_BYTES = 32'h0004;
    localparam logic [ADDR_WIDTH-1:0] REG_TCDM_BASE   = 32'h0008;
    localparam logic [ADDR_WIDTH-1:0] REG_TCDM_BYTES  = 32'h000C;
    localparam logic [ADDR_WIDTH-1:0] REG_START       = 32'h0010;
    localparam logic [ADDR_WIDTH-1:0] REG_STATUS      = 32'h0014;
    localparam logic [ADDR_WIDTH-1:0] REG_FAIL_CODE   = 32'h0018;
    localparam logic [ADDR_WIDTH-1:0] REG_FAIL_PTR    = 32'h001C;
    localparam logic [ADDR_WIDTH-1:0] REG_DONE_COUNT  = 32'h0020;

    logic [31:0] l2_base_q;
    logic [31:0] total_bytes_q;
    logic [31:0] tcdm_base_q;
    logic [31:0] tcdm_bytes_q;
    logic [31:0] start_q;
    logic [31:0] status_q;
    logic [31:0] fail_code_q;
    logic [31:0] fail_ptr_q;
    logic [31:0] done_count_q;

    logic [ADDR_WIDTH-1:0] host_raddr_q;
    logic [ADDR_WIDTH-1:0] snitch_raddr_q;

    assign host_gnt_o = 1'b1;
    assign snitch_gnt_o = 1'b1;

    function automatic logic [ADDR_WIDTH-1:0] local_addr(input logic [ADDR_WIDTH-1:0] addr);
        local_addr = (addr - BASE_ADDR) & ADDR_WIDTH'(32'h0000_0FFF);
    endfunction

    function automatic logic [31:0] reg_read(input logic [ADDR_WIDTH-1:0] addr);
        logic [ADDR_WIDTH-1:0] loc;
        begin
            loc = local_addr(addr);
            unique case (loc)
                REG_L2_BASE:     reg_read = l2_base_q;
                REG_TOTAL_BYTES: reg_read = total_bytes_q;
                REG_TCDM_BASE:   reg_read = tcdm_base_q;
                REG_TCDM_BYTES:  reg_read = tcdm_bytes_q;
                REG_START:       reg_read = start_q;
                REG_STATUS:      reg_read = status_q;
                REG_FAIL_CODE:   reg_read = fail_code_q;
                REG_FAIL_PTR:    reg_read = fail_ptr_q;
                REG_DONE_COUNT:  reg_read = done_count_q;
                default:         reg_read = 32'h0;
            endcase
        end
    endfunction

    task automatic apply_write(
        input logic [ADDR_WIDTH-1:0] addr,
        input logic [DATA_WIDTH-1:0] data,
        input logic [(DATA_WIDTH/8)-1:0] be
    );
        logic [ADDR_WIDTH-1:0] loc;
        logic [31:0] wdata_word;
        begin
            loc = local_addr(addr);
            wdata_word = data[31:0];
            if (|be[3:0]) begin
                unique case (loc)
                    REG_L2_BASE:     l2_base_q <= wdata_word;
                    REG_TOTAL_BYTES: total_bytes_q <= wdata_word;
                    REG_TCDM_BASE:   tcdm_base_q <= wdata_word;
                    REG_TCDM_BYTES:  tcdm_bytes_q <= wdata_word;
                    REG_START:       start_q <= wdata_word;
                    REG_STATUS:      status_q <= wdata_word;
                    REG_FAIL_CODE:   fail_code_q <= wdata_word;
                    REG_FAIL_PTR:    fail_ptr_q <= wdata_word;
                    REG_DONE_COUNT:  done_count_q <= wdata_word;
                    default: begin
                    end
                endcase
            end
        end
    endtask

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            l2_base_q       <= 32'h8004_0000;
            total_bytes_q   <= 32'h0;
            tcdm_base_q     <= DEFAULT_TCDM_BASE;
            tcdm_bytes_q    <= DEFAULT_TCDM_BYTES;
            start_q         <= 32'h0;
            status_q        <= 32'h0;
            fail_code_q     <= 32'h0;
            fail_ptr_q      <= 32'h0;
            done_count_q    <= 32'h0;
            host_raddr_q    <= '0;
            snitch_raddr_q  <= '0;
            host_rvalid_o   <= 1'b0;
            snitch_rvalid_o <= 1'b0;
        end else begin
            host_rvalid_o <= 1'b0;
            snitch_rvalid_o <= 1'b0;

            if (host_req_i && host_gnt_o) begin
                if (host_we_i) begin
                    apply_write(host_addr_i, host_wdata_i, host_be_i);
                end else begin
                    host_raddr_q <= host_addr_i & ~(32'(DATA_BYTES - 1));
                end
                host_rvalid_o <= 1'b1;
            end

            if (snitch_req_i && snitch_gnt_o) begin
                if (snitch_we_i) begin
                    apply_write(snitch_addr_i, snitch_wdata_i, snitch_be_i);
                end else begin
                    snitch_raddr_q <= snitch_addr_i & ~(32'(DATA_BYTES - 1));
                end
                snitch_rvalid_o <= 1'b1;
            end
        end
    end

    always_comb begin
        host_rdata_o = '0;
        if (host_rvalid_o) begin
            host_rdata_o[31:0] = reg_read(host_raddr_q);
        end
    end

    always_comb begin
        snitch_rdata_o = '0;
        if (snitch_rvalid_o) begin
            snitch_rdata_o[31:0] = reg_read(snitch_raddr_q);
        end
    end

endmodule
