`default_nettype none

module npu_pmu #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 32,
    parameter int unsigned NUM_COUNTERS = 32,
    parameter int unsigned INC_WIDTH = 16
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

    input  logic [NUM_COUNTERS-1:0][INC_WIDTH-1:0] event_inc_i
);

    localparam int unsigned DATA_BYTES = DATA_WIDTH / 8;
    localparam logic [ADDR_WIDTH-1:0] REG_CTRL        = 32'h0000;
    localparam logic [ADDR_WIDTH-1:0] REG_STATUS      = 32'h0004;
    localparam logic [ADDR_WIDTH-1:0] REG_NUM_COUNTER = 32'h0008;
    localparam logic [ADDR_WIDTH-1:0] REG_COUNTER_BASE = 32'h0100;

    logic [NUM_COUNTERS-1:0][63:0] counter_q;
    logic [NUM_COUNTERS-1:0][63:0] snapshot_q;
    logic [NUM_COUNTERS-1:0] overflow_q;
    logic enable_q;
    logic snapshot_valid_q;
    logic [ADDR_WIDTH-1:0] r_addr_q;
    logic [ADDR_WIDTH-1:0] wr_local_addr;
    logic [ADDR_WIDTH-1:0] rd_local_addr;
    logic [ADDR_WIDTH-1:0] rd_counter_offset;
    logic [31:0] rd_counter_idx;
    logic [63:0] rd_counter_value;
    logic ctrl_write;
    logic [31:0] ctrl_wdata;

    assign gnt_o = 1'b1;
    assign wr_local_addr = addr_i & ADDR_WIDTH'(32'h0000_0FFF);
    assign rd_local_addr = r_addr_q & ADDR_WIDTH'(32'h0000_0FFF);
    assign rd_counter_offset = rd_local_addr - REG_COUNTER_BASE;
    assign rd_counter_idx = 32'(rd_counter_offset >> 3);

    always_comb begin
        ctrl_write = req_i && gnt_o && we_i && (|be_i) && (wr_local_addr == REG_CTRL);
        ctrl_wdata = wdata_i[31:0];
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            counter_q        <= '0;
            snapshot_q       <= '0;
            overflow_q       <= '0;
            enable_q         <= 1'b0;
            snapshot_valid_q <= 1'b0;
            r_addr_q         <= '0;
            rvalid_o         <= 1'b0;
        end else begin
            if (req_i && gnt_o) begin
                if (!we_i) begin
                    r_addr_q <= addr_i & ~(32'(DATA_BYTES - 1));
                end
                rvalid_o <= 1'b1;
            end else begin
                rvalid_o <= 1'b0;
            end

            if (ctrl_write) begin
                enable_q <= ctrl_wdata[0];
            end

            if (ctrl_write && ctrl_wdata[1]) begin
                counter_q        <= '0;
                snapshot_q       <= '0;
                overflow_q       <= '0;
                snapshot_valid_q <= 1'b0;
            end else begin
                if (enable_q) begin
                    for (int idx = 0; idx < NUM_COUNTERS; idx++) begin
                        logic [64:0] next_counter;
                        next_counter = {1'b0, counter_q[idx]} + 65'(event_inc_i[idx]);
                        counter_q[idx] <= next_counter[63:0];
                        overflow_q[idx] <= overflow_q[idx] | next_counter[64];
                    end
                end
                if (ctrl_write && ctrl_wdata[2]) begin
                    snapshot_q <= counter_q;
                    snapshot_valid_q <= 1'b1;
                end
            end
        end
    end

    always_comb begin
        rd_counter_value = '0;
        if (rd_counter_idx < NUM_COUNTERS) begin
            rd_counter_value = snapshot_valid_q ? snapshot_q[rd_counter_idx] : counter_q[rd_counter_idx];
        end

        rdata_o = '0;
        if (rvalid_o) begin
            unique case (rd_local_addr)
                REG_CTRL: begin
                    rdata_o[31:0] = {29'd0, snapshot_valid_q, 1'b0, enable_q};
                end
                REG_STATUS: begin
                    rdata_o[31:0] = overflow_q[31:0];
                end
                REG_NUM_COUNTER: begin
                    rdata_o[31:0] = 32'(NUM_COUNTERS);
                end
                default: begin
                    if (rd_local_addr >= REG_COUNTER_BASE &&
                        rd_local_addr < (REG_COUNTER_BASE + (NUM_COUNTERS * 8))) begin
                        rdata_o[31:0] = rd_local_addr[2] ? rd_counter_value[63:32] :
                                                           rd_counter_value[31:0];
                    end
                end
            endcase
        end
    end

endmodule
