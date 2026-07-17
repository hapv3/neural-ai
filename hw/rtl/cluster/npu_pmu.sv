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

    logic [NUM_COUNTERS-1:0] overflow_q;
    logic [NUM_COUNTERS-1:0][DATA_WIDTH-1:0] counter_read_data;
    logic enable_q;
    logic snapshot_valid_q;
    logic write_resp_q;
    logic read_pending_s0_q;
    logic read_pending_s1_q;
    logic read_pending_s2_q;
    logic [DATA_WIDTH-1:0] read_data_d;
    logic [DATA_WIDTH-1:0] read_counter_data;
    logic [ADDR_WIDTH-1:0] wr_local_addr;
    logic [31:0] read_counter_idx;
    logic [NUM_COUNTERS-1:0] read_counter_sel_d;
    logic [NUM_COUNTERS-1:0] read_counter_sel_q;
    logic [NUM_COUNTERS-1:0] read_counter_high_q;
    logic [NUM_COUNTERS-1:0] read_snapshot_q;
    logic read_ctrl_s0_q;
    logic read_status_s0_q;
    logic read_num_counter_s0_q;
    logic read_ctrl_s1_q;
    logic read_status_s1_q;
    logic read_num_counter_s1_q;
    logic read_ctrl_s2_q;
    logic read_status_s2_q;
    logic read_num_counter_s2_q;
    logic ctrl_write_d;
    logic ctrl_write_q;
    logic [2:0] ctrl_bits_d;
    logic [2:0] ctrl_bits_q;
    logic ctrl_clear_q;
    logic ctrl_snapshot_q;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [DATA_WIDTH-4:0] unused_wdata;
    /* verilator lint_on UNUSEDSIGNAL */

    assign gnt_o = 1'b1;
    assign wr_local_addr = (addr_i & ~(ADDR_WIDTH'(DATA_BYTES - 1))) &
                           ADDR_WIDTH'(32'h0000_0FFF);
    assign read_counter_idx = 32'((wr_local_addr - REG_COUNTER_BASE) >> 3);
    assign ctrl_clear_q = ctrl_write_q && ctrl_bits_q[1];
    assign ctrl_snapshot_q = ctrl_write_q && !ctrl_bits_q[1] && ctrl_bits_q[2];
    assign unused_wdata = wdata_i[DATA_WIDTH-1:3];

    for (genvar idx = 0; idx < NUM_COUNTERS; idx++) begin : gen_counters
        npu_pmu_counter #(
            .INC_WIDTH  (INC_WIDTH),
            .DATA_WIDTH (DATA_WIDTH)
        ) u_counter (
            .clk_i           (clk_i),
            .rst_ni          (rst_ni),
            .ctrl_write_i    (ctrl_write_q),
            .ctrl_enable_i   (ctrl_bits_q[0]),
            .ctrl_clear_i    (ctrl_clear_q),
            .ctrl_snapshot_i (ctrl_snapshot_q),
            .read_sel_i      (read_counter_sel_q[idx]),
            .read_high_i     (read_counter_high_q[idx]),
            .read_snapshot_i (read_snapshot_q[idx]),
            .event_inc_i     (event_inc_i[idx]),
            .read_data_o     (counter_read_data[idx]),
            .overflow_o      (overflow_q[idx])
        );
    end

    always_comb begin
        ctrl_write_d = req_i && gnt_o && we_i && (|be_i) && (wr_local_addr == REG_CTRL);
        ctrl_bits_d = wdata_i[2:0];
    end

    always_comb begin
        read_counter_sel_d = '0;
        if (req_i && gnt_o && !we_i &&
            (wr_local_addr >= REG_COUNTER_BASE) &&
            (wr_local_addr < (REG_COUNTER_BASE + (NUM_COUNTERS * 8)))) begin
            for (int idx = 0; idx < NUM_COUNTERS; idx++) begin
                read_counter_sel_d[idx] = (read_counter_idx == 32'(idx));
            end
        end
    end

    always_comb begin
        read_counter_data = '0;
        for (int idx = 0; idx < NUM_COUNTERS; idx++) begin
            read_counter_data |= counter_read_data[idx];
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            enable_q         <= 1'b0;
            snapshot_valid_q <= 1'b0;
            write_resp_q     <= 1'b0;
            read_pending_s0_q <= 1'b0;
            read_pending_s1_q <= 1'b0;
            read_pending_s2_q <= 1'b0;
            read_counter_sel_q <= '0;
            read_counter_high_q <= '0;
            read_snapshot_q  <= '0;
            ctrl_write_q     <= 1'b0;
            ctrl_bits_q      <= '0;
            read_ctrl_s0_q   <= 1'b0;
            read_status_s0_q <= 1'b0;
            read_num_counter_s0_q <= 1'b0;
            read_ctrl_s1_q   <= 1'b0;
            read_status_s1_q <= 1'b0;
            read_num_counter_s1_q <= 1'b0;
            read_ctrl_s2_q   <= 1'b0;
            read_status_s2_q <= 1'b0;
            read_num_counter_s2_q <= 1'b0;
            rvalid_o         <= 1'b0;
            rdata_o          <= '0;
        end else begin
            rvalid_o <= read_pending_s2_q || write_resp_q;
            if (read_pending_s2_q) begin
                rdata_o <= read_data_d;
            end else if (write_resp_q) begin
                rdata_o <= '0;
            end
            read_pending_s2_q <= read_pending_s1_q;
            read_ctrl_s2_q <= read_ctrl_s1_q;
            read_status_s2_q <= read_status_s1_q;
            read_num_counter_s2_q <= read_num_counter_s1_q;

            read_pending_s1_q <= read_pending_s0_q;
            read_ctrl_s1_q <= read_ctrl_s0_q;
            read_status_s1_q <= read_status_s0_q;
            read_num_counter_s1_q <= read_num_counter_s0_q;

            read_pending_s0_q <= 1'b0;
            read_ctrl_s0_q <= 1'b0;
            read_status_s0_q <= 1'b0;
            read_num_counter_s0_q <= 1'b0;
            read_counter_sel_q <= '0;
            read_counter_high_q <= '0;
            read_snapshot_q <= '0;
            write_resp_q <= 1'b0;
            ctrl_write_q <= ctrl_write_d;
            ctrl_bits_q <= ctrl_bits_d;

            if (req_i && gnt_o) begin
                if (we_i) begin
                    write_resp_q <= 1'b1;
                end else begin
                    read_pending_s0_q <= 1'b1;
                    read_ctrl_s0_q <= (wr_local_addr == REG_CTRL);
                    read_status_s0_q <= (wr_local_addr == REG_STATUS);
                    read_num_counter_s0_q <= (wr_local_addr == REG_NUM_COUNTER);
                    read_counter_sel_q <= read_counter_sel_d;
                    for (int idx = 0; idx < NUM_COUNTERS; idx++) begin
                        read_counter_high_q[idx] <= read_counter_sel_d[idx] && wr_local_addr[2];
                        read_snapshot_q[idx] <= read_counter_sel_d[idx] && snapshot_valid_q;
                    end
                end
            end

            if (ctrl_write_q) begin
                enable_q <= ctrl_bits_q[0];
                if (ctrl_bits_q[1]) begin
                    snapshot_valid_q <= 1'b0;
                end else if (ctrl_bits_q[2]) begin
                    snapshot_valid_q <= 1'b1;
                end
            end
        end
    end

    always_comb begin
        read_data_d = '0;
        if (read_ctrl_s2_q) begin
            read_data_d[31:0] = {29'd0, snapshot_valid_q, 1'b0, enable_q};
        end else if (read_status_s2_q) begin
            read_data_d[31:0] = overflow_q[31:0];
        end else if (read_num_counter_s2_q) begin
            read_data_d[31:0] = 32'(NUM_COUNTERS);
        end else begin
            read_data_d = read_counter_data;
        end
    end

endmodule

/* verilator lint_off DECLFILENAME */
(* keep_hierarchy = "yes" *)
module npu_pmu_counter #(
    parameter int unsigned INC_WIDTH = 16,
    parameter int unsigned DATA_WIDTH = 32
)(
    input  logic                  clk_i,
    input  logic                  rst_ni,
    input  logic                  ctrl_write_i,
    input  logic                  ctrl_enable_i,
    input  logic                  ctrl_clear_i,
    input  logic                  ctrl_snapshot_i,
    input  logic                  read_sel_i,
    input  logic                  read_high_i,
    input  logic                  read_snapshot_i,
    input  logic [INC_WIDTH-1:0]  event_inc_i,
    output logic [DATA_WIDTH-1:0] read_data_o,
    output logic                  overflow_o
);

    logic enable_q;
    logic clear_q;
    logic snapshot_cmd_q;
    logic read_sel_q;
    logic read_high_q;
    logic read_snapshot_q;
    logic [63:0] counter_q;
    logic [63:0] snapshot_q;
    logic [64:0] next_counter;
    logic [63:0] selected_counter;

    assign next_counter = {1'b0, counter_q} + 65'(event_inc_i);
    assign selected_counter = read_snapshot_q ? snapshot_q : counter_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            enable_q    <= 1'b0;
            clear_q     <= 1'b0;
            snapshot_cmd_q <= 1'b0;
            read_sel_q  <= 1'b0;
            read_high_q <= 1'b0;
            read_snapshot_q <= 1'b0;
            counter_q   <= '0;
            snapshot_q  <= '0;
            read_data_o <= '0;
            overflow_o  <= 1'b0;
        end else begin
            clear_q        <= ctrl_clear_i;
            snapshot_cmd_q <= ctrl_snapshot_i;
            read_sel_q     <= read_sel_i;
            read_high_q    <= read_high_i;
            read_snapshot_q <= read_snapshot_i;

            if (ctrl_write_i) begin
                enable_q <= ctrl_enable_i;
            end

            if (clear_q) begin
                counter_q   <= '0;
                snapshot_q  <= '0;
                overflow_o <= 1'b0;
            end else begin
                if (enable_q) begin
                    counter_q  <= next_counter[63:0];
                    overflow_o <= overflow_o | next_counter[64];
                end
                if (snapshot_cmd_q) begin
                    snapshot_q <= counter_q;
                end
            end

            read_data_o <= '0;
            if (read_sel_q) begin
                read_data_o <= read_high_q ? selected_counter[63:32] : selected_counter[31:0];
            end
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
