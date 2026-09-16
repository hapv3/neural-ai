`timescale 1ns/1ps
`default_nettype none

module tb_npu_pmu;
    localparam int unsigned NUM_COUNTERS = 163;

    logic clk;
    logic rst_n;
    logic req;
    logic gnt;
    logic [31:0] addr;
    logic we;
    logic [3:0] be;
    logic [31:0] wdata;
    logic rvalid;
    logic [31:0] rdata;
    logic [NUM_COUNTERS-1:0][31:0] event_inc;
    logic filter_enable;
    logic [31:0] filter_context;

    always #0.5 clk = ~clk;

    npu_pmu #(
        .NUM_COUNTERS(NUM_COUNTERS),
        .INC_WIDTH(32),
        .MAX_COUNTER_MASK(NUM_COUNTERS'(1) << 3)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .req_i(req),
        .gnt_o(gnt),
        .addr_i(addr),
        .we_i(we),
        .be_i(be),
        .wdata_i(wdata),
        .rvalid_o(rvalid),
        .rdata_o(rdata),
        .event_inc_i(event_inc),
        .context_id_i(32'd17),
        .context_active_i(1'b1),
        .phase_i(4'd6),
        .filter_enable_o(filter_enable),
        .filter_context_o(filter_context)
    );

    task automatic mmio_write(input logic [31:0] address, input logic [31:0] data);
        begin
            @(negedge clk);
            req = 1'b1;
            we = 1'b1;
            addr = address;
            wdata = data;
            @(negedge clk);
            req = 1'b0;
            we = 1'b0;
            while (!rvalid) @(negedge clk);
        end
    endtask

    task automatic mmio_read(input logic [31:0] address, output logic [31:0] data);
        begin
            @(negedge clk);
            req = 1'b1;
            we = 1'b0;
            addr = address;
            @(negedge clk);
            req = 1'b0;
            while (!rvalid) @(negedge clk);
            data = rdata;
        end
    endtask

    logic [31:0] value;
    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        req = 1'b0;
        addr = '0;
        we = 1'b0;
        be = 4'hf;
        wdata = '0;
        event_inc = '0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        mmio_write(32'h20, 32'd17);
        mmio_write(32'h1c, 32'd1);
        if (!filter_enable || filter_context != 32'd17) $fatal(1, "PMU filter write failed");

        mmio_write(32'h00, 32'h2);
        mmio_write(32'h00, 32'h1);
        repeat (3) @(negedge clk);
        event_inc[0] = 32'd1;
        event_inc[1] = 32'd3;
        event_inc[3] = 32'd7;
        event_inc[162] = 32'd4;
        repeat (5) @(negedge clk);
        event_inc[0] = 32'd0;
        event_inc[1] = 32'd0;
        event_inc[3] = 32'd2;
        event_inc[162] = 32'd0;
        repeat (3) @(negedge clk);

        mmio_write(32'h00, 32'h5);
        repeat (3) @(negedge clk);
        mmio_write(32'h00, 32'h0);
        mmio_read(32'h100, value);
        if (value != 32'd5) $fatal(1, "sum counter mismatch: %0d", value);
        mmio_read(32'h108, value);
        if (value != 32'd15) $fatal(1, "increment counter mismatch: %0d", value);
        mmio_read(32'h118, value);
        if (value != 32'd7) $fatal(1, "max counter mismatch: %0d", value);
        mmio_read(32'h610, value);
        if (value != 32'd20) $fatal(1, "high counter decode mismatch: %0d", value);
        mmio_read(32'h08, value);
        if (value != NUM_COUNTERS) $fatal(1, "counter count mismatch: %0d", value);
        mmio_read(32'h18, value);
        if (value != 32'h0002_0001) $fatal(1, "PMU version mismatch");
        mmio_read(32'h24, value);
        if (value != 32'd17) $fatal(1, "PMU context mismatch");
        mmio_read(32'h28, value);
        if (value != 32'h16) $fatal(1, "PMU phase mismatch: 0x%0h", value);

        $display("PASS: npu_pmu sum/max/snapshot/filter/context");
        $finish;
    end
endmodule

`default_nettype wire
