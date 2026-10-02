`timescale 1ns/1ps
`include "axi/typedef.svh"

// Diagnostic A/B test: the memory and NPU RTL are not modified. A synchronous
// consumer toggles RREADY at clock edges, like the DMA's FIFO backpressure.
module axi_mem_handshake_case #(
    parameter realtime ApplyDelay = 0ps,
    parameter realtime AcquireDelay = 0ps,
    parameter int Beats = 57,
    parameter int Phase = 0
)(
    input logic clk_i,
    input logic rst_ni,
    output logic done_o,
    output logic failed_o
);
    typedef logic [31:0] addr_t;
    typedef logic [255:0] data_t;
    typedef logic [3:0] id_t;
    typedef logic [31:0] strb_t;
    typedef logic user_t;
    `AXI_TYPEDEF_ALL(axi, addr_t, id_t, data_t, strb_t, user_t)
    axi_req_t req;
    axi_resp_t rsp;
    localparam addr_t Base = 32'h810008e0;
    int unsigned cycle_q;
    int unsigned accepted;
    logic ar_sent_q;
    logic ready_q;
    logic held;
    logic [255:0] held_data;
    logic held_last;
    logic saw_last;

    function automatic logic [7:0] pattern(input int index);
        return 8'((index * 29 + (index >> 5) * 7 + 11) ^ (index >> 8));
    endfunction

    always_comb begin
        req = '0;
        req.ar.addr = Base;
        req.ar.len = 8'(Beats - 1);
        req.ar.size = 3'd5;
        req.ar.burst = axi_pkg::BURST_INCR;
        req.ar_valid = rst_ni && !ar_sent_q && !done_o;
        req.r_ready = ready_q;
    end

    axi_sim_mem #(
        .AddrWidth(32), .DataWidth(256), .IdWidth(4), .UserWidth(1),
        .NumPorts(1), .axi_req_t(axi_req_t), .axi_rsp_t(axi_resp_t),
        .UninitializedData("zeros"),
        .ApplDelay(ApplyDelay), .AcqDelay(AcquireDelay)
    ) mem (
        .clk_i, .rst_ni, .axi_req_i(req), .axi_rsp_o(rsp),
        .mon_w_valid_o(), .mon_w_addr_o(), .mon_w_data_o(), .mon_w_id_o(),
        .mon_w_user_o(), .mon_w_beat_count_o(), .mon_w_last_o(),
        .mon_r_valid_o(), .mon_r_addr_o(), .mon_r_data_o(), .mon_r_id_o(),
        .mon_r_user_o(), .mon_r_beat_count_o(), .mon_r_last_o()
    );

    initial begin
        for (int index = 0; index < Beats * 32; index++)
            mem.mem[Base + index] = pattern(index);
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cycle_q <= 0;
            ar_sent_q <= 0;
            ready_q <= 0;
        end else begin
            cycle_q <= cycle_q + 1;
            if (req.ar_valid && rsp.ar_ready) ar_sent_q <= 1;
            // Exercise alternating ready/stall, including late-burst stalls.
            ready_q <= ((cycle_q + Phase) % 5) != 3;
        end
    end

    // Check the stable values just before the next active edge, matching the
    // passive trace used on the full cluster. The first violation is retained.
    always @(negedge clk_i) begin
        if (!rst_ni) begin
            accepted = 0;
            done_o = 0;
            failed_o = 0;
            held = 0;
            saw_last = 0;
        end else if (!done_o) begin
            if (held && (!rsp.r_valid || rsp.r.data !== held_data || rsp.r.last !== held_last)) begin
                $display("HOLD_VIOLATION apply=%0t acquire=%0t beats=%0d phase=%0d accepted=%0d time=%0t",
                         ApplyDelay, AcquireDelay, Beats, Phase, accepted, $time);
                failed_o = 1;
                done_o = 1;
            end
            if (!done_o && rsp.r_valid && req.r_ready) begin
                for (int byte_index = 0; byte_index < 32; byte_index++) begin
                    if (rsp.r.data[byte_index*8 +: 8] !== pattern(accepted * 32 + byte_index))
                        failed_o = 1;
                end
                if (rsp.r.last !== (accepted == Beats - 1)) failed_o = 1;
                accepted++;
                if (rsp.r.last) saw_last = 1;
                if (failed_o || saw_last) begin
                    done_o = 1;
                    $display("DATA_RESULT apply=%0t acquire=%0t beats=%0d phase=%0d accepted=%0d last=%0b failed=%0b",
                             ApplyDelay, AcquireDelay, Beats, Phase, accepted, saw_last, failed_o);
                end
            end
            held = rsp.r_valid && !req.r_ready;
            held_data = rsp.r.data;
            held_last = rsp.r.last;
            if (cycle_q > 400 && !done_o) begin
                failed_o = 1;
                done_o = 1;
                $display("TIMEOUT apply=%0t acquire=%0t beats=%0d phase=%0d accepted=%0d",
                         ApplyDelay, AcquireDelay, Beats, Phase, accepted);
            end
        end
    end
endmodule

module tb_axi_mem_handshake;
    logic clk = 0;
    logic rst_n = 0;
    logic [4:0] zero_done, zero_failed, phased_done, phased_failed;
    always #500ps clk = !clk;
    initial begin
        #4ns;
        rst_n = 1;
    end
    for (genvar phase = 0; phase < 5; phase++) begin : gen_case
        axi_mem_handshake_case #(.Phase(phase)) zero_delay (
            .clk_i(clk), .rst_ni(rst_n), .done_o(zero_done[phase]), .failed_o(zero_failed[phase])
        );
        axi_mem_handshake_case #(.ApplyDelay(100ps), .AcquireDelay(400ps), .Phase(phase)) phased (
            .clk_i(clk), .rst_ni(rst_n), .done_o(phased_done[phase]), .failed_o(phased_failed[phase])
        );
    end
    initial begin
        wait (&zero_done && &phased_done);
        $display("SUMMARY zero_delay_failures=%b phased_failures=%b", zero_failed, phased_failed);
        if (|phased_failed) $fatal(1, "Separated timing failed byte-exact/handshake checks");
        if (!(|zero_failed)) $fatal(1, "Zero-delay negative control did not reproduce the issue");
        $display("PASS: zero-delay failure reproduced; separated timing passes 5 phases byte-exact");
        $finish;
    end
endmodule
