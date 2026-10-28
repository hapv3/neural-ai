`timescale 1ns/1ps
module tb_mixed_tcdm_interconnect;
    localparam int M = 20;
    localparam int B = 64;
    localparam int WIDTHS[M] = '{32,32,32,32,32,32,32,32,64,64,128,128,
                                256,256,256,256,256,256,256,256};
    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    logic [M-1:0] req, gnt, we, valid;
    logic [M-1:0][31:0] addr, be;
    logic [M-1:0][255:0] wdata, rdata;
    logic [B-1:0] breq, bwe, conflict;
    logic [B-1:0][31:0] baddr, bwdata, brdata;
    logic [B-1:0][3:0] bbe;
    logic [31:0] memory[B][2048];
    logic [31:0] expected_memory[B][2048];
    logic [255:0] expected_read[M];
    logic [M-1:0] last_accepted;
    int served[M];
    int unsigned rng = 32'h384d7281;
    int cycles = 0;
    logic [2:0] class_req, class_gnt;

    mixed_tcdm_interconnect #(.NUM_MASTERS(M), .NUM_BANKS(B),
        .MASTER_DATA_WIDTH(WIDTHS)) dut (
        .clk_i(clk), .rst_ni(rst_n), .master_req_i(req), .master_gnt_o(gnt),
        .master_addr_i(addr), .master_we_i(we), .master_be_i(be),
        .master_wdata_i(wdata), .master_rvalid_o(valid), .master_rdata_o(rdata),
        .bank_req_o(breq), .bank_addr_o(baddr), .bank_we_o(bwe),
        .bank_be_o(bbe), .bank_wdata_o(bwdata), .bank_rdata_i(brdata),
        .perf_bank_conflict_o(conflict));
    mixed_tcdm_class_arbiter #(.N(3), .HWPE_MASK(3'b001),
        .DMA_MASK(3'b010), .CORE_MASK(3'b100)) priority_dut (
        .clk_i(clk), .rst_ni(rst_n), .req_i(class_req), .gnt_o(class_gnt));

    always @(posedge clk) begin
        for (int b = 0; b < B; b++) begin
            if (breq[b]) begin
                brdata[b] <= memory[b][baddr[b][12:2]];
                if (bwe[b]) for (int byte_lane = 0; byte_lane < 4; byte_lane++)
                    if (bbe[b][byte_lane])
                        memory[b][baddr[b][12:2]][byte_lane*8 +: 8] <= bwdata[b][byte_lane*8 +: 8];
            end
        end
    end

    function automatic int unsigned random_word();
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    endfunction

    // An independent byte/word-address scoreboard verifies every accepted beat,
    // not just final memory. Two accepted beats must never own the same bank.
    task automatic step(input int minimum_grants = 0);
        bit [B-1:0] owned;
        logic [M-1:0] accepted;
        int base, bank, row, words;
        #1;
        assert ((gnt & ~req) == 0) else $fatal(1, "unsolicited grant");
        assert ($countones(gnt) >= minimum_grants) else $fatal(1, "lost parallelism");
        owned = '0;
        accepted = gnt;
        for (int m = 0; m < M; m++) begin
            expected_read[m] = '0;
            if (gnt[m]) begin
                served[m]++;
                words = WIDTHS[m]/32;
                base = (addr[m] & 32'h7ffff) & ~(WIDTHS[m]/8-1);
                for (int w = 0; w < words; w++) begin
                    bank = ((base/4)+w)%B;
                    row = ((base/4)+w)/B;
                    if (!we[m] || (|be[m][w*4 +: 4])) begin
                        assert (!owned[bank]) else $fatal(1, "non-atomic bank collision");
                        owned[bank] = 1;
                        assert (breq[bank] && bwe[bank] == we[m] && baddr[bank] == ((addr[m] >> 8) << 2))
                            else $fatal(1, "bank decode m=%0d bank=%0d", m, bank);
                        expected_read[m][w*32 +: 32] = expected_memory[bank][row];
                        if (we[m]) begin
                            assert (bbe[bank] == be[m][w*4 +: 4] && bwdata[bank] == wdata[m][w*32 +: 32])
                                else $fatal(1, "write lane mapping");
                            for (int k = 0; k < 4; k++) if (be[m][w*4+k])
                                expected_memory[bank][row][k*8 +: 8] = wdata[m][w*32+k*8 +: 8];
                        end
                    end
                end
            end
        end
        assert (breq == owned) else $fatal(1, "unowned SRAM transaction");
        @(posedge clk); #1;
        assert (valid == accepted) else $fatal(1, "response latency");
        for (int m = 0; m < M; m++) if (accepted[m] && !we[m])
            assert (rdata[m] == expected_read[m]) else $fatal(1, "read mismatch m=%0d cycle=%0d", m, cycles);
        cycles++;
        last_accepted = accepted;
        @(negedge clk);
    endtask

    initial begin
        req = '0; we = '0; addr = '0; be = '0; wdata = '0; class_req = '0;
        for (int m = 0; m < M; m++) served[m] = 0;
        for (int b = 0; b < B; b++) for (int r = 0; r < 2048; r++) begin
            memory[b][r] = 32'hb8710000 ^ (r*B+b);
            expected_memory[b][r] = memory[b][r];
        end
        repeat (3) @(negedge clk);
        rst_n = 1;
        // Eight adjacent 32-bit accesses must all complete in one cycle.
        req = 20'hff;
        for (int m = 0; m < 8; m++) addr[m] = 32'h10100000 + 4*m;
        step(8);
        // Parallel wide beats in separate groups, with unaligned bus addresses.
        req = '1;
        for (int m = 0; m < M; m++) begin
            addr[m] = 32'h10100000 + (m%8)*32 + (m%4);
            we[m] = 1; be[m] = (m%3 == 0) ? 0 : 32'h55555555;
            for (int w = 0; w < 8; w++) wdata[m][w*32 +: 32] = random_word();
        end
        repeat (50) step();
        we = '0; be = '0; // Reads must ignore byte enables.
        repeat (50) step();
        for (int m = 0; m < M; m++) served[m] = 0;
        addr = '0;
        repeat (80) step();
        for (int m = 0; m < M; m++) assert (served[m] > 0) else $fatal(1, "RR starvation m=%0d", m);
        // Every denied request remains stable. Accepted requesters can issue
        // another transaction immediately, with random width, byte strobes,
        // aliases, offsets and same/different-bank contention.
        for (int n = 0; n < 2500; n++) begin
            for (int m = 0; m < M; m++) if (!req[m] || last_accepted[m]) begin
                req[m] = (random_word()%5 != 0);
                addr[m] = 32'h10100000 + ((random_word()%4 == 0) ? (random_word()%256) : (random_word()%524288));
                we[m] = random_word() & 1;
                be[m] = random_word();
                if (n%17 == 0) be[m] = 0;
                for (int w = 0; w < 8; w++) wdata[m][w*32 +: 32] = random_word();
            end
            step();
        end
        req = '0; step();
        class_req = 3'b111; #1; assert (class_gnt == 3'b001) else $fatal(1, "HWPE priority");
        class_req = 3'b110; #1; assert (class_gnt == 3'b010) else $fatal(1, "DMA priority");
        class_req = 3'b100; #1; assert (class_gnt == 3'b100) else $fatal(1, "core priority");
        rst_n = 0; #1; assert (valid == 0) else $fatal(1, "reset outstanding response");
        for (int b = 0; b < B; b++) for (int r = 0; r < 2048; r++)
            assert (memory[b][r] == expected_memory[b][r]) else $fatal(1, "final SRAM scoreboard");
        $display("PASS native mixed TCDM: %0d cycles, 32/64/128/256-bit, eight native ports, atomic grants, byte scoreboard", cycles);
        $finish;
    end
    initial begin #1000000; $fatal(1, "unit timeout"); end
endmodule
