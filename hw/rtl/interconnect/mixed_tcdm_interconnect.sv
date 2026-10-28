`default_nettype none

// Native mixed-width, single-cycle SRAM interconnect. Narrow ports use the low
// bits of the packed interface; they are never upsized into wide transactions.
// A wide beat is granted atomically across its adjacent 32-bit banks. Arbitration
// is hierarchical: a group guide arbitrates wide beats against narrow cohorts,
// then independent bank arbiters serve narrow traffic in the remaining banks.
module mixed_tcdm_interconnect #(
    parameter int unsigned NUM_MASTERS = 20,
    parameter int unsigned NUM_BANKS = 64,
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned BANK_DATA_WIDTH = 32,
    parameter int unsigned MASTER_DATA_WIDTH [NUM_MASTERS] = '{default: DATA_WIDTH},
    parameter logic [NUM_MASTERS-1:0] HWPE_MASTER_MASK = '0,
    parameter logic [NUM_MASTERS-1:0] DMA_MASTER_MASK = '0,
    parameter logic [NUM_MASTERS-1:0] CORE_MASTER_MASK = '0
) (
    input logic clk_i, rst_ni,
    input logic [NUM_MASTERS-1:0] master_req_i,
    output logic [NUM_MASTERS-1:0] master_gnt_o,
    input logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0] master_addr_i,
    input logic [NUM_MASTERS-1:0] master_we_i,
    input logic [NUM_MASTERS-1:0][DATA_WIDTH/8-1:0] master_be_i,
    input logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0] master_wdata_i,
    output logic [NUM_MASTERS-1:0] master_rvalid_o,
    output logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0] master_rdata_o,
    output logic [NUM_BANKS-1:0] bank_req_o,
    output logic [NUM_BANKS-1:0][ADDR_WIDTH-1:0] bank_addr_o,
    output logic [NUM_BANKS-1:0] bank_we_o,
    output logic [NUM_BANKS-1:0][BANK_DATA_WIDTH/8-1:0] bank_be_o,
    output logic [NUM_BANKS-1:0][BANK_DATA_WIDTH-1:0] bank_wdata_o,
    input logic [NUM_BANKS-1:0][BANK_DATA_WIDTH-1:0] bank_rdata_i,
    output logic [NUM_BANKS-1:0] perf_bank_conflict_o
);
    localparam int BANK_BYTES = BANK_DATA_WIDTH / 8;
    localparam int BANK_SHIFT = $clog2(BANK_BYTES);
    localparam int BANK_BITS = $clog2(NUM_BANKS);
    localparam int GROUP_WORDS = DATA_WIDTH / BANK_DATA_WIDTH;
    localparam int GROUP_BYTES = DATA_WIDTH / 8;
    localparam int GROUP_COUNT = NUM_BANKS / GROUP_WORDS;
    localparam int GROUP_BITS = $clog2(GROUP_COUNT);
    localparam bit GROUPS_CONFIGURED = |(HWPE_MASTER_MASK | DMA_MASTER_MASK | CORE_MASTER_MASK);
    localparam logic [NUM_MASTERS-1:0] HWPE_MASK = GROUPS_CONFIGURED ? HWPE_MASTER_MASK : '0;
    localparam logic [NUM_MASTERS-1:0] DMA_MASK = GROUPS_CONFIGURED ? DMA_MASTER_MASK : '0;
    localparam logic [NUM_MASTERS-1:0] CORE_MASK = GROUPS_CONFIGURED ? CORE_MASTER_MASK : '1;
    logic [NUM_MASTERS-1:0][GROUP_BITS-1:0] target_group;
    logic [NUM_MASTERS-1:0][$clog2(GROUP_WORDS)-1:0] first_lane;
    logic [GROUP_COUNT-1:0][NUM_MASTERS-1:0] group_req, guide;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] bank_wants, narrow_req, wide_owner, bank_owner;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] narrow_gnt, narrow_offer, guided_narrow;
    logic [NUM_MASTERS-1:0][GROUP_WORDS-1:0][BANK_BITS-1:0] response_bank_q;

    for (genvar m = 0; m < NUM_MASTERS; m++) begin : gen_master_decode
        localparam int WORDS = MASTER_DATA_WIDTH[m] / BANK_DATA_WIDTH;
        assign target_group[m] = (master_addr_i[m] >> $clog2(GROUP_BYTES));
        // Preserve the existing aligned-bus-word semantics independently per port.
        assign first_lane[m] = (master_addr_i[m] >> BANK_SHIFT) & (GROUP_WORDS-WORDS);
        for (genvar g = 0; g < GROUP_COUNT; g++) begin : gen_group_request
            assign group_req[g][m] = master_req_i[m] && (target_group[m] == g);
        end
        for (genvar b = 0; b < NUM_BANKS; b++) begin : gen_bank_request
            localparam int LANE = b % GROUP_WORDS;
            wire covers = (target_group[m] == b/GROUP_WORDS) &&
                (LANE >= first_lane[m]) && (LANE < first_lane[m] + WORDS);
            wire writes_lane = |(master_be_i[m] >> ((LANE-first_lane[m])*BANK_BYTES) & ((1 << BANK_BYTES)-1));
            assign bank_wants[b][m] = master_req_i[m] && covers && (!master_we_i[m] || writes_lane);
            if (WORDS == 1) begin : gen_native_narrow
                assign narrow_req[b][m] = bank_wants[b][m];
                assign guided_narrow[b][m] = bank_wants[b][m] && guide[b/GROUP_WORDS][m];
                assign wide_owner[b][m] = 1'b0;
            end else begin : gen_native_wide
                assign narrow_req[b][m] = 1'b0;
                assign guided_narrow[b][m] = 1'b0;
                assign wide_owner[b][m] = bank_wants[b][m] && guide[b/GROUP_WORDS][m];
            end
        end
        always_comb begin
            master_gnt_o[m] = 1'b0;
            if (WORDS > 1) master_gnt_o[m] = guide[target_group[m]][m];
            else begin
                for (int b = 0; b < NUM_BANKS; b++) master_gnt_o[m] |= narrow_gnt[b][m];
                // A zero-byte store is an acknowledged no-op, not a deadlock.
                if (master_req_i[m] && master_we_i[m] && !(|master_be_i[m][BANK_BYTES-1:0]))
                    master_gnt_o[m] = guide[target_group[m]][m];
            end
        end
        always_ff @(posedge clk_i or negedge rst_ni) begin
            if (!rst_ni) begin
                master_rvalid_o[m] <= 1'b0;
                response_bank_q[m] <= '0;
            end else begin
                master_rvalid_o[m] <= master_gnt_o[m];
                if (master_gnt_o[m]) begin
                    for (int w = 0; w < WORDS; w++)
                        response_bank_q[m][w] <= target_group[m]*GROUP_WORDS + first_lane[m] + w;
                end
            end
        end
        for (genvar w = 0; w < GROUP_WORDS; w++) begin : gen_response_word
            if (w < WORDS)
                assign master_rdata_o[m][w*BANK_DATA_WIDTH +: BANK_DATA_WIDTH] = bank_rdata_i[response_bank_q[m][w]];
            else assign master_rdata_o[m][w*BANK_DATA_WIDTH +: BANK_DATA_WIDTH] = '0;
        end
        initial begin
            assert (MASTER_DATA_WIDTH[m] >= BANK_DATA_WIDTH && MASTER_DATA_WIDTH[m] <= DATA_WIDTH &&
                (MASTER_DATA_WIDTH[m] & (MASTER_DATA_WIDTH[m]-1)) == 0)
                else $fatal(1, "Unsupported native master data width");
        end
    end

    for (genvar g = 0; g < GROUP_COUNT; g++) begin : gen_group_arb
        mixed_tcdm_class_arbiter #(
            .N(NUM_MASTERS), .HWPE_MASK(HWPE_MASK), .DMA_MASK(DMA_MASK), .CORE_MASK(CORE_MASK)
        ) u_guide (.clk_i, .rst_ni, .req_i(group_req[g]), .gnt_o(guide[g]));
    end
    for (genvar b = 0; b < NUM_BANKS; b++) begin : gen_bank_arb
        // Guarantee the guide's progress, while servicing every other available
        // narrow bank concurrently. No partially accepted wide transactions.
        assign narrow_offer[b] = (|wide_owner[b]) ? '0 :
            ((|guided_narrow[b]) ? guided_narrow[b] : narrow_req[b]);
        mixed_tcdm_class_arbiter #(
            .N(NUM_MASTERS), .HWPE_MASK(HWPE_MASK), .DMA_MASK(DMA_MASK), .CORE_MASK(CORE_MASK)
        ) u_narrow (.clk_i, .rst_ni, .req_i(narrow_offer[b]), .gnt_o(narrow_gnt[b]));
        assign bank_owner[b] = wide_owner[b] | narrow_gnt[b];
        assign bank_req_o[b] = |bank_owner[b];
        assign perf_bank_conflict_o[b] = $countones(bank_wants[b]) > 1;
        always_comb begin
            bank_addr_o[b] = '0;
            bank_we_o[b] = 1'b0;
            bank_be_o[b] = '0;
            bank_wdata_o[b] = '0;
            for (int m = 0; m < NUM_MASTERS; m++) begin
                if (bank_owner[b][m]) begin
                    bank_addr_o[b] = (master_addr_i[m] >> (BANK_SHIFT+BANK_BITS)) << BANK_SHIFT;
                    bank_we_o[b] = master_we_i[m];
                    bank_be_o[b] = master_be_i[m] >> ((b%GROUP_WORDS-first_lane[m])*BANK_BYTES);
                    bank_wdata_o[b] = master_wdata_i[m] >> ((b%GROUP_WORDS-first_lane[m])*BANK_DATA_WIDTH);
                end
            end
        end
    end
    initial begin
        assert (BANK_DATA_WIDTH == 32 && DATA_WIDTH >= 64 &&
            NUM_BANKS >= 2*GROUP_WORDS && (NUM_BANKS & (NUM_BANKS-1)) == 0)
            else $fatal(1, "Invalid mixed TCDM geometry");
        assert (!(|(HWPE_MASK & DMA_MASK)) && !(|(HWPE_MASK & CORE_MASK)) &&
            !(|(DMA_MASK & CORE_MASK)) && (&(HWPE_MASK | DMA_MASK | CORE_MASK)))
            else $fatal(1, "Traffic classes must be disjoint and cover all masters");
    end
endmodule

// The same traffic-class policy is used at both levels; round-robin is updated
// only for the class actually served. Strict priority matches the baseline.
module mixed_tcdm_class_arbiter #(
    parameter int N = 20,
    parameter logic [N-1:0] HWPE_MASK = '0,
    parameter logic [N-1:0] DMA_MASK = '0,
    parameter logic [N-1:0] CORE_MASK = '1
) (
    input logic clk_i, rst_ni,
    input logic [N-1:0] req_i,
    output logic [N-1:0] gnt_o
);
    wire [2:0][N-1:0] requests = {req_i & CORE_MASK, req_i & DMA_MASK, req_i & HWPE_MASK};
    wire [2:0] selected = {!(|requests[0]) && !(|requests[1]) && (|requests[2]),
        !(|requests[0]) && (|requests[1]), |requests[0]};
    wire [2:0][N-1:0] grants;
    wire [N-1:0][0:0] unused_data = '0;
    for (genvar c = 0; c < 3; c++) begin : gen_class
        rr_arb_tree #(.NumIn(N), .DataWidth(1), .ExtPrio(1'b0), .AxiVldRdy(1'b1),
            .LockIn(1'b0), .FairArb(1'b1)) u_rr (
            .clk_i, .rst_ni, .flush_i(1'b0), .rr_i('0), .req_i(requests[c]),
            .gnt_o(grants[c]), .data_i(unused_data), .req_o(), .gnt_i(selected[c]),
            .data_o(), .idx_o()
        );
    end
    assign gnt_o = grants[0] | grants[1] | grants[2];
endmodule
