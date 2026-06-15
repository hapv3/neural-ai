`default_nettype none

module tcdm_interconnect #(
    parameter int unsigned NUM_MASTERS = 4,
    parameter int unsigned NUM_BANKS   = 8,
    parameter int unsigned ADDR_WIDTH  = 32,
    parameter int unsigned DATA_WIDTH  = 256
)(
    input  logic                                        clk_i,
    input  logic                                        rst_ni,

    // OBI-like Master Interfaces
    input  logic [NUM_MASTERS-1:0]                      master_req_i,
    output logic [NUM_MASTERS-1:0]                      master_gnt_o,
    input  logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0]      master_addr_i,
    input  logic [NUM_MASTERS-1:0]                      master_we_i,
    input  logic [NUM_MASTERS-1:0][(DATA_WIDTH/8)-1:0]  master_be_i,
    input  logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]      master_wdata_i,
    output logic [NUM_MASTERS-1:0]                      master_rvalid_o,
    output logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]      master_rdata_o,

    // SRAM Bank Interfaces (TCDM)
    output logic [NUM_BANKS-1:0]                        bank_req_o,
    output logic [NUM_BANKS-1:0][ADDR_WIDTH-1:0]        bank_addr_o,
    output logic [NUM_BANKS-1:0]                        bank_we_o,
    output logic [NUM_BANKS-1:0][(DATA_WIDTH/8)-1:0]    bank_be_o,
    output logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]        bank_wdata_o,
    input  logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]        bank_rdata_i
);

    localparam int unsigned BANK_SEL_BITS = $clog2(NUM_BANKS);
    localparam int unsigned BYTE_SEL_BITS = $clog2(DATA_WIDTH / 8);

    // Signals for arbitration
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] bank_req_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] bank_gnt_matrix;
    
    // 1. Address decoding (Word-interleaved banking)
    for (genvar m = 0; m < NUM_MASTERS; m++) begin : gen_addr_decode
        logic [BANK_SEL_BITS-1:0] target_bank;
        assign target_bank = master_addr_i[m][BYTE_SEL_BITS +: BANK_SEL_BITS];
        
        for (genvar b = 0; b < NUM_BANKS; b++) begin : gen_req_matrix
            assign bank_req_matrix[b][m] = master_req_i[m] & (target_bank == b);
        end
    end

    // 2. Simple fixed-priority arbitration per bank (Master 0 has highest priority)
    // In production, this can be replaced by `rr_arb_tree` from PULP common_cells
    for (genvar b = 0; b < NUM_BANKS; b++) begin : gen_bank_arb
        logic [NUM_MASTERS-1:0] higher_pri_reqs;
        
        assign higher_pri_reqs[0] = 1'b0;
        for (genvar m = 1; m < NUM_MASTERS; m++) begin : gen_pri
            assign higher_pri_reqs[m] = higher_pri_reqs[m-1] | bank_req_matrix[b][m-1];
        end

        for (genvar m = 0; m < NUM_MASTERS; m++) begin : gen_gnt
            assign bank_gnt_matrix[b][m] = bank_req_matrix[b][m] & ~higher_pri_reqs[m];
        end

        // Bank outputs
        assign bank_req_o[b] = |bank_req_matrix[b];

        logic [NUM_MASTERS-1:0] grant_oh;
        assign grant_oh = bank_gnt_matrix[b];

        // Muxing logic to SRAM banks
        always_comb begin
            bank_addr_o[b]  = '0;
            bank_we_o[b]    = 1'b0;
            bank_be_o[b]    = '0;
            bank_wdata_o[b] = '0;
            for (int m = 0; m < NUM_MASTERS; m++) begin
                if (grant_oh[m]) begin
                    bank_addr_o[b]  = master_addr_i[m];
                    bank_we_o[b]    = master_we_i[m];
                    bank_be_o[b]    = master_be_i[m];
                    bank_wdata_o[b] = master_wdata_i[m];
                end
            end
        end
    end

    // 3. Routing responses back to masters
    // Assuming SRAM has exactly 1 cycle read latency
    logic [NUM_MASTERS-1:0]                      master_req_q;
    logic [NUM_MASTERS-1:0][BANK_SEL_BITS-1:0]   master_bank_sel_q;
    logic [NUM_MASTERS-1:0]                      master_we_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            master_req_q      <= '0;
            master_bank_sel_q <= '0;
            master_we_q       <= '0;
        end else begin
            for (int m = 0; m < NUM_MASTERS; m++) begin
                if (master_gnt_o[m]) begin
                    master_req_q[m]      <= 1'b1;
                    master_bank_sel_q[m] <= master_addr_i[m][BYTE_SEL_BITS +: BANK_SEL_BITS];
                    master_we_q[m]       <= master_we_i[m];
                end else begin
                    master_req_q[m]      <= 1'b0;
                end
            end
        end
    end

    for (genvar m = 0; m < NUM_MASTERS; m++) begin : gen_master_resp
        logic gnt;
        // Master grant if any bank granted it
        always_comb begin
            gnt = 1'b0;
            for (int b = 0; b < NUM_BANKS; b++) begin
                gnt |= bank_gnt_matrix[b][m];
            end
            master_gnt_o[m] = gnt;
        end

        // Master read valid and data
        assign master_rvalid_o[m] = master_req_q[m];
        assign master_rdata_o[m]  = bank_rdata_i[master_bank_sel_q[m]];
    end

endmodule
