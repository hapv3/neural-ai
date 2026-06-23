`default_nettype none

module tcdm_interconnect #(
    parameter int unsigned NUM_MASTERS = 4,
    parameter int unsigned NUM_BANKS   = 8,
    parameter int unsigned ADDR_WIDTH  = 32,
    parameter int unsigned DATA_WIDTH  = 256,
    parameter logic [NUM_MASTERS-1:0] HWPE_MASTER_MASK = '0,
    parameter logic [NUM_MASTERS-1:0] DMA_MASTER_MASK  = '0,
    parameter logic [NUM_MASTERS-1:0] CORE_MASTER_MASK = '0
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
    localparam int unsigned MASTER_SEL_BITS = (NUM_MASTERS > 1) ? $clog2(NUM_MASTERS) : 1;
    localparam logic GROUP_MASKS_CONFIGURED = |(HWPE_MASTER_MASK | DMA_MASTER_MASK | CORE_MASTER_MASK);
    localparam logic [NUM_MASTERS-1:0] HWPE_MASK =
        GROUP_MASKS_CONFIGURED ? HWPE_MASTER_MASK : '0;
    localparam logic [NUM_MASTERS-1:0] DMA_MASK =
        GROUP_MASKS_CONFIGURED ? DMA_MASTER_MASK : '0;
    localparam logic [NUM_MASTERS-1:0] CORE_MASK =
        GROUP_MASKS_CONFIGURED ? CORE_MASTER_MASK : '1;

    typedef logic [MASTER_SEL_BITS-1:0] master_sel_t;

    // Signals for arbitration
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] bank_req_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] bank_gnt_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] hwpe_req_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] dma_req_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] core_req_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] hwpe_gnt_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] dma_gnt_matrix;
    logic [NUM_BANKS-1:0][NUM_MASTERS-1:0] core_gnt_matrix;
    logic [NUM_BANKS-1:0] hwpe_req;
    logic [NUM_BANKS-1:0] dma_req;
    logic [NUM_BANKS-1:0] core_req;
    master_sel_t [NUM_BANKS-1:0] hwpe_rr_q, hwpe_rr_d;
    master_sel_t [NUM_BANKS-1:0] dma_rr_q, dma_rr_d;
    master_sel_t [NUM_BANKS-1:0] core_rr_q, core_rr_d;

    function automatic logic [NUM_MASTERS-1:0] pick_rr(
        input logic [NUM_MASTERS-1:0] req,
        input master_sel_t            rr_ptr
    );
        logic [NUM_MASTERS-1:0] grant;
        logic                   found;
        int unsigned            idx;
        begin
            grant = '0;
            found = 1'b0;
            for (int unsigned offset = 0; offset < NUM_MASTERS; offset++) begin
                idx = int'(rr_ptr) + offset;
                if (idx >= NUM_MASTERS) begin
                    idx = idx - NUM_MASTERS;
                end
                if (!found && req[idx]) begin
                    grant[idx] = 1'b1;
                    found = 1'b1;
                end
            end
            return grant;
        end
    endfunction

    function automatic master_sel_t next_rr(
        input logic [NUM_MASTERS-1:0] grant
    );
        master_sel_t next_ptr;
        begin
            next_ptr = '0;
            for (int unsigned m = 0; m < NUM_MASTERS; m++) begin
                if (grant[m]) begin
                    if (m == NUM_MASTERS - 1) begin
                        next_ptr = '0;
                    end else begin
                        next_ptr = master_sel_t'(m + 1);
                    end
                end
            end
            return next_ptr;
        end
    endfunction
    
    // 1. Address decoding (Word-interleaved banking)
    for (genvar m = 0; m < NUM_MASTERS; m++) begin : gen_addr_decode
        logic [BANK_SEL_BITS-1:0] target_bank;
        assign target_bank = (master_addr_i[m] >> BYTE_SEL_BITS) % NUM_BANKS;
        
        for (genvar b = 0; b < NUM_BANKS; b++) begin : gen_req_matrix
            assign bank_req_matrix[b][m] = master_req_i[m] & (target_bank == b);
        end
    end

    // 2. Grouped arbitration per bank.
    //    - First stage: round-robin inside each traffic class.
    //    - Final stage: strict priority HWPE > DMA > CORE.
    for (genvar b = 0; b < NUM_BANKS; b++) begin : gen_bank_arb
        for (genvar m = 0; m < NUM_MASTERS; m++) begin : gen_group_req
            assign hwpe_req_matrix[b][m] = bank_req_matrix[b][m] & HWPE_MASK[m];
            assign dma_req_matrix[b][m]  = bank_req_matrix[b][m] & DMA_MASK[m];
            assign core_req_matrix[b][m] = bank_req_matrix[b][m] & CORE_MASK[m];
        end

        assign hwpe_req[b] = |hwpe_req_matrix[b];
        assign dma_req[b]  = |dma_req_matrix[b];
        assign core_req[b] = |core_req_matrix[b];

        assign hwpe_gnt_matrix[b] = pick_rr(hwpe_req_matrix[b], hwpe_rr_q[b]);
        assign dma_gnt_matrix[b]  = pick_rr(dma_req_matrix[b], dma_rr_q[b]);
        assign core_gnt_matrix[b] = pick_rr(core_req_matrix[b], core_rr_q[b]);

        assign bank_gnt_matrix[b] =
            hwpe_req[b] ? hwpe_gnt_matrix[b] :
            dma_req[b]  ? dma_gnt_matrix[b]  :
            core_req[b] ? core_gnt_matrix[b] :
                          '0;

        // Bank outputs
        assign bank_req_o[b] = hwpe_req[b] | dma_req[b] | core_req[b];

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
                    bank_addr_o[b]  = ((master_addr_i[m] >> BYTE_SEL_BITS) / NUM_BANKS) << BYTE_SEL_BITS;
                    bank_we_o[b]    = master_we_i[m];
                    bank_be_o[b]    = master_be_i[m];
                    bank_wdata_o[b] = master_wdata_i[m];
                end
            end
        end
    end

    always_comb begin
        hwpe_rr_d = hwpe_rr_q;
        dma_rr_d  = dma_rr_q;
        core_rr_d = core_rr_q;

        for (int unsigned b = 0; b < NUM_BANKS; b++) begin
            if (hwpe_req[b]) begin
                hwpe_rr_d[b] = next_rr(hwpe_gnt_matrix[b]);
            end else if (dma_req[b]) begin
                dma_rr_d[b] = next_rr(dma_gnt_matrix[b]);
            end else if (core_req[b]) begin
                core_rr_d[b] = next_rr(core_gnt_matrix[b]);
            end
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            hwpe_rr_q <= '0;
            dma_rr_q  <= '0;
            core_rr_q <= '0;
        end else begin
            hwpe_rr_q <= hwpe_rr_d;
            dma_rr_q  <= dma_rr_d;
            core_rr_q <= core_rr_d;
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
                    master_bank_sel_q[m] <= (master_addr_i[m] >> BYTE_SEL_BITS) % NUM_BANKS;
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
