`default_nettype none

module conv2d_feeder_cache #(
    parameter int unsigned ADDR_WIDTH    = 32,
    parameter int unsigned DATA_WIDTH    = 256,
    parameter int unsigned CACHE_ENTRIES = 128,
    parameter int unsigned CACHE_BANKS   = 4
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic flush_req_i,
    output logic flush_done_o,

    input  logic                              lookup_req_i,
    input  logic [ADDR_WIDTH-$clog2(DATA_WIDTH/8)-1:0] lookup_line_i,
    input  logic [$clog2(DATA_WIDTH/8)-1:0]              lookup_byte_offset_i,
    output logic                              lookup_valid_o,
    output logic                              lookup_hit_o,
    output logic [7:0]                        lookup_byte_o,

    input  logic                              fill_req_i,
    input  logic [ADDR_WIDTH-$clog2(DATA_WIDTH/8)-1:0] fill_line_i,
    input  logic [DATA_WIDTH-1:0]             fill_data_i
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BEAT_OFF_W = $clog2(BEAT_BYTES);
    localparam int unsigned CACHE_LINES_PER_BANK = CACHE_ENTRIES / CACHE_BANKS;
    localparam int unsigned CACHE_BANK_W = $clog2(CACHE_BANKS);
    localparam int unsigned CACHE_INDEX_W = $clog2(CACHE_LINES_PER_BANK);
    localparam int unsigned CACHE_LINE_W = ADDR_WIDTH - BEAT_OFF_W;
    localparam int unsigned CACHE_TAG_W = CACHE_LINE_W - CACHE_BANK_W - CACHE_INDEX_W;
    localparam int unsigned CACHE_META_W = CACHE_TAG_W + 1;

    typedef logic [CACHE_LINE_W-1:0] line_t;
    typedef logic [CACHE_BANK_W-1:0] bank_t;
    typedef logic [CACHE_INDEX_W-1:0] index_t;
    typedef logic [CACHE_TAG_W-1:0] tag_t;
    typedef logic [CACHE_META_W-1:0] meta_t;
    typedef logic [DATA_WIDTH-1:0] data_t;
    typedef logic [DATA_WIDTH/8-1:0] data_be_t;
    typedef logic [(CACHE_META_W+7)/8-1:0] meta_be_t;

    logic flush_active_q;
    index_t flush_index_q;

    bank_t lookup_bank_q;
    tag_t lookup_tag_q;
    logic [BEAT_OFF_W-1:0] lookup_offset_q;
    logic lookup_valid_q;

    bank_t lookup_bank;
    index_t lookup_index;
    tag_t lookup_tag;
    bank_t fill_bank;
    index_t fill_index;
    tag_t fill_tag;

    logic [CACHE_BANKS-1:0][1:0] data_req;
    logic [CACHE_BANKS-1:0][1:0] data_we;
    index_t [CACHE_BANKS-1:0][1:0] data_addr;
    data_t [CACHE_BANKS-1:0][1:0] data_wdata;
    data_be_t [CACHE_BANKS-1:0][1:0] data_be;
    data_t [CACHE_BANKS-1:0][1:0] data_rdata;

    logic [CACHE_BANKS-1:0][1:0] meta_req;
    logic [CACHE_BANKS-1:0][1:0] meta_we;
    index_t [CACHE_BANKS-1:0][1:0] meta_addr;
    meta_t [CACHE_BANKS-1:0][1:0] meta_wdata;
    meta_be_t [CACHE_BANKS-1:0][1:0] meta_be;
    meta_t [CACHE_BANKS-1:0][1:0] meta_rdata;

    data_t lookup_data;
    meta_t lookup_meta;

    assign flush_done_o = !flush_active_q && !flush_req_i;

    assign lookup_bank = lookup_line_i[CACHE_BANK_W-1:0];
    assign lookup_index = lookup_line_i[CACHE_BANK_W +: CACHE_INDEX_W];
    assign lookup_tag = lookup_line_i[CACHE_LINE_W-1:CACHE_BANK_W+CACHE_INDEX_W];
    assign fill_bank = fill_line_i[CACHE_BANK_W-1:0];
    assign fill_index = fill_line_i[CACHE_BANK_W +: CACHE_INDEX_W];
    assign fill_tag = fill_line_i[CACHE_LINE_W-1:CACHE_BANK_W+CACHE_INDEX_W];

    always_comb begin
        data_req = '0;
        data_we = '0;
        data_addr = '0;
        data_wdata = '0;
        data_be = '0;
        meta_req = '0;
        meta_we = '0;
        meta_addr = '0;
        meta_wdata = '0;
        meta_be = '0;

        if (flush_active_q) begin
            for (int unsigned bank_idx = 0; bank_idx < CACHE_BANKS; bank_idx++) begin
                meta_req[bank_idx][0] = 1'b1;
                meta_we[bank_idx][0] = 1'b1;
                meta_addr[bank_idx][0] = flush_index_q;
                meta_wdata[bank_idx][0] = '0;
                meta_be[bank_idx][0] = '1;
            end
        end else if (fill_req_i) begin
            data_req[fill_bank][0] = 1'b1;
            data_we[fill_bank][0] = 1'b1;
            data_addr[fill_bank][0] = fill_index;
            data_wdata[fill_bank][0] = fill_data_i;
            data_be[fill_bank][0] = '1;

            meta_req[fill_bank][0] = 1'b1;
            meta_we[fill_bank][0] = 1'b1;
            meta_addr[fill_bank][0] = fill_index;
            meta_wdata[fill_bank][0] = {1'b1, fill_tag};
            meta_be[fill_bank][0] = '1;
        end

        if (lookup_req_i && !flush_active_q) begin
            data_req[lookup_bank][1] = 1'b1;
            data_we[lookup_bank][1] = 1'b0;
            data_addr[lookup_bank][1] = lookup_index;
            data_be[lookup_bank][1] = '1;

            meta_req[lookup_bank][1] = 1'b1;
            meta_we[lookup_bank][1] = 1'b0;
            meta_addr[lookup_bank][1] = lookup_index;
            meta_be[lookup_bank][1] = '1;
        end
    end

    always_comb begin
        lookup_data = data_rdata[lookup_bank_q][1];
        lookup_meta = meta_rdata[lookup_bank_q][1];
    end

    assign lookup_valid_o = lookup_valid_q;
    assign lookup_hit_o = lookup_valid_q &&
                          lookup_meta[CACHE_META_W-1] &&
                          (lookup_meta[CACHE_TAG_W-1:0] == lookup_tag_q);
    assign lookup_byte_o = lookup_data[lookup_offset_q * 8 +: 8];

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            flush_active_q <= 1'b0;
            flush_index_q <= '0;
            lookup_bank_q <= '0;
            lookup_tag_q <= '0;
            lookup_offset_q <= '0;
            lookup_valid_q <= 1'b0;
        end else begin
            lookup_valid_q <= lookup_req_i && !flush_active_q;
            if (lookup_req_i && !flush_active_q) begin
                lookup_bank_q <= lookup_bank;
                lookup_tag_q <= lookup_tag;
                lookup_offset_q <= lookup_byte_offset_i;
            end

            if (flush_req_i && !flush_active_q) begin
                flush_active_q <= 1'b1;
                flush_index_q <= '0;
            end else if (flush_active_q) begin
                if (flush_index_q == index_t'(CACHE_LINES_PER_BANK - 1)) begin
                    flush_active_q <= 1'b0;
                end else begin
                    flush_index_q <= flush_index_q + 1'b1;
                end
            end
        end
    end

    generate
        for (genvar bank_idx = 0; bank_idx < CACHE_BANKS; bank_idx++) begin : gen_cache_bank
            tc_sram #(
                .NumWords  (CACHE_LINES_PER_BANK),
                .DataWidth (DATA_WIDTH),
                .NumPorts  (2),
                .Latency   (1),
                .SimInit   ("none")
            ) i_data_sram (
                .clk_i   (clk_i),
                .rst_ni  (rst_ni),
                .req_i   (data_req[bank_idx]),
                .we_i    (data_we[bank_idx]),
                .addr_i  (data_addr[bank_idx]),
                .wdata_i (data_wdata[bank_idx]),
                .be_i    (data_be[bank_idx]),
                .rdata_o (data_rdata[bank_idx])
            );

            tc_sram #(
                .NumWords  (CACHE_LINES_PER_BANK),
                .DataWidth (CACHE_META_W),
                .NumPorts  (2),
                .Latency   (1),
                .SimInit   ("none")
            ) i_meta_sram (
                .clk_i   (clk_i),
                .rst_ni  (rst_ni),
                .req_i   (meta_req[bank_idx]),
                .we_i    (meta_we[bank_idx]),
                .addr_i  (meta_addr[bank_idx]),
                .wdata_i (meta_wdata[bank_idx]),
                .be_i    (meta_be[bank_idx]),
                .rdata_o (meta_rdata[bank_idx])
            );
        end
    endgenerate

endmodule
