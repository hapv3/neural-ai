`default_nettype none

module conv_linebuf_row_store #(
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ROW_SLOTS = 7,
    parameter int unsigned BANKS = 14,
    parameter int unsigned BANK_DEPTH = 320,
    parameter int unsigned BANK_ADDR_WIDTH = $clog2(BANK_DEPTH),
    parameter int unsigned ROW_PENDING_WIDTH = 11
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic        job_start_i,
    input  logic        job_full_mode_i,
    input  logic [15:0] job_c_base_i,
    input  logic        invalidate_i,
    input  logic        cached_c_base_set_i,
    input  logic [15:0] cached_c_base_i,

    input  logic alloc_main_valid_i,
    input  logic [$clog2(ROW_SLOTS)-1:0] alloc_main_slot_i,
    input  logic [15:0] alloc_main_ih_i,
    input  logic alloc_background_valid_i,
    input  logic [$clog2(ROW_SLOTS)-1:0] alloc_background_slot_i,
    input  logic [15:0] alloc_background_ih_i,

    input  logic beat_push_i,
    input  logic [$clog2(ROW_SLOTS)-1:0] beat_push_slot_i,
    input  logic beat_push_last_for_row_i,
    input  logic beat_pop_i,
    input  logic [$clog2(ROW_SLOTS)-1:0] beat_pop_slot_i,

    input  logic [BANKS-1:0] bank_write_req_i,
    input  logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_write_addr_i,
    input  logic [BANKS-1:0][DATA_WIDTH-1:0] bank_write_data_i,
    input  logic [BANKS-1:0] bank_read_req_i,
    input  logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_read_addr_i,
    output logic [BANKS-1:0][DATA_WIDTH-1:0] bank_read_data_o,

    input  logic [$clog2(ROW_SLOTS)-1:0] query_main_slot_i,
    input  logic [15:0] query_main_ih_i,
    output logic query_main_cached_o,
    output logic query_main_pending_o,
    input  logic [$clog2(ROW_SLOTS)-1:0] query_background_slot_i,
    input  logic [15:0] query_background_ih_i,
    output logic query_background_cached_o,
    output logic query_background_pending_o,

    output logic        row_cache_full_o,
    output logic [15:0] cached_c_base_o
);

    logic row_cache_full_q;
    logic [15:0] cached_c_base_q;
    logic [ROW_SLOTS-1:0] row_slot_valid_q;
    logic [ROW_SLOTS-1:0] row_fetch_active_q;
    logic [ROW_SLOTS-1:0] row_fetch_done_q;
    logic [ROW_SLOTS-1:0][15:0] row_slot_ih_q;
    logic [ROW_SLOTS-1:0][ROW_PENDING_WIDTH-1:0] row_pending_q;
    logic [ROW_SLOTS-1:0][ROW_PENDING_WIDTH-1:0] row_pending_next;
    logic [ROW_SLOTS-1:0] row_ready_next;

    logic [DATA_WIDTH/8-1:0] bank_be;
    logic [BANKS-1:0][DATA_WIDTH-1:0] bank_write_read_data_unused;

    assign bank_be = '1;
    assign row_cache_full_o = row_cache_full_q;
    assign cached_c_base_o = cached_c_base_q;

    assign query_main_cached_o = row_slot_valid_q[query_main_slot_i] &&
                                 (row_slot_ih_q[query_main_slot_i] == query_main_ih_i);
    assign query_main_pending_o = row_fetch_active_q[query_main_slot_i] &&
                                  (row_slot_ih_q[query_main_slot_i] == query_main_ih_i);
    assign query_background_cached_o = row_slot_valid_q[query_background_slot_i] &&
                                       (row_slot_ih_q[query_background_slot_i] ==
                                        query_background_ih_i);
    assign query_background_pending_o = row_fetch_active_q[query_background_slot_i] &&
                                        (row_slot_ih_q[query_background_slot_i] ==
                                         query_background_ih_i);

    for (genvar bank = 0; bank < BANKS; bank++) begin : gen_line_banks
        tc_sram #(
            .NumWords    (BANK_DEPTH),
            .DataWidth   (DATA_WIDTH),
            .ByteWidth   (8),
            .NumPorts    (2),
            .Latency     (1),
            .SimInit     ("none"),
            .PrintSimCfg (1'b0)
        ) i_bank_sram (
            .clk_i   (clk_i),
            .rst_ni  (rst_ni),
            .req_i   ({bank_read_req_i[bank],  bank_write_req_i[bank]}),
            .we_i    ({1'b0,                    1'b1}),
            .addr_i  ({bank_read_addr_i[bank], bank_write_addr_i[bank]}),
            .wdata_i ({DATA_WIDTH'(0),          bank_write_data_i[bank]}),
            .be_i    ({bank_be,                 bank_be}),
            .rdata_o ({bank_read_data_o[bank],  bank_write_read_data_unused[bank]})
        );
    end

    always_comb begin
        row_pending_next = row_pending_q;
        if (beat_push_i) begin
            row_pending_next[beat_push_slot_i] =
                row_pending_next[beat_push_slot_i] + ROW_PENDING_WIDTH'(1);
        end
        if (beat_pop_i && row_fetch_active_q[beat_pop_slot_i]) begin
            row_pending_next[beat_pop_slot_i] =
                row_pending_next[beat_pop_slot_i] - ROW_PENDING_WIDTH'(1);
        end

        row_ready_next = '0;
        for (int unsigned slot = 0; slot < ROW_SLOTS; slot++) begin
            row_ready_next[slot] = row_fetch_active_q[slot] &&
                                   row_fetch_done_q[slot] &&
                                   (row_pending_next[slot] == '0);
        end
    end

    task automatic allocate_row(
        input logic [$clog2(ROW_SLOTS)-1:0] slot_i,
        input logic [15:0] row_ih_i
    );
        begin
            row_slot_valid_q[slot_i] <= 1'b0;
            row_fetch_active_q[slot_i] <= 1'b1;
            row_fetch_done_q[slot_i] <= 1'b0;
            row_pending_q[slot_i] <= '0;
            row_slot_ih_q[slot_i] <= row_ih_i;
        end
    endtask

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            row_cache_full_q <= 1'b0;
            cached_c_base_q <= '0;
            row_slot_valid_q <= '0;
            row_fetch_active_q <= '0;
            row_fetch_done_q <= '0;
            row_slot_ih_q <= '0;
            row_pending_q <= '0;
        end else begin
            row_pending_q <= row_pending_next;
            if (beat_push_i && beat_push_last_for_row_i) begin
                row_fetch_done_q[beat_push_slot_i] <= 1'b1;
            end
            for (int unsigned slot = 0; slot < ROW_SLOTS; slot++) begin
                if (row_ready_next[slot]) begin
                    row_slot_valid_q[slot] <= 1'b1;
                    row_fetch_active_q[slot] <= 1'b0;
                    row_fetch_done_q[slot] <= 1'b0;
                end
            end

            if (alloc_main_valid_i) begin
                allocate_row(alloc_main_slot_i, alloc_main_ih_i);
            end
            if (invalidate_i) begin
                row_slot_valid_q <= '0;
                row_fetch_active_q <= '0;
                row_fetch_done_q <= '0;
                row_pending_q <= '0;
            end
            if (cached_c_base_set_i) begin
                cached_c_base_q <= cached_c_base_i;
            end
            if (alloc_background_valid_i) begin
                allocate_row(alloc_background_slot_i, alloc_background_ih_i);
            end

            if (job_start_i) begin
                row_cache_full_q <= job_full_mode_i;
                cached_c_base_q <= job_c_base_i;
                row_slot_valid_q <= '0;
                row_fetch_active_q <= '0;
                row_fetch_done_q <= '0;
                row_pending_q <= '0;
            end
        end
    end

endmodule

`default_nettype wire
