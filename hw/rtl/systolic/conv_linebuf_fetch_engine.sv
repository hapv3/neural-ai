`default_nettype none

module conv_linebuf_fetch_engine #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned ROW_SLOTS = 7,
    parameter int unsigned BANKS = 14,
    parameter int unsigned BANK_ADDR_WIDTH = 9,
    parameter int unsigned BEAT_FIFO_DEPTH = 4
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic clear_count_i,
    input  logic reset_background_i,
    input  logic row_ring_mode_i,
    input  logic c32_blocked_mode_i,
    input  logic [15:0] input_h_i,
    input  logic [15:0] input_w_i,
    input  logic [15:0] kernel_h_i,
    input  logic [31:0] pixel_stride_bytes_i,
    input  logic [31:0] row_stride_bytes_i,
    input  logic [31:0] channel_addr_offset_i,

    input  logic main_start_i,
    input  logic [ADDR_WIDTH-1:0] main_base_addr_i,
    input  logic [15:0] main_row_slot_i,
    input  logic [15:0] main_row_ih_i,
    input  logic [5:0] main_valid_bytes_i,
    input  logic main_row_ready_i,
    output logic main_request_accepted_o,
    output logic [1:0] main_next_phase_o,
    output logic main_done_o,
    output logic main_alloc_valid_o,
    output logic [$clog2(ROW_SLOTS)-1:0] main_alloc_slot_o,
    output logic [15:0] main_alloc_ih_o,

    input  logic background_start_i,
    input  logic signed [31:0] background_base_ih_i,
    input  logic [ADDR_WIDTH-1:0] background_row_base_addr_i,
    output logic background_idle_o,
    output logic [$clog2(ROW_SLOTS)-1:0] background_query_slot_o,
    output logic [15:0] background_query_ih_o,
    input  logic background_row_cached_i,
    input  logic background_row_pending_i,
    output logic background_alloc_valid_o,
    output logic [$clog2(ROW_SLOTS)-1:0] background_alloc_slot_o,
    output logic [15:0] background_alloc_ih_o,

    output logic obi_req_o,
    input  logic obi_gnt_i,
    output logic [ADDR_WIDTH-1:0] obi_addr_o,
    input  logic obi_rvalid_i,
    input  logic [DATA_WIDTH-1:0] obi_rdata_i,

    output logic [31:0] fetch_beats_o,

    output logic beat_push_o,
    output logic [$clog2(ROW_SLOTS)-1:0] beat_push_slot_o,
    output logic beat_push_last_for_row_o,
    output logic beat_pop_o,
    output logic [$clog2(ROW_SLOTS)-1:0] beat_pop_slot_o,

    output logic [BANKS-1:0] bank_write_req_o,
    output logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_write_addr_o,
    output logic [BANKS-1:0][DATA_WIDTH-1:0] bank_write_data_o
);

    localparam int unsigned BEAT_BYTES = DATA_WIDTH / 8;
    localparam int unsigned BYTE_SEL_BITS = $clog2(BEAT_BYTES);
    localparam int unsigned FIFO_PTR_WIDTH = $clog2(BEAT_FIFO_DEPTH);
    localparam int unsigned FIFO_COUNT_WIDTH = FIFO_PTR_WIDTH + 1;

    typedef enum logic [1:0] {
        MAIN_REQ0,
        MAIN_REQ1,
        MAIN_DRAIN,
        MAIN_IDLE
    } main_state_e;

    typedef enum logic [2:0] {
        BG_IDLE,
        BG_SCAN,
        BG_REQ0,
        BG_REQ1,
        BG_DRAIN
    } background_state_e;

    typedef struct packed {
        logic [BYTE_SEL_BITS-1:0] addr_lsb;
        logic [5:0] valid_bytes;
        logic [15:0] row_slot;
        logic [15:0] x;
        logic is_beat0_of_cross;
        logic is_solo;
    } beat_meta_t;

    main_state_e main_state_q;
    background_state_e background_state_q;
    logic [15:0] main_x_q;
    logic [ADDR_WIDTH-1:0] main_addr_q;
    logic [ADDR_WIDTH-1:0] main_pending_beat_addr_q;
    logic [15:0] main_row_slot_q;
    logic [5:0] main_valid_bytes_q;

    logic signed [31:0] background_base_ih_q;
    logic [ADDR_WIDTH-1:0] background_row_base_addr_q;
    logic [15:0] background_kh_q;
    logic [15:0] background_x_q;
    logic [ADDR_WIDTH-1:0] background_addr_q;
    logic [ADDR_WIDTH-1:0] background_pending_beat_addr_q;
    logic [5:0] background_valid_bytes_q;

    beat_meta_t [BEAT_FIFO_DEPTH-1:0] beat_fifo_q;
    logic [FIFO_PTR_WIDTH-1:0] write_ptr_q;
    logic [FIFO_PTR_WIDTH-1:0] read_ptr_q;
    logic [FIFO_COUNT_WIDTH-1:0] count_q;
    logic [DATA_WIDTH-1:0] response_beat0_q;
    logic [31:0] fetch_beats_q;

    beat_meta_t response_meta;
    beat_meta_t accepted_meta;
    logic main_crosses;
    logic background_crosses;
    logic main_req;
    logic background_req;
    logic select_main;
    logic select_background;
    logic accept_main;
    logic accept_background;
    logic accept_request;
    logic pop_response;
    logic fifo_full;
    logic fifo_empty;
    logic background_row_in_bounds;
    logic signed [31:0] background_ih;
    logic [15:0] background_row_slot;
    logic response_write;
    logic [$clog2(BANKS)-1:0] response_bank;
    logic [BANK_ADDR_WIDTH-1:0] response_address;
    logic [DATA_WIDTH-1:0] response_data;

    assign fifo_full = count_q >= FIFO_COUNT_WIDTH'(BEAT_FIFO_DEPTH - 1);
    assign fifo_empty = count_q == '0;
    assign fetch_beats_o = fetch_beats_q;
    assign response_meta = beat_fifo_q[read_ptr_q];
    assign background_idle_o = background_state_q == BG_IDLE;

    function automatic logic [ADDR_WIDTH-1:0] beat_base(
        input logic [ADDR_WIDTH-1:0] addr
    );
        beat_base = {addr[ADDR_WIDTH-1:BYTE_SEL_BITS], {BYTE_SEL_BITS{1'b0}}};
    endfunction

    function automatic logic [DATA_WIDTH-1:0] merge_beats(
        input logic [DATA_WIDTH-1:0] beat0,
        input logic [DATA_WIDTH-1:0] beat1,
        input logic [BYTE_SEL_BITS-1:0] addr_lsb,
        input logic [5:0] valid_bytes
    );
        logic [DATA_WIDTH-1:0] merged;
        logic [BYTE_SEL_BITS:0] byte_sel;
        begin
            merged = '0;
            for (int unsigned lane = 0; lane < ARRAY_DIM; lane++) begin
                byte_sel = {1'b0, addr_lsb} + (BYTE_SEL_BITS+1)'(lane);
                if (lane < valid_bytes) begin
                    if (byte_sel < (BYTE_SEL_BITS+1)'(BEAT_BYTES)) begin
                        merged[(lane << 3) +: 8] =
                            beat0[{byte_sel[BYTE_SEL_BITS-1:0], 3'b000} +: 8];
                    end else begin
                        merged[(lane << 3) +: 8] =
                            beat1[{byte_sel[BYTE_SEL_BITS-1:0], 3'b000} +: 8];
                    end
                end
            end
            merge_beats = merged;
        end
    endfunction

    function automatic logic [$clog2(BANKS)-1:0] bank_index(
        input logic [15:0] row_slot,
        input logic [15:0] x
    );
        bank_index = ($clog2(BANKS))'(({16'd0, row_slot} << 1) + {31'd0, x[0]});
    endfunction

    function automatic logic [BANK_ADDR_WIDTH-1:0] bank_word_addr(
        input logic [15:0] x
    );
        bank_word_addr = BANK_ADDR_WIDTH'(x >> 1);
    endfunction

    function automatic logic [2:0] mod7_u16(input logic [15:0] value);
        logic [5:0] sum0;
        logic [5:0] rem0;
        logic [5:0] rem1;
        logic [5:0] rem2;
        begin
            sum0 = {3'd0, value[2:0]} +
                   {3'd0, value[5:3]} +
                   {3'd0, value[8:6]} +
                   {3'd0, value[11:9]} +
                   {3'd0, value[14:12]} +
                   {5'd0, value[15]};
            rem0 = (sum0 >= 6'd28) ? (sum0 - 6'd28) : sum0;
            rem1 = (rem0 >= 6'd14) ? (rem0 - 6'd14) : rem0;
            rem2 = (rem1 >= 6'd7) ? (rem1 - 6'd7) : rem1;
            mod7_u16 = rem2[2:0];
        end
    endfunction

    function automatic logic [15:0] cache_row_slot(input logic [15:0] ih);
        cache_row_slot = {13'd0, mod7_u16(ih)};
    endfunction

    function automatic logic [31:0] row_stride_offset(input logic [15:0] kh);
        logic [31:0] stride_x2;
        logic [31:0] stride_x4;
        begin
            stride_x2 = row_stride_bytes_i << 1;
            stride_x4 = row_stride_bytes_i << 2;
            unique case (kh[2:0])
                3'd0: row_stride_offset = 32'd0;
                3'd1: row_stride_offset = row_stride_bytes_i;
                3'd2: row_stride_offset = stride_x2;
                3'd3: row_stride_offset = stride_x2 + row_stride_bytes_i;
                3'd4: row_stride_offset = stride_x4;
                default: row_stride_offset = 32'd0;
            endcase
        end
    endfunction

    always_comb begin
        main_crosses = ({2'b00, main_addr_q[BYTE_SEL_BITS-1:0]} +
                        {1'b0, main_valid_bytes_q}) >
                       (BYTE_SEL_BITS+2)'(BEAT_BYTES);
        background_crosses =
            ({2'b00, background_addr_q[BYTE_SEL_BITS-1:0]} +
             {1'b0, background_valid_bytes_q}) >
            (BYTE_SEL_BITS+2)'(BEAT_BYTES);

        background_ih = background_base_ih_q + $signed({16'd0, background_kh_q});
        background_row_in_bounds = (background_kh_q < kernel_h_i) &&
                                   (background_ih >= 32'sd0) &&
                                   (background_ih < $signed({16'd0, input_h_i}));
        background_row_slot = background_row_in_bounds ?
                              cache_row_slot(background_ih[15:0]) : 16'd0;
        background_query_slot_o =
            background_row_slot[$clog2(ROW_SLOTS)-1:0];
        background_query_ih_o = background_ih[15:0];
    end

    always_comb begin
        main_alloc_valid_o = main_start_i && row_ring_mode_i;
        main_alloc_slot_o = main_row_slot_i[$clog2(ROW_SLOTS)-1:0];
        main_alloc_ih_o = main_row_ih_i;
        background_alloc_valid_o = (background_state_q == BG_SCAN) &&
                                   (background_kh_q != kernel_h_i) &&
                                   background_row_in_bounds &&
                                   !background_row_cached_i &&
                                   !background_row_pending_i &&
                                   (background_valid_bytes_q != 6'd0) &&
                                   row_ring_mode_i;
        background_alloc_slot_o =
            background_row_slot[$clog2(ROW_SLOTS)-1:0];
        background_alloc_ih_o = background_ih[15:0];
    end

    always_comb begin
        main_req = (main_state_q == MAIN_REQ0) || (main_state_q == MAIN_REQ1);
        background_req = (background_state_q == BG_REQ0) ||
                         (background_state_q == BG_REQ1);
        select_main = main_req && !fifo_full;
        select_background = background_req && !fifo_full && !select_main;
        obi_req_o = 1'b0;
        obi_addr_o = '0;

        if (select_main) begin
            obi_req_o = 1'b1;
            obi_addr_o = main_pending_beat_addr_q;
        end else if (select_background) begin
            obi_req_o = 1'b1;
            obi_addr_o = background_pending_beat_addr_q;
        end
    end

    always_comb begin
        accept_main = select_main && obi_gnt_i;
        accept_background = select_background && obi_gnt_i;
        accept_request = accept_main || accept_background;
        main_request_accepted_o = accept_main;
        main_next_phase_o = main_state_q;
        if (accept_main) begin
            unique case (main_state_q)
                MAIN_REQ0: begin
                    if (main_crosses) begin
                        main_next_phase_o = MAIN_REQ1;
                    end else if ((main_x_q + 16'd1) == input_w_i) begin
                        main_next_phase_o = MAIN_DRAIN;
                    end else begin
                        main_next_phase_o = MAIN_REQ0;
                    end
                end
                MAIN_REQ1: begin
                    main_next_phase_o = ((main_x_q + 16'd1) == input_w_i) ?
                                        MAIN_DRAIN : MAIN_REQ0;
                end
                default: main_next_phase_o = main_state_q;
            endcase
        end
        main_done_o = (main_state_q == MAIN_DRAIN) &&
                      ((row_ring_mode_i && main_row_ready_i) ||
                       (!row_ring_mode_i && fifo_empty));

        accepted_meta = '0;
        if (accept_main) begin
            accepted_meta.addr_lsb = main_addr_q[BYTE_SEL_BITS-1:0];
            accepted_meta.valid_bytes = main_valid_bytes_q;
            accepted_meta.row_slot = main_row_slot_q;
            accepted_meta.x = main_x_q;
            accepted_meta.is_beat0_of_cross =
                (main_state_q == MAIN_REQ0) && main_crosses;
            accepted_meta.is_solo = (main_state_q == MAIN_REQ0) && !main_crosses;
        end else if (accept_background) begin
            accepted_meta.addr_lsb = background_addr_q[BYTE_SEL_BITS-1:0];
            accepted_meta.valid_bytes = background_valid_bytes_q;
            accepted_meta.row_slot = background_row_slot;
            accepted_meta.x = background_x_q;
            accepted_meta.is_beat0_of_cross =
                (background_state_q == BG_REQ0) && background_crosses;
            accepted_meta.is_solo = (background_state_q == BG_REQ0) &&
                                    !background_crosses;
        end

        beat_push_o = row_ring_mode_i && accept_request;
        beat_push_slot_o = accepted_meta.row_slot[$clog2(ROW_SLOTS)-1:0];
        beat_push_last_for_row_o = accept_main ?
            (((main_state_q == MAIN_REQ0) && !main_crosses &&
              ((main_x_q + 16'd1) == input_w_i)) ||
             ((main_state_q == MAIN_REQ1) &&
              ((main_x_q + 16'd1) == input_w_i))) :
            (((background_state_q == BG_REQ0) && !background_crosses &&
              ((background_x_q + 16'd1) == input_w_i)) ||
             ((background_state_q == BG_REQ1) &&
              ((background_x_q + 16'd1) == input_w_i)));

        pop_response = obi_rvalid_i && !fifo_empty;
        beat_pop_o = pop_response;
        beat_pop_slot_o = response_meta.row_slot[$clog2(ROW_SLOTS)-1:0];
    end

    always_comb begin
        bank_write_req_o = '0;
        bank_write_addr_o = '0;
        bank_write_data_o = '0;
        response_write = 1'b0;
        response_bank = '0;
        response_address = '0;
        response_data = '0;

        if (pop_response && !response_meta.is_beat0_of_cross) begin
            response_write = 1'b1;
            response_bank = bank_index(response_meta.row_slot, response_meta.x);
            response_address = bank_word_addr(response_meta.x);
            if (response_meta.is_solo) begin
                response_data = (c32_blocked_mode_i &&
                                 (response_meta.addr_lsb == '0) &&
                                 (response_meta.valid_bytes == 6'(BEAT_BYTES))) ?
                                obi_rdata_i :
                                merge_beats(obi_rdata_i, '0,
                                            response_meta.addr_lsb,
                                            response_meta.valid_bytes);
            end else begin
                response_data = merge_beats(response_beat0_q, obi_rdata_i,
                                            response_meta.addr_lsb,
                                            response_meta.valid_bytes);
            end
        end

        if (response_write) begin
            bank_write_req_o[response_bank] = 1'b1;
            bank_write_addr_o[response_bank] = response_address;
            bank_write_data_o[response_bank] = response_data;
        end
    end

    task automatic tick_main_fetch;
        begin
            if (main_start_i) begin
                main_x_q <= '0;
                main_addr_q <= main_base_addr_i;
                main_pending_beat_addr_q <= beat_base(main_base_addr_i);
                main_row_slot_q <= main_row_slot_i;
                main_valid_bytes_q <= main_valid_bytes_i;
                main_state_q <= MAIN_REQ0;
            end else begin
                unique case (main_state_q)
                    MAIN_REQ0: begin
                        if (accept_main) begin
                            if (main_crosses) begin
                                main_pending_beat_addr_q <=
                                    beat_base(main_addr_q) + ADDR_WIDTH'(BEAT_BYTES);
                                main_state_q <= MAIN_REQ1;
                            end else if ((main_x_q + 16'd1) == input_w_i) begin
                                main_state_q <= MAIN_DRAIN;
                            end else begin
                                main_x_q <= main_x_q + 16'd1;
                                main_addr_q <= main_addr_q + pixel_stride_bytes_i;
                                main_pending_beat_addr_q <=
                                    beat_base(main_addr_q + pixel_stride_bytes_i);
                            end
                        end
                    end
                    MAIN_REQ1: begin
                        if (accept_main) begin
                            if ((main_x_q + 16'd1) == input_w_i) begin
                                main_state_q <= MAIN_DRAIN;
                            end else begin
                                main_x_q <= main_x_q + 16'd1;
                                main_addr_q <= main_addr_q + pixel_stride_bytes_i;
                                main_pending_beat_addr_q <=
                                    beat_base(main_addr_q + pixel_stride_bytes_i);
                                main_state_q <= MAIN_REQ0;
                            end
                        end
                    end
                    MAIN_DRAIN: begin
                        if (main_done_o) begin
                            main_state_q <= MAIN_IDLE;
                        end
                    end
                    default: begin
                    end
                endcase
            end
        end
    endtask

    task automatic tick_background_fetch;
        begin
            if (reset_background_i) begin
                background_state_q <= BG_IDLE;
            end else begin
                unique case (background_state_q)
                    BG_IDLE: begin
                        if (background_start_i) begin
                            background_base_ih_q <= background_base_ih_i;
                            background_row_base_addr_q <= background_row_base_addr_i;
                            background_kh_q <= '0;
                            background_x_q <= '0;
                            background_valid_bytes_q <= main_valid_bytes_i;
                            background_state_q <= BG_SCAN;
                        end
                    end
                    BG_SCAN: begin
                        if (background_kh_q == kernel_h_i) begin
                            background_state_q <= BG_IDLE;
                        end else if (!background_row_in_bounds ||
                                     background_row_cached_i ||
                                     background_row_pending_i ||
                                     (background_valid_bytes_q == 6'd0)) begin
                            background_kh_q <= background_kh_q + 16'd1;
                        end else begin
                            background_x_q <= '0;
                            background_addr_q <=
                                background_row_base_addr_q +
                                row_stride_offset(background_kh_q) +
                                channel_addr_offset_i;
                            background_pending_beat_addr_q <= beat_base(
                                background_row_base_addr_q +
                                row_stride_offset(background_kh_q) +
                                channel_addr_offset_i
                            );
                            background_state_q <= BG_REQ0;
                        end
                    end
                    BG_REQ0: begin
                        if (accept_background) begin
                            if (background_crosses) begin
                                background_pending_beat_addr_q <=
                                    beat_base(background_addr_q) + ADDR_WIDTH'(BEAT_BYTES);
                                background_state_q <= BG_REQ1;
                            end else if ((background_x_q + 16'd1) == input_w_i) begin
                                background_state_q <= BG_DRAIN;
                            end else begin
                                background_x_q <= background_x_q + 16'd1;
                                background_addr_q <=
                                    background_addr_q + pixel_stride_bytes_i;
                                background_pending_beat_addr_q <= beat_base(
                                    background_addr_q + pixel_stride_bytes_i
                                );
                            end
                        end
                    end
                    BG_REQ1: begin
                        if (accept_background) begin
                            if ((background_x_q + 16'd1) == input_w_i) begin
                                background_state_q <= BG_DRAIN;
                            end else begin
                                background_x_q <= background_x_q + 16'd1;
                                background_addr_q <=
                                    background_addr_q + pixel_stride_bytes_i;
                                background_pending_beat_addr_q <= beat_base(
                                    background_addr_q + pixel_stride_bytes_i
                                );
                                background_state_q <= BG_REQ0;
                            end
                        end
                    end
                    BG_DRAIN: begin
                        if (background_row_cached_i) begin
                            background_kh_q <= background_kh_q + 16'd1;
                            background_state_q <= BG_SCAN;
                        end
                    end
                    default: background_state_q <= BG_IDLE;
                endcase
            end
        end
    endtask

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            main_state_q <= MAIN_IDLE;
            main_x_q <= '0;
            main_addr_q <= '0;
            main_pending_beat_addr_q <= '0;
            main_row_slot_q <= '0;
            main_valid_bytes_q <= '0;
            background_state_q <= BG_IDLE;
            background_base_ih_q <= '0;
            background_row_base_addr_q <= '0;
            background_kh_q <= '0;
            background_x_q <= '0;
            background_addr_q <= '0;
            background_pending_beat_addr_q <= '0;
            background_valid_bytes_q <= '0;
            beat_fifo_q <= '0;
            write_ptr_q <= '0;
            read_ptr_q <= '0;
            count_q <= '0;
            response_beat0_q <= '0;
            fetch_beats_q <= '0;
        end else begin
            tick_main_fetch();
            tick_background_fetch();

            if (clear_count_i) begin
                fetch_beats_q <= '0;
            end
            if (accept_request) begin
                beat_fifo_q[write_ptr_q] <= accepted_meta;
                write_ptr_q <= write_ptr_q + FIFO_PTR_WIDTH'(1);
            end
            if (pop_response) begin
                read_ptr_q <= read_ptr_q + FIFO_PTR_WIDTH'(1);
                fetch_beats_q <= fetch_beats_q + 32'd1;
                if (response_meta.is_beat0_of_cross) begin
                    response_beat0_q <= obi_rdata_i;
                end
            end
            unique case ({accept_request, pop_response})
                2'b10: count_q <= count_q + FIFO_COUNT_WIDTH'(1);
                2'b01: count_q <= count_q - FIFO_COUNT_WIDTH'(1);
                default: count_q <= count_q;
            endcase
        end
    end

endmodule

`default_nettype wire
