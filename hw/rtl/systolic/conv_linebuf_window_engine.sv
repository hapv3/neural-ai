`default_nettype none

module conv_linebuf_window_engine #(
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned K_MAX = 5,
    parameter int unsigned BANKS = 14,
    parameter int unsigned BANK_ADDR_WIDTH = 9
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  logic clear_i,
    input  logic load_request_i,
    input  logic load_capture_i,
    input  logic [15:0] load_request_kw_i,
    input  logic [15:0] load_capture_kw_i,
    input  logic slide_request_i,
    input  logic signed [31:0] slide_from_iw_i,
    input  logic slide_commit_i,

    input  logic [15:0] input_h_i,
    input  logic [15:0] input_w_i,
    input  logic [15:0] kernel_h_i,
    input  logic [15:0] kernel_w_i,
    input  logic [15:0] stride_w_i,
    input  logic signed [31:0] base_ih_i,
    input  logic signed [31:0] base_iw_i,
    input  logic row_ring_mode_i,
    input  logic row_cache_full_i,
    input  logic [DATA_WIDTH-1:0] pad_vector_i,

    output logic [BANKS-1:0] bank_read_req_o,
    output logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] bank_read_addr_o,
    input  logic [BANKS-1:0][DATA_WIDTH-1:0] bank_read_data_i,

    output logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_o
);

    typedef logic [K_MAX-1:0][K_MAX-1:0][DATA_WIDTH-1:0] window_t;

    window_t window_q;
    window_t slide_window;

    assign window_o = window_q;

    function automatic logic [2:0] mod7_u16(input logic [15:0] value);
        logic [5:0] sum0;
        logic [5:0] rem0;
        logic [5:0] rem1;
        logic [5:0] rem2;
        begin
            // ROW_SLOTS is fixed at 7. Since 8 mod 7 == 1, n mod 7 is the
            // modulo-7 sum of its 3-bit chunks, avoiding a divider.
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

    task automatic request_column(
        input logic signed [31:0] base_iw,
        input logic [15:0] target_kw,
        input logic add_stride,
        inout logic [BANKS-1:0] read_req,
        inout logic [BANKS-1:0][BANK_ADDR_WIDTH-1:0] read_addr
    );
        logic signed [31:0] cell_ih;
        logic signed [31:0] cell_iw;
        logic [$clog2(BANKS)-1:0] bank;
        begin
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                cell_ih = base_ih_i + $signed(32'(kh));
                cell_iw = base_iw + $signed({16'd0, target_kw});
                if (add_stride) begin
                    cell_iw = cell_iw + $signed({16'd0, stride_w_i});
                end
                if ((kh < kernel_h_i) &&
                    (target_kw < kernel_w_i) &&
                    (cell_ih >= 32'sd0) &&
                    (cell_iw >= 32'sd0) &&
                    (cell_ih < $signed({16'd0, input_h_i})) &&
                    (cell_iw < $signed({16'd0, input_w_i}))) begin
                    bank = bank_index(
                        row_ring_mode_i ? cache_row_slot(cell_ih[15:0]) :
                        (row_cache_full_i ? cell_ih[15:0] : 16'(kh)),
                        cell_iw[15:0]
                    );
                    read_req[bank] = 1'b1;
                    read_addr[bank] = bank_word_addr(cell_iw[15:0]);
                end
            end
        end
    endtask

    always_comb begin
        logic [15:0] target_kw;
        bank_read_req_o = '0;
        bank_read_addr_o = '0;
        target_kw = '0;

        if (load_request_i) begin
            request_column(
                base_iw_i, load_request_kw_i, 1'b0,
                bank_read_req_o, bank_read_addr_o
            );
        end else if (slide_request_i) begin
            if (stride_w_i == 16'd1) begin
                target_kw = (kernel_w_i == 16'd1) ? 16'd0 : kernel_w_i - 16'd1;
                request_column(
                    slide_from_iw_i, target_kw, 1'b1,
                    bank_read_req_o, bank_read_addr_o
                );
            end else if (kernel_w_i == 16'd1) begin
                request_column(
                    slide_from_iw_i, 16'd0, 1'b1,
                    bank_read_req_o, bank_read_addr_o
                );
            end else begin
                target_kw = (stride_w_i >= kernel_w_i) ? 16'd0 : kernel_w_i - 16'd2;
                request_column(
                    slide_from_iw_i, target_kw, 1'b1,
                    bank_read_req_o, bank_read_addr_o
                );
                target_kw = (stride_w_i >= kernel_w_i) ? 16'd1 : kernel_w_i - 16'd1;
                request_column(
                    slide_from_iw_i, target_kw, 1'b1,
                    bank_read_req_o, bank_read_addr_o
                );
            end
        end
    end

    task automatic capture_slide_column(
        input logic [15:0] target_kw,
        inout window_t next_window
    );
        logic signed [31:0] cell_ih;
        logic signed [31:0] cell_iw;
        logic [$clog2(BANKS)-1:0] bank;
        begin
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                if (kh < kernel_h_i) begin
                    cell_ih = base_ih_i + $signed(32'(kh));
                    cell_iw = base_iw_i + $signed({16'd0, stride_w_i}) +
                              $signed({16'd0, target_kw});
                    if ((cell_ih >= 32'sd0) &&
                        (cell_iw >= 32'sd0) &&
                        (cell_ih < $signed({16'd0, input_h_i})) &&
                        (cell_iw < $signed({16'd0, input_w_i}))) begin
                        bank = bank_index(
                            row_ring_mode_i ? cache_row_slot(cell_ih[15:0]) :
                            (row_cache_full_i ? cell_ih[15:0] : 16'(kh)),
                            cell_iw[15:0]
                        );
                        next_window[kh][target_kw[2:0]] = bank_read_data_i[bank];
                    end else begin
                        next_window[kh][target_kw[2:0]] = pad_vector_i;
                    end
                end
            end
        end
    endtask

    always_comb begin
        logic [15:0] target_kw;
        slide_window = '0;
        target_kw = '0;
        if (stride_w_i >= kernel_w_i) begin
            slide_window = '0;
        end else if (stride_w_i == 16'd1) begin
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                if (kh < kernel_h_i) begin
                    slide_window[kh][0] = (kernel_w_i > 16'd1) ? window_q[kh][1] : pad_vector_i;
                    slide_window[kh][1] = (kernel_w_i > 16'd2) ? window_q[kh][2] : pad_vector_i;
                    slide_window[kh][2] = (kernel_w_i > 16'd3) ? window_q[kh][3] : pad_vector_i;
                    slide_window[kh][3] = (kernel_w_i > 16'd4) ? window_q[kh][4] : pad_vector_i;
                    slide_window[kh][4] = pad_vector_i;
                end
            end
        end else begin
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                if (kh < kernel_h_i) begin
                    slide_window[kh][0] = (kernel_w_i > 16'd2) ? window_q[kh][2] : pad_vector_i;
                    slide_window[kh][1] = (kernel_w_i > 16'd3) ? window_q[kh][3] : pad_vector_i;
                    slide_window[kh][2] = (kernel_w_i > 16'd4) ? window_q[kh][4] : pad_vector_i;
                    slide_window[kh][3] = pad_vector_i;
                    slide_window[kh][4] = pad_vector_i;
                end
            end
        end

        if (stride_w_i == 16'd1) begin
            target_kw = (kernel_w_i == 16'd1) ? 16'd0 : kernel_w_i - 16'd1;
            capture_slide_column(target_kw, slide_window);
        end else if (kernel_w_i == 16'd1) begin
            capture_slide_column(16'd0, slide_window);
        end else begin
            target_kw = (stride_w_i >= kernel_w_i) ? 16'd0 : kernel_w_i - 16'd2;
            capture_slide_column(target_kw, slide_window);
            target_kw = (stride_w_i >= kernel_w_i) ? 16'd1 : kernel_w_i - 16'd1;
            capture_slide_column(target_kw, slide_window);
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            window_q <= '0;
        end else if (clear_i) begin
            window_q <= '0;
        end else if (load_capture_i) begin
            for (int unsigned kh = 0; kh < K_MAX; kh++) begin
                logic signed [31:0] cell_ih;
                logic signed [31:0] cell_iw;
                logic [$clog2(BANKS)-1:0] bank;
                cell_ih = base_ih_i + $signed(32'(kh));
                cell_iw = base_iw_i + $signed({16'd0, load_capture_kw_i});
                if ((kh < kernel_h_i) &&
                    (load_capture_kw_i < kernel_w_i) &&
                    (cell_ih >= 32'sd0) &&
                    (cell_iw >= 32'sd0) &&
                    (cell_ih < $signed({16'd0, input_h_i})) &&
                    (cell_iw < $signed({16'd0, input_w_i}))) begin
                    bank = bank_index(
                        row_ring_mode_i ? cache_row_slot(cell_ih[15:0]) :
                        (row_cache_full_i ? cell_ih[15:0] : 16'(kh)),
                        cell_iw[15:0]
                    );
                    window_q[kh][load_capture_kw_i[2:0]] <= bank_read_data_i[bank];
                end else begin
                    window_q[kh][load_capture_kw_i[2:0]] <= pad_vector_i;
                end
            end
        end else if (slide_commit_i) begin
            window_q <= slide_window;
        end
    end

endmodule

`default_nettype wire
