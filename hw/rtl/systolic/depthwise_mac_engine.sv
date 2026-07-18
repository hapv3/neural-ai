`default_nettype none

module depthwise_mac_engine #(
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned ACC_WIDTH = 32
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic flush_i,

    input  logic                                         in_valid_i,
    output logic                                         in_ready_o,
    input  logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0]   ifm_i,
    input  logic [ARRAY_DIM-1:0][INPUT_ELEM_WIDTH-1:0]   weight_i,
    input  logic                                         clear_i,
    input  logic                                         last_i,
    input  logic [5:0]                                   valid_lanes_i,

    output logic                                         out_valid_o,
    input  logic                                         out_ready_i,
    output logic [ARRAY_DIM-1:0][ACC_WIDTH-1:0]          acc_o
);

    typedef logic [ARRAY_DIM-1:0][ACC_WIDTH-1:0] acc_row_t;
    typedef logic signed [15:0] product_t;

    logic      s1_valid_q;
    logic      s1_valid_n;
    product_t  s1_product_q [ARRAY_DIM];
    product_t  s1_product_n [ARRAY_DIM];
    logic      s1_clear_q, s1_clear_n;
    logic      s1_last_q, s1_last_n;
    logic [5:0] s1_valid_lanes_q, s1_valid_lanes_n;

    acc_row_t  acc_q;
    acc_row_t  acc_n;

    logic      out_valid_q;
    logic      out_valid_n;
    acc_row_t  out_acc_q;
    acc_row_t  out_acc_n;

    logic      s1_ready;
    logic      add_ready;

    assign add_ready = !out_valid_q || out_ready_i;
    assign s1_ready = add_ready || !s1_valid_q;
    assign in_ready_o = s1_ready;
    assign out_valid_o = out_valid_q;
    assign acc_o = out_acc_q;

    function automatic product_t mul_i8_signed(
        input logic signed [INPUT_ELEM_WIDTH-1:0] lhs,
        input logic signed [INPUT_ELEM_WIDTH-1:0] rhs
    );
        begin
            mul_i8_signed = product_t'(lhs) * product_t'(rhs);
        end
    endfunction

    function automatic logic [ACC_WIDTH-1:0] extend_product(input product_t product);
        begin
            extend_product = {{(ACC_WIDTH-16){product[15]}}, product};
        end
    endfunction

    always_comb begin
        logic [ACC_WIDTH-1:0] product_ext;
        acc_row_t next_acc;

        s1_valid_n = s1_valid_q;
        s1_clear_n = s1_clear_q;
        s1_last_n = s1_last_q;
        s1_valid_lanes_n = s1_valid_lanes_q;
        for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
            s1_product_n[ch] = s1_product_q[ch];
        end

        acc_n = acc_q;
        out_valid_n = out_valid_q;
        out_acc_n = out_acc_q;
        next_acc = acc_q;
        product_ext = '0;

        if (out_valid_q && out_ready_i) begin
            out_valid_n = 1'b0;
        end

        if (add_ready) begin
            if (s1_valid_q) begin
                for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
                    if (ch < s1_valid_lanes_q) begin
                        product_ext = extend_product(s1_product_q[ch]);
                        next_acc[ch] = s1_clear_q ? product_ext : (acc_q[ch] + product_ext);
                    end else begin
                        next_acc[ch] = '0;
                    end
                end
                acc_n = next_acc;
                if (s1_last_q) begin
                    out_valid_n = 1'b1;
                    out_acc_n = next_acc;
                end
            end else begin
                s1_valid_n = 1'b0;
            end
        end

        if (s1_ready) begin
            s1_valid_n = in_valid_i;
            s1_clear_n = clear_i;
            s1_last_n = last_i;
            s1_valid_lanes_n = valid_lanes_i;
            for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
                if (ch < valid_lanes_i) begin
                    s1_product_n[ch] = mul_i8_signed($signed(ifm_i[ch]), $signed(weight_i[ch]));
                end else begin
                    s1_product_n[ch] = '0;
                end
            end
        end

        if (flush_i) begin
            s1_valid_n = 1'b0;
            out_valid_n = 1'b0;
            acc_n = '0;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            s1_valid_q <= 1'b0;
            s1_clear_q <= 1'b0;
            s1_last_q <= 1'b0;
            s1_valid_lanes_q <= '0;
            acc_q <= '0;
            out_valid_q <= 1'b0;
            out_acc_q <= '0;
            for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
                s1_product_q[ch] <= '0;
            end
        end else begin
            s1_valid_q <= s1_valid_n;
            s1_clear_q <= s1_clear_n;
            s1_last_q <= s1_last_n;
            s1_valid_lanes_q <= s1_valid_lanes_n;
            acc_q <= acc_n;
            out_valid_q <= out_valid_n;
            out_acc_q <= out_acc_n;
            for (int unsigned ch = 0; ch < ARRAY_DIM; ch++) begin
                s1_product_q[ch] <= s1_product_n[ch];
            end
        end
    end

endmodule

`default_nettype wire
