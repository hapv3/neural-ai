`default_nettype none

module afu_fifo_ff #(
    parameter string       NAME       = "FIFO",
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned DEPTH      = 2
) (
    input  logic                  clk_i,
    input  logic                  rst_ni,
    input  logic                  flush_i,
    
    output logic                  full_o,
    output logic                  almost_full_o,
    output logic                  empty_o,
    output logic                  all_empty_o,
    
    input  logic [DATA_WIDTH-1:0] data_i,
    input  logic                  push_i,
    
    output logic [DATA_WIDTH-1:0] data_o,
    input  logic                  pop_i
);
    localparam int unsigned ADDR_WIDTH = $clog2(DEPTH);
    
    logic [DEPTH-1:0][DATA_WIDTH-1:0] mem_q;
    logic [ADDR_WIDTH-1:0]            w_ptr_q, r_ptr_q;
    logic [ADDR_WIDTH:0]              cnt_q;
    
    assign full_o        = (cnt_q == (ADDR_WIDTH+1)'(DEPTH));
    // almost_full is used to stop issuing new reads if in-flight reads might overflow the FIFO.
    // Since OBI has 1 cycle delay from gnt to rvalid, we stop at DEPTH-1.
    assign almost_full_o = (cnt_q >= (ADDR_WIDTH+1)'(DEPTH - 1));
    assign empty_o       = (cnt_q == 0);
    assign all_empty_o   = empty_o;
    
    assign data_o        = mem_q[r_ptr_q];
    
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            w_ptr_q <= '0;
            r_ptr_q <= '0;
            cnt_q   <= '0;
            mem_q   <= '0;
        end else if (flush_i) begin
            w_ptr_q <= '0;
            r_ptr_q <= '0;
            cnt_q   <= '0;
        end else begin
            logic push_accept;
            logic pop_accept;
            
            push_accept = push_i && (!full_o || (pop_i && !empty_o));
            pop_accept  = pop_i && !empty_o;
            
            if (push_accept) begin
                mem_q[w_ptr_q] <= data_i;
                w_ptr_q        <= (w_ptr_q == (ADDR_WIDTH)'(DEPTH-1)) ? '0 : w_ptr_q + 1;
            end
            
            if (pop_accept) begin
                r_ptr_q        <= (r_ptr_q == (ADDR_WIDTH)'(DEPTH-1)) ? '0 : r_ptr_q + 1;
            end
            
            if (push_accept && !pop_accept) begin
                cnt_q <= cnt_q + 1;
            end else if (!push_accept && pop_accept) begin
                cnt_q <= cnt_q - 1;
            end
        end
    end
endmodule
