`default_nettype none

//-------------------------------------------------------------------------
// Processing Element (PE) for the Systolic Array
// Performs: PSum_out = PSum_in + (Weight * IFM)
// Mechanism: Weight-Stationary
//-------------------------------------------------------------------------
module npu_pe (
    input  logic        clk_i,
    input  logic        rst_ni,
    
    // Control signals
    input  logic        clear_acc_i,   // Reset the accumulator
    input  logic        weight_load_i, // Load weight from top
    
    // Data inputs
    input  logic signed [7:0]  weight_i,  // Weight from top neighbor
    input  logic signed [7:0]  ifm_i,     // Input Feature Map from left neighbor
    input  logic signed [31:0] psum_i,    // Partial Sum from top neighbor
    
    // Data outputs
    output logic signed [7:0]  weight_o,  // Pass weight to bottom neighbor
    output logic signed [7:0]  ifm_o,     // Pass IFM to right neighbor
    output logic signed [31:0] psum_o     // Pass accumulated PSum to bottom neighbor
);

    logic signed [7:0]  weight_q;
    logic signed [7:0]  ifm_q;
    logic signed [31:0] psum_q;

    // Weight Stationary Logic
    // During weight_load_i = 1, weights shift down through the array.
    // When weight_load_i = 0, the weight is locked in weight_q.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            weight_q <= '0;
        end else if (weight_load_i) begin
            weight_q <= weight_i;
        end
    end
    assign weight_o = weight_q; // Shift weight down

    // Compute & Dataflow Logic
    // IFM shifts right, PSum shifts down every cycle
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            ifm_q  <= '0;
            psum_q <= '0;
        end else begin
            ifm_q <= ifm_i; // Shift IFM right
            
            if (clear_acc_i) begin
                // New accumulation sequence
                psum_q <= psum_i + (weight_q * ifm_i);
            end else begin
                // Ongoing accumulation
                psum_q <= psum_i + (weight_q * ifm_i);
            end
        end
    end

    assign ifm_o  = ifm_q;
    assign psum_o = psum_q;

endmodule

//-------------------------------------------------------------------------
// NPU Systolic Array Top Level
// A configurable 2D grid of PEs (default 32x32).
//-------------------------------------------------------------------------
module npu_systolic_array #(
    parameter int unsigned ARRAY_DIM = 32
)(
    input  logic                                clk_i,
    input  logic                                rst_ni,
    
    // Control signals from Custom Matrix Controller
    input  logic                                weight_load_en_i,
    input  logic                                clear_acc_i,
    input  logic                                compute_en_i,

    // Vector Inputs
    input  logic signed [ARRAY_DIM-1:0][7:0]    weight_data_i, // Top edge (32 x 8-bit)
    input  logic signed [ARRAY_DIM-1:0][7:0]    ifm_data_i,    // Left edge (32 x 8-bit)
    input  logic signed [ARRAY_DIM-1:0][31:0]   psum_data_i,   // Top edge (32 x 32-bit, usually 0)
    
    // Vector Output
    output logic signed [ARRAY_DIM-1:0][31:0]   ofm_data_o,    // Bottom edge (32 x 32-bit)
    output logic                                ofm_valid_o
);

    // 2D wire arrays to connect the PE grid
    logic signed [7:0]  weight_wire [ARRAY_DIM+1][ARRAY_DIM];
    logic signed [7:0]  ifm_wire    [ARRAY_DIM][ARRAY_DIM+1];
    logic signed [31:0] psum_wire   [ARRAY_DIM+1][ARRAY_DIM];

    // 1. Connect boundary inputs
    for (genvar c = 0; c < ARRAY_DIM; c++) begin : gen_top_edge
        assign weight_wire[0][c] = weight_data_i[c];
        assign psum_wire[0][c]   = psum_data_i[c];
    end

    for (genvar r = 0; r < ARRAY_DIM; r++) begin : gen_left_edge
        assign ifm_wire[r][0] = ifm_data_i[r];
    end

    // 2. Connect boundary outputs
    for (genvar c = 0; c < ARRAY_DIM; c++) begin : gen_bottom_edge
        assign ofm_data_o[c] = psum_wire[ARRAY_DIM][c];
    end

    // 3. Generate 32x32 Grid of Processing Elements (PEs)
    for (genvar r = 0; r < ARRAY_DIM; r++) begin : gen_row
        for (genvar c = 0; c < ARRAY_DIM; c++) begin : gen_col
            npu_pe u_pe (
                .clk_i         ( clk_i ),
                .rst_ni        ( rst_ni ),
                .clear_acc_i   ( clear_acc_i ),
                .weight_load_i ( weight_load_en_i ),
                .weight_i      ( weight_wire[r][c] ),
                .ifm_i         ( ifm_wire[r][c] ),
                .psum_i        ( psum_wire[r][c] ),
                .weight_o      ( weight_wire[r+1][c] ),
                .ifm_o         ( ifm_wire[r][c+1] ),
                .psum_o        ( psum_wire[r+1][c] )
            );
        end
    end

    // 4. Validity Shift Register
    // Data takes ARRAY_DIM cycles to propagate from top-left to bottom-right.
    logic [ARRAY_DIM-1:0] valid_sr;
    
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_sr <= '0;
        end else begin
            valid_sr <= {valid_sr[ARRAY_DIM-2:0], compute_en_i};
        end
    end
    
    assign ofm_valid_o = valid_sr[ARRAY_DIM-1];

endmodule
