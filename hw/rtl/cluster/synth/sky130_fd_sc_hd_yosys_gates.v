(* techmap_celltype = "$_AND_" *)
module sky130_map_and(input A, input B, output Y);
    sky130_fd_sc_hd__and2_1 _TECHMAP_REPLACE_ (.A(A), .B(B), .X(Y));
endmodule

(* techmap_celltype = "$_OR_" *)
module sky130_map_or(input A, input B, output Y);
    sky130_fd_sc_hd__or2_1 _TECHMAP_REPLACE_ (.A(A), .B(B), .X(Y));
endmodule

(* techmap_celltype = "$_XOR_" *)
module sky130_map_xor(input A, input B, output Y);
    sky130_fd_sc_hd__xor2_1 _TECHMAP_REPLACE_ (.A(A), .B(B), .X(Y));
endmodule

(* techmap_celltype = "$_XNOR_" *)
module sky130_map_xnor(input A, input B, output Y);
    sky130_fd_sc_hd__xnor2_1 _TECHMAP_REPLACE_ (.A(A), .B(B), .X(Y));
endmodule

(* techmap_celltype = "$_NAND_" *)
module sky130_map_nand(input A, input B, output Y);
    sky130_fd_sc_hd__nand2_1 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
endmodule

(* techmap_celltype = "$_NOR_" *)
module sky130_map_nor(input A, input B, output Y);
    sky130_fd_sc_hd__nor2_1 _TECHMAP_REPLACE_ (.A(A), .B(B), .Y(Y));
endmodule

(* techmap_celltype = "$_NOT_" *)
module sky130_map_not(input A, output Y);
    sky130_fd_sc_hd__inv_1 _TECHMAP_REPLACE_ (.A(A), .Y(Y));
endmodule

(* techmap_celltype = "$_BUF_" *)
module sky130_map_buf(input A, output Y);
    sky130_fd_sc_hd__buf_1 _TECHMAP_REPLACE_ (.A(A), .X(Y));
endmodule

(* techmap_celltype = "$_MUX_" *)
module sky130_map_mux(input A, input B, input S, output Y);
    sky130_fd_sc_hd__mux2_1 _TECHMAP_REPLACE_ (.A0(A), .A1(B), .S(S), .X(Y));
endmodule

(* techmap_celltype = "$_NMUX_" *)
module sky130_map_nmux(input A, input B, input S, output Y);
    wire mux_y;
    sky130_fd_sc_hd__mux2_1 _TECHMAP_REPLACE_.mux (.A0(A), .A1(B), .S(S), .X(mux_y));
    sky130_fd_sc_hd__inv_1 _TECHMAP_REPLACE_.inv (.A(mux_y), .Y(Y));
endmodule

(* techmap_celltype = "$_ANDNOT_" *)
module sky130_map_andnot(input A, input B, output Y);
    wire b_n;
    sky130_fd_sc_hd__inv_1 _TECHMAP_REPLACE_.inv (.A(B), .Y(b_n));
    sky130_fd_sc_hd__and2_1 _TECHMAP_REPLACE_.and2 (.A(A), .B(b_n), .X(Y));
endmodule

(* techmap_celltype = "$_ORNOT_" *)
module sky130_map_ornot(input A, input B, output Y);
    wire b_n;
    sky130_fd_sc_hd__inv_1 _TECHMAP_REPLACE_.inv (.A(B), .Y(b_n));
    sky130_fd_sc_hd__or2_1 _TECHMAP_REPLACE_.or2 (.A(A), .B(b_n), .X(Y));
endmodule

(* techmap_celltype = "$_ALDFF_PN_" *)
module sky130_map_aldff_pn(input D, input C, input L, output Q);
    parameter AD = 1'b0;
    wire _TECHMAP_FAIL_ = (AD !== 1'b0) && (AD !== 1'b1);
    generate
        if (AD == 1'b0) begin : gen_reset
            sky130_fd_sc_hd__dfrtp_1 _TECHMAP_REPLACE_ (.CLK(C), .D(D), .RESET_B(L), .Q(Q));
        end else begin : gen_set
            sky130_fd_sc_hd__dfstp_2 _TECHMAP_REPLACE_ (.CLK(C), .D(D), .SET_B(L), .Q(Q));
        end
    endgenerate
endmodule
