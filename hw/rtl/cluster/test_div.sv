module test_div;
  initial begin
    $display("Result=%h", ((32'h1010_0000 >> 5) / 12) << 5);
    $finish;
  end
endmodule
