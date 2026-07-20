set lib_path "tools/skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib"
set netlist_path "build/synth/npu_systolic_array_sky130_tt_sta.v"
set top_name "npu_pe"

if {[info exists ::env(PERIOD_NS)]} {
    set period_ns $::env(PERIOD_NS)
} else {
    set period_ns 8.8
}

read_liberty $lib_path
read_verilog $netlist_path
link_design $top_name

create_clock -name clk -period $period_ns [get_ports clk_i]
set_false_path -from [get_ports rst_ni]

report_checks -path_delay max -group_path_count 5 -fields {slew cap input fanout}
report_wns
report_tns
