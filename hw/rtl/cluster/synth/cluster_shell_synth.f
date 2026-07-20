+define+TARGET_SYNTHESIS
+define+TARGET_SPATZ
+define+SYNTHESIS_SRAM_BLACKBOX
+define+SYNTH_REAL_LOCAL_CLUSTER_RTL

+incdir+../../../../hw/common_cells/include
+incdir+../../../../hw/axi/include
+incdir+../../../../hw/snitch_cluster/hw/snitch/include
+incdir+../../../../hw/snitch_cluster/hw/snitch/include/cv_x_if
+incdir+../../../../hw/snitch_cluster/hw/reqrsp_interface/include
+incdir+../../../../hw/snitch_cluster/hw/tcdm_interface/include
+incdir+../../../../hw/snitch_cluster/hw/snitch_fp_ss/include
+incdir+../../../../hw/spatz/hw/ip/reqrsp_interface/include
+incdir+../../../../hw/spatz/hw/ip/tcdm_interface/include
+incdir+../../../../hw/obi/include
+incdir+../../../../hw/register_interface/include
+incdir+../../../../hw/axi_stream/include
+incdir+../../../../hw/idma/src/include

../../../../hw/rtl/cluster/synth/sram_blackboxes.sv

../../../../hw/axi/src/axi_pkg.sv
../../../../hw/obi/src/obi_pkg.sv
../../../../hw/spatz/hw/ip/reqrsp_interface/src/reqrsp_pkg.sv
../../../../hw/riscv-dbg/src/dm_pkg.sv
../../../../hw/fpnew/src/fpnew_pkg.sv

../../../../hw/common_cells/src/cf_math_pkg.sv
../../../../hw/common_cells/src/lzc.sv
../../../../hw/common_cells/src/onehot_to_bin.sv
../../../../hw/common_cells/src/popcount.sv
../../../../hw/common_cells/src/fifo_v3.sv
../../../../hw/common_cells/src/spill_register_flushable.sv
../../../../hw/common_cells/src/spill_register.sv
../../../../hw/common_cells/src/stream_fifo.sv
../../../../hw/common_cells/src/passthrough_stream_fifo.sv
../../../../hw/common_cells/src/fall_through_register.sv
../../../../hw/common_cells/src/stream_register.sv
../../../../hw/common_cells/src/id_queue.sv
../../../../hw/common_cells/src/rr_arb_tree.sv
../../../../hw/common_cells/src/delta_counter.sv
../../../../hw/common_cells/src/counter.sv
../../../../hw/common_cells/src/stream_fifo_optimal_wrap.sv
../../../../hw/common_cells/src/deprecated/rrarbiter.sv
../../../../hw/common_cells/src/deprecated/find_first_one.sv

../../../../hw/tech_cells/src/rtl/tc_clk.sv

../../../../hw/snitch_cluster/hw/snitch/src/snitch_pma_pkg.sv
../../../../hw/rtl/cluster/riscv_instr_npu.sv
../../../../hw/spatz/hw/ip/snitch/src/snitch_pkg.sv
../../../../hw/rtl/cluster/synth/cluster_ip_blackboxes.sv

../../../../hw/rtl/cluster/npu_cluster_pkg.sv
../../../../hw/rtl/cluster/obi_arbiter_2to1.sv
../../../../hw/rtl/cluster/obi_req_register_slice.sv
../../../../hw/rtl/cluster/obi_demux_1to4.sv
../../../../hw/rtl/cluster/obi_demux_1to5.sv
../../../../hw/rtl/cluster/obi_narrow_to_wide.sv
../../../../hw/rtl/cluster/axi_lite_to_obi.sv
../../../../hw/rtl/systolic/systolic_ctrl_regs.sv
../../../../hw/rtl/cluster/npu_interrupt_ctrl.sv
../../../../hw/rtl/cluster/npu_pmu.sv
../../../../hw/rtl/cluster/npu_cmd_ctrl.sv

../../../../hw/rtl/afu/afu_fifo_ff.sv
../../../../hw/rtl/afu/afu_frontend.sv
../../../../hw/rtl/afu/afu_backend.sv
../../../../hw/rtl/afu/afu_core.sv
../../../../hw/rtl/afu/afu.sv
../../../../hw/rtl/interconnect/tcdm_interconnect.sv
../../../../hw/rtl/systolic/requant_pipeline.sv
../../../../hw/rtl/systolic/conv_linebuf_stream_packer.sv
../../../../hw/rtl/systolic/npu_systolic_array.sv
../../../../hw/rtl/systolic/systolic_controller.sv

../../../../hw/rtl/cluster/tcdm_to_obi_bridge.sv
../../../../hw/rtl/cluster/obi_snitch_if_adapter.sv
../../../../hw/rtl/cluster/npu_cluster.sv
