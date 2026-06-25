// ============================================================
// snitch_minimal.F  —  Minimal compile list for snitch_core.sv
//
// Dùng đường dẫn tương đối qua biến $(REPO_ROOT).
// Các dependency IP là git submodule tại hw/<name>/:
//   hw/common_cells  (pulp-platform/common_cells @ snitch branch)
//   hw/fpnew         (pulp-platform/cvfpu        @ pulp-v0.2.3)
//   hw/tech_cells    (pulp-platform/tech_cells_generic @ v0.2.13)
//   hw/riscv-dbg     (pulp-platform/riscv-dbg    @ v0.8.1)
//   hw/axi           (pulp-platform/axi           @ existing submodule)
//   hw/snitch_cluster (contains snitch/ snitch_ipu/ etc.)
// ============================================================

// -----------------------------------------------------------
// Include Directories
// -----------------------------------------------------------
+define+TARGET_VERILATOR
+define+TARGET_SYNTHESIS
+define+TARGET_SPATZ

+incdir+$(REPO_ROOT)/hw/common_cells/include
+incdir+$(REPO_ROOT)/hw/axi/include
+incdir+$(REPO_ROOT)/hw/snitch_cluster/hw/snitch/include
+incdir+$(REPO_ROOT)/hw/snitch_cluster/hw/snitch/include/cv_x_if
+incdir+$(REPO_ROOT)/hw/snitch_cluster/hw/reqrsp_interface/include
+incdir+$(REPO_ROOT)/hw/snitch_cluster/hw/tcdm_interface/include
+incdir+$(REPO_ROOT)/hw/snitch_cluster/hw/snitch_fp_ss/include
+incdir+$(REPO_ROOT)/hw/spatz/hw/ip/reqrsp_interface/include
+incdir+$(REPO_ROOT)/hw/spatz/hw/ip/tcdm_interface/include
+incdir+$(REPO_ROOT)/hw/obi/include
+incdir+$(REPO_ROOT)/hw/register_interface/include
+incdir+$(REPO_ROOT)/hw/axi_stream/include
+incdir+$(REPO_ROOT)/hw/idma/src/include
+incdir+$(REPO_ROOT)/hw/idma/target/rtl/include

// -----------------------------------------------------------
// 1. AXI package + simulation model (needed by tb_npu_cluster)
// -----------------------------------------------------------
$(REPO_ROOT)/hw/axi/src/axi_pkg.sv
$(REPO_ROOT)/hw/axi/src/axi_sim_mem.sv
$(REPO_ROOT)/hw/obi/src/obi_pkg.sv
$(REPO_ROOT)/hw/spatz/hw/ip/reqrsp_interface/src/reqrsp_pkg.sv

// -----------------------------------------------------------
// 2. Debug Module package (dm:: types used in snitch.sv)
// -----------------------------------------------------------
$(REPO_ROOT)/hw/riscv-dbg/src/dm_pkg.sv

// -----------------------------------------------------------
// 3. FPnew package (fpnew_pkg imported by snitch even with FP disabled)
// -----------------------------------------------------------
$(REPO_ROOT)/hw/fpnew/src/fpnew_pkg.sv

// -----------------------------------------------------------
// 4. Common Cells — only primitives used by snitch/snitch_lsu
// -----------------------------------------------------------
$(REPO_ROOT)/hw/common_cells/src/cf_math_pkg.sv
$(REPO_ROOT)/hw/common_cells/src/lzc.sv
$(REPO_ROOT)/hw/common_cells/src/onehot_to_bin.sv
$(REPO_ROOT)/hw/common_cells/src/popcount.sv
$(REPO_ROOT)/hw/common_cells/src/fifo_v3.sv
$(REPO_ROOT)/hw/common_cells/src/spill_register_flushable.sv
$(REPO_ROOT)/hw/common_cells/src/spill_register.sv
$(REPO_ROOT)/hw/common_cells/src/stream_fifo.sv
$(REPO_ROOT)/hw/common_cells/src/passthrough_stream_fifo.sv
$(REPO_ROOT)/hw/common_cells/src/fall_through_register.sv
$(REPO_ROOT)/hw/common_cells/src/stream_register.sv
$(REPO_ROOT)/hw/common_cells/src/id_queue.sv
$(REPO_ROOT)/hw/common_cells/src/rr_arb_tree.sv
$(REPO_ROOT)/hw/common_cells/src/delta_counter.sv
$(REPO_ROOT)/hw/common_cells/src/counter.sv
$(REPO_ROOT)/hw/common_cells/src/stream_fifo_optimal_wrap.sv
$(REPO_ROOT)/hw/common_cells/src/deprecated/rrarbiter.sv
$(REPO_ROOT)/hw/common_cells/src/deprecated/find_first_one.sv

// -----------------------------------------------------------
// 5. Tech Cells — tc_sram for simulation
// -----------------------------------------------------------
$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_clk.sv
$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram.sv
$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram_impl.sv

// -----------------------------------------------------------
// 6. Snitch IPU (integer processing unit: mul/div)
// -----------------------------------------------------------
$(REPO_ROOT)/hw/snitch_cluster/hw/snitch_ipu/src/snitch_multiplier.sv
$(REPO_ROOT)/hw/snitch_cluster/hw/snitch_ipu/src/snitch_serial_divider.sv
$(REPO_ROOT)/hw/snitch_cluster/hw/snitch_ipu/src/snitch_ipu.sv

// -----------------------------------------------------------
// 7. Snitch Core files
// -----------------------------------------------------------
$(REPO_ROOT)/hw/snitch_cluster/hw/snitch/src/snitch_pma_pkg.sv
$(REPO_ROOT)/hw/rtl/cluster/riscv_instr_npu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/snitch/src/snitch_pkg.sv
$(REPO_ROOT)/hw/spatz/hw/ip/snitch/src/snitch_regfile_ff.sv
$(REPO_ROOT)/hw/spatz/hw/ip/snitch/src/snitch_lsu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/snitch/src/snitch_l0_tlb.sv
$(REPO_ROOT)/hw/spatz/hw/ip/snitch/src/snitch.sv

// -----------------------------------------------------------
// 8. NPU Cluster RTL (our files — ordered by dependency)
// -----------------------------------------------------------
$(REPO_ROOT)/hw/rtl/cluster/npu_cluster_pkg.sv
$(REPO_ROOT)/hw/rtl/cluster/cluster_sram_bank.sv
$(REPO_ROOT)/hw/rtl/cluster/obi_arbiter_2to1.sv
$(REPO_ROOT)/hw/rtl/cluster/obi_demux_1to4.sv
$(REPO_ROOT)/hw/rtl/cluster/obi_demux_1to5.sv
$(REPO_ROOT)/hw/rtl/cluster/obi_narrow_to_wide.sv
$(REPO_ROOT)/hw/rtl/cluster/axi_lite_to_obi.sv
$(REPO_ROOT)/hw/rtl/cluster/cluster_ctrl_regs.sv
$(REPO_ROOT)/hw/rtl/cluster/npu_interrupt_ctrl.sv
$(REPO_ROOT)/hw/idma/src/idma_pkg.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_axil_read.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_axil_write.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_axi_read.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_axi_write.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_axis_read.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_axis_write.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_channel_coupler.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_dataflow_element.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_error_handler.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_init_read.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_init_write.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_legalizer_page_splitter.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_legalizer_pow2_splitter.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_obi_read.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_obi_write.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_tilelink_read.sv
$(REPO_ROOT)/hw/idma/src/backend/idma_tilelink_write.sv
$(REPO_ROOT)/hw/idma/src/midend/idma_nd_midend.sv
$(REPO_ROOT)/hw/idma/src/frontend/idma_transfer_id_gen.sv
$(REPO_ROOT)/hw/idma/target/rtl/idma_generated.sv
$(REPO_ROOT)/hw/register_interface/vendor/lowrisc_opentitan/src/prim_subreg_arb.sv
$(REPO_ROOT)/hw/register_interface/vendor/lowrisc_opentitan/src/prim_subreg_ext.sv
$(REPO_ROOT)/hw/register_interface/vendor/lowrisc_opentitan/src/prim_subreg.sv
$(REPO_ROOT)/hw/rtl/cluster/npu_pulp_idma_ctrl_mm.sv
$(REPO_ROOT)/hw/rtl/afu/afu_fifo_ff.sv
$(REPO_ROOT)/hw/rtl/afu/afu_frontend.sv
$(REPO_ROOT)/hw/rtl/afu/afu_backend.sv
$(REPO_ROOT)/hw/rtl/afu/afu_core.sv
$(REPO_ROOT)/hw/rtl/afu/afu.sv
$(REPO_ROOT)/hw/rtl/interconnect/tcdm_interconnect.sv
$(REPO_ROOT)/hw/rtl/systolic/requant_pipeline.sv
$(REPO_ROOT)/hw/rtl/systolic/systolic_controller.sv
$(REPO_ROOT)/hw/rtl/systolic/npu_systolic_array.sv
$(REPO_ROOT)/hw/rtl/cluster/snitch_core.sv

// -----------------------------------------------------------
// 9. Spatz Vector Engine (Phase 3B-B)
// -----------------------------------------------------------
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/reorder_buffer.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/rvv_pkg.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/generated/spatz_pkg.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_serdiv.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_decoder.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_simd_lane.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/vregfile.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_ipu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vfu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vlsu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_doublebw_vlsu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vrf.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_vsldu.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz_controller.sv
$(REPO_ROOT)/hw/spatz/hw/ip/spatz/src/spatz.sv
$(REPO_ROOT)/hw/rtl/cluster/tcdm_to_obi_bridge.sv

$(REPO_ROOT)/hw/rtl/cluster/obi_snitch_if_adapter.sv
$(REPO_ROOT)/hw/rtl/cluster/npu_cluster.sv
$(REPO_ROOT)/hw/rtl/cluster/tb/tb_npu_cluster.sv
