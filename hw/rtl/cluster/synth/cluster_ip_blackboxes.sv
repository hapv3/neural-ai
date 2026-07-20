`default_nettype none

`include "snitch/typedef.svh"

(* blackbox *)
module snitch_core #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned I_DATA_WIDTH = 256,
    parameter int unsigned D_DATA_WIDTH = 64,
    parameter logic [31:0] BOOT_ADDR = 32'h0000_1000
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic [31:0] hart_id_i,
    input  snitch_pkg::interrupts_t irq_i,
    output logic                      obi_i_req_o,
    input  logic                      obi_i_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_i_addr_o,
    output logic                      obi_i_we_o,
    output logic [(I_DATA_WIDTH/8)-1:0] obi_i_be_o,
    output logic [I_DATA_WIDTH-1:0]     obi_i_wdata_o,
    input  logic                      obi_i_rvalid_i,
    input  logic [I_DATA_WIDTH-1:0]     obi_i_rdata_i,
    output logic                      obi_d_req_o,
    input  logic                      obi_d_gnt_i,
    output logic [ADDR_WIDTH-1:0]     obi_d_addr_o,
    output logic                      obi_d_we_o,
    output logic [(D_DATA_WIDTH/8)-1:0] obi_d_be_o,
    output logic [D_DATA_WIDTH-1:0]     obi_d_wdata_o,
    input  logic                      obi_d_rvalid_i,
    input  logic [D_DATA_WIDTH-1:0]     obi_d_rdata_i,
    output logic                      acc_qvalid_o,
    input  logic                      acc_qready_i,
    output logic [31:0]               acc_qdata_op_o,
    output logic [D_DATA_WIDTH-1:0]   acc_qdata_arga_o,
    output logic [D_DATA_WIDTH-1:0]   acc_qdata_argb_o,
    output logic [ADDR_WIDTH-1:0]     acc_qdata_argc_o,
    output logic [4:0]                acc_qid_o,
    input  logic                      acc_qaccept_i,
    input  logic                      acc_qwriteback_i,
    input  logic                      acc_qloadstore_i,
    input  logic                      acc_qexception_i,
    input  logic                      acc_qisfloat_i,
    input  logic [1:0]                acc_mem_finished_i,
    input  logic [1:0]                acc_mem_str_finished_i,
    input  logic                      acc_pvalid_i,
    output logic                      acc_pready_o,
    input  logic [4:0]                acc_pid_i,
    input  logic [D_DATA_WIDTH-1:0]   acc_pdata_i,
    input  logic                      acc_perror_i,
    output logic [2:0]                fpu_rnd_mode_o,
    output logic                      fpu_fmt_mode_o,
    input  logic [4:0]                fpu_status_i,
    output snitch_pkg::core_events_t  core_events_o
);
endmodule

`ifndef SYNTH_REAL_LOCAL_CLUSTER_RTL
(* blackbox *)
module afu #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned CFG_DATA_WIDTH = 32,
    parameter int unsigned MEM_DATA_WIDTH = 256,
    parameter int unsigned LUT_LANES = 4
)(
    input  logic                          clk_i,
    input  logic                          rst_ni,
    input  logic                          obi_s_req_i,
    output logic                          obi_s_gnt_o,
    input  logic [ADDR_WIDTH-1:0]         obi_s_addr_i,
    input  logic                          obi_s_we_i,
    input  logic [(CFG_DATA_WIDTH/8)-1:0] obi_s_be_i,
    input  logic [CFG_DATA_WIDTH-1:0]     obi_s_wdata_i,
    output logic                          obi_s_rvalid_o,
    output logic [CFG_DATA_WIDTH-1:0]     obi_s_rdata_o,
    output logic                          obi_m_req_o,
    input  logic                          obi_m_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_m_addr_o,
    output logic                          obi_m_we_o,
    output logic [(MEM_DATA_WIDTH/8)-1:0] obi_m_be_o,
    output logic [MEM_DATA_WIDTH-1:0]     obi_m_wdata_o,
    input  logic                          obi_m_rvalid_i,
    input  logic [MEM_DATA_WIDTH-1:0]     obi_m_rdata_i,
    output logic                          obi_rhs_req_o,
    input  logic                          obi_rhs_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_rhs_addr_o,
    output logic                          obi_rhs_we_o,
    output logic [(MEM_DATA_WIDTH/8)-1:0] obi_rhs_be_o,
    output logic [MEM_DATA_WIDTH-1:0]     obi_rhs_wdata_o,
    input  logic                          obi_rhs_rvalid_i,
    input  logic [MEM_DATA_WIDTH-1:0]     obi_rhs_rdata_i,
    output logic                          done_o
);
endmodule
`endif

`ifndef SYNTH_REAL_LOCAL_CLUSTER_RTL
(* blackbox *)
module tcdm_interconnect #(
    parameter int unsigned NUM_MASTERS = 4,
    parameter int unsigned NUM_BANKS = 8,
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter logic [NUM_MASTERS-1:0] HWPE_MASTER_MASK = '0,
    parameter logic [NUM_MASTERS-1:0] DMA_MASTER_MASK = '0,
    parameter logic [NUM_MASTERS-1:0] CORE_MASTER_MASK = '0
)(
    input  logic                                       clk_i,
    input  logic                                       rst_ni,
    input  logic [NUM_MASTERS-1:0]                     master_req_i,
    output logic [NUM_MASTERS-1:0]                     master_gnt_o,
    input  logic [NUM_MASTERS-1:0][ADDR_WIDTH-1:0]     master_addr_i,
    input  logic [NUM_MASTERS-1:0]                     master_we_i,
    input  logic [NUM_MASTERS-1:0][(DATA_WIDTH/8)-1:0] master_be_i,
    input  logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]     master_wdata_i,
    output logic [NUM_MASTERS-1:0]                     master_rvalid_o,
    output logic [NUM_MASTERS-1:0][DATA_WIDTH-1:0]     master_rdata_o,
    output logic [NUM_BANKS-1:0]                       bank_req_o,
    output logic [NUM_BANKS-1:0][ADDR_WIDTH-1:0]       bank_addr_o,
    output logic [NUM_BANKS-1:0]                       bank_we_o,
    output logic [NUM_BANKS-1:0][(DATA_WIDTH/8)-1:0]   bank_be_o,
    output logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]       bank_wdata_o,
    input  logic [NUM_BANKS-1:0][DATA_WIDTH-1:0]       bank_rdata_i
);
endmodule
`endif

module spatz #(
    parameter int unsigned NrMemPorts = 1,
    parameter int unsigned NumOutstandingLoads = 1,
    parameter bit RegisterRsp = 1'b0,
    parameter type dreq_t = logic,
    parameter type drsp_t = logic,
    parameter type spatz_mem_req_t = logic,
    parameter type spatz_mem_rsp_t = logic,
    parameter type spatz_issue_req_t = logic,
    parameter type spatz_issue_rsp_t = logic,
    parameter type spatz_rsp_t = logic
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic testmode_i,
    input  logic [31:0] hart_id_i,
    input  logic issue_valid_i,
    output logic issue_ready_o,
    input  spatz_issue_req_t issue_req_i,
    output spatz_issue_rsp_t issue_rsp_o,
    output logic rsp_valid_o,
    input  logic rsp_ready_i,
    output spatz_rsp_t rsp_o,
    output spatz_mem_req_t [NrMemPorts-1:0] spatz_mem_req_o,
    output logic [NrMemPorts-1:0] spatz_mem_req_valid_o,
    input  logic [NrMemPorts-1:0] spatz_mem_req_ready_i,
    input  spatz_mem_rsp_t [NrMemPorts-1:0] spatz_mem_rsp_i,
    input  logic [NrMemPorts-1:0] spatz_mem_rsp_valid_i,
    output logic [NrMemPorts-1:0] spatz_mem_finished_o,
    output logic [NrMemPorts-1:0] spatz_mem_str_finished_o,
    output dreq_t fp_lsu_mem_req_o,
    input  drsp_t fp_lsu_mem_rsp_i,
    input  fpnew_pkg::roundmode_e fpu_rnd_mode_i,
    input  fpnew_pkg::fmt_mode_t fpu_fmt_mode_i,
    output logic [4:0] fpu_status_o
);
endmodule

(* blackbox *)
module npu_pulp_idma_ctrl_mm #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned CFG_DATA_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned IDMA_JOB_FIFO_DEPTH = 16,
    parameter logic [ADDR_WIDTH-1:0] BASE_ADDR = 32'h2000_1000
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic                          req_i,
    output logic                          gnt_o,
    input  logic [ADDR_WIDTH-1:0]         addr_i,
    input  logic                          we_i,
    input  logic [(CFG_DATA_WIDTH/8)-1:0] be_i,
    input  logic [CFG_DATA_WIDTH-1:0]     wdata_i,
    output logic                          rvalid_o,
    output logic [CFG_DATA_WIDTH-1:0]     rdata_o,
    output logic [31:0]                   axi_aw_addr_o,
    output logic [7:0]                    axi_aw_len_o,
    output logic [2:0]                    axi_aw_size_o,
    output logic [1:0]                    axi_aw_burst_o,
    output logic                          axi_aw_valid_o,
    input  logic                          axi_aw_ready_i,
    output logic [DATA_WIDTH-1:0]         axi_w_data_o,
    output logic [(DATA_WIDTH/8)-1:0]     axi_w_strb_o,
    output logic                          axi_w_last_o,
    output logic                          axi_w_valid_o,
    input  logic                          axi_w_ready_i,
    input  logic [1:0]                    axi_b_resp_i,
    input  logic                          axi_b_valid_i,
    output logic                          axi_b_ready_o,
    output logic [31:0]                   axi_ar_addr_o,
    output logic [7:0]                    axi_ar_len_o,
    output logic [2:0]                    axi_ar_size_o,
    output logic [1:0]                    axi_ar_burst_o,
    output logic                          axi_ar_valid_o,
    input  logic                          axi_ar_ready_i,
    input  logic [DATA_WIDTH-1:0]         axi_r_data_i,
    input  logic [1:0]                    axi_r_resp_i,
    input  logic                          axi_r_last_i,
    input  logic                          axi_r_valid_i,
    output logic                          axi_r_ready_o,
    output logic                          obi_read_req_o,
    input  logic                          obi_read_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_read_addr_o,
    output logic                          obi_read_we_o,
    output logic [(DATA_WIDTH/8)-1:0]     obi_read_be_o,
    output logic [DATA_WIDTH-1:0]         obi_read_wdata_o,
    input  logic                          obi_read_rvalid_i,
    input  logic [DATA_WIDTH-1:0]         obi_read_rdata_i,
    output logic                          obi_write_req_o,
    input  logic                          obi_write_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_write_addr_o,
    output logic                          obi_write_we_o,
    output logic [(DATA_WIDTH/8)-1:0]     obi_write_be_o,
    output logic [DATA_WIDTH-1:0]         obi_write_wdata_o,
    input  logic                          obi_write_rvalid_i,
    input  logic [DATA_WIDTH-1:0]         obi_write_rdata_i,
    output logic                          irq_a2o_busy_o,
    output logic                          irq_a2o_start_o,
    output logic                          irq_a2o_done_o,
    output logic                          irq_a2o_error_o,
    output logic                          irq_o2a_busy_o,
    output logic                          irq_o2a_start_o,
    output logic                          irq_o2a_done_o,
    output logic                          irq_o2a_error_o
);
endmodule

`ifndef SYNTH_REAL_LOCAL_CLUSTER_RTL
(* blackbox *)
module systolic_controller #(
    parameter int unsigned ADDR_WIDTH = 32,
    parameter int unsigned DATA_WIDTH = 256,
    parameter int unsigned CFG_DATA_WIDTH = 32,
    parameter int unsigned ARRAY_DIM = 32,
    parameter int unsigned INPUT_ELEM_WIDTH = 8,
    parameter int unsigned OFM_ELEM_WIDTH = 32,
    parameter int unsigned INPUT_FIFO_DEPTH = 4,
    parameter int unsigned OFM_FIFO_DEPTH = 128
)(
    input  logic clk_i,
    input  logic rst_ni,
    input  logic                          ctrl_req_i,
    output logic                          ctrl_gnt_o,
    input  logic [ADDR_WIDTH-1:0]         ctrl_addr_i,
    input  logic                          ctrl_we_i,
    input  logic [(CFG_DATA_WIDTH/8)-1:0] ctrl_be_i,
    input  logic [CFG_DATA_WIDTH-1:0]     ctrl_wdata_i,
    output logic                          ctrl_rvalid_o,
    output logic [CFG_DATA_WIDTH-1:0]     ctrl_rdata_o,
    output logic                          cfg_sys_done_o,
    output logic                          obi_i_req_o,
    input  logic                          obi_i_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_i_addr_o,
    output logic                          obi_i_we_o,
    output logic [(DATA_WIDTH/8)-1:0]     obi_i_be_o,
    output logic [DATA_WIDTH-1:0]         obi_i_wdata_o,
    input  logic                          obi_i_rvalid_i,
    input  logic [DATA_WIDTH-1:0]         obi_i_rdata_i,
    output logic                          obi_w_req_o,
    input  logic                          obi_w_gnt_i,
    output logic [ADDR_WIDTH-1:0]         obi_w_addr_o,
    output logic                          obi_w_we_o,
    output logic [(DATA_WIDTH/8)-1:0]     obi_w_be_o,
    output logic [DATA_WIDTH-1:0]         obi_w_wdata_o,
    input  logic                          obi_w_rvalid_i,
    input  logic [DATA_WIDTH-1:0]         obi_w_rdata_i,
    output logic [3:0]                    obi_o_req_o,
    input  logic [3:0]                    obi_o_gnt_i,
    output logic [3:0][ADDR_WIDTH-1:0]    obi_o_addr_o,
    output logic [3:0]                    obi_o_we_o,
    output logic [3:0][(DATA_WIDTH/8)-1:0] obi_o_be_o,
    output logic [3:0][DATA_WIDTH-1:0]    obi_o_wdata_o,
    input  logic [3:0]                    obi_o_rvalid_i,
    input  logic [3:0][DATA_WIDTH-1:0]    obi_o_rdata_i,
    output logic                          perf_weight_load_en_o,
    output logic                          perf_compute_en_o,
    output logic                          perf_ofm_valid_o,
    output logic                          perf_ofm_ready_o,
    output logic [2:0]                    debug_state_o,
    output logic [1:0]                    debug_drain_state_o,
    output logic [4:0]                    debug_linebuf_state_o
);
endmodule
`endif

`default_nettype wire
