# Makefile for Cocotb Simulation of NPU Cluster

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

PWD=$(shell pwd)
REPO_ROOT=$(PWD)/../../..

# RTL sources
VERILOG_SOURCES = \
    $(PWD)/npu_cluster_pkg.sv \
    $(PWD)/cluster_ctrl_regs.sv \
    $(PWD)/cluster_sram_bank.sv \
    $(REPO_ROOT)/hw/rtl/interconnect/tcdm_interconnect.sv \
    $(REPO_ROOT)/hw/rtl/dma/dma_engine.sv \
    $(REPO_ROOT)/hw/rtl/systolic/npu_systolic_array.sv \
    $(REPO_ROOT)/hw/rtl/vector/spatz_wrapper.sv \
    $(PWD)/npu_cluster.sv \
    $(REPO_ROOT)/hw/common_cells/src/cdc_2phase.sv \
    $(REPO_ROOT)/hw/axi/src/axi_pkg.sv \
    $(REPO_ROOT)/hw/axi/src/axi_sim_mem.sv \
    $(PWD)/tb/tb_npu_cluster.sv

# MODULE is the basename of the Python test file
export PYTHONPATH := $(PWD)/tb:$(PWD)/tb/tests:$(PYTHONPATH)
# Python test module
MODULE = test_dma_tcm

# Set build directory to tb/sim/<test_name>
SIM_BUILD = $(PWD)/tb/sim/$(MODULE)

# TOPLEVEL is the name of the toplevel module in your Verilog
TOPLEVEL = tb_npu_cluster

# Extra args for verilator
EXTRA_ARGS += -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT +incdir+$(REPO_ROOT)/hw/axi/include --timing -j 0

# Enable VCD dumping if DEBUG=1
DEBUG ?= 0
ifeq ($(DEBUG), 1)
    EXTRA_ARGS += --trace
    export WAVES = 1
endif

# Include Cocotb's make rules
include $(shell cocotb-config --makefiles)/Makefile.sim

# Clean output dir
clean::
	@rm -rf tb/sim __pycache__ tb/__pycache__ results.xml dump.vcd
