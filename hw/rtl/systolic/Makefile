# Makefile for Cocotb Simulation of Systolic Array

SIM ?= verilator
TOPLEVEL_LANG ?= verilog
VERILATOR_JOBS ?= 8

PWD=$(shell pwd)
REPO_ROOT=$(PWD)/../../..

# MODULE is the basename of the Python test file
export PYTHONPATH := $(PWD)/tb:$(PYTHONPATH)
MODULE ?= test_systolic

# RTL sources
VERILOG_SOURCES = $(PWD)/npu_systolic_array.sv

# Set build directory to tb/sim/<test_name>
SIM_BUILD = $(PWD)/tb/sim/$(MODULE)

# TOPLEVEL is the name of the toplevel module in your Verilog
TOPLEVEL = npu_systolic_array

# Extra args for verilator
EXTRA_ARGS += -Wno-fatal --timing -j $(VERILATOR_JOBS)
BUILD_ARGS += -j$(VERILATOR_JOBS)

ifneq ($(filter test_systolic_controller%,$(MODULE)),)
TOPLEVEL = tb_systolic_controller
VERILOG_SOURCES = \
	$(REPO_ROOT)/hw/common_cells/src/fifo_v3.sv \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_clk.sv \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram.sv \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram_impl.sv \
	$(PWD)/requant_pipeline.sv \
	$(PWD)/npu_systolic_array.sv \
	$(PWD)/conv_channel_linebuf_packer.sv \
	$(PWD)/conv_linebuf_stream_packer.sv \
	$(PWD)/systolic_ctrl_regs.sv \
	$(PWD)/systolic_controller.sv \
	$(PWD)/tb/tb_systolic_controller.sv
EXTRA_ARGS += +incdir+$(REPO_ROOT)/hw/common_cells/include
endif

ifeq ($(MODULE),test_conv_channel_linebuf_packer)
TOPLEVEL = conv_channel_linebuf_packer
VERILOG_SOURCES = \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_clk.sv \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram.sv \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram_impl.sv \
	$(PWD)/conv_channel_linebuf_packer.sv
endif

ifeq ($(MODULE),test_conv_linebuf_stream_packer)
TOPLEVEL = conv_linebuf_stream_packer
VERILOG_SOURCES = \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_clk.sv \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram.sv \
	$(REPO_ROOT)/hw/tech_cells/src/rtl/tc_sram_impl.sv \
	$(PWD)/conv_linebuf_stream_packer.sv
endif

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
