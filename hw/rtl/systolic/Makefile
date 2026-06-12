# Makefile for Cocotb Simulation of Systolic Array

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

PWD=$(shell pwd)

# RTL sources
VERILOG_SOURCES = $(PWD)/npu_systolic_array.sv

# MODULE is the basename of the Python test file
export PYTHONPATH := $(PWD)/tb:$(PYTHONPATH)
MODULE ?= test_systolic

# Set build directory to tb/sim/<test_name>
SIM_BUILD = $(PWD)/tb/sim/$(MODULE)

# TOPLEVEL is the name of the toplevel module in your Verilog
TOPLEVEL = npu_systolic_array

# Extra args for verilator
EXTRA_ARGS += -Wno-fatal

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
