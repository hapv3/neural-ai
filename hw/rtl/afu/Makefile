VERILATOR ?= verilator
export CCACHE_DIR ?= /tmp/ccache
export CCACHE_TEMPDIR ?= /tmp/ccache-tmp
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

TOP      ?= tb_afu
SIM_ROOT ?= tb/sim
SIM_DIR  ?= $(SIM_ROOT)/$(TOP)
SIM_BIN  ?= $(SIM_DIR)/V$(TOP)
SIM_LOG  ?= $(SIM_DIR)/sim.log

SRC = afu_binary_requant_pipeline.sv afu_binary_requant_engine.sv \
      afu_frontend.sv afu_backend.sv afu_core.sv afu.sv \
      ../../tech_cells/src/rtl/tc_sram_impl.sv \
      ../../tech_cells/src/rtl/tc_sram.sv \
      tb/tb_afu.sv

VERILATOR_FLAGS += -sv --timing --binary -j 0
VERILATOR_FLAGS += --top-module $(TOP)
VERILATOR_FLAGS += --Mdir $(SIM_DIR)
VERILATOR_FLAGS += -Wall -Wno-fatal -Wno-BLKSEQ -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL

.PHONY: all sim clean

all: sim

sim: $(SRC)
	mkdir -p $(CCACHE_DIR) $(CCACHE_TEMPDIR) $(SIM_DIR)
	{ $(VERILATOR) $(VERILATOR_FLAGS) $(SRC) && ./$(SIM_BIN); } 2>&1 | tee $(SIM_LOG)

clean:
	rm -rf $(SIM_ROOT)
