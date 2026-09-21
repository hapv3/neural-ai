VERILATOR ?= verilator
VERILATOR_FLAGS ?= --binary --timing -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
CCACHE_DIR ?= /tmp/ccache
CCACHE_TEMPDIR ?= /tmp/ccache-tmp
REPO_ROOT ?= ../../..
COMMON_CELLS := $(REPO_ROOT)/hw/common_cells
TEST_NAME ?= tb_tcdm_interconnect_grouped
SIM_ROOT ?= tb/sim
SIM_DIR ?= $(SIM_ROOT)/$(TEST_NAME)
SIM_BIN ?= $(SIM_DIR)/V$(TEST_NAME)
SIM_LOG ?= $(SIM_DIR)/sim.log

.PHONY: test clean

test:
	mkdir -p $(CCACHE_DIR) $(CCACHE_TEMPDIR) $(SIM_DIR)
	CCACHE_DIR=$(CCACHE_DIR) CCACHE_TEMPDIR=$(CCACHE_TEMPDIR) \
	$(VERILATOR) $(VERILATOR_FLAGS) \
		+incdir+$(COMMON_CELLS)/include \
		--top-module $(TEST_NAME) --Mdir $(SIM_DIR) \
		$(COMMON_CELLS)/src/cf_math_pkg.sv \
		$(COMMON_CELLS)/src/lzc.sv \
		$(COMMON_CELLS)/src/rr_arb_tree.sv \
		tcdm_interconnect.sv tb_tcdm_interconnect_grouped.sv
	./$(SIM_BIN) 2>&1 | tee $(SIM_LOG)

clean:
	rm -rf tb/sim
