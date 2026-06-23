VERILATOR ?= verilator
VERILATOR_FLAGS ?= --binary --timing -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT
CCACHE_DIR ?= /tmp/ccache
CCACHE_TEMPDIR ?= /tmp/ccache-tmp
REPO_ROOT ?= ../../..
COMMON_CELLS := $(REPO_ROOT)/hw/common_cells

.PHONY: test clean

test:
	CCACHE_DIR=$(CCACHE_DIR) CCACHE_TEMPDIR=$(CCACHE_TEMPDIR) \
	$(VERILATOR) $(VERILATOR_FLAGS) \
		+incdir+$(COMMON_CELLS)/include \
		--top-module tb_tcdm_interconnect_grouped \
		$(COMMON_CELLS)/src/cf_math_pkg.sv \
		$(COMMON_CELLS)/src/lzc.sv \
		$(COMMON_CELLS)/src/rr_arb_tree.sv \
		tcdm_interconnect.sv tb_tcdm_interconnect_grouped.sv
	./obj_dir/Vtb_tcdm_interconnect_grouped

clean:
	rm -rf obj_dir
