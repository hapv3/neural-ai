VERILATOR ?= verilator
VERILATOR_FLAGS ?= --binary --timing -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT
CCACHE_DIR ?= /tmp/ccache
CCACHE_TEMPDIR ?= /tmp/ccache-tmp

.PHONY: test clean

test:
	CCACHE_DIR=$(CCACHE_DIR) CCACHE_TEMPDIR=$(CCACHE_TEMPDIR) \
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module tb_tcdm_interconnect_grouped \
		tcdm_interconnect.sv tb_tcdm_interconnect_grouped.sv
	./obj_dir/Vtb_tcdm_interconnect_grouped

clean:
	rm -rf obj_dir
