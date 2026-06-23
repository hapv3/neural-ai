VERILATOR ?= verilator
export CCACHE_DIR ?= /tmp/ccache
export CCACHE_TEMPDIR ?= /tmp/ccache-tmp

TOP      ?= tb_afu
SIM_DIR  ?= obj_dir
SIM_BIN  ?= $(SIM_DIR)/V$(TOP)

SRC = afu_frontend.sv afu_backend.sv afu_core.sv afu.sv \
      ../../tech_cells/src/rtl/tc_sram_impl.sv \
      ../../tech_cells/src/rtl/tc_sram.sv \
      tb/tb_afu.sv

VERILATOR_FLAGS += -sv --timing --binary -j 0
VERILATOR_FLAGS += --top-module $(TOP)
VERILATOR_FLAGS += -Wall -Wno-fatal -Wno-BLKSEQ -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL

.PHONY: all sim clean

all: sim

sim: $(SRC)
	mkdir -p $(CCACHE_DIR) $(CCACHE_TEMPDIR)
	$(VERILATOR) $(VERILATOR_FLAGS) $(SRC)
	./$(SIM_BIN)

clean:
	rm -rf $(SIM_DIR)
