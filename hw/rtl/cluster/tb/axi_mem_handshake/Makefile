REPO_ROOT := $(abspath ../../../../..)
SIM_DIR ?= $(abspath ../sim/axi_mem_handshake)
JOBS ?= 10
TOP := tb_axi_mem_handshake
SOURCES := $(REPO_ROOT)/hw/axi/src/axi_pkg.sv \
           $(REPO_ROOT)/hw/axi/src/axi_sim_mem.sv $(CURDIR)/$(TOP).sv
FLAGS := --timing --timescale 1ns/1ps --top-module $(TOP) \
         -I$(REPO_ROOT)/hw/axi/include -Wno-fatal -Wno-WIDTH -Wno-TIMESCALEMOD

.PHONY: build lint run
lint:
	verilator --lint-only $(FLAGS) $(SOURCES)

build:
	mkdir -p $(SIM_DIR)
	verilator --binary -j $(JOBS) $(FLAGS) --Mdir $(SIM_DIR) $(SOURCES)

run: build
	@$(SIM_DIR)/V$(TOP) > $(SIM_DIR)/sim.log 2>&1; status=$$?; \
	cat $(SIM_DIR)/sim.log; exit $$status
