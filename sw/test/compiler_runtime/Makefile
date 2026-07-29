CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
NEURAL_COMPILER_ROOT ?= $(abspath ../../../..)/neural-compiler
SIM_BUILD ?= /tmp/neural-ai-compiled-model
COCOTB_TESTCASE ?=

.PHONY: all check check_cross_repo_abi firmware sim clean

all: test_model_abi test_cmd_v2 test_copy_layout test_quant_buffer

test_model_abi: test_model_abi.c ../../lib/npu_model_abi.h ../../lib/npu_cmd_desc_v2.h ../../lib/npu_model_loader.h ../../lib/npu_model_loader.c ../../lib/npu_memory_map.h
	$(CC) $(CFLAGS) -I../../lib test_model_abi.c ../../lib/npu_model_loader.c -o $@

check: test_model_abi test_cmd_v2 test_copy_layout test_quant_buffer check_cross_repo_abi
	./test_model_abi
	./test_cmd_v2
	./test_copy_layout
	./test_quant_buffer

check_cross_repo_abi:
	NEURAL_COMPILER_ROOT="$(NEURAL_COMPILER_ROOT)" python3 ./check_cross_repo_abi.py

firmware:
	$(MAKE) -C ../../runtime/neural_ai all

sim: firmware
	PYTHONPATH="$(CURDIR)" NEURAL_COMPILER_ROOT="$(NEURAL_COMPILER_ROOT)" \
	$(MAKE) -C ../../../hw/rtl/cluster sim \
		COCOTB_TEST_MODULES=test_compiled_model \
		COCOTB_TESTCASE="$(COCOTB_TESTCASE)" \
		CLUSTER_SIM_NAME=compiled_model \
		SIM_BUILD="$(SIM_BUILD)"

test_cmd_v2: test_cmd_v2.c ../../lib/npu_cmd_desc_v2.h ../../lib/npu_cmd_desc_v2.c ../../lib/npu_model_loader.c
	$(CC) $(CFLAGS) -I../../lib test_cmd_v2.c ../../lib/npu_cmd_desc_v2.c ../../lib/npu_model_loader.c -o $@

test_copy_layout: test_copy_layout.c ../../lib/npu_layout_ops.h ../../lib/npu_layout_ops.c
	$(CC) $(CFLAGS) -I../../lib test_copy_layout.c ../../lib/npu_layout_ops.c -o $@

test_quant_buffer: test_quant_buffer.c ../../lib/npu_quant_buffer.h ../../lib/npu_quant_buffer.c
	$(CC) $(CFLAGS) -DNAI_QUANT_BUFFER_HOST_TEST -I../../lib test_quant_buffer.c ../../lib/npu_quant_buffer.c -o $@

clean:
	rm -f test_model_abi test_cmd_v2 test_copy_layout test_quant_buffer
