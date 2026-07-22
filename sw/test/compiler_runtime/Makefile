CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror

.PHONY: all check clean

all: test_model_abi test_cmd_v2 test_copy_layout test_quant_buffer

test_model_abi: test_model_abi.c ../../lib/npu_model_abi.h ../../lib/npu_cmd_desc_v2.h ../../lib/npu_model_loader.h ../../lib/npu_model_loader.c ../../lib/npu_memory_map.h
	$(CC) $(CFLAGS) -I../../lib test_model_abi.c ../../lib/npu_model_loader.c -o $@

check: test_model_abi test_cmd_v2 test_copy_layout test_quant_buffer
	./test_model_abi
	./test_cmd_v2
	./test_copy_layout
	./test_quant_buffer

test_cmd_v2: test_cmd_v2.c ../../lib/npu_cmd_desc_v2.h ../../lib/npu_cmd_desc_v2.c ../../lib/npu_model_loader.c
	$(CC) $(CFLAGS) -I../../lib test_cmd_v2.c ../../lib/npu_cmd_desc_v2.c ../../lib/npu_model_loader.c -o $@

test_copy_layout: test_copy_layout.c ../../lib/npu_layout_ops.h ../../lib/npu_layout_ops.c
	$(CC) $(CFLAGS) -I../../lib test_copy_layout.c ../../lib/npu_layout_ops.c -o $@

test_quant_buffer: test_quant_buffer.c ../../lib/npu_quant_buffer.h ../../lib/npu_quant_buffer.c
	$(CC) $(CFLAGS) -DNAI_QUANT_BUFFER_HOST_TEST -I../../lib test_quant_buffer.c ../../lib/npu_quant_buffer.c -o $@

clean:
	rm -f test_model_abi test_cmd_v2 test_copy_layout test_quant_buffer
