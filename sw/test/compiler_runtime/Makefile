CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror

.PHONY: all check clean

all: test_model_abi

test_model_abi: test_model_abi.c ../../lib/npu_model_abi.h ../../lib/npu_cmd_desc_v2.h ../../lib/npu_model_loader.h ../../lib/npu_model_loader.c ../../lib/npu_memory_map.h
	$(CC) $(CFLAGS) -I../../lib test_model_abi.c ../../lib/npu_model_loader.c -o $@

check: test_model_abi
	./test_model_abi

clean:
	rm -f test_model_abi
