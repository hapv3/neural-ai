SPATZ_INSTALL ?= ../../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy
SIZE = $(LLVM_BIN)/llvm-size

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -Os -fno-builtin -fno-jump-tables -ffunction-sections -fdata-sections \
         -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib -I../../lib -T link.ld \
         -Wl,--no-relax -Wl,--gc-sections

SRCS = start.S main.c \
       ../../lib/npu_cmd_desc.c ../../lib/npu_cmd_desc_v2.c \
       ../../lib/npu_model_loader.c ../../lib/npu_model_runtime.c \
       ../../lib/npu_quant_buffer.c \
       ../../lib/npu_layout_ops.c ../../lib/npu_runtime_ops.c \
       ../../lib/spatz_ops.c ../../lib/spatz_ops.S \
       ../../lib/npu_runtime_memory.c \
       ../../lib/hal_systolic.c

HDRS = ../../lib/npu_cmd_desc.h ../../lib/npu_cmd_desc_v2.h \
       ../../lib/npu_model_abi.h ../../lib/npu_model_loader.h \
       ../../lib/npu_model_runtime.h ../../lib/npu_layout_ops.h \
       ../../lib/npu_quant_buffer.h \
       ../../lib/npu_runtime_ops.h ../../lib/npu_memory_map.h \
       ../../lib/npu_types.h ../../lib/idma_mm_utils.h ../../lib/hal_systolic.h

.PHONY: all size clean

all: neural_ai.bin neural_ai.hex size

neural_ai.elf: $(SRCS) $(HDRS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

neural_ai.bin: neural_ai.elf
	$(OBJCOPY) -O binary $< $@

neural_ai.hex: neural_ai.elf
	$(HEXOBJCOPY) -O verilog $< $@

size: neural_ai.elf
	$(SIZE) -A $<

clean:
	rm -f neural_ai.elf neural_ai.bin neural_ai.hex
