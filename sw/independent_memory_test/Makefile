SPATZ_INSTALL ?= ../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -fno-builtin -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../lib -T link.ld -Wl,--no-relax

SRCS = start.S main.c ../lib/spatz_rt.c
HDRS = ../lib/idma_mm_utils.h ../lib/npu_memory_map.h ../lib/npu_types.h ../lib/spatz_rt.h

all: independent_memory.bin independent_memory.hex

independent_memory.elf: $(SRCS) $(HDRS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

independent_memory.bin: independent_memory.elf
	$(OBJCOPY) -O binary $< $@

independent_memory.hex: independent_memory.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
