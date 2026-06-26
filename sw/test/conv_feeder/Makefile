SPATZ_INSTALL ?= ../../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -fno-builtin -fno-jump-tables -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../../lib -T link.ld -Wl,--no-relax

SRCS = start.S main.c ../../lib/conv2d_feeder_sw.c ../../lib/hal_systolic.c ../../lib/spatz_rt.c
HDRS = ../../lib/conv2d_feeder_sw.h ../../lib/hal_systolic.h ../../lib/idma_mm_utils.h ../../lib/npu_memory_map.h ../../lib/npu_types.h ../../lib/spatz_rt.h

all: conv_feeder.bin conv_feeder.hex

conv_feeder.elf: $(SRCS) $(HDRS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

conv_feeder.bin: conv_feeder.elf
	$(OBJCOPY) -O binary $< $@

conv_feeder.hex: conv_feeder.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
