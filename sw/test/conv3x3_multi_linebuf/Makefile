SPATZ_INSTALL ?= ../../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -fno-jump-tables -ffunction-sections -fdata-sections \
         -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../../lib -T link.ld -Wl,--no-relax -Wl,--gc-sections

SRCS = start.S main.c ../../lib/conv2d_packed.c ../../lib/spatz_rt.c \
       ../../lib/hal_systolic.c ../../lib/spatz_ops.c ../../lib/spatz_ops.S

all: conv3x3_multi_linebuf.bin conv3x3_multi_linebuf.hex

conv3x3_multi_linebuf.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

conv3x3_multi_linebuf.bin: conv3x3_multi_linebuf.elf
	$(OBJCOPY) -O binary $< $@

conv3x3_multi_linebuf.hex: conv3x3_multi_linebuf.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
