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

SRCS = start.S main.c ../../lib/npu_graph.c ../../lib/conv2d_packed.c ../../lib/spatz_rt.c \
       ../../lib/hal_systolic.c ../../lib/spatz_ops.c ../../lib/spatz_ops.S

all: pointwise_conv.bin pointwise_conv.hex pointwise_conv_c64_c128.bin pointwise_conv_c64_c128.hex

pointwise_conv.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

pointwise_conv_c64_c128.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DH=16u -DW=16u -DIC=64u -DOC=128u $(SRCS) -o $@

pointwise_conv.bin: pointwise_conv.elf
	$(OBJCOPY) -O binary $< $@

pointwise_conv_c64_c128.bin: pointwise_conv_c64_c128.elf
	$(OBJCOPY) -O binary $< $@

pointwise_conv.hex: pointwise_conv.elf
	$(HEXOBJCOPY) -O verilog $< $@

pointwise_conv_c64_c128.hex: pointwise_conv_c64_c128.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
