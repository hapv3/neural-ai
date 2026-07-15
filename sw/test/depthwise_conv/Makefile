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
CFLAGS_C64 = $(CFLAGS) -DDEPTHWISE_H=24u -DDEPTHWISE_W=24u -DDEPTHWISE_C=64u
CFLAGS_C33_48 = $(CFLAGS) -DDEPTHWISE_H=48u -DDEPTHWISE_W=48u -DDEPTHWISE_C=33u
CFLAGS_C64_48 = $(CFLAGS) -DDEPTHWISE_H=48u -DDEPTHWISE_W=48u -DDEPTHWISE_C=64u
CFLAGS_C65_48_S2 = $(CFLAGS) -DDEPTHWISE_H=48u -DDEPTHWISE_W=48u -DDEPTHWISE_C=65u -DDEPTHWISE_STRIDE=2u
CFLAGS_C96_48 = $(CFLAGS) -DDEPTHWISE_H=48u -DDEPTHWISE_W=48u -DDEPTHWISE_C=96u
CFLAGS_S2 = $(CFLAGS) -DDEPTHWISE_H=48u -DDEPTHWISE_W=48u -DDEPTHWISE_C=32u -DDEPTHWISE_STRIDE=2u

SRCS = start.S main.c ../../lib/npu_graph.c ../../lib/conv2d_packed.c ../../lib/spatz_rt.c \
       ../../lib/hal_systolic.c ../../lib/spatz_ops.c ../../lib/spatz_ops.S

all: depthwise_conv.bin depthwise_conv.hex \
     depthwise_conv_c64.bin depthwise_conv_c64.hex \
     depthwise_conv_c33_48.bin depthwise_conv_c33_48.hex \
     depthwise_conv_c64_48.bin depthwise_conv_c64_48.hex \
     depthwise_conv_c65_48_s2.bin depthwise_conv_c65_48_s2.hex \
     depthwise_conv_c96_48.bin depthwise_conv_c96_48.hex \
     depthwise_conv_s2.bin depthwise_conv_s2.hex

depthwise_conv.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

depthwise_conv_c64.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS_C64) $(SRCS) -o $@

depthwise_conv_c33_48.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS_C33_48) $(SRCS) -o $@

depthwise_conv_c64_48.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS_C64_48) $(SRCS) -o $@

depthwise_conv_c65_48_s2.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS_C65_48_S2) $(SRCS) -o $@

depthwise_conv_c96_48.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS_C96_48) $(SRCS) -o $@

depthwise_conv_s2.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS_S2) $(SRCS) -o $@

depthwise_conv.bin: depthwise_conv.elf
	$(OBJCOPY) -O binary $< $@

depthwise_conv_c64.bin: depthwise_conv_c64.elf
	$(OBJCOPY) -O binary $< $@

depthwise_conv_c33_48.bin: depthwise_conv_c33_48.elf
	$(OBJCOPY) -O binary $< $@

depthwise_conv_c64_48.bin: depthwise_conv_c64_48.elf
	$(OBJCOPY) -O binary $< $@

depthwise_conv_c65_48_s2.bin: depthwise_conv_c65_48_s2.elf
	$(OBJCOPY) -O binary $< $@

depthwise_conv_c96_48.bin: depthwise_conv_c96_48.elf
	$(OBJCOPY) -O binary $< $@

depthwise_conv_s2.bin: depthwise_conv_s2.elf
	$(OBJCOPY) -O binary $< $@

depthwise_conv.hex: depthwise_conv.elf
	$(HEXOBJCOPY) -O verilog $< $@

depthwise_conv_c64.hex: depthwise_conv_c64.elf
	$(HEXOBJCOPY) -O verilog $< $@

depthwise_conv_c33_48.hex: depthwise_conv_c33_48.elf
	$(HEXOBJCOPY) -O verilog $< $@

depthwise_conv_c64_48.hex: depthwise_conv_c64_48.elf
	$(HEXOBJCOPY) -O verilog $< $@

depthwise_conv_c65_48_s2.hex: depthwise_conv_c65_48_s2.elf
	$(HEXOBJCOPY) -O verilog $< $@

depthwise_conv_c96_48.hex: depthwise_conv_c96_48.elf
	$(HEXOBJCOPY) -O verilog $< $@

depthwise_conv_s2.hex: depthwise_conv_s2.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
