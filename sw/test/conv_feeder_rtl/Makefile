TOOLCHAIN ?= ../../../hw/spatz/install/llvm/bin
GCC_TOOLCHAIN ?= ../../../hw/spatz/install/riscv-gcc/bin
CC := $(TOOLCHAIN)/clang
OBJCOPY := $(TOOLCHAIN)/llvm-objcopy
GNU_OBJCOPY := $(GCC_TOOLCHAIN)/riscv32-unknown-elf-objcopy

CFLAGS := --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 -O2 \
          -fno-builtin -fno-jump-tables -mno-relax -fuse-ld=lld \
          -nodefaultlibs -nostdlib -I../../lib -T link.ld -Wl,--no-relax

SOURCES := start.S main.c ../../lib/hal_conv2d_feeder.c ../../lib/hal_systolic.c ../../lib/spatz_rt.c

all: conv_feeder_rtl.bin conv_feeder_rtl.hex

conv_feeder_rtl.elf: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $@

conv_feeder_rtl.bin: conv_feeder_rtl.elf
	$(OBJCOPY) -O binary $< $@

conv_feeder_rtl.hex: conv_feeder_rtl.elf
	$(GNU_OBJCOPY) -O verilog $< $@

clean:
	rm -f conv_feeder_rtl.elf conv_feeder_rtl.bin conv_feeder_rtl.hex
