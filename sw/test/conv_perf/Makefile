SPATZ_INSTALL ?= ../../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CONV_PERF_GROUP ?= 0

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -fno-builtin -fno-jump-tables -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../../lib -T link.ld -Wl,--no-relax -DCONV_PERF_GROUP=$(CONV_PERF_GROUP)

SRCS = start.S main.c ../../lib/conv2d_feeder_sw.c ../../lib/conv2d_packed.c ../../lib/hal_systolic.c ../../lib/spatz_ops.S ../../lib/spatz_rt.c
HDRS = ../../lib/conv2d_feeder_sw.h ../../lib/conv2d_packed.h ../../lib/hal_systolic.h ../../lib/idma_mm_utils.h ../../lib/npu_memory_map.h ../../lib/npu_types.h ../../lib/spatz_ops.h ../../lib/spatz_rt.h

all: conv_perf.bin conv_perf.hex

conv_perf.elf: $(SRCS) $(HDRS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

conv_perf.bin: conv_perf.elf
	$(OBJCOPY) -O binary $< $@

conv_perf.hex: conv_perf.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
