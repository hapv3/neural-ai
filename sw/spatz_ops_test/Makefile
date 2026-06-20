SPATZ_INSTALL ?= ../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../lib -T link.ld -Wl,--no-relax

SRCS = start.S main.c ../lib/spatz_ops.c ../lib/spatz_ops.S

all: spatz_ops_test.bin spatz_ops_test.hex

spatz_ops_test.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

spatz_ops_test.bin: spatz_ops_test.elf
	$(OBJCOPY) -O binary $< $@

spatz_ops_test.hex: spatz_ops_test.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
