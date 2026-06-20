SPATZ_INSTALL ?= ../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
OBJDUMP = $(LLVM_BIN)/llvm-objdump
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -T link.ld -Wl,--no-relax

TESTS = basic_mem_arith

.SECONDARY: $(addsuffix .elf,$(TESTS))

all: $(addsuffix .bin,$(TESTS)) $(addsuffix .hex,$(TESTS))

%.elf: tests/%.S link.ld
	$(CC) $(CFLAGS) $< -o $@

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

%.hex: %.elf
	$(HEXOBJCOPY) -O verilog $< $@

dump: $(addsuffix .elf,$(TESTS))
	$(OBJDUMP) -d --mcpu=snitch --mattr=+v $^

clean:
	rm -f *.elf *.bin *.hex
