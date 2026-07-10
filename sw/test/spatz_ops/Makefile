SPATZ_INSTALL ?= ../../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../../lib -T link.ld -Wl,--no-relax

SRCS = start.S main.c ../../lib/spatz_ops.c ../../lib/spatz_ops.S

OP_TESTS = copy relu requant add mul logistic maxpool upsample concat
OP_BINS = $(addprefix spatz_ops_,$(addsuffix .bin,$(OP_TESTS)))
OP_HEXS = $(addprefix spatz_ops_,$(addsuffix .hex,$(OP_TESTS)))

all: spatz_ops_test.bin spatz_ops_test.hex $(OP_BINS) $(OP_HEXS)

spatz_ops_test.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

spatz_ops_test.bin: spatz_ops_test.elf
	$(OBJCOPY) -O binary $< $@

spatz_ops_test.hex: spatz_ops_test.elf
	$(HEXOBJCOPY) -O verilog $< $@

spatz_ops_copy.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=1 $(SRCS) -o $@

spatz_ops_relu.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=2 $(SRCS) -o $@

spatz_ops_requant.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=3 $(SRCS) -o $@

spatz_ops_add.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=4 $(SRCS) -o $@

spatz_ops_mul.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=5 $(SRCS) -o $@

spatz_ops_logistic.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=6 $(SRCS) -o $@

spatz_ops_maxpool.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=7 $(SRCS) -o $@

spatz_ops_upsample.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=8 $(SRCS) -o $@

spatz_ops_concat.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=9 $(SRCS) -o $@

spatz_ops_%.bin: spatz_ops_%.elf
	$(OBJCOPY) -O binary $< $@

spatz_ops_%.hex: spatz_ops_%.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
