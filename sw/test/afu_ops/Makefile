SPATZ_INSTALL ?= ../../../hw/spatz/install
LLVM_BIN = $(SPATZ_INSTALL)/llvm/bin
GCC_BIN = $(SPATZ_INSTALL)/riscv-gcc/bin

CC = $(LLVM_BIN)/clang
OBJCOPY = $(LLVM_BIN)/llvm-objcopy
HEXOBJCOPY = $(GCC_BIN)/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../../lib -T ../spatz_ops/link.ld -Wl,--no-relax

SRCS = ../spatz_ops/start.S ../spatz_ops/main.c ../../lib/spatz_ops.c ../../lib/spatz_ops.S

AFU_TESTS = add_full mul_q7_full logistic logistic_full clamp_relu6 dfl_fused class_sigmoid global_avgpool
AFU_BINS = $(addprefix afu_ops_,$(addsuffix .bin,$(AFU_TESTS)))
AFU_HEXS = $(addprefix afu_ops_,$(addsuffix .hex,$(AFU_TESTS)))

all: $(AFU_BINS) $(AFU_HEXS)

afu_ops_add_full.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=12 $(SRCS) -o $@

afu_ops_mul_q7_full.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=11 $(SRCS) -o $@

afu_ops_logistic.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=6 $(SRCS) -o $@

afu_ops_logistic_full.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=10 $(SRCS) -o $@

afu_ops_clamp_relu6.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=17 $(SRCS) -o $@

afu_ops_dfl_fused.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=14 $(SRCS) -o $@

afu_ops_class_sigmoid.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=15 $(SRCS) -o $@

afu_ops_global_avgpool.elf: $(SRCS) ../spatz_ops/link.ld
	$(CC) $(CFLAGS) -DSPATZ_OP_TEST_ID=16 $(SRCS) -o $@

afu_ops_%.bin: afu_ops_%.elf
	$(OBJCOPY) -O binary $< $@

afu_ops_%.hex: afu_ops_%.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
