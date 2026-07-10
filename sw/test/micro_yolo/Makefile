CC = ../../../hw/spatz/install/llvm/bin/clang
OBJCOPY = ../../../hw/spatz/install/llvm/bin/llvm-objcopy
HEXOBJCOPY = ../../../hw/spatz/install/riscv-gcc/bin/riscv32-unknown-elf-objcopy

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -fno-jump-tables -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../../lib -T link.ld -Wl,--no-relax

SRCS = start.S main.c ../../lib/npu_graph.c ../../lib/hal_systolic.c \
       ../../lib/spatz_ops.c ../../lib/spatz_ops.S

all: micro_yolo.bin micro_yolo.hex

micro_yolo.elf: $(SRCS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

micro_yolo.bin: micro_yolo.elf
	$(OBJCOPY) -O binary $< $@

micro_yolo.hex: micro_yolo.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
