CC = riscv64-unknown-elf-gcc
OBJCOPY = riscv64-unknown-elf-objcopy

CFLAGS = -march=rv32ima -mabi=ilp32 -O2 -nostartfiles -nodefaultlibs -nostdlib -I../../lib -T link.ld

SRCS = main.c start.S
HDRS = ../../lib/idma_mm_utils.h ../../lib/npu_memory_map.h ../../lib/npu_types.h

all: matmul.bin matmul.hex

matmul.elf: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(SRCS) -o $@

matmul.bin: matmul.elf
	$(OBJCOPY) -O binary $< $@

matmul.hex: matmul.elf
	$(OBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
