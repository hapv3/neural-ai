CC = riscv64-unknown-elf-gcc
OBJCOPY = riscv64-unknown-elf-objcopy

CFLAGS = -march=rv32ima -mabi=ilp32 -O2 -nostartfiles -nodefaultlibs -nostdlib -I../../lib -T link.ld

SRCS = main.c start.S
HDRS = ../../lib/hal_afu.h ../../lib/npu_memory_map.h ../../lib/npu_types.h

all: afu.bin afu.hex

afu.elf: $(SRCS) $(HDRS) link.ld
	$(CC) $(CFLAGS) $(SRCS) -o $@

afu.bin: afu.elf
	$(OBJCOPY) -O binary $< $@

afu.hex: afu.elf
	$(OBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
