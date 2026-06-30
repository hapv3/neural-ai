CC = riscv64-unknown-elf-gcc
OBJCOPY = riscv64-unknown-elf-objcopy

CFLAGS = -march=rv32ima -mabi=ilp32 -O2 -nostartfiles -nodefaultlibs -nostdlib -I../../lib -T ../boot/link.ld

SRCS = main.c ../boot/start.S
HDRS = ../../lib/npu_memory_map.h ../../lib/npu_types.h

all: pmu.bin pmu.hex

pmu.elf: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(SRCS) -o $@

pmu.bin: pmu.elf
	$(OBJCOPY) -O binary $< $@

pmu.hex: pmu.elf
	$(OBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
