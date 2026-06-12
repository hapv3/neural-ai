CC = riscv64-unknown-elf-gcc
OBJCOPY = riscv64-unknown-elf-objcopy

CFLAGS = -march=rv32ima -mabi=ilp32 -O2 -nostartfiles -nodefaultlibs -nostdlib -I../lib -T link.ld

SRCS = main.c ../lib/hal_dma.c start.S

all: boot.bin boot.hex

boot.elf: $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@

boot.bin: boot.elf
	$(OBJCOPY) -O binary $< $@

boot.hex: boot.elf
	$(OBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex
