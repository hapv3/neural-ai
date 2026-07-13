CC = ../../../hw/spatz/install/llvm/bin/clang
OBJCOPY = ../../../hw/spatz/install/llvm/bin/llvm-objcopy
HEXOBJCOPY = ../../../hw/spatz/install/riscv-gcc/bin/riscv32-unknown-elf-objcopy

MICRO_YOLO_C2F_TILE_OH ?= 16
MICRO_YOLO_C2F_TILE_OW ?= 16
MICRO_YOLO_DOWN_TILE_OH ?= 16
MICRO_YOLO_DOWN_TILE_OW ?= 16
MICRO_YOLO_HEAD_TILE_OH ?= 16
MICRO_YOLO_HEAD_TILE_OW ?= 16
LINEBUF_PRECOMPUTE_HDR = micro_yolo_linebuf_precompute.h
LINEBUF_PRECOMPUTE_GEN = ../../../tools/npu_linebuf_precompute.py

CFLAGS = --target=riscv32-unknown-elf -march=rv32im_zve32x -mabi=ilp32 \
         -O2 -fno-jump-tables -ffunction-sections -fdata-sections \
         -mno-relax -fuse-ld=lld -nodefaultlibs -nostdlib \
         -I../../lib -T link.ld -Wl,--no-relax -Wl,--gc-sections \
         -DMICRO_YOLO_C2F_TILE_OH=$(MICRO_YOLO_C2F_TILE_OH) \
         -DMICRO_YOLO_C2F_TILE_OW=$(MICRO_YOLO_C2F_TILE_OW) \
         -DMICRO_YOLO_DOWN_TILE_OH=$(MICRO_YOLO_DOWN_TILE_OH) \
         -DMICRO_YOLO_DOWN_TILE_OW=$(MICRO_YOLO_DOWN_TILE_OW) \
         -DMICRO_YOLO_HEAD_TILE_OH=$(MICRO_YOLO_HEAD_TILE_OH) \
         -DMICRO_YOLO_HEAD_TILE_OW=$(MICRO_YOLO_HEAD_TILE_OW)

SRCS = start.S main.c ../../lib/npu_graph.c ../../lib/conv2d_packed.c ../../lib/spatz_rt.c \
       ../../lib/hal_systolic.c \
       ../../lib/spatz_ops.c ../../lib/spatz_ops.S

all: micro_yolo.bin micro_yolo.hex

$(LINEBUF_PRECOMPUTE_HDR): $(LINEBUF_PRECOMPUTE_GEN)
	python3 $(LINEBUF_PRECOMPUTE_GEN) micro-yolo-header --output $@ \
	    --c2f-tile-oh $(MICRO_YOLO_C2F_TILE_OH) \
	    --c2f-tile-ow $(MICRO_YOLO_C2F_TILE_OW) \
	    --down-tile-oh $(MICRO_YOLO_DOWN_TILE_OH) \
	    --down-tile-ow $(MICRO_YOLO_DOWN_TILE_OW) \
	    --head-tile-oh $(MICRO_YOLO_HEAD_TILE_OH) \
	    --head-tile-ow $(MICRO_YOLO_HEAD_TILE_OW)

micro_yolo.elf: $(SRCS) link.ld $(LINEBUF_PRECOMPUTE_HDR)
	$(CC) $(CFLAGS) $(SRCS) -o $@

micro_yolo.bin: micro_yolo.elf
	$(OBJCOPY) -O binary $< $@

micro_yolo.hex: micro_yolo.elf
	$(HEXOBJCOPY) -O verilog $< $@

clean:
	rm -f *.elf *.bin *.hex $(LINEBUF_PRECOMPUTE_HDR)
