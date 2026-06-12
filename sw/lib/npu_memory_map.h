#ifndef NPU_MEMORY_MAP_H
#define NPU_MEMORY_MAP_H

#include "npu_types.h"

// Base Addresses
#define NPU_TCDM_BASE     0x10000000
#define NPU_MMIO_BASE     0x20000000

// Cluster Control Registers Offsets (32-byte aligned)
#define REG_DMA_SRC       0x00
#define REG_DMA_DST       0x20
#define REG_DMA_LEN       0x40
#define REG_DMA_START     0x60
#define REG_DMA_DONE      0x80

#define REG_SYS_W_PTR     0xA0
#define REG_SYS_I_PTR     0xC0
#define REG_SYS_O_PTR     0xE0
#define REG_SYS_DIM_M     0x100
#define REG_SYS_START     0x120
#define REG_SYS_DONE      0x140

// Utility Macros
#define REG_WRITE(offset, val)  (*((volatile uint32_t*)(NPU_MMIO_BASE + (offset))) = (val))
#define REG_READ(offset)        (*((volatile uint32_t*)(NPU_MMIO_BASE + (offset))))

#endif // NPU_MEMORY_MAP_H
