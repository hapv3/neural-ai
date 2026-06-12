#ifndef NPU_MEMORY_MAP_H
#define NPU_MEMORY_MAP_H

#include "npu_types.h"

// 1. I-TCM (32 KB) - Instruction Memory
#define NPU_ITCM_BASE   0x10000000
#define NPU_ITCM_SIZE   0x00008000

// 2. D-TCM (8 KB) - Snitch Local Data
#define NPU_DTCM_BASE   0x10008000
#define NPU_DTCM_SIZE   0x00002000

// 3. Shared Data TCDM (512 KB) - Compute Data
#define NPU_TCDM_BASE   0x10100000
#define NPU_TCDM_SIZE   0x00080000

// 4. MMIO Control Registers (64 KB)
#define NPU_CTRL_BASE   0x20000000

// DMA Configuration Registers
#define REG_DMA_START   (NPU_CTRL_BASE + 0x00)
#define REG_DMA_SRC     (NPU_CTRL_BASE + 0x20)
#define REG_DMA_DST     (NPU_CTRL_BASE + 0x40)
#define REG_DMA_LEN     (NPU_CTRL_BASE + 0x60)
#define REG_DMA_DONE    (NPU_CTRL_BASE + 0x80)

// Register Access Macros
#define REG_WRITE(addr, val) *((volatile uint32_t*)(addr)) = (val)
#define REG_READ(addr)       *((volatile uint32_t*)(addr))

#endif // NPU_MEMORY_MAP_H
