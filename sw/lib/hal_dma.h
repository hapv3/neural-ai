#ifndef HAL_DMA_H
#define HAL_DMA_H

#include "npu_types.h"
#include "npu_memory_map.h"

// Initialize a 1D DMA transfer through the iDMA-style MMIO window.
// Local-to-local copies fall back to the legacy compatibility registers.
void dma_start_transfer(uint32_t src_addr, uint32_t dst_addr, uint32_t length);

// Wait for the last DMA transfer to complete.
void dma_wait_done(void);

#endif // HAL_DMA_H
