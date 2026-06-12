#include "hal_dma.h"

void dma_start_transfer(uint32_t src_addr, uint32_t dst_addr, uint32_t length) {
    REG_WRITE(REG_DMA_SRC, src_addr);
    REG_WRITE(REG_DMA_DST, dst_addr);
    REG_WRITE(REG_DMA_LEN, length);
    // Pulse start
    REG_WRITE(REG_DMA_START, 1);
}

void dma_wait_done(void) {
    // Poll the done register until it reads 1
    while(REG_READ(REG_DMA_DONE) == 0) {
        // Busy wait
    }
    // Clear the done flag by writing to it
    REG_WRITE(REG_DMA_DONE, 0);
}
