#include "hal_dma.h"

static uint32_t last_dma_dir;
static uint32_t last_dma_id;
static uint32_t last_dma_legacy;

static uint32_t dma_is_l1_addr(uint32_t addr) {
    return ((addr >> 24) == 0x10u);
}

static void legacy_dma_start_transfer(uint32_t src_addr, uint32_t dst_addr, uint32_t length) {
    REG_WRITE(REG_DMA_SRC, src_addr);
    REG_WRITE(REG_DMA_DST, dst_addr);
    REG_WRITE(REG_DMA_LEN, length);
    REG_WRITE(REG_DMA_DONE, 0);
    REG_WRITE(REG_DMA_START, 1);
}

void dma_start_transfer(uint32_t src_addr, uint32_t dst_addr, uint32_t length) {
    uint32_t src_is_l1 = dma_is_l1_addr(src_addr);
    uint32_t dst_is_l1 = dma_is_l1_addr(dst_addr);

    if (src_is_l1 == dst_is_l1) {
        last_dma_legacy = 1;
        legacy_dma_start_transfer(src_addr, dst_addr, length);
        return;
    }

    last_dma_legacy = 0;
    last_dma_dir = src_is_l1 ? IDMA_DIR_OBI2AXI : IDMA_DIR_AXI2OBI;

    REG_WRITE(IDMA_CONF(last_dma_dir), 0);
    REG_WRITE(IDMA_DST_ADDR_LOW(last_dma_dir), dst_addr);
    REG_WRITE(IDMA_SRC_ADDR_LOW(last_dma_dir), src_addr);
    REG_WRITE(IDMA_LENGTH_LOW(last_dma_dir), length);
    REG_WRITE(IDMA_DST_STRIDE_2(last_dma_dir), 0);
    REG_WRITE(IDMA_SRC_STRIDE_2(last_dma_dir), 0);
    REG_WRITE(IDMA_REPS_2(last_dma_dir), 1);
    REG_WRITE(IDMA_DST_STRIDE_3(last_dma_dir), 0);
    REG_WRITE(IDMA_SRC_STRIDE_3(last_dma_dir), 0);
    REG_WRITE(IDMA_REPS_3(last_dma_dir), 1);

    last_dma_id = REG_READ(IDMA_NEXT_ID(last_dma_dir));
}

void dma_wait_done(void) {
    if (last_dma_legacy) {
        while(REG_READ(REG_DMA_DONE) == 0) {
        }
        REG_WRITE(REG_DMA_DONE, 0);
        return;
    }

    while(REG_READ(IDMA_STATUS(last_dma_dir)) != 0) {
    }

    while(REG_READ(IDMA_DONE_ID(last_dma_dir)) != last_dma_id) {
    }
}
