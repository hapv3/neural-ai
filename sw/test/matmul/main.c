#include "idma_mm_utils.h"

/*
 * Scenario: legacy systolic matmul application regression.
 * Target: accept one M dimension from D-TCM, move fixtures through DMA, run the
 * raw systolic register interface, and return Mx32 INT32 output to L2.
 */
#define WEIGHT_PING_ADDR 0x10110000
#define IFM_PING_ADDR    0x10120000
#define OFM_PING_ADDR    0x10200000

#define EXT_MEM_WEIGHT   0x80000000
#define EXT_MEM_IFM      0x80001000
#define EXT_MEM_OFM      0x80002000

void dma_transfer(uint32_t src, uint32_t dst, uint32_t len) {
    // Keep this wrapper blocking so register-level systolic sequencing is deterministic.
    if (!idma_memcpy_blocking(src, dst, len)) {
        while (1) {
        }
    }
}

int main() {
    volatile uint32_t *start_flag = (volatile uint32_t*)0x10008004;
    volatile uint32_t *done_flag = (volatile uint32_t*)0x10008008;

    while (1) {
        // Testbench owns fixtures and kicks one transaction by setting start_flag.
        while (*start_flag == 0) {
            // wait
        }
        *start_flag = 0; // clear for next run

        uint32_t dim_m = *(volatile uint32_t*)0x10008000;
        // This legacy app limits M to the non-tiled raw-controller range.
        if (dim_m == 0 || dim_m > 128) dim_m = 32;

        uint32_t weight_len = 32 * 32; // 1024 bytes (8-bit), fixed K=32
        uint32_t ifm_len = dim_m * 32; // 32 bytes per row * dim_m rows
        uint32_t ofm_len = dim_m * 32 * 4; // 128 bytes per row * dim_m rows

        // Stage fixed 32x32 INT8 weights into I-TCDM.
        dma_transfer(EXT_MEM_WEIGHT, WEIGHT_PING_ADDR, weight_len);

        // Stage Mx32 INT8 activations into I-TCDM.
        dma_transfer(EXT_MEM_IFM, IFM_PING_ADDR, ifm_len);

        // Program raw systolic MMIO registers; HAL tiling is intentionally bypassed here.
        REG_WRITE(REG_SYS_W_PTR, WEIGHT_PING_ADDR);
        REG_WRITE(REG_SYS_I_PTR, IFM_PING_ADDR);
        REG_WRITE(REG_SYS_O_PTR, OFM_PING_ADDR);
        REG_WRITE(REG_SYS_DIM_M, dim_m);
        REG_WRITE(REG_SYS_DONE, 0);
        REG_WRITE(REG_SYS_START, 1);

        // Poll completion because there is no interrupt path in this firmware.
        while (REG_READ(REG_SYS_DONE) == 0) {
            // Wait
        }

        // Return full INT32 output tensor to L2 for cocotb comparison.
        dma_transfer(OFM_PING_ADDR, EXT_MEM_OFM, ofm_len);

        // Signal finish by setting a done flag in D-TCM
        *done_flag = 1;
    }

    return 0;
}
