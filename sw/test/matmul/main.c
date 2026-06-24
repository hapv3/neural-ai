#include "idma_mm_utils.h"

/*
 * Scenario: legacy systolic matmul application regression.
 * Target: run one M=64 systolic GEMM fixture from L2, return Mx32 INT32 output
 * to L2, and notify the host through the interrupt controller.
 */
#define WEIGHT_PING_ADDR 0x10110000
#define IFM_PING_ADDR    0x10120000
#define OFM_PING_ADDR    0x10200000

#define EXT_MEM_WEIGHT   0x80000000
#define EXT_MEM_IFM      0x80001000
#define EXT_MEM_OFM      0x80002000

#define MATMUL_DIM_M     64u
#define PASS_SIGNATURE   0xDEADBEEFu
#define FAIL_SIGNATURE   0xBAD30000u

void dma_transfer(uint32_t src, uint32_t dst, uint32_t len) {
    // Keep this wrapper blocking so register-level systolic sequencing is deterministic.
    if (!idma_memcpy_blocking(src, dst, len)) {
        while (1) {
        }
    }
}

int main() {
    uint32_t dim_m = MATMUL_DIM_M;
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

    // Poll completion locally, then notify the external host/testbench by IRQ.
    while (REG_READ(REG_SYS_DONE) == 0) {
        // Wait
    }

    // Return full INT32 output tensor to L2 for cocotb comparison.
    dma_transfer(OFM_PING_ADDR, EXT_MEM_OFM, ofm_len);

    REG_WRITE(NPU_IRQ_HOST_NOTIFY, PASS_SIGNATURE);
    while (1) {
    }

    return 0;
}
