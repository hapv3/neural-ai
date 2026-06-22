#include "hal_systolic.h"
#include "npu_memory_map.h"

static void systolic_gemm32_tile(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    REG_WRITE(REG_SYS_W_PTR, weight_addr);
    REG_WRITE(REG_SYS_I_PTR, ifm_addr);
    REG_WRITE(REG_SYS_O_PTR, ofm_addr);
    REG_WRITE(REG_SYS_DIM_M, dim_m);
    REG_WRITE(REG_SYS_DONE, 0);
    REG_WRITE(REG_SYS_START, 1);

    while (REG_READ(REG_SYS_DONE) == 0) {
    }

    REG_WRITE(REG_SYS_DONE, 0);
}

void systolic_gemm32(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    uint32_t row = 0;

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_TILE_M;
        }

        systolic_gemm32_tile(weight_addr,
                             ifm_addr + row * 32u,
                             ofm_addr + row * 32u * 4u,
                             tile_m);
        row += tile_m;
    }
}
