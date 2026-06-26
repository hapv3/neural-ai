#include "hal_systolic.h"
#include "npu_memory_map.h"

static void systolic_gemm32_tile_ex(uint32_t weight_addr,
                                    uint32_t ifm_addr,
                                    uint32_t psum_addr,
                                    uint32_t ofm_addr,
                                    uint32_t dim_m,
                                    uint32_t accum_en) {
    REG_WRITE(REG_SYS_W_PTR, weight_addr);
    REG_WRITE(REG_SYS_I_PTR, ifm_addr);
    REG_WRITE(REG_SYS_O_PTR, ofm_addr);
    REG_WRITE(REG_SYS_PSUM_PTR, psum_addr);
    REG_WRITE(REG_SYS_DIM_M, dim_m);
    REG_WRITE(REG_SYS_ACCUM_CTRL, accum_en ? REG_SYS_ACCUM_CTRL_EN : 0u);
    REG_WRITE(REG_SYS_DONE, 0);
    REG_WRITE(REG_SYS_START, 1);

    while (REG_READ(REG_SYS_DONE) == 0) {
    }

    REG_WRITE(REG_SYS_DONE, 0);
    REG_WRITE(REG_SYS_ACCUM_CTRL, 0u);
}

static void systolic_gemm32_tile(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    systolic_gemm32_tile_ex(weight_addr, ifm_addr, 0u, ofm_addr, dim_m, 0u);
}

void systolic_requant_disable(void) {
    REG_WRITE(REG_RQ_CTRL, 0u);
}

void systolic_requant_config_per_channel(const int32_t *bias,
                                         const int32_t *multiplier,
                                         const uint8_t *shift,
                                         const int32_t *zero_point,
                                         int32_t clamp_min,
                                         int32_t clamp_max) {
    REG_WRITE(REG_RQ_CTRL, 0u);
    REG_WRITE(REG_RQ_CMIN, (uint32_t)clamp_min);
    REG_WRITE(REG_RQ_CMAX, (uint32_t)clamp_max);

    for (uint32_t ch = 0; ch < SYSTOLIC_GEMM32_N; ch++) {
        REG_WRITE(REG_RQ_BIAS(ch), (uint32_t)bias[ch]);
        REG_WRITE(REG_RQ_MULT(ch), (uint32_t)multiplier[ch]);
        REG_WRITE(REG_RQ_SHIFT(ch), (uint32_t)shift[ch]);
        REG_WRITE(REG_RQ_ZP(ch), (uint32_t)zero_point[ch]);
    }

    REG_WRITE(REG_RQ_CTRL, REG_RQ_CTRL_EN);
}

void systolic_gemm32(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    uint32_t row = 0;
    systolic_requant_disable();

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

void systolic_gemm32_accumulate(uint32_t weight_addr,
                                uint32_t ifm_addr,
                                uint32_t psum_addr,
                                uint32_t ofm_addr,
                                uint32_t dim_m) {
    uint32_t row = 0;
    systolic_requant_disable();

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_ACCUM_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_ACCUM_TILE_M;
        }

        systolic_gemm32_tile_ex(weight_addr,
                                ifm_addr + row * 32u,
                                psum_addr + row * 32u * 4u,
                                ofm_addr + row * 32u * 4u,
                                tile_m,
                                1u);
        row += tile_m;
    }
}

void systolic_gemm32_accumulate_requant(uint32_t weight_addr,
                                        uint32_t ifm_addr,
                                        uint32_t psum_addr,
                                        uint32_t ofm_addr,
                                        uint32_t dim_m) {
    uint32_t row = 0;

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_ACCUM_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_ACCUM_TILE_M;
        }

        systolic_gemm32_tile_ex(weight_addr,
                                ifm_addr + row * 32u,
                                psum_addr + row * 32u * 4u,
                                ofm_addr + row * 32u,
                                tile_m,
                                1u);
        row += tile_m;
    }
}

void systolic_gemm32_requant(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m) {
    uint32_t row = 0;

    while (row < dim_m) {
        uint32_t tile_m = dim_m - row;
        if (tile_m > SYSTOLIC_GEMM32_TILE_M) {
            tile_m = SYSTOLIC_GEMM32_TILE_M;
        }

        systolic_gemm32_tile(weight_addr,
                             ifm_addr + row * 32u,
                             ofm_addr + row * 32u,
                             tile_m);
        row += tile_m;
    }
}
