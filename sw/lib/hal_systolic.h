#ifndef HAL_SYSTOLIC_H
#define HAL_SYSTOLIC_H

#include "npu_types.h"

#define SYSTOLIC_GEMM32_K 32u
#define SYSTOLIC_GEMM32_N 32u
#define SYSTOLIC_GEMM32_TILE_M 1024u

void systolic_gemm32(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m);

#endif
