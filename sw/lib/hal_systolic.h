#ifndef HAL_SYSTOLIC_H
#define HAL_SYSTOLIC_H

#include "npu_types.h"

void systolic_gemm32(uint32_t weight_addr, uint32_t ifm_addr, uint32_t ofm_addr, uint32_t dim_m);

#endif
