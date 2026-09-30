#ifndef IDMA_TRANSFER_PLAN_H
#define IDMA_TRANSFER_PLAN_H

#include <stdint.h>

#define IDMA_AXI_BOUNDARY_BYTES 4096u

static inline uint32_t idma_limit_axi_4k_chunk(
    uint32_t external_address, uint32_t requested_bytes)
{
    const uint32_t boundary_bytes = IDMA_AXI_BOUNDARY_BYTES -
        (external_address & (IDMA_AXI_BOUNDARY_BYTES - 1u));
    return requested_bytes < boundary_bytes ? requested_bytes : boundary_bytes;
}

#endif
