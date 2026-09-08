#ifndef IDMA_TRANSFER_ID_H
#define IDMA_TRANSFER_ID_H

#include <stdint.h>

static inline uint32_t idma_mm_transfer_completed(uint32_t done_id, uint32_t transfer_id)
{
    return transfer_id != 0u && (uint32_t)(done_id - transfer_id) < 0x80000000u;
}

#endif
