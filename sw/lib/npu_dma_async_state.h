#ifndef NPU_DMA_ASYNC_STATE_H
#define NPU_DMA_ASYNC_STATE_H

#include <stdint.h>

#define NPU_DMA_ASYNC_DIRECTIONS 2u
#define NPU_DMA_JOB_QUEUE_DEPTH 16u

typedef struct {
    uint32_t last_transfer_id[NPU_DMA_ASYNC_DIRECTIONS];
    uint32_t queued_transfers[NPU_DMA_ASYNC_DIRECTIONS];
    uint32_t systolic_pending;
} nai_dma_async_state_t;

static inline uint32_t nai_dma_async_can_submit(
    const nai_dma_async_state_t *state, uint32_t direction)
{
    return state != 0 && direction < NPU_DMA_ASYNC_DIRECTIONS &&
        state->queued_transfers[direction] < NPU_DMA_JOB_QUEUE_DEPTH;
}

static inline uint32_t nai_dma_async_record(
    nai_dma_async_state_t *state, uint32_t direction, uint32_t transfer_id)
{
    if (!nai_dma_async_can_submit(state, direction) || transfer_id == 0u) return 1u;
    state->last_transfer_id[direction] = transfer_id;
    state->queued_transfers[direction]++;
    return 0u;
}

static inline uint32_t nai_dma_async_take_last(
    nai_dma_async_state_t *state, uint32_t direction, uint32_t *transfer_id)
{
    if (state == 0 || transfer_id == 0 || direction >= NPU_DMA_ASYNC_DIRECTIONS ||
        state->queued_transfers[direction] == 0u) return 1u;
    *transfer_id = state->last_transfer_id[direction];
    state->last_transfer_id[direction] = 0u;
    state->queued_transfers[direction] = 0u;
    return 0u;
}

#endif
