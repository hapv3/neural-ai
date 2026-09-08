#include "npu_dma_async_state.h"
#include "idma_transfer_id.h"

#include <assert.h>
#include <string.h>

int main(void)
{
    nai_dma_async_state_t state;
    uint32_t transfer_id = 0u;
    memset(&state, 0, sizeof(state));

    assert(!nai_dma_async_can_submit(0, 0u));
    assert(!nai_dma_async_can_submit(&state, NPU_DMA_ASYNC_DIRECTIONS));
    assert(nai_dma_async_record(&state, 0u, 0u) != 0u);

    for (uint32_t index = 0u; index < NPU_DMA_JOB_QUEUE_DEPTH; index++) {
        assert(nai_dma_async_can_submit(&state, 0u));
        assert(nai_dma_async_record(&state, 0u, index + 2u) == 0u);
    }
    assert(!nai_dma_async_can_submit(&state, 0u));
    assert(nai_dma_async_record(&state, 0u, 99u) != 0u);

    assert(nai_dma_async_can_submit(&state, 1u));
    assert(nai_dma_async_record(&state, 1u, 0xffffffffu) == 0u);
    assert(nai_dma_async_record(&state, 1u, 2u) == 0u);

    assert(nai_dma_async_take_last(&state, 0u, &transfer_id) == 0u);
    assert(transfer_id == NPU_DMA_JOB_QUEUE_DEPTH + 1u);
    assert(state.queued_transfers[0] == 0u);
    assert(state.last_transfer_id[0] == 0u);
    assert(nai_dma_async_can_submit(&state, 0u));
    assert(nai_dma_async_take_last(&state, 0u, &transfer_id) != 0u);

    assert(nai_dma_async_take_last(&state, 1u, &transfer_id) == 0u);
    assert(transfer_id == 2u);
    assert(state.queued_transfers[1] == 0u);

    assert(!idma_mm_transfer_completed(1u, 2u));
    assert(idma_mm_transfer_completed(2u, 2u));
    assert(idma_mm_transfer_completed(3u, 2u));
    assert(!idma_mm_transfer_completed(3u, 4u));
    assert(idma_mm_transfer_completed(2u, 0xffffffffu));
    assert(!idma_mm_transfer_completed(0xffffffffu, 2u));
    assert(!idma_mm_transfer_completed(3u, 0u));
    return 0;
}
