#ifndef NPU_STREAM_PREFETCH_H
#define NPU_STREAM_PREFETCH_H

#include "npu_types.h"

/* Private scratch above required_tcdm_bytes, never the 4 KiB weight/qparam
 * staging area. No space is taken from a compiler-owned tensor allocation. */
#define NAI_STREAM_PREFETCH_SLOTS 2u
#define NAI_STREAM_PREFETCH_MAX_BYTES 4096u
#define NAI_STREAM_PREFETCH_MIN_BYTES 1024u

typedef struct {
    void *context;
    uint32_t (*submit)(void *context, uint32_t source, void *destination, uint32_t bytes);
    uint32_t (*wait)(void *context, uint32_t transfer_id);
} nai_stream_prefetch_ops_t;

typedef struct {
    uint8_t *data;
    uint32_t offset;
    uint32_t bytes;
    uint32_t used;
    uint32_t transfer_id;
} nai_stream_prefetch_slot_t;

typedef struct {
    nai_stream_prefetch_ops_t ops;
    nai_stream_prefetch_slot_t slots[NAI_STREAM_PREFETCH_SLOTS];
    uint32_t base;
    uint32_t read_offset;
    uint32_t next_offset;
    uint32_t end;
    uint32_t slot_bytes;
    uint32_t current;
    uint32_t failed;
} nai_stream_prefetch_t;

/* Forward copying also handles the dispatch window's overlapping retained
 * tail. Volatile reads are required for memory written by DMA. Unroll scalar
 * words without using Spatz while another command may still own the VPU. */
static inline void nai_stream_copy(void *destination, const volatile void *source, uint32_t bytes)
{
    uint8_t *output = (uint8_t *)destination;
    const volatile uint8_t *input = (const volatile uint8_t *)source;
    if ((((unsigned long)output | (unsigned long)input) & 3u) == 0u) {
        uint32_t *words = (uint32_t *)output;
        const volatile uint32_t *reads = (const volatile uint32_t *)input;
        while (bytes >= 16u) {
            words[0] = reads[0];
            words[1] = reads[1];
            words[2] = reads[2];
            words[3] = reads[3];
            words += 4;
            reads += 4;
            bytes -= 16u;
        }
        while (bytes >= 4u) {
            *words++ = *reads++;
            bytes -= 4u;
        }
        output = (uint8_t *)words;
        input = (const volatile uint8_t *)reads;
    }
    while (bytes-- != 0u) *output++ = *input++;
}

static inline uint32_t nai_stream_prefetch_capacity(uint32_t available)
{
    uint32_t bytes = NAI_STREAM_PREFETCH_MAX_BYTES;
    while (bytes >= NAI_STREAM_PREFETCH_MIN_BYTES) {
        if (available >= NAI_STREAM_PREFETCH_SLOTS * bytes) return bytes;
        bytes >>= 1;
    }
    return 0u;
}

static inline uint32_t nai_stream_prefetch_fill(nai_stream_prefetch_t *state, uint32_t index)
{
    nai_stream_prefetch_slot_t *slot = &state->slots[index];
    if (state->next_offset == state->end) return 0u;
    uint32_t bytes = state->end - state->next_offset;
    if (bytes > state->slot_bytes) bytes = state->slot_bytes;
    const uint32_t source = state->base + state->next_offset;
    const uint32_t boundary = 4096u - (source & 4095u);
    if (bytes > boundary) bytes = boundary;
    slot->offset = state->next_offset;
    slot->used = 0u;
    slot->bytes = bytes;
    slot->transfer_id = state->ops.submit(state->ops.context, source, slot->data, bytes);
    if (slot->transfer_id == 0u) {
        slot->bytes = 0u;
        state->failed = 1u;
        return 1u;
    }
    state->next_offset += bytes;
    return 0u;
}

static inline uint32_t nai_stream_prefetch_init(nai_stream_prefetch_t *state,
    const nai_stream_prefetch_ops_t *ops, uint32_t base, uint32_t begin, uint32_t end,
    void *scratch, uint32_t slot_bytes)
{
    *state = (nai_stream_prefetch_t){0};
    if (ops == 0 || ops->submit == 0 || ops->wait == 0 || scratch == 0 ||
        begin > end || base > 0xffffffffu - end ||
        slot_bytes < NAI_STREAM_PREFETCH_MIN_BYTES ||
        slot_bytes > NAI_STREAM_PREFETCH_MAX_BYTES) return 1u;
    state->ops = *ops;
    state->base = base;
    state->read_offset = state->next_offset = begin;
    state->end = end;
    state->slot_bytes = slot_bytes;
    for (uint32_t index = 0u; index < NAI_STREAM_PREFETCH_SLOTS; index++) {
        state->slots[index].data = (uint8_t *)scratch + index * slot_bytes;
        if (nai_stream_prefetch_fill(state, index) != 0u) return 1u;
    }
    return 0u;
}

static inline uint32_t nai_stream_prefetch_read(nai_stream_prefetch_t *state,
    uint32_t offset, void *destination, uint32_t bytes)
{
    if (state->failed || offset != state->read_offset || offset > state->end ||
        bytes > state->end - offset || (bytes != 0u && destination == 0)) return 1u;
    uint8_t *output = (uint8_t *)destination;
    while (bytes != 0u) {
        nai_stream_prefetch_slot_t *slot = &state->slots[state->current];
        if (slot->bytes == 0u || slot->offset + slot->used != state->read_offset) return 1u;
        if (slot->transfer_id != 0u) {
            if (!state->ops.wait(state->ops.context, slot->transfer_id)) {
                state->failed = 1u;
                return 1u;
            }
            slot->transfer_id = 0u;
        }
        uint32_t copy = slot->bytes - slot->used;
        if (copy > bytes) copy = bytes;
        nai_stream_copy(output, slot->data + slot->used, copy);
        slot->used += copy;
        state->read_offset += copy;
        output += copy;
        bytes -= copy;
        if (slot->used == slot->bytes) {
            slot->bytes = 0u;
            const uint32_t released = state->current;
            state->current ^= 1u;
            if (nai_stream_prefetch_fill(state, released) != 0u) return 1u;
        }
    }
    return 0u;
}

/* On END/error/re-entry, join every private job, including a partially filled
 * pair after submit failure. Never leave a DMA writing a reused scratch area. */
static inline uint32_t nai_stream_prefetch_finish(nai_stream_prefetch_t *state)
{
    uint32_t failed = 0u;
    for (uint32_t index = 0u; index < NAI_STREAM_PREFETCH_SLOTS; index++) {
        nai_stream_prefetch_slot_t *slot = &state->slots[index];
        if (slot->transfer_id != 0u) {
            if (state->ops.wait != 0 && state->ops.wait(state->ops.context, slot->transfer_id))
                slot->transfer_id = 0u;
            else failed = 1u;
        }
    }
    return failed;
}

#endif
