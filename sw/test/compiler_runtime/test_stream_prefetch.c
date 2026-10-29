#include "npu_stream_prefetch.h"

#include <assert.h>
#include <string.h>

typedef struct {
    uint32_t id;
    uint32_t source;
    void *destination;
    uint32_t bytes;
} job_t;

typedef struct {
    uint8_t model[65536];
    uint8_t staging[8224];
    job_t jobs[16];
    uint32_t head, count, next_id, done_id;
    uint32_t submits, waits, max_pending;
    uint32_t fail_submit, fail_wait;
} backend_t;

static backend_t s_backend;

static void complete_one(backend_t *backend)
{
    assert(backend->count != 0u);
    job_t *job = &backend->jobs[backend->head];
    assert(job->source >= 0x81000000u && job->source - 0x81000000u <= sizeof(backend->model) - job->bytes);
    memcpy(job->destination, backend->model + job->source - 0x81000000u, job->bytes);
    backend->done_id = job->id;
    backend->head = (backend->head + 1u) % 16u;
    backend->count--;
}

static uint32_t submit(void *context, uint32_t source, void *destination, uint32_t bytes)
{
    backend_t *backend = context;
    assert(bytes && bytes <= 4096u - (source & 4095u));
    if (++backend->submits == backend->fail_submit) return 0u;
    /* The register frontend backpressures when its shared FIFO is full. */
    if (backend->count == 16u) complete_one(backend);
    const uint32_t index = (backend->head + backend->count) % 16u;
    const uint32_t id = backend->next_id++;
    if (backend->next_id == 0u) backend->next_id = 1u;
    backend->jobs[index] = (job_t){id, source, destination, bytes};
    backend->count++;
    if (backend->count > backend->max_pending) backend->max_pending = backend->count;
    return id;
}

static uint32_t wait_job(void *context, uint32_t id)
{
    backend_t *backend = context;
    backend->waits++;
    if (backend->waits == backend->fail_wait) return 0u;
    assert(id != 0u);
    while ((int32_t)(backend->done_id - id) < 0) complete_one(backend);
    return 1u;
}

static void reset_backend(void)
{
    memset(&s_backend, 0, sizeof(s_backend));
    s_backend.next_id = 1u;
    for (uint32_t i = 0; i < sizeof(s_backend.model); i++)
        s_backend.model[i] = (uint8_t)(i * 37u + (i >> 8) + 11u);
    memset(s_backend.staging, 0xa5, sizeof(s_backend.staging));
}

static void check_case(uint32_t begin, uint32_t end, uint32_t capacity, uint32_t seed)
{
    nai_stream_prefetch_t state;
    nai_stream_prefetch_ops_t ops = {&s_backend, submit, wait_job};
    uint8_t actual[2057];
    uint8_t unrelated[32];
    reset_backend();
    if (seed & 1u) {
        s_backend.next_id = 0xfffffffeu;
        s_backend.done_id = 0xfffffffdu;
    }
    assert(nai_stream_prefetch_init(&state, &ops, 0x81000000u, begin, end,
        s_backend.staging + 16, capacity) == 0u);
    assert(s_backend.waits == 0u); /* Submit ahead; never wait at prefetch issue. */
    uint32_t offset = begin;
    while (offset < end) {
        const uint32_t step = 1u + (seed * 97u + offset * 13u) % 2049u;
        const uint32_t bytes = step < end - offset ? step : end - offset;
        memset(actual, 0xa5, sizeof(actual));
        /* A data transfer can share the direction and finish the private IDs
         * before the reader asks for them; IDs are not private to prefetch. */
        if ((offset & 7u) == 0u) {
            uint32_t id = submit(&s_backend, 0x81000000u, unrelated, sizeof(unrelated));
            assert(wait_job(&s_backend, id));
        }
        assert(nai_stream_prefetch_read(&state, offset, actual + 1, bytes) == 0u);
        assert(memcmp(actual + 1, s_backend.model + offset, bytes) == 0);
        assert(actual[0] == 0xa5u && actual[bytes + 1] == 0xa5u);
        offset += bytes;
    }
    assert(nai_stream_prefetch_finish(&state) == 0u && s_backend.count == 0u);
    assert(nai_stream_prefetch_read(&state, end, 0, 0) == 0u);
    assert(nai_stream_prefetch_read(&state, end, actual, 1) != 0u);
    for (uint32_t i = 0; i < 16; i++) {
        assert(s_backend.staging[i] == 0xa5u);
        assert(s_backend.staging[16 + 2 * capacity + i] == 0xa5u);
    }
}

int main(void)
{
    assert(nai_stream_prefetch_capacity(0u) == 0u);
    assert(nai_stream_prefetch_capacity(2047u) == 0u);
    assert(nai_stream_prefetch_capacity(2048u) == 1024u);
    assert(nai_stream_prefetch_capacity(4096u) == 2048u);
    assert(nai_stream_prefetch_capacity(8192u) == 4096u);
    assert(nai_stream_prefetch_capacity(0xffffffffu) == 4096u);
    for (uint32_t trial = 0; trial < 300; trial++) {
        uint32_t begin = (trial * 127u) % 8192u;
        uint32_t end = begin + (trial * 503u) % 49152u;
        check_case(begin, end, 1024u << (trial % 3u), trial);
    }
    check_case(4095, 4096, 4096, 0);
    check_case(32, 32, 1024, 0);

    nai_stream_prefetch_t state;
    nai_stream_prefetch_ops_t ops = {&s_backend, submit, wait_job};
    uint8_t actual[32];
    reset_backend();
    s_backend.fail_submit = 2;
    assert(nai_stream_prefetch_init(&state, &ops, 0x81000000u, 0, 16384,
        s_backend.staging + 16, 4096) != 0u);
    assert(s_backend.count == 1u);
    assert(nai_stream_prefetch_finish(&state) == 0u && s_backend.count == 0u);

    reset_backend();
    assert(nai_stream_prefetch_init(&state, &ops, 0x81000000u, 0, 16384,
        s_backend.staging + 16, 4096) == 0u);
    s_backend.fail_wait = 1;
    assert(nai_stream_prefetch_read(&state, 0, actual, sizeof(actual)) != 0u);
    assert(nai_stream_prefetch_finish(&state) == 0u && s_backend.count == 0u);

    reset_backend();
    assert(nai_stream_prefetch_init(&state, &ops, 0x81000000u, 0, 16384,
        s_backend.staging + 16, 4096) == 0u);
    assert(nai_stream_prefetch_read(&state, 1, actual, 1) != 0u);
    assert(nai_stream_prefetch_finish(&state) == 0u && s_backend.count == 0u);
    assert(nai_stream_prefetch_init(&state, &ops, 0xfffff000u, 0, 8192,
        s_backend.staging + 16, 4096) != 0u);

    /* A full shared direction queue must backpressure safely, not overwrite
     * a hardware descriptor or assume that only prefetch owns transfer IDs. */
    reset_backend();
    uint8_t destinations[16][32];
    for (uint32_t i = 0; i < 16; i++)
        assert(submit(&s_backend, 0x81000000u, destinations[i], 32) != 0u);
    assert(nai_stream_prefetch_init(&state, &ops, 0x81000000u, 0, 16384,
        s_backend.staging + 16, 4096) == 0u);
    assert(nai_stream_prefetch_read(&state, 0, actual, sizeof(actual)) == 0u);
    assert(nai_stream_prefetch_finish(&state) == 0u && s_backend.count == 0u);
    assert(s_backend.max_pending == 16u);
    for (uint32_t i = 0; i < 16; i++)
        assert(memcmp(destinations[i], s_backend.model, 32) == 0);

    /* Forward overlapping window tails, aligned and byte paths. */
    uint8_t reference[256], copy[256];
    for (uint32_t shift = 1; shift < 33; shift++) {
        for (uint32_t bytes = 0; bytes < 200; bytes++) {
            memset(reference, 0x5a, sizeof(reference));
            for (uint32_t i = 0; i < sizeof(copy); i++) copy[i] = reference[i] = (uint8_t)i;
            memmove(reference, reference + shift, bytes);
            nai_stream_copy(copy, copy + shift, bytes);
            assert(memcmp(copy, reference, sizeof(copy)) == 0);
        }
    }
    return 0;
}
