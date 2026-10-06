#include "npu_cmd_desc_v2.h"

#include <assert.h>
#include <string.h>

extern const nai_cmd_header_v2_t *volatile g_nai_current_command;

static uint8_t g_model[16416];
static nai_cmd_header_v2_t g_expected[512];
static uint32_t g_expected_count;
static uint32_t g_calls;
static uint32_t g_begins;
static uint32_t g_ends;

typedef struct {
    uint32_t next;
    uint32_t end;
    uint32_t reads;
    uint32_t fail_read;
} reader_state_t;

static uint32_t read_model(void *context, uint32_t offset, void *destination, uint32_t bytes)
{
    reader_state_t *state = context;
    assert(offset == state->next); /* No reread/discard of a retained tail. */
    assert(bytes != 0 && bytes <= state->end - offset);
    if (++state->reads == state->fail_read) return 1;
    memcpy(destination, g_model + offset, bytes);
    state->next += bytes;
    return 0;
}

void nai_pmu_command_begin(uint32_t id)
{
    assert(id == g_calls && id == g_begins++);
    assert(memcmp(g_nai_current_command, &g_expected[id], sizeof(g_expected[id])) == 0);
}

void nai_pmu_command_end(uint32_t id)
{
    assert(id + 1 == g_calls && id == g_ends++);
}

static uint32_t barrier(void *context)
{
    (void)context;
    assert(g_calls < g_expected_count);
    assert(memcmp(g_nai_current_command, &g_expected[g_calls], sizeof(g_expected[g_calls])) == 0);
    g_calls++;
    return 0;
}

static uint32_t dma(void *context, uint32_t source, uint32_t destination,
    uint32_t bytes, uint32_t direction)
{
    assert(source == 0x81004000u && destination == 0x10100000u);
    assert(bytes == 32 && direction == NAI_DMA_EXTERNAL_TO_LOCAL);
    return barrier(context);
}

static uint32_t append_control(uint32_t offset, uint16_t type, uint32_t tile)
{
    nai_cmd_control_v2_t command = {0};
    command.header.type = type;
    command.header.size_bytes = sizeof(command);
    command.header.tile_id = tile;
    memcpy(g_model + offset, &command, sizeof(command));
    if (type != NAI_CMD_END) g_expected[g_expected_count++] = command.header;
    return offset + sizeof(command);
}

static uint32_t build_mixed(void)
{
    uint32_t offset = 32;
    memset(g_model, 0, sizeof(g_model));
    g_expected_count = 0;
    for (uint32_t index = 0; index < 120; index++) {
        if (index % 3 == 0) {
            offset = append_control(offset, NAI_CMD_BARRIER, index * 4);
        } else if (index % 3 == 1) {
            nai_cmd_dma_1d_v2_t command = {0};
            command.header.type = NAI_CMD_DMA_1D;
            command.header.size_bytes = sizeof(command);
            command.header.tile_id = index * 4;
            command.source.region = NAI_REGION_MODEL_CONSTANTS;
            command.destination.region = NAI_REGION_TCDM_SCRATCH;
            command.length = 32;
            command.direction = NAI_DMA_EXTERNAL_TO_LOCAL;
            memcpy(g_model + offset, &command, sizeof(command));
            g_expected[g_expected_count++] = command.header;
            offset += sizeof(command);
        } else {
            nai_cmd_affine_loop_v2_t loop = {0};
            nai_cmd_affine_patch_v2_t patch = {3, 1}; /* Increment tile ID. */
            loop.header.type = NAI_CMD_AFFINE_LOOP;
            loop.header.size_bytes = 64;
            loop.iteration_count = 3;
            loop.body_command_count = 1;
            loop.body_bytes = sizeof(nai_cmd_control_v2_t);
            loop.patch_count = 1;
            memcpy(g_model + offset, &loop, sizeof(loop));
            memcpy(g_model + offset + sizeof(loop), &patch, sizeof(patch));
            offset = append_control(offset + 64, NAI_CMD_BARRIER, index * 4);
            for (uint32_t iteration = 1; iteration < 3; iteration++) {
                g_expected[g_expected_count] = g_expected[g_expected_count - 1];
                g_expected[g_expected_count++].tile_id++;
            }
        }
    }
    return append_control(offset, NAI_CMD_END, 0);
}

static void run_window(uint32_t bytes, uint32_t capacity, uint32_t fail_read,
    uint32_t entry_skip, uint32_t expected_count, uint32_t unchanged_prefix)
{
    uint8_t original[sizeof(g_model)];
    uint32_t storage[1026];
    uint8_t *buffer = (uint8_t *)(storage + 1);
    nai_model_header_v1_t header = {0};
    nai_section_v1_t commands = {0};
    nai_section_v1_t constants = {0};
    nai_model_view_v1_t view = {0};
    nai_resolver_v1_t resolver = {0};
    nai_runtime_ops_v2_t ops = {0};
    reader_state_t state = {32 + entry_skip, bytes, 0, fail_read};
    nai_model_reader_v1_t reader = {&state, read_model};
    uint32_t completed = 0, failure = 0;

    assert(capacity <= 4096 && capacity % 4 == 0);
    memset(storage, 0xa5, sizeof(storage));
    memcpy(original, g_model, sizeof(original));
    header.entry_command_off = 32 + entry_skip;
    header.command_count = expected_count;
    commands.offset = 32;
    commands.size = bytes - 32;
    constants.offset = 16384;
    constants.size = 32;
    view.header = &header;
    view.commands = &commands;
    view.constants = &constants;
    view.model = g_model;
    view.model_bytes = 16416;
    resolver.model_base = 0x81000000u;
    resolver.model_bytes = sizeof(g_model);
    resolver.tcdm_scratch_base = 0x10100000u;
    resolver.tcdm_scratch_bytes = 0x7f000;
    ops.barrier = barrier;
    ops.dma_1d = dma;
    g_calls = g_begins = g_ends = 0;
    nai_dispatch_status_v2_t status = nai_cmd_dispatch_stream_v2(
        &view, &resolver, &ops, &reader, buffer, capacity, &completed, &failure);
    if (fail_read) {
        assert(status == NAI_DISPATCH_BAD_STREAM);
        assert(state.reads == fail_read);
        assert(completed == g_calls && failure >= header.entry_command_off);
    } else {
        assert(status == NAI_DISPATCH_OK);
        assert(completed == expected_count && g_calls == completed);
        assert(state.next == bytes);
        assert(failure == 0);
    }
    assert(memcmp(original, g_model, sizeof(original)) == 0);
    assert(storage[0] == 0xa5a5a5a5u && storage[capacity / 4 + 1] == 0xa5a5a5a5u);
    if (unchanged_prefix) {
        assert(state.reads == 1);
        assert(memcmp(buffer, g_model + header.entry_command_off, unchanged_prefix) == 0);
    }
#if defined(NAI_PMU_PROFILE) && NAI_PMU_PROFILE
    assert(g_begins == g_calls && g_ends == g_calls);
#endif
}

int main(void)
{
    uint32_t bytes = build_mixed();
    const uint32_t capacities[] = {96, 100, 112, 128, 160, 192, 224, 2048, 4096};
    for (uint32_t index = 0; index < sizeof(capacities) / sizeof(capacities[0]); index++)
        run_window(bytes, capacities[index], 0, 0, g_expected_count, 0);
    run_window(bytes, 128, 1, 0, g_expected_count, 0);
    run_window(bytes, 128, 2, 0, g_expected_count, 0);
    run_window(bytes, 128, 3, 0, g_expected_count, 0);

    /* Exactly one window: no descriptor, including affine metadata, is shifted. */
    run_window(append_control(224, NAI_CMD_END, 0), 2048, 0, 0, 5, 32);

    memset(g_model, 0, sizeof(g_model));
    g_expected_count = 0;
    uint32_t offset = 32;
    for (uint32_t index = 0; index < 63; index++)
        offset = append_control(offset, NAI_CMD_BARRIER, index);
    bytes = append_control(offset, NAI_CMD_END, 0);
    run_window(bytes, 2048, 0, 0, 63, 2048);
    /* Nonzero entry skips the first command, without reading it. */
    memmove(g_expected, g_expected + 1, 62 * sizeof(g_expected[0]));
    g_expected_count = 62;
    run_window(bytes, 2048, 0, 32, 62, 2016);
    return 0;
}
