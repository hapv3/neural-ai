#include "npu_cmd_desc_v2.h"

#include <assert.h>
#include <string.h>

typedef struct {
    uint32_t calls;
    uint32_t source;
    uint32_t destination;
    uint32_t length;
    uint32_t qparam_address;
    uint32_t qparam_count;
    uint32_t qparam_block;
} mock_state_t;

typedef struct {
    const uint8_t *data;
    uint32_t bytes;
    uint32_t largest_read;
    uint32_t reads;
} memory_reader_t;

static uint32_t memory_read(void *context, uint32_t offset, void *destination, uint32_t bytes)
{
    memory_reader_t *reader = (memory_reader_t *)context;
    if (offset > reader->bytes || bytes > reader->bytes - offset) return 1u;
    memcpy(destination, reader->data + offset, bytes);
    if (bytes > reader->largest_read) reader->largest_read = bytes;
    reader->reads++;
    return 0u;
}

static uint32_t mock_dma_1d(void *context, uint32_t source, uint32_t destination,
                            uint32_t length, uint32_t direction)
{
    mock_state_t *state = (mock_state_t *)context;
    (void)direction;
    state->calls++;
    state->source = source;
    state->destination = destination;
    state->length = length;
    return 0u;
}

static uint32_t mock_rq_load(void *context, uint32_t qparam_address,
                             uint32_t qparam_count, uint32_t qparam_block)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->qparam_address = qparam_address;
    state->qparam_count = qparam_count;
    state->qparam_block = qparam_block;
    return 0u;
}

static void make_dma_model(uint8_t model[1408])
{
    nai_model_header_v1_t *header;
    nai_section_v1_t *sections;
    nai_cmd_dma_1d_v2_t *dma;
    nai_cmd_header_v2_t *end;
    nai_binding_v1_t *binding;

    memset(model, 0, 1408);
    header = (nai_model_header_v1_t *)model;
    sections = (nai_section_v1_t *)(model + 64);
    dma = (nai_cmd_dma_1d_v2_t *)(model + 224);
    end = (nai_cmd_header_v2_t *)(model + 288);
    binding = (nai_binding_v1_t *)(model + 1344);

    header->magic = NAI_MODEL_MAGIC;
    header->abi_major = NAI_ABI_MAJOR;
    header->target_id = NAI_TARGET_ID;
    header->total_bytes = 1408;
    header->section_count = 5;
    header->section_table_off = 64;
    header->entry_command_off = 224;
    header->command_count = 1;
    header->required_tcdm_align = 32;
    header->output_count = 1;

    sections[0] = (nai_section_v1_t){NAI_SECTION_COMMANDS, 0, 224, 96, 32, 2, 0, 0};
    sections[1] = (nai_section_v1_t){NAI_SECTION_CONSTANTS, 0, 320, 1024, 32, 1, 0, 0};
    sections[2] = (nai_section_v1_t){NAI_SECTION_TENSORS, 0, 1344, 0, 32, 0, 0, 0};
    sections[3] = (nai_section_v1_t){NAI_SECTION_BINDINGS, 0, 1344, 64, 32, 1, 0, 0};
    sections[4] = (nai_section_v1_t){NAI_SECTION_QPARAMS, 0, 1408, 0, 32, 0, 0, 0};

    dma->header.type = NAI_CMD_DMA_1D;
    dma->header.size_bytes = sizeof(*dma);
    dma->source.region = NAI_REGION_MODEL_CONSTANTS;
    dma->destination.region = NAI_REGION_OUTPUT_BINDING;
    dma->length = 32;
    end->type = NAI_CMD_END;
    end->size_bytes = 32;

    binding->direction = NAI_BINDING_OUTPUT;
    binding->data_type = NAI_DTYPE_I8;
    binding->layout = NAI_LAYOUT_NHWC;
    binding->rank = 4;
    binding->dimensions[0] = 1;
    binding->dimensions[1] = 1;
    binding->dimensions[2] = 1;
    binding->dimensions[3] = 32;
    binding->byte_size = 32;

    for (uint32_t index = 0; index < 5u; index++) {
        sections[index].crc32 = nai_crc32(model + sections[index].offset, sections[index].size);
    }
}

int main(void)
{
    uint8_t model[1408];
    nai_model_view_v1_t view;
    nai_model_view_v1_t stream_view;
    nai_model_stream_storage_v1_t stream_storage;
    nai_binding_address_v1_t address = {NAI_BINDING_OUTPUT, 0, 0x80001000u, 32, 0};
    nai_resolver_v1_t resolver = {0x80000000u, 1408, &address, 1, 0x10100000u, 0x7f000u, 0, 0};
    mock_state_t state = {0};
    nai_runtime_ops_v2_t ops = {0};
    uint32_t completed;
    uint32_t failure;
    uint8_t scratch[64];
    uint8_t command_buffer[96];
    memory_reader_t memory = {model, sizeof(model), 0, 0};
    nai_model_reader_v1_t reader = {&memory, memory_read};
    uint8_t rq_model[1088] = {0};
    nai_model_header_v1_t rq_header = {0};
    nai_section_v1_t rq_commands = {NAI_SECTION_COMMANDS, 0, 0, 64, 32, 2, 0, 0};
    nai_section_v1_t rq_qparams = {NAI_SECTION_QPARAMS, 0, 64, 1024, 32, 32, 0, 0};
    nai_model_view_v1_t rq_view = {0};
    nai_resolver_v1_t rq_resolver = {0x80010000u, sizeof(rq_model), 0, 0,
        0x10100000u, 0x7f000u, 0, 0};
    nai_runtime_ops_v2_t rq_ops = {0};
    nai_cmd_rq_load_v2_t *rq_command = (nai_cmd_rq_load_v2_t *)rq_model;
    nai_cmd_control_v2_t *rq_end = (nai_cmd_control_v2_t *)(rq_model + 32);

    ops.context = &state;
    ops.dma_1d = mock_dma_1d;
    make_dma_model(model);
    assert(nai_model_open_v1(model, sizeof(model), NAI_TARGET_ID, &view) == NAI_LOADER_OK);
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops, &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u);
    assert(state.calls == 1u);
    assert(state.source == 0x80000140u);
    assert(state.destination == 0x80001000u);
    assert(state.length == 32u);

    state = (mock_state_t){0};
    assert(nai_model_open_stream_v1(&reader, sizeof(model), NAI_TARGET_ID,
        scratch, sizeof(scratch), &stream_storage, &stream_view) == NAI_LOADER_OK);
    assert(nai_cmd_dispatch_stream_v2(&stream_view, &resolver, &ops, &reader,
        command_buffer, sizeof(command_buffer), &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u);
    assert(state.calls == 1u);
    assert(state.source == 0x80000140u);
    assert(memory.reads > 20u);
    assert(memory.largest_read == 160u);

    ((nai_cmd_header_v2_t *)(model + 224))->type = 0xffffu;
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops, &completed, &failure) == NAI_DISPATCH_UNSUPPORTED);
    assert(completed == 0u);
    assert(failure == 224u);

    ((nai_cmd_header_v2_t *)(model + 224))->flags = NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE;
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops, &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u);

    rq_header.command_count = 1;
    rq_header.entry_command_off = 0;
    rq_header.total_bytes = sizeof(rq_model);
    rq_command->header.type = NAI_CMD_RQ_LOAD;
    rq_command->header.size_bytes = sizeof(*rq_command);
    rq_command->qparam_count = 32;
    rq_command->qparam_block = 9;
    rq_end->header.type = NAI_CMD_END;
    rq_end->header.size_bytes = sizeof(*rq_end);
    rq_view.model = rq_model;
    rq_view.model_bytes = sizeof(rq_model);
    rq_view.header = &rq_header;
    rq_view.commands = &rq_commands;
    rq_view.qparams = &rq_qparams;
    rq_ops.context = &state;
    rq_ops.rq_load = mock_rq_load;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&rq_view, &rq_resolver, &rq_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.qparam_address == 0x80010040u);
    assert(state.qparam_count == 32u && state.qparam_block == 9u);
    rq_command->qparam_count = 31;
    assert(nai_cmd_dispatch_v2(&rq_view, &rq_resolver, &rq_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    return 0;
}
