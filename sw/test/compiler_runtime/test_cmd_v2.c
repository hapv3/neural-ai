#include "npu_cmd_desc_v2.h"

#include <assert.h>
#include <string.h>

typedef struct {
    uint32_t calls;
    uint32_t source;
    uint32_t source2;
    uint32_t destination;
    uint32_t length;
    uint32_t direction;
    uint32_t mode;
    uint32_t qparam_address;
    uint32_t qparam_count;
    uint32_t qparam_block;
    uint32_t partial_sums;
    uint32_t ofm;
    uint32_t rows;
    uint32_t input_groups;
    uint32_t output_groups;
    uint32_t input_h;
    uint32_t input_w;
    uint32_t output_h;
    uint32_t output_w;
    uint32_t channels;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t linebuf_rows;
    uint32_t linebuf_k_tiles;
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
    state->calls++;
    state->source = source;
    state->destination = destination;
    state->length = length;
    state->direction = direction;
    return 0u;
}

static uint32_t mock_dma_2d(void *context, uint32_t source, uint32_t destination,
                            uint32_t length, uint32_t source_stride,
                            uint32_t destination_stride, uint32_t repetitions,
                            uint32_t direction)
{
    (void)source_stride;
    (void)destination_stride;
    (void)repetitions;
    return mock_dma_1d(context, source, destination, length, direction);
}

static uint32_t mock_dma_3d(void *context, uint32_t source, uint32_t destination,
                            uint32_t length, uint32_t source_stride_2,
                            uint32_t destination_stride_2, uint32_t repetitions_2,
                            uint32_t source_stride_3,
                            uint32_t destination_stride_3, uint32_t repetitions_3,
                            uint32_t direction)
{
    (void)source_stride_2;
    (void)destination_stride_2;
    (void)repetitions_2;
    (void)source_stride_3;
    (void)destination_stride_3;
    (void)repetitions_3;
    return mock_dma_1d(context, source, destination, length, direction);
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

static uint32_t mock_gemm32(void *context, const nai_cmd_gemm32_v2_t *command,
                            uint32_t weights, uint32_t ifm,
                            uint32_t partial_sums, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    (void)weights;
    (void)ifm;
    state->calls++;
    state->partial_sums = partial_sums;
    state->ofm = ofm;
    state->rows = command->dim_m;
    return 0u;
}

static uint32_t mock_pointwise_c32(void *context, const nai_cmd_pointwise_c32_v2_t *command,
                                   uint32_t weights, uint32_t ifm,
                                   uint32_t partial_sums, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    (void)weights;
    (void)ifm;
    state->calls++;
    state->partial_sums = partial_sums;
    state->ofm = ofm;
    state->rows = command->rows;
    state->input_groups = command->input_c32_groups;
    state->output_groups = command->output_c32_groups;
    return 0u;
}

static uint32_t mock_depthwise_c32(void *context, const nai_cmd_depthwise_c32_v2_t *command,
                                   uint32_t weights, uint32_t ifm, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    (void)weights;
    (void)ifm;
    state->calls++;
    state->ofm = ofm;
    state->input_h = command->input_h;
    state->input_w = command->input_w;
    state->output_h = command->output_h;
    state->output_w = command->output_w;
    state->channels = command->channels;
    state->stride_h = command->stride_h;
    state->stride_w = command->stride_w;
    state->pad_h = command->pad_h;
    state->pad_w = command->pad_w;
    state->qparam_block = command->qparam_block;
    return 0u;
}

static uint32_t mock_afu_binary(void *context, const nai_cmd_afu_binary_v2_t *command,
                                uint32_t lhs, uint32_t rhs, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = lhs;
    state->source2 = rhs;
    state->destination = ofm;
    state->length = command->length;
    state->mode = command->mode;
    return 0u;
}

static uint32_t mock_spatz_add(void *context, const nai_cmd_spatz_add_v2_t *command,
                               uint32_t lhs, uint32_t rhs, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = lhs;
    state->source2 = rhs;
    state->destination = ofm;
    state->length = command->length;
    state->mode = command->double_round_shift;
    return 0u;
}

static uint32_t mock_afu_lut(void *context, const nai_cmd_afu_lut_v2_t *command,
                             uint32_t ifm, uint32_t ofm, uint32_t lut)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = ifm;
    state->destination = ofm;
    state->source2 = lut;
    state->length = command->length;
    return 0u;
}

static uint32_t mock_afu_global_avgpool(
    void *context, const nai_cmd_afu_global_avgpool_v2_t *command,
    uint32_t ifm, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = ifm;
    state->destination = ofm;
    state->input_h = command->input_h;
    state->input_w = command->input_w;
    state->channels = command->channels;
    return 0u;
}

static uint32_t mock_upsample_nearest(
    void *context, const nai_cmd_upsample_nearest_v2_t *command,
    uint32_t ifm, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = ifm;
    state->destination = ofm;
    state->input_h = command->input_h;
    state->input_w = command->input_w;
    state->channels = command->channels;
    state->stride_h = command->scale_h;
    state->stride_w = command->scale_w;
    return 0u;
}

static uint32_t mock_maxpool(
    void *context, const nai_cmd_maxpool_v2_t *command,
    uint32_t ifm, uint32_t ofm)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = ifm;
    state->destination = ofm;
    state->input_h = command->input_h;
    state->input_w = command->input_w;
    state->channels = command->channels;
    state->stride_h = command->stride_h;
    state->stride_w = command->stride_w;
    state->pad_h = command->pad_h;
    state->pad_w = command->pad_w;
    return 0u;
}

static uint32_t mock_linebuf_job(void *context, const nai_cmd_linebuf_job_v2_t *command)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->linebuf_rows = command->job.rows;
    state->linebuf_k_tiles = command->job.k_tiles;
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

    sections[0] = (nai_section_v1_t){NAI_SECTION_COMMANDS, 0, 224, 96, 32, 2, {0, 0}};
    sections[1] = (nai_section_v1_t){NAI_SECTION_CONSTANTS, 0, 320, 1024, 32, 1, {0, 0}};
    sections[2] = (nai_section_v1_t){NAI_SECTION_TENSORS, 0, 1344, 0, 32, 0, {0, 0}};
    sections[3] = (nai_section_v1_t){NAI_SECTION_BINDINGS, 0, 1344, 64, 32, 1, {0, 0}};
    sections[4] = (nai_section_v1_t){NAI_SECTION_QPARAMS, 0, 1408, 0, 32, 0, {0, 0}};

    dma->header.type = NAI_CMD_DMA_1D;
    dma->header.size_bytes = sizeof(*dma);
    dma->source.region = NAI_REGION_MODEL_CONSTANTS;
    dma->destination.region = NAI_REGION_TCDM_SCRATCH;
    dma->length = 32;
    dma->direction = NAI_DMA_EXTERNAL_TO_LOCAL;
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
    uint8_t command_buffer[96];
    memory_reader_t memory = {model, sizeof(model), 0, 0};
    nai_model_reader_v1_t reader = {&memory, memory_read};
    uint8_t rq_model[1088] = {0};
    nai_model_header_v1_t rq_header = {0};
    nai_section_v1_t rq_commands = {NAI_SECTION_COMMANDS, 0, 0, 64, 32, 2, {0, 0}};
    nai_section_v1_t rq_qparams = {NAI_SECTION_QPARAMS, 0, 64, 1024, 32, 32, {0, 0}};
    nai_model_view_v1_t rq_view = {0};
    nai_resolver_v1_t rq_resolver = {0x80010000u, sizeof(rq_model), 0, 0,
        0x10100000u, 0x7f000u, 0, 0};
    nai_runtime_ops_v2_t rq_ops = {0};
    nai_cmd_rq_load_v2_t *rq_command = (nai_cmd_rq_load_v2_t *)rq_model;
    nai_cmd_control_v2_t *rq_end = (nai_cmd_control_v2_t *)(rq_model + 32);
    uint8_t gemm_model[1152] = {0};
    nai_model_header_v1_t gemm_header = {0};
    nai_section_v1_t gemm_commands = {NAI_SECTION_COMMANDS, 0, 0, 128, 32, 2, {0, 0}};
    nai_section_v1_t gemm_constants = {NAI_SECTION_CONSTANTS, 0, 128, 1024, 32, 1, {0, 0}};
    nai_model_view_v1_t gemm_view = {0};
    nai_resolver_v1_t gemm_resolver = {0x80020000u, sizeof(gemm_model), 0, 0,
        0x10100000u, 0x7f000u, 0, 0};
    nai_runtime_ops_v2_t gemm_ops = {0};
    nai_cmd_gemm32_v2_t *gemm_command = (nai_cmd_gemm32_v2_t *)gemm_model;
    nai_cmd_control_v2_t *gemm_end = (nai_cmd_control_v2_t *)(gemm_model + 96);
    memory_reader_t gemm_memory = {gemm_model, sizeof(gemm_model), 0, 0};
    nai_model_reader_v1_t gemm_reader = {&gemm_memory, memory_read};

    ops.context = &state;
    ops.dma_1d = mock_dma_1d;
    make_dma_model(model);
    assert(nai_model_open_v1(model, sizeof(model), NAI_TARGET_ID, &view) == NAI_LOADER_OK);
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops, &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u);
    assert(state.calls == 1u);
    assert(state.source == 0x80000140u);
    assert(state.destination == 0x10100000u);
    assert(state.length == 32u);
    assert(state.direction == NAI_DMA_EXTERNAL_TO_LOCAL);

    state = (mock_state_t){0};
    assert(nai_model_open_stream_v1(&reader, sizeof(model), NAI_TARGET_ID,
        &stream_storage, &stream_view) == NAI_LOADER_OK);
    assert(nai_cmd_dispatch_stream_v2(&stream_view, &resolver, &ops, &reader,
        command_buffer, sizeof(command_buffer), &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u);
    assert(state.calls == 1u);
    assert(state.source == 0x80000140u);
    assert(memory.reads == 7u);
    assert(memory.largest_read == 160u);

    nai_cmd_dma_1d_v2_t *dma = (nai_cmd_dma_1d_v2_t *)(model + 224);
    dma->direction = NAI_DMA_LOCAL_TO_EXTERNAL;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    assert(nai_cmd_dispatch_stream_v2(&stream_view, &resolver, &ops, &reader,
        command_buffer, sizeof(command_buffer), &completed, &failure) ==
        NAI_DISPATCH_BAD_COMMAND);

    dma->destination.region = NAI_REGION_OUTPUT_BINDING;
    dma->direction = NAI_DMA_EXTERNAL_TO_LOCAL;
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    dma->source.region = NAI_REGION_TCDM_SCRATCH;
    dma->source.offset = 3u;
    dma->destination.region = NAI_REGION_OUTPUT_BINDING;
    dma->destination.offset = 1u;
    dma->length = 31u;
    dma->direction = NAI_DMA_LOCAL_TO_EXTERNAL;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(state.source == 0x10100003u && state.destination == 0x80001001u);
    assert(state.length == 31u);
    assert(state.direction == NAI_DMA_LOCAL_TO_EXTERNAL);

    dma->source.region = NAI_REGION_TCDM_SCRATCH;
    dma->source.offset = 3u;
    dma->destination.region = NAI_REGION_TCDM_SCRATCH;
    dma->destination.offset = 37u;
    dma->direction = NAI_DMA_LOCAL_TO_LOCAL;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(state.source == 0x10100003u && state.destination == 0x10100025u);
    assert(state.direction == NAI_DMA_LOCAL_TO_LOCAL);

    nai_cmd_dma_2d_v2_t *dma_2d = (nai_cmd_dma_2d_v2_t *)(model + 224);
    memset(dma_2d, 0, sizeof(*dma_2d));
    dma_2d->header.type = NAI_CMD_DMA_2D;
    dma_2d->header.size_bytes = sizeof(*dma_2d);
    dma_2d->source.region = NAI_REGION_MODEL_CONSTANTS;
    dma_2d->source.offset = 3u;
    dma_2d->destination.region = NAI_REGION_TCDM_SCRATCH;
    dma_2d->destination.offset = 0u;
    dma_2d->length = 3u;
    dma_2d->source_stride_2 = 3u;
    dma_2d->destination_stride_2 = 32u;
    dma_2d->repetitions_2 = 11u;
    dma_2d->direction = NAI_DMA_EXTERNAL_TO_LOCAL;
    ops.dma_2d = mock_dma_2d;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(state.calls == 1u && state.direction == NAI_DMA_EXTERNAL_TO_LOCAL);
    assert(state.source == 0x80000143u && state.destination == 0x10100000u);
    assert(state.length == 3u);
    dma_2d->direction = NAI_DMA_LOCAL_TO_EXTERNAL;
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    nai_cmd_dma_3d_v2_t *dma_3d = (nai_cmd_dma_3d_v2_t *)(model + 224);
    memset(dma_3d, 0, sizeof(*dma_3d));
    dma_3d->header.type = NAI_CMD_DMA_3D;
    dma_3d->header.size_bytes = sizeof(*dma_3d);
    dma_3d->source.region = NAI_REGION_MODEL_CONSTANTS;
    dma_3d->source.offset = 3u;
    dma_3d->destination.region = NAI_REGION_TCDM_SCRATCH;
    dma_3d->destination.offset = 0u;
    dma_3d->length = 31u;
    dma_3d->source_stride_2 = 31u;
    dma_3d->destination_stride_2 = 32u;
    dma_3d->repetitions_2 = 2u;
    dma_3d->source_stride_3 = 67u;
    dma_3d->destination_stride_3 = 96u;
    dma_3d->repetitions_3 = 2u;
    dma_3d->direction = NAI_DMA_EXTERNAL_TO_LOCAL;
    ops.dma_3d = mock_dma_3d;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(state.calls == 1u && state.direction == NAI_DMA_EXTERNAL_TO_LOCAL);
    assert(state.source == 0x80000143u && state.destination == 0x10100000u);
    assert(state.length == 31u);
    dma_3d->destination.region = NAI_REGION_OUTPUT_BINDING;
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

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

    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_header.total_bytes = sizeof(gemm_model);
    gemm_command->header.type = NAI_CMD_GEMM32_REQUANT;
    gemm_command->header.size_bytes = sizeof(*gemm_command);
    gemm_command->weights.region = NAI_REGION_MODEL_CONSTANTS;
    gemm_command->ifm.region = NAI_REGION_TCDM_SCRATCH;
    gemm_command->ofm.region = NAI_REGION_TCDM_SCRATCH;
    gemm_command->ofm.offset = 0x1000u;
    gemm_command->dim_m = 2;
    gemm_command->ofm_row_stride = 64;
    gemm_command->partial_sum_row_stride = 128;
    gemm_end->header.type = NAI_CMD_END;
    gemm_end->header.size_bytes = sizeof(*gemm_end);
    gemm_view.model = gemm_model;
    gemm_view.model_bytes = sizeof(gemm_model);
    gemm_view.header = &gemm_header;
    gemm_view.commands = &gemm_commands;
    gemm_view.constants = &gemm_constants;
    gemm_ops.context = &state;
    gemm_ops.gemm32 = mock_gemm32;
    const uint32_t valid_dim_m_values[] = {1u, 31u, 32u, 33u, 255u, 256u};
    for (uint32_t index = 0u;
         index < sizeof(valid_dim_m_values) / sizeof(valid_dim_m_values[0]);
         index++) {
        gemm_command->dim_m = valid_dim_m_values[index];
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
            &completed, &failure) == NAI_DISPATCH_OK);
        assert(state.calls == 1u && state.partial_sums == 0u);
        assert(state.ofm == 0x10101000u);
        assert(state.rows == valid_dim_m_values[index]);
    }

    for (uint32_t invalid_dim_m = 257u; invalid_dim_m <= 511u; invalid_dim_m += 254u) {
        gemm_command->dim_m = invalid_dim_m;
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
            &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
        assert(completed == 0u && state.calls == 0u);
    }
    gemm_command->dim_m = 2u;

    gemm_command->partial_sums.region = NAI_REGION_TCDM_SCRATCH;
    gemm_command->partial_sums.offset = 0x2000u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(state.partial_sums == 0x10102000u);

    gemm_command->partial_sums.region = 0u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 96;
    gemm_commands.element_count = 2;
    gemm_constants.offset = 96;
    gemm_constants.size = 256;
    nai_cmd_afu_lut_v2_t *afu_lut = (nai_cmd_afu_lut_v2_t *)gemm_model;
    nai_cmd_control_v2_t *afu_lut_end = (nai_cmd_control_v2_t *)(gemm_model + 64);
    afu_lut->header.type = NAI_CMD_AFU_LUT;
    afu_lut->header.size_bytes = sizeof(*afu_lut);
    afu_lut->ifm.region = NAI_REGION_TCDM_SCRATCH;
    afu_lut->ofm.region = NAI_REGION_TCDM_SCRATCH;
    afu_lut->ofm.offset = 0x100u;
    afu_lut->lut.region = NAI_REGION_MODEL_CONSTANTS;
    afu_lut->length = 64u;
    afu_lut_end->header.type = NAI_CMD_END;
    afu_lut_end->header.size_bytes = sizeof(*afu_lut_end);
    gemm_ops.afu_lut = mock_afu_lut;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.destination == 0x10100100u);
    assert(state.source2 == 0x80020060u && state.length == 64u);
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_stream_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &gemm_reader, command_buffer, sizeof(command_buffer), &completed, &failure) ==
        NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source2 == 0x80020060u && state.length == 64u);
    afu_lut->ofm.offset = 0x20u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    afu_lut->ofm.offset = 0x100u;
    afu_lut->reserved[0] = 1u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    gemm_constants.offset = 128;
    gemm_constants.size = 1024;
    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_header.total_bytes = sizeof(gemm_model);
    gemm_commands.size = 128;
    gemm_commands.element_count = 2;
    nai_cmd_pointwise_c32_v2_t *pointwise = (nai_cmd_pointwise_c32_v2_t *)gemm_model;
    nai_cmd_control_v2_t *pointwise_end = (nai_cmd_control_v2_t *)(gemm_model + 96);
    pointwise->header.type = NAI_CMD_POINTWISE_C32;
    pointwise->header.size_bytes = sizeof(*pointwise);
    pointwise->weights.region = NAI_REGION_MODEL_CONSTANTS;
    pointwise->ifm.region = NAI_REGION_TCDM_SCRATCH;
    pointwise->ofm.region = NAI_REGION_TCDM_SCRATCH;
    pointwise->ofm.offset = 0x1000u;
    pointwise->rows = 2u;
    pointwise->input_c32_groups = 1u;
    pointwise->output_c32_groups = 1u;
    pointwise->input_group_stride_bytes = 64u;
    pointwise->output_group_stride_bytes = 64u;
    pointwise_end->header.type = NAI_CMD_END;
    pointwise_end->header.size_bytes = sizeof(*pointwise_end);
    gemm_view.header = &gemm_header;
    gemm_view.commands = &gemm_commands;
    gemm_view.constants = &gemm_constants;
    gemm_ops.pointwise_c32 = mock_pointwise_c32;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.rows == 2u && state.input_groups == 1u && state.output_groups == 1u);
    assert(state.partial_sums == 0u && state.ofm == 0x10101000u);
    for (uint32_t invalid_rows = 257u; invalid_rows <= 511u; invalid_rows += 254u) {
        pointwise->rows = invalid_rows;
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
            &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
        assert(completed == 0u && state.calls == 0u);
    }
    pointwise->rows = 2u;
    pointwise->input_c32_groups = 2u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    pointwise->output_c32_groups = 2u;
    pointwise->input_c32_groups = 1u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    pointwise->output_c32_groups = 1u;

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_header.total_bytes = sizeof(gemm_model);
    gemm_commands.size = 128;
    gemm_commands.element_count = 2;
    nai_cmd_depthwise_c32_v2_t *depthwise = (nai_cmd_depthwise_c32_v2_t *)gemm_model;
    nai_cmd_control_v2_t *depthwise_end = (nai_cmd_control_v2_t *)(gemm_model + 96);
    depthwise->header.type = NAI_CMD_DEPTHWISE_C32;
    depthwise->header.size_bytes = sizeof(*depthwise);
    depthwise->weights.region = NAI_REGION_MODEL_CONSTANTS;
    depthwise->ifm.region = NAI_REGION_TCDM_SCRATCH;
    depthwise->ofm.region = NAI_REGION_TCDM_SCRATCH;
    depthwise->ofm.offset = 0x1000u;
    depthwise->input_h = 4u;
    depthwise->input_w = 4u;
    depthwise->output_h = 2u;
    depthwise->output_w = 2u;
    depthwise->channels = 32u;
    depthwise->stride_h = 2u;
    depthwise->stride_w = 2u;
    depthwise->pad_h = 0u;
    depthwise->pad_w = 0u;
    depthwise->qparam_block = 3u;
    depthwise_end->header.type = NAI_CMD_END;
    depthwise_end->header.size_bytes = sizeof(*depthwise_end);
    gemm_ops.depthwise_c32 = mock_depthwise_c32;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.ofm == 0x10101000u);
    assert(state.input_h == 4u && state.input_w == 4u);
    assert(state.output_h == 2u && state.output_w == 2u);
    assert(state.channels == 32u && state.stride_h == 2u && state.stride_w == 2u);
    assert(state.pad_h == 0u && state.pad_w == 0u && state.qparam_block == 3u);
    depthwise->input_h = 5u;
    depthwise->input_w = 5u;
    depthwise->output_h = 3u;
    depthwise->output_w = 3u;
    depthwise->pad_h = 1u;
    depthwise->pad_w = 1u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.input_h == 5u && state.input_w == 5u);
    assert(state.output_h == 3u && state.output_w == 3u);
    assert(state.pad_h == 1u && state.pad_w == 1u);
    depthwise->input_h = 10u;
    depthwise->input_w = 10u;
    depthwise->output_h = 8u;
    depthwise->output_w = 8u;
    depthwise->stride_h = 1u;
    depthwise->stride_w = 1u;
    depthwise->pad_h = 0u;
    depthwise->pad_w = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.input_h == 10u && state.input_w == 10u);
    assert(state.output_h == 8u && state.output_w == 8u);
    assert(state.stride_h == 1u && state.stride_w == 1u);
    assert(state.pad_h == 0u && state.pad_w == 0u);
    depthwise->channels = 33u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    depthwise->channels = 32u;
    depthwise->output_w = 4u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 192;
    gemm_commands.element_count = 2;
    nai_cmd_linebuf_job_v2_t *linebuf = (nai_cmd_linebuf_job_v2_t *)gemm_model;
    nai_cmd_control_v2_t *linebuf_end = (nai_cmd_control_v2_t *)(gemm_model + 160);
    linebuf->header.type = NAI_CMD_LINEBUF_JOB;
    linebuf->header.size_bytes = sizeof(*linebuf);
    linebuf->job.rows = 4u;
    linebuf->job.k_tiles = 9u;
    linebuf->job.linebuf.input_h = 3u;
    linebuf->job.linebuf.input_w = 3u;
    linebuf->job.linebuf.input_c = 32u;
    linebuf->job.linebuf.output_w = 2u;
    linebuf->job.linebuf.stride_h = 1u;
    linebuf->job.linebuf.stride_w = 1u;
    linebuf->job.linebuf.pad_h = 1u;
    linebuf->job.linebuf.pad_w = 1u;
    linebuf->job.linebuf.row_stride_bytes = 96u;
    linebuf->job.linebuf.pixel_stride_bytes = 32u;
    linebuf->job.linebuf.ow_step_bytes = 32u;
    linebuf->job.linebuf.oh_step_bytes = 96u;
    linebuf->job.linebuf.kernel_h = 3u;
    linebuf->job.linebuf.kernel_w = 3u;
    linebuf->job.linebuf.block_valid_bytes = 32u;
    linebuf->job.linebuf.k_tiles = 9u;
    linebuf->job.linebuf.spatial_m = 4u;
    linebuf->job.gemm.dim_m = 4u;
    linebuf->job.gemm.ofm_row_stride_bytes = 64u;
    linebuf->job.gemm.ofm_tile_cols = 2u;
    linebuf_end->header.type = NAI_CMD_END;
    linebuf_end->header.size_bytes = sizeof(*linebuf_end);
    gemm_ops.linebuf_job = mock_linebuf_job;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.linebuf_rows == 4u && state.linebuf_k_tiles == 9u);
    linebuf->job.rows = 1024u;
    linebuf->job.linebuf.spatial_m = 1024u;
    linebuf->job.gemm.dim_m = 1024u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u && state.linebuf_rows == 1024u);
    linebuf->job.rows = 1025u;
    linebuf->job.linebuf.spatial_m = 1025u;
    linebuf->job.gemm.dim_m = 1025u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    linebuf->job.rows = 257u;
    linebuf->job.linebuf.spatial_m = 257u;
    linebuf->job.gemm.dim_m = 257u;
    linebuf->job.gemm.accum_en = 3u;
    linebuf->job.gemm.psum_row_stride_bytes = 256u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    linebuf->job.rows = 4u;
    linebuf->job.linebuf.spatial_m = 4u;
    linebuf->job.gemm.dim_m = 4u;
    linebuf->job.gemm.accum_en = 0u;
    linebuf->job.gemm.psum_row_stride_bytes = 0u;
    linebuf->reserved[0] = 1u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    linebuf->reserved[0] = 0u;
    linebuf->job.linebuf.k_tiles = 8u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    linebuf->job.linebuf.k_tiles = 9u;
    linebuf->job.linebuf.c32_fast = 2u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    linebuf->job.linebuf.c32_fast = 1u;
    linebuf->job.linebuf.coalesce = 1u;
    linebuf->job.linebuf.kgen = 1u;
    linebuf->job.linebuf.c32_group_stationary = 1u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    linebuf->job.gemm.accum_en = 3u;
    linebuf->job.gemm.psum_row_stride_bytes = 256u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    linebuf->job.gemm.accum_en = 4u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    linebuf->job.gemm.accum_en = 0u;
    linebuf->job.gemm.psum_row_stride_bytes = 0u;
    linebuf->job.linebuf.c32_group_stationary = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 96;
    gemm_commands.element_count = 2;
    nai_cmd_afu_binary_v2_t *afu_binary = (nai_cmd_afu_binary_v2_t *)gemm_model;
    nai_cmd_control_v2_t *afu_binary_end = (nai_cmd_control_v2_t *)(gemm_model + 64);
    afu_binary->header.type = NAI_CMD_AFU_BINARY;
    afu_binary->header.size_bytes = sizeof(*afu_binary);
    afu_binary->lhs.region = NAI_REGION_TCDM_SCRATCH;
    afu_binary->rhs.region = NAI_REGION_TCDM_SCRATCH;
    afu_binary->rhs.offset = 0x100u;
    afu_binary->ofm.region = NAI_REGION_TCDM_SCRATCH;
    afu_binary->ofm.offset = 0x200u;
    afu_binary->length = 64u;
    afu_binary->mode = NAI_AFU_BINARY_ADD_I8;
    afu_binary_end->header.type = NAI_CMD_END;
    afu_binary_end->header.size_bytes = sizeof(*afu_binary_end);
    gemm_ops.afu_binary = mock_afu_binary;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.source2 == 0x10100100u);
    assert(state.destination == 0x10100200u && state.length == 64u);
    assert(state.mode == NAI_AFU_BINARY_ADD_I8);

    afu_binary->ofm.offset = 0x120u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    afu_binary->ofm.offset = 0x200u;
    afu_binary->mode = 0u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 128;
    gemm_commands.element_count = 2;
    nai_cmd_spatz_add_v2_t *spatz_add = (nai_cmd_spatz_add_v2_t *)gemm_model;
    nai_cmd_control_v2_t *spatz_add_end = (nai_cmd_control_v2_t *)(gemm_model + 96);
    spatz_add->header.type = NAI_CMD_SPATZ_ADD;
    spatz_add->header.size_bytes = sizeof(*spatz_add);
    spatz_add->lhs.region = NAI_REGION_TCDM_SCRATCH;
    spatz_add->rhs.region = NAI_REGION_TCDM_SCRATCH;
    spatz_add->rhs.offset = 0x100u;
    spatz_add->ofm.region = NAI_REGION_TCDM_SCRATCH;
    spatz_add->ofm.offset = 0x200u;
    spatz_add->length = 64u;
    spatz_add->lhs_scale = 0x60000000;
    spatz_add->lhs_shift = 20u;
    spatz_add->rhs_scale = 0x40000000;
    spatz_add->rhs_shift = 20u;
    spatz_add->output_scale = 0x40000000;
    spatz_add->output_shift = 41u;
    spatz_add->lhs_zero_point = -3;
    spatz_add->rhs_zero_point = 5;
    spatz_add->output_zero_point = 7;
    spatz_add->clamp_min = -100;
    spatz_add->clamp_max = 100;
    spatz_add->double_round_shift = 20u;
    spatz_add_end->header.type = NAI_CMD_END;
    spatz_add_end->header.size_bytes = sizeof(*spatz_add_end);
    gemm_ops.spatz_add = mock_spatz_add;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.source2 == 0x10100100u);
    assert(state.destination == 0x10100200u && state.length == 64u);
    assert(state.mode == 20u);

    gemm_memory.data = gemm_model;
    gemm_memory.bytes = sizeof(gemm_model);
    gemm_memory.largest_read = 0u;
    gemm_memory.reads = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_stream_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &gemm_reader, command_buffer, sizeof(command_buffer), &completed, &failure) ==
        NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.source2 == 0x10100100u);
    assert(state.destination == 0x10100200u && state.length == 64u);
    assert(state.mode == 20u);

    spatz_add->ofm.offset = 0x120u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    spatz_add->ofm.offset = 0x200u;
    spatz_add->lhs_scale = 0;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    spatz_add->lhs_scale = 0x60000000;
    spatz_add->output_shift = 64u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 96;
    gemm_commands.element_count = 2;
    nai_cmd_afu_global_avgpool_v2_t *global_avgpool =
        (nai_cmd_afu_global_avgpool_v2_t *)gemm_model;
    nai_cmd_control_v2_t *global_avgpool_end =
        (nai_cmd_control_v2_t *)(gemm_model + 64);
    global_avgpool->header.type = NAI_CMD_AFU_GLOBAL_AVGPOOL;
    global_avgpool->header.size_bytes = sizeof(*global_avgpool);
    global_avgpool->ifm.region = NAI_REGION_TCDM_SCRATCH;
    global_avgpool->ofm.region = NAI_REGION_TCDM_SCRATCH;
    global_avgpool->ofm.offset = 0x200u;
    global_avgpool->input_h = 2u;
    global_avgpool->input_w = 3u;
    global_avgpool->channels = 33u;
    global_avgpool_end->header.type = NAI_CMD_END;
    global_avgpool_end->header.size_bytes = sizeof(*global_avgpool_end);
    gemm_ops.afu_global_avgpool = mock_afu_global_avgpool;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.destination == 0x10100200u);
    assert(state.input_h == 2u && state.input_w == 3u && state.channels == 33u);

    global_avgpool->ofm.offset = 0x20u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    global_avgpool->ofm.offset = 0x100u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    global_avgpool->ofm.offset = 0x200u;
    global_avgpool->input_h = 0u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 96;
    gemm_commands.element_count = 2;
    nai_cmd_upsample_nearest_v2_t *upsample =
        (nai_cmd_upsample_nearest_v2_t *)gemm_model;
    nai_cmd_control_v2_t *upsample_end =
        (nai_cmd_control_v2_t *)(gemm_model + 64);
    upsample->header.type = NAI_CMD_UPSAMPLE_NEAREST;
    upsample->header.size_bytes = sizeof(*upsample);
    upsample->ifm.region = NAI_REGION_TCDM_SCRATCH;
    upsample->ofm.region = NAI_REGION_TCDM_SCRATCH;
    upsample->ofm.offset = 0x1000u;
    upsample->input_h = 2u;
    upsample->input_w = 3u;
    upsample->channels = 32u;
    upsample->scale_h = 2u;
    upsample->scale_w = 2u;
    upsample_end->header.type = NAI_CMD_END;
    upsample_end->header.size_bytes = sizeof(*upsample_end);
    gemm_ops.upsample_nearest = mock_upsample_nearest;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.destination == 0x10101000u);
    assert(state.input_h == 2u && state.input_w == 3u && state.channels == 32u);
    assert(state.stride_h == 2u && state.stride_w == 2u);

    gemm_memory.data = gemm_model;
    gemm_memory.bytes = sizeof(gemm_model);
    gemm_memory.largest_read = 0u;
    gemm_memory.reads = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_stream_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &gemm_reader, command_buffer, sizeof(command_buffer), &completed, &failure) ==
        NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);

    upsample->ofm.offset = 0x80u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    upsample->ofm.offset = 0x1000u;
    upsample->scale_w = 3u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 128;
    gemm_commands.element_count = 2;
    nai_cmd_maxpool_v2_t *maxpool = (nai_cmd_maxpool_v2_t *)gemm_model;
    nai_cmd_control_v2_t *maxpool_end =
        (nai_cmd_control_v2_t *)(gemm_model + 96);
    maxpool->header.type = NAI_CMD_MAXPOOL;
    maxpool->header.size_bytes = sizeof(*maxpool);
    maxpool->ifm.region = NAI_REGION_TCDM_SCRATCH;
    maxpool->ofm.region = NAI_REGION_TCDM_SCRATCH;
    maxpool->ofm.offset = 0x1000u;
    maxpool->input_h = 4u;
    maxpool->input_w = 4u;
    maxpool->channels = 32u;
    maxpool->kernel_h = 5u;
    maxpool->kernel_w = 5u;
    maxpool->stride_h = 1u;
    maxpool->stride_w = 1u;
    maxpool->pad_h = 2u;
    maxpool->pad_w = 2u;
    maxpool_end->header.type = NAI_CMD_END;
    maxpool_end->header.size_bytes = sizeof(*maxpool_end);
    gemm_ops.maxpool = mock_maxpool;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.destination == 0x10101000u);
    assert(state.input_h == 4u && state.input_w == 4u && state.channels == 32u);
    assert(state.stride_h == 1u && state.stride_w == 1u);
    assert(state.pad_h == 2u && state.pad_w == 2u);

    gemm_memory.largest_read = 0u;
    gemm_memory.reads = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_stream_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &gemm_reader, command_buffer, sizeof(command_buffer), &completed, &failure) ==
        NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);

    maxpool->ofm.offset = 0x100u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    maxpool->ofm.offset = 0x1000u;
    maxpool->kernel_w = 3u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    return 0;
}
