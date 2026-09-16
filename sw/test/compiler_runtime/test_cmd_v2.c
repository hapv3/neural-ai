#include "npu_cmd_desc_v2.h"

#include <assert.h>
#include <string.h>

#if defined(NAI_PMU_PROFILE) && NAI_PMU_PROFILE
static uint32_t g_pmu_begin_count;
static uint32_t g_pmu_end_count;
static uint32_t g_pmu_last_begin;
static uint32_t g_pmu_last_end;

void nai_pmu_command_begin(uint32_t command_id)
{
    g_pmu_begin_count++;
    g_pmu_last_begin = command_id;
}

void nai_pmu_command_end(uint32_t command_id)
{
    g_pmu_end_count++;
    g_pmu_last_end = command_id;
}
#endif

typedef struct {
    uint32_t calls;
    uint32_t source;
    uint32_t source2;
    uint32_t destination;
    uint32_t length;
    uint32_t direction;
    uint32_t mode;
    int32_t bias;
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
    uint32_t binary_rhs;
    uint32_t binary_lhs_shift;
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

static uint32_t mock_dma_wait(void *context, uint32_t direction)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->direction = direction;
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
    state->bias = command->bias;
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
    state->mode = command->header.flags;
    return 0u;
}

static uint32_t mock_afu_dfl16(void *context, const nai_cmd_afu_dfl16_v2_t *command,
                               uint32_t source, uint32_t destination, uint32_t scratch,
                               uint32_t exp_lut, uint32_t recip_lut)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = source;
    state->destination = destination;
    state->partial_sums = scratch;
    state->source2 = exp_lut;
    state->ofm = recip_lut;
    state->length = command->locations;
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

static uint32_t mock_linebuf_binary_job(
    void *context, const nai_cmd_linebuf_binary_v2_t *command)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->linebuf_rows = command->job.rows;
    state->linebuf_k_tiles = command->job.k_tiles;
    state->binary_rhs = command->binary.rhs_addr;
    state->binary_lhs_shift = command->binary.lhs_shift;
    state->mode = command->binary.mode;
    return 0u;
}

static uint32_t mock_systolic_wait(void *context)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    return 0u;
}

static uint32_t mock_copy_layout(void *context, const nai_cmd_copy_layout_v2_t *command,
                                 uint32_t source, uint32_t destination)
{
    mock_state_t *state = (mock_state_t *)context;
    state->calls++;
    state->source = source;
    state->destination = destination;
    state->mode = command->mode;
    state->channels = command->valid_channels;
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
    nai_cmd_dma_1d_v2_t *dma;
    uint32_t completed;
    uint32_t failure;
    uint8_t command_buffer[160];
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
    uint8_t gemm_model[2176] = {0};
    nai_model_header_v1_t gemm_header = {0};
    nai_section_v1_t gemm_commands = {NAI_SECTION_COMMANDS, 0, 0, 128, 32, 2, {0, 0}};
    nai_section_v1_t gemm_constants = {NAI_SECTION_CONSTANTS, 0, 128, 2048, 32, 1, {0, 0}};
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

    dma = (nai_cmd_dma_1d_v2_t *)(model + 224);
    dma->header.flags = NAI_CMD_FLAG_AFU_LUT_REUSE;
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dma->header.flags = 0u;

    dma->header.type = NAI_CMD_DMA_SUBMIT_1D;
    ops.dma_submit_1d = mock_dma_1d;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x80000140u && state.destination == 0x10100000u);
    dma->header.type = NAI_CMD_DMA_1D;

    state = (mock_state_t){0};
    assert(nai_model_open_stream_v1(&reader, sizeof(model), NAI_TARGET_ID,
        &stream_storage, &stream_view) == NAI_LOADER_OK);
    assert(nai_cmd_dispatch_stream_v2(&stream_view, &resolver, &ops, &reader,
        command_buffer, sizeof(command_buffer), &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u);
    assert(state.calls == 1u);
    assert(state.source == 0x80000140u);
    /* Stream dispatch prefetches each command once instead of reading its
       header and full record through separate model-reader transactions. */
    assert(memory.reads == 5u);
    assert(memory.largest_read == 160u);

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
    dma_2d->header.type = NAI_CMD_DMA_SUBMIT_2D;
    ops.dma_submit_2d = mock_dma_2d;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(state.calls == 1u && state.direction == NAI_DMA_EXTERNAL_TO_LOCAL);
    dma_2d->header.type = NAI_CMD_DMA_2D;
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
    dma_3d->header.type = NAI_CMD_DMA_SUBMIT_3D;
    ops.dma_submit_3d = mock_dma_3d;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(state.calls == 1u && state.direction == NAI_DMA_EXTERNAL_TO_LOCAL);
    dma_3d->header.type = NAI_CMD_DMA_3D;
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

    make_dma_model(model);
    {
        nai_model_header_v1_t *header = (nai_model_header_v1_t *)model;
        nai_section_v1_t *sections = (nai_section_v1_t *)(model + 64);
        nai_cmd_dma_wait_v2_t *wait = (nai_cmd_dma_wait_v2_t *)(model + 224);
        nai_cmd_control_v2_t *end = (nai_cmd_control_v2_t *)(model + 256);
        memset(model + 224, 0, 96);
        sections[0].size = 64;
        wait->header.type = NAI_CMD_DMA_WAIT;
        wait->header.size_bytes = sizeof(*wait);
        wait->direction = NAI_DMA_LOCAL_TO_EXTERNAL;
        end->header.type = NAI_CMD_END;
        end->header.size_bytes = sizeof(*end);
        header->command_count = 1;
        ops.dma_wait = mock_dma_wait;
        assert(nai_model_open_v1(model, sizeof(model), NAI_TARGET_ID, &view) == NAI_LOADER_OK);
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
            &completed, &failure) == NAI_DISPATCH_OK);
        assert(completed == 1u && state.calls == 1u);
        assert(state.direction == NAI_DMA_LOCAL_TO_EXTERNAL);
        memory = (memory_reader_t){model, sizeof(model), 0, 0};
        assert(nai_model_open_stream_v1(&reader, sizeof(model), NAI_TARGET_ID,
            &stream_storage, &stream_view) == NAI_LOADER_OK);
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_stream_v2(&stream_view, &resolver, &ops, &reader,
            command_buffer, sizeof(command_buffer), &completed, &failure) == NAI_DISPATCH_OK);
        assert(completed == 1u && state.calls == 1u);
        assert(state.direction == NAI_DMA_LOCAL_TO_EXTERNAL);
        wait->direction = NAI_DMA_LOCAL_TO_LOCAL;
        assert(nai_cmd_dispatch_v2(&view, &resolver, &ops,
            &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    }

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
    gemm_commands.size = 128;
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
    afu_lut->header.flags = NAI_CMD_FLAG_AFU_LUT_REUSE;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.mode == NAI_CMD_FLAG_AFU_LUT_REUSE);
    afu_lut->header.flags = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_stream_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &gemm_reader, command_buffer, sizeof(command_buffer), &completed, &failure) ==
        NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source2 == 0x80020060u && state.length == 64u);
    afu_lut->ofm.offset = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == state.destination);
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
    /* M may exceed the 256-row systolic limit; dispatch keeps the complete
       tensor references while the runtime callback stripes the operation. */
    pointwise->rows = 511u;
    pointwise->input_group_stride_bytes = 511u * 32u;
    pointwise->output_group_stride_bytes = 511u * 32u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u && state.rows == 511u);
    pointwise->rows = 2u;
    pointwise->input_group_stride_bytes = 64u;
    pointwise->output_group_stride_bytes = 64u;

    pointwise->ofm.offset = 0u;
    gemm_resolver.tcdm_scratch_base = 0xfffffff0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_REFERENCE);
    assert(completed == 0u && state.calls == 0u);
    pointwise->ofm.offset = 0x1000u;
    gemm_resolver.tcdm_scratch_base = 0x10100000u;
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
    linebuf->job.linebuf.c32_fast = 0u;

    linebuf->job.linebuf.block_valid_bytes = 16u;
    linebuf->job.linebuf.c32_group_stationary =
        SYSTOLIC_LINEBUF_SCHEDULE_GENERIC_LINEAR_K32;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    linebuf->job.linebuf.c32_group_stationary = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);

    memset(linebuf->reserved, 0, sizeof(linebuf->reserved));
    linebuf->header.type = NAI_CMD_LINEBUF_SUBMIT;
    linebuf->job.rows = 4u;
    linebuf->job.linebuf.spatial_m = 4u;
    linebuf->job.gemm.dim_m = 4u;
    linebuf->job.gemm.accum_en = 0u;
    linebuf->job.gemm.psum_row_stride_bytes = 0u;
    linebuf->job.linebuf.k_tiles = 9u;
    linebuf->job.linebuf.c32_group_stationary =
        SYSTOLIC_LINEBUF_SCHEDULE_GENERIC_LINEAR_K32;
    nai_cmd_control_v2_t *systolic_wait =
        (nai_cmd_control_v2_t *)(gemm_model + 160);
    linebuf_end = (nai_cmd_control_v2_t *)(gemm_model + 192);
    systolic_wait->header.type = NAI_CMD_SYSTOLIC_WAIT;
    systolic_wait->header.size_bytes = sizeof(*systolic_wait);
    linebuf_end->header.type = NAI_CMD_END;
    linebuf_end->header.size_bytes = sizeof(*linebuf_end);
    gemm_header.command_count = 2;
    gemm_commands.size = 224;
    gemm_commands.element_count = 3;
    gemm_ops.linebuf_submit = mock_linebuf_job;
    gemm_ops.systolic_wait = mock_systolic_wait;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 2u && state.calls == 2u);
    assert(state.linebuf_rows == 4u && state.linebuf_k_tiles == 9u);
    gemm_memory.data = gemm_model;
    gemm_memory.bytes = sizeof(gemm_model);
    gemm_memory.largest_read = 0u;
    gemm_memory.reads = 0u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_stream_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &gemm_reader, command_buffer, sizeof(command_buffer), &completed, &failure) ==
        NAI_DISPATCH_OK);
    assert(completed == 2u && state.calls == 2u);
    assert(state.linebuf_rows == 4u && state.linebuf_k_tiles == 9u);
    systolic_wait->reserved[0] = 1u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 1u && state.calls == 1u);

    {
        nai_linebuf_job_wire_v1_t binary_job = linebuf->job;
        uint8_t binary_command_buffer[sizeof(nai_cmd_linebuf_binary_v2_t)];
        nai_cmd_linebuf_binary_v2_t *binary;
        nai_cmd_control_v2_t *binary_end;

        memset(gemm_model, 0, sizeof(gemm_model));
        binary = (nai_cmd_linebuf_binary_v2_t *)gemm_model;
        binary_end = (nai_cmd_control_v2_t *)(gemm_model + sizeof(*binary));
        binary->header.type = NAI_CMD_LINEBUF_BINARY;
        binary->header.size_bytes = sizeof(*binary);
        binary->job = binary_job;
        binary->binary.rhs_addr = 0x4000u;
        binary->binary.rhs_row_stride_bytes = 64u;
        binary->binary.rhs_tile_cols = 2u;
        binary->binary.lhs_multiplier = 17;
        binary->binary.lhs_shift = 8u;
        binary->binary.rhs_multiplier = 19;
        binary->binary.rhs_shift = 9u;
        binary->binary.output_multiplier = 23;
        binary->binary.output_shift = 10u;
        binary->binary.lhs_zero_point = -7;
        binary->binary.rhs_zero_point = 5;
        binary->binary.output_zero_point = -3;
        binary->binary.clamp_min = -100;
        binary->binary.clamp_max = 99;
        binary->binary.double_round_shift = 1u;
        binary->binary.mode = SYSTOLIC_BINARY_SUB;
        binary_end->header.type = NAI_CMD_END;
        binary_end->header.size_bytes = sizeof(*binary_end);
        gemm_header.command_count = 1;
        gemm_header.total_bytes = sizeof(*binary) + sizeof(*binary_end);
        gemm_commands.size = gemm_header.total_bytes;
        gemm_commands.element_count = 2;
        gemm_ops.linebuf_binary_job = mock_linebuf_binary_job;
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
            &completed, &failure) == NAI_DISPATCH_OK);
        assert(completed == 1u && state.calls == 1u);
        assert(state.linebuf_rows == 4u && state.linebuf_k_tiles == 9u);
        assert(state.binary_rhs == 0x4000u && state.binary_lhs_shift == 8u);
        assert(state.mode == SYSTOLIC_BINARY_SUB);

        gemm_memory.data = gemm_model;
        gemm_memory.bytes = gemm_header.total_bytes;
        gemm_memory.largest_read = 0u;
        gemm_memory.reads = 0u;
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_stream_v2(&gemm_view, &gemm_resolver, &gemm_ops,
            &gemm_reader, binary_command_buffer, sizeof(binary_command_buffer),
            &completed, &failure) == NAI_DISPATCH_OK);
        assert(completed == 1u && state.calls == 1u);
        assert(gemm_memory.largest_read == sizeof(binary_command_buffer));

        binary->binary.lhs_shift = 64u;
        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
            &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
        assert(completed == 0u && state.calls == 0u);
    }

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
    assert(state.bias == 0);

    afu_binary->mode = NAI_AFU_BINARY_ADD_I8_BIAS;
    afu_binary->bias = 110;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.mode == NAI_AFU_BINARY_ADD_I8_BIAS && state.bias == 110);

    afu_binary->bias = 384;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    afu_binary->bias = 0;

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
    spatz_add->mode = NAI_SPATZ_BINARY_SUBTRACT;
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
    assert(spatz_add->mode == NAI_SPATZ_BINARY_SUBTRACT);

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
    spatz_add->mode = 2u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    spatz_add->mode = NAI_SPATZ_BINARY_SUBTRACT;
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

    upsample->channels = 128u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u && state.channels == 128u);
    upsample->channels = 256u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u && state.channels == 256u);
    upsample->channels = 64u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    upsample->channels = 32u;

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

    maxpool->channels = 128u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u && state.channels == 128u);
    maxpool->channels = 64u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    maxpool->channels = 32u;

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

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 128;
    gemm_commands.element_count = 2;
    nai_cmd_copy_layout_v2_t *head_pack = (nai_cmd_copy_layout_v2_t *)gemm_model;
    nai_cmd_control_v2_t *head_pack_end =
        (nai_cmd_control_v2_t *)(gemm_model + 96);
    head_pack->header.type = NAI_CMD_COPY_LAYOUT;
    head_pack->header.size_bytes = sizeof(*head_pack);
    head_pack->source.region = NAI_REGION_TCDM_SCRATCH;
    head_pack->destination.region = NAI_REGION_TCDM_SCRATCH;
    head_pack->destination.offset = 0x5000u;
    head_pack->mode = NAI_COPY_C32_TO_CHW;
    head_pack->source_layout = NAI_LAYOUT_C32_BLOCKED;
    head_pack->destination_layout = NAI_LAYOUT_NHWC;
    head_pack->data_type = NAI_DTYPE_I8;
    head_pack->dimensions[0] = 1u;
    head_pack->dimensions[1] = 10u;
    head_pack->dimensions[2] = 10u;
    head_pack->dimensions[3] = 144u;
    head_pack->valid_channels = 144u;
    head_pack->source_row_stride = 100u * 32u;
    head_pack->destination_row_stride = 100u;
    head_pack_end->header.type = NAI_CMD_END;
    head_pack_end->header.size_bytes = sizeof(*head_pack_end);
    gemm_ops.copy_layout = mock_copy_layout;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    assert(state.source == 0x10100000u && state.destination == 0x10105000u);
    assert(state.mode == NAI_COPY_C32_TO_CHW && state.channels == 144u);

    head_pack->dimensions[1] = 8u;
    head_pack->dimensions[2] = 8u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    assert(completed == 0u && state.calls == 0u);
    head_pack->dimensions[1] = 10u;
    head_pack->dimensions[2] = 10u;
    head_pack->destination.region = NAI_REGION_OUTPUT_BINDING;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    head_pack->destination.region = NAI_REGION_TCDM_SCRATCH;
    head_pack->destination.offset = 0x100u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    memset(gemm_model, 0, sizeof(gemm_model));
    gemm_header.command_count = 1;
    gemm_header.entry_command_off = 0;
    gemm_commands.size = 128;
    gemm_commands.element_count = 2;
    gemm_constants.offset = 128;
    gemm_constants.size = 2048;
    nai_cmd_afu_dfl16_v2_t *dfl16 = (nai_cmd_afu_dfl16_v2_t *)gemm_model;
    nai_cmd_control_v2_t *dfl16_end = (nai_cmd_control_v2_t *)(gemm_model + 96);
    dfl16->header.type = NAI_CMD_AFU_DFL16;
    dfl16->header.size_bytes = sizeof(*dfl16);
    dfl16->source.region = NAI_REGION_TCDM_SCRATCH;
    dfl16->destination.region = NAI_REGION_TCDM_SCRATCH;
    dfl16->destination.offset = 0x50000u;
    dfl16->scratch.region = NAI_REGION_TCDM_SCRATCH;
    dfl16->scratch.offset = 0x53000u;
    dfl16->exp_lut.region = NAI_REGION_MODEL_CONSTANTS;
    dfl16->recip_lut.region = NAI_REGION_MODEL_CONSTANTS;
    dfl16->recip_lut.offset = 0x400u;
    dfl16->locations = 2100u;
    dfl16->output_multiplier = 11381;
    dfl16->output_shift = 17u;
    dfl16->output_zero_point = -128;
    dfl16->clamp_min = -128;
    dfl16->clamp_max = 127;
    dfl16_end->header.type = NAI_CMD_END;
    dfl16_end->header.size_bytes = sizeof(*dfl16_end);
    gemm_ops.afu_dfl16 = mock_afu_dfl16;
    {
        const uint32_t selected_locations[] = {100u, 400u, 1600u, 2100u};
        for (uint32_t index = 0u;
             index < sizeof(selected_locations) / sizeof(selected_locations[0]); index++) {
            dfl16->locations = selected_locations[index];
            state = (mock_state_t){0};
            assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
                &completed, &failure) == NAI_DISPATCH_OK);
            assert(completed == 1u && state.calls == 1u &&
                   state.length == selected_locations[index]);
            assert(state.source == 0x10100000u && state.destination == 0x10150000u);
            assert(state.partial_sums == 0x10153000u && state.source2 == 0x80020080u);
            assert(state.ofm == 0x80020480u);
        }
    }

    dfl16->locations = 1600u;
    dfl16->source_layout = 1u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u && state.length == 1600u);
    dfl16->source_layout = 2u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->source_layout = 0u;
    dfl16->locations = 0u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->locations = 2101u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->locations = 2100u;
    dfl16->output_multiplier = 65536;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->output_multiplier = 11381;
    dfl16->output_shift = 16u;
    state = (mock_state_t){0};
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_OK);
    assert(completed == 1u && state.calls == 1u);
    dfl16->output_shift = 32u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->output_shift = 17u;
    dfl16->scratch.offset = 0x49000u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->scratch.offset = 0x53000u;
    dfl16->destination.region = NAI_REGION_MODEL_CONSTANTS;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->destination.region = NAI_REGION_TCDM_SCRATCH;
    dfl16->source.region = NAI_REGION_INPUT_BINDING;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->source.region = NAI_REGION_TCDM_SCRATCH;
    dfl16->recip_lut.offset = 0x420u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);
    dfl16->recip_lut.offset = 0x400u;
    dfl16->exp_lut.offset = 0xfffffc00u;
    assert(nai_cmd_dispatch_v2(&gemm_view, &gemm_resolver, &gemm_ops,
        &completed, &failure) == NAI_DISPATCH_BAD_COMMAND);

    {
        uint8_t affine_model[288] = {0};
        uint8_t affine_buffer[NAI_AFFINE_LOOP_MAX_RECORD_BYTES];
        nai_model_header_v1_t affine_header = {0};
        nai_section_v1_t affine_commands = {
            NAI_SECTION_COMMANDS, 0, 0, 160, 32, 4, {0, 0}};
        nai_section_v1_t affine_constants = {
            NAI_SECTION_CONSTANTS, 0, 160, 128, 32, 1, {0, 0}};
        nai_model_view_v1_t affine_view = {0};
        nai_resolver_v1_t affine_resolver = {
            0x80030000u, sizeof(affine_model), 0, 0,
            0x10100000u, 0x7f000u, 0, 0};
        nai_cmd_affine_loop_v2_t *loop =
            (nai_cmd_affine_loop_v2_t *)affine_model;
        nai_cmd_affine_patch_v2_t *patches =
            (nai_cmd_affine_patch_v2_t *)(affine_model + sizeof(*loop));
        nai_cmd_dma_1d_v2_t *body =
            (nai_cmd_dma_1d_v2_t *)(affine_model + 64);
        nai_cmd_control_v2_t *end =
            (nai_cmd_control_v2_t *)(affine_model + 128);
        memory_reader_t affine_memory = {
            affine_model, sizeof(affine_model), 0, 0};
        nai_model_reader_v1_t affine_reader = {
            &affine_memory, memory_read};

        affine_header.command_count = 4;
        affine_view.model = affine_model;
        affine_view.model_bytes = sizeof(affine_model);
        affine_view.header = &affine_header;
        affine_view.commands = &affine_commands;
        affine_view.constants = &affine_constants;

        loop->header.type = NAI_CMD_AFFINE_LOOP;
        loop->header.size_bytes = 64;
        loop->iteration_count = 4;
        loop->body_command_count = 1;
        loop->body_bytes = sizeof(*body);
        loop->patch_count = 2;
        patches[0] = (nai_cmd_affine_patch_v2_t){3, 1};
        patches[1] = (nai_cmd_affine_patch_v2_t){5, 32};
        body->header.type = NAI_CMD_DMA_1D;
        body->header.size_bytes = sizeof(*body);
        body->source.region = NAI_REGION_MODEL_CONSTANTS;
        body->destination.region = NAI_REGION_TCDM_SCRATCH;
        body->length = 32;
        body->direction = NAI_DMA_EXTERNAL_TO_LOCAL;
        end->header.type = NAI_CMD_END;
        end->header.size_bytes = sizeof(*end);

        state = (mock_state_t){0};
        ops.context = &state;
        assert(nai_cmd_dispatch_v2(&affine_view, &affine_resolver, &ops,
            &completed, &failure) == NAI_DISPATCH_OK);
        assert(completed == 4u && state.calls == 4u);
        assert(state.source == 0x80030100u);

        state = (mock_state_t){0};
        assert(nai_cmd_dispatch_stream_v2(&affine_view, &affine_resolver, &ops,
            &affine_reader, affine_buffer, sizeof(affine_buffer),
            &completed, &failure) == NAI_DISPATCH_OK);
        assert(completed == 4u && state.calls == 4u);
        assert(state.source == 0x80030100u);
        assert(affine_memory.reads == 2u);

        patches[0].body_word_offset = 0;
        assert(nai_cmd_dispatch_v2(&affine_view, &affine_resolver, &ops,
            &completed, &failure) == NAI_DISPATCH_BAD_STREAM);
    }
#if defined(NAI_PMU_PROFILE) && NAI_PMU_PROFILE
    assert(g_pmu_begin_count != 0u);
    assert(g_pmu_begin_count == g_pmu_end_count);
    assert(g_pmu_last_begin == g_pmu_last_end);
#endif
    return 0;
}
