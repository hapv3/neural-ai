#include "npu_model_runtime.h"

#include "idma_mm_utils.h"
#include "npu_cmd_desc.h"
#include "npu_cmd_desc_v2.h"
#include "npu_memory_map.h"
#include "npu_model_loader.h"
#include "npu_quant_buffer.h"
#include "npu_runtime_ops.h"

typedef struct {
    uint32_t base;
    uint32_t bytes;
    uint32_t staging_base;
    uint32_t staging_bytes;
} l2_reader_context_t;

static nai_model_stream_storage_v1_t g_model_storage;
static nai_binding_address_v1_t g_binding_addresses[NAI_MAX_PUBLIC_BINDINGS_V1];
/* The stream reader must accommodate the largest v2 record.  LINEBUF_JOB is
   160 bytes, larger than the 96-byte GEMM/pointwise/depthwise records. */
static uint8_t g_command_buffer[sizeof(nai_cmd_linebuf_job_v2_t)];

static uint32_t l2_read(void *context_pointer, uint32_t offset, void *destination, uint32_t bytes)
{
    l2_reader_context_t *context = (l2_reader_context_t *)context_pointer;
    uint8_t *output = (uint8_t *)destination;
    volatile const uint8_t *staging = (volatile const uint8_t *)(unsigned long)context->staging_base;

    if (bytes == 0u) return 0u;
    if (offset > context->bytes || bytes > context->bytes - offset || bytes > context->staging_bytes ||
        context->base > 0xffffffffu - offset) return 1u;
    if (!idma_memcpy_blocking(context->base + offset, context->staging_base, bytes)) return 1u;
    for (uint32_t index = 0; index < bytes; index++) output[index] = staging[index];
    return 0u;
}

static uint32_t all_zero(const uint32_t *words, uint32_t count)
{
    for (uint32_t index = 0; index < count; index++) {
        if (words[index] != 0u) return 0u;
    }
    return 1u;
}

static uint32_t fail(uint32_t code, uint32_t pointer, uint32_t completed)
{
    REG_WRITE(NPU_CMD_FAIL_CODE, code);
    REG_WRITE(NPU_CMD_FAIL_PTR, pointer);
    REG_WRITE(NPU_CMD_DONE_COUNT, completed);
    REG_WRITE(NPU_CMD_STATUS, NPU_CMD_STATUS_FAIL);
    return code;
}

static uint32_t validate_invocation(const nai_invocation_v1_t *invocation,
                                    uint32_t invocation_bytes)
{
    if (invocation->magic != NAI_INVOCATION_MAGIC || invocation->abi_major != NAI_ABI_MAJOR ||
        invocation->abi_minor > NAI_ABI_MINOR || invocation->total_bytes != sizeof(*invocation) ||
        invocation->total_bytes > invocation_bytes || invocation->flags != 0u ||
        !all_zero(invocation->reserved, 8u)) return 0u;
    if ((invocation->model_base & 31u) != 0u || (invocation->model_bytes & 31u) != 0u ||
        invocation->model_bytes < sizeof(nai_model_header_v1_t) ||
        (invocation->binding_table_base & 31u) != 0u ||
        invocation->binding_count > NAI_MAX_PUBLIC_BINDINGS_V1) return 0u;
    if ((invocation->model_bytes != 0u && invocation->model_base > 0xffffffffu - invocation->model_bytes) ||
        (invocation->binding_count != 0u && invocation->binding_table_base >
            0xffffffffu - invocation->binding_count * sizeof(nai_binding_address_v1_t))) return 0u;
    return 1u;
}

static uint32_t validate_runtime_bindings(const nai_model_view_v1_t *view,
                                          const nai_invocation_v1_t *invocation)
{
    uint32_t public_count = view->header->input_count + view->header->output_count;
    if (invocation->binding_count < public_count) return 0u;

    for (uint32_t index = 0; index < invocation->binding_count; index++) {
        const nai_binding_address_v1_t *address = &g_binding_addresses[index];
        uint32_t required_alignment = NAI_ALIGNMENT_BYTES;
        /* Public NHWC is a compact byte-addressed contract.  Native command
           references still request 32-byte alignment when they need it; the
           binding table itself must not reject a valid compact base. */
        if (address->direction == NAI_BINDING_INPUT || address->direction == NAI_BINDING_OUTPUT) {
            for (uint32_t public_index = 0u; public_index < public_count; public_index++) {
                const nai_binding_v1_t *binding = &view->public_bindings[public_index];
                if (binding->direction == address->direction && binding->index == address->index &&
                    binding->layout == NAI_LAYOUT_NHWC) {
                    required_alignment = 1u;
                    break;
                }
            }
        }
        if (address->flags != 0u || (address->base % required_alignment) != 0u || address->byte_size == 0u ||
            address->base > 0xffffffffu - address->byte_size) return 0u;
        if (address->direction != NAI_BINDING_INPUT && address->direction != NAI_BINDING_OUTPUT &&
            address->direction != NAI_BINDING_L2_TEMPORARY) return 0u;
        for (uint32_t previous = 0; previous < index; previous++) {
            if (g_binding_addresses[previous].direction == address->direction &&
                g_binding_addresses[previous].index == address->index) return 0u;
        }
    }

    for (uint32_t model_index = 0; model_index < public_count; model_index++) {
        const nai_binding_v1_t *binding = &view->public_bindings[model_index];
        uint32_t found = 0u;
        for (uint32_t runtime_index = 0; runtime_index < invocation->binding_count; runtime_index++) {
            const nai_binding_address_v1_t *address = &g_binding_addresses[runtime_index];
            if (address->direction == binding->direction && address->index == binding->index) {
                if (address->byte_size < binding->byte_size) return 0u;
                found = 1u;
                break;
            }
        }
        if (!found) return 0u;
    }
    return 1u;
}

uint32_t nai_runtime_dispatch_from_ctrl(uint32_t invocation_base,
                                        uint32_t invocation_bytes,
                                        uint32_t staging_base,
                                        uint32_t staging_bytes)
{
    nai_invocation_v1_t invocation;
    nai_model_view_v1_t view;
    nai_resolver_v1_t resolver;
    l2_reader_context_t invocation_context = {invocation_base, invocation_bytes, staging_base, staging_bytes};
    nai_model_reader_v1_t invocation_reader = {&invocation_context, l2_read};
    l2_reader_context_t model_context;
    nai_model_reader_v1_t model_reader;
    uint32_t completed = 0u;
    uint32_t failure_offset = 0u;

    if (staging_base != NPU_CMD_TCDM_BASE || staging_bytes < NPU_CMD_TCDM_SIZE ||
        invocation_reader.read(invocation_reader.context, 0u, &invocation, sizeof(invocation)) != 0u ||
        !validate_invocation(&invocation, invocation_bytes))
        return fail(NPU_CMD_FAIL_BAD_INVOCATION, invocation_base, 0u);

    model_context = (l2_reader_context_t){invocation.model_base, invocation.model_bytes,
        staging_base, NPU_CMD_TCDM_SIZE};
    model_reader = (nai_model_reader_v1_t){&model_context, l2_read};
    if (nai_model_open_stream_v1(&model_reader, invocation.model_bytes, NAI_TARGET_ID,
        &g_model_storage, &view) != NAI_LOADER_OK)
        return fail(NPU_CMD_FAIL_BAD_MODEL, invocation.model_base, 0u);

    if (invocation.binding_count != 0u) {
        l2_reader_context_t binding_context = {invocation.binding_table_base,
            invocation.binding_count * sizeof(nai_binding_address_v1_t), staging_base, NPU_CMD_TCDM_SIZE};
        nai_model_reader_v1_t binding_reader = {&binding_context, l2_read};
        if (binding_reader.read(binding_reader.context, 0u, g_binding_addresses,
            binding_context.bytes) != 0u) return fail(NPU_CMD_FAIL_BAD_BINDING,
                invocation.binding_table_base, 0u);
    }
    if (!validate_runtime_bindings(&view, &invocation))
        return fail(NPU_CMD_FAIL_BAD_BINDING, invocation.binding_table_base, 0u);

    resolver = (nai_resolver_v1_t){invocation.model_base, invocation.model_bytes,
        g_binding_addresses, invocation.binding_count, NPU_TCDM_BASE,
        view.header->required_tcdm_bytes, 0u, 0u};
    nai_quant_buffer_reset_v1();
    REG_WRITE(NPU_CMD_STATUS, NPU_CMD_STATUS_RUNNING);
    nai_dispatch_status_v2_t status = nai_cmd_dispatch_stream_v2(&view, &resolver,
        nai_default_runtime_ops_v2(), &model_reader, g_command_buffer, sizeof(g_command_buffer),
        &completed, &failure_offset);
    if (status != NAI_DISPATCH_OK) return fail(NPU_CMD_FAIL_V2_DISPATCH + (uint32_t)status,
        invocation.model_base + failure_offset, completed);

    REG_WRITE(NPU_CMD_DONE_COUNT, completed);
    REG_WRITE(NPU_CMD_FAIL_CODE, NPU_CMD_FAIL_NONE);
    REG_WRITE(NPU_CMD_FAIL_PTR, 0u);
    REG_WRITE(NPU_CMD_STATUS, NPU_CMD_STATUS_PASS);
    return NPU_CMD_FAIL_NONE;
}
