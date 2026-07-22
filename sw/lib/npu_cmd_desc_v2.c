#include "npu_cmd_desc_v2.h"

static uint32_t all_zero(const uint32_t *words, uint32_t count)
{
    for (uint32_t index = 0; index < count; index++) {
        if (words[index] != 0u) return 0u;
    }
    return 1u;
}

static uint32_t valid_range(uint32_t offset, uint32_t size, uint32_t total)
{
    return offset <= total && size <= total - offset;
}

static nai_dispatch_status_v2_t resolve(const nai_model_view_v1_t *view,
                                        const nai_resolver_v1_t *resolver,
                                        const nai_ref_v1_t *ref, uint32_t bytes,
                                        uint32_t alignment, uint32_t *address)
{
    nai_loader_status_t status = nai_resolve_ref_v1(view, resolver, ref, bytes, alignment, address);
    return status == NAI_LOADER_OK ? NAI_DISPATCH_OK : NAI_DISPATCH_BAD_REFERENCE;
}

static uint32_t multiply(uint32_t lhs, uint32_t rhs, uint32_t *result)
{
    if (lhs != 0u && rhs > 0xffffffffu / lhs) return 0u;
    *result = lhs * rhs;
    return 1u;
}

static nai_dispatch_status_v2_t run_dma_1d(const nai_cmd_dma_1d_v2_t *command,
                                           const nai_model_view_v1_t *view,
                                           const nai_resolver_v1_t *resolver,
                                           const nai_runtime_ops_v2_t *ops)
{
    uint32_t source;
    uint32_t destination;
    if (command->length == 0u || !all_zero(command->reserved, 6u) || ops->dma_1d == 0 ||
        resolve(view, resolver, &command->source, command->length, NAI_ALIGNMENT_BYTES, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, command->length, NAI_ALIGNMENT_BYTES, &destination) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return ops->dma_1d(ops->context, source, destination, command->length, command->direction) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_dma_2d(const nai_cmd_dma_2d_v2_t *command,
                                           const nai_model_view_v1_t *view,
                                           const nai_resolver_v1_t *resolver,
                                           const nai_runtime_ops_v2_t *ops)
{
    uint32_t source;
    uint32_t destination;
    uint32_t source_bytes;
    uint32_t destination_bytes;
    if (command->length == 0u || command->repetitions_2 == 0u || !all_zero(command->reserved, 3u) ||
        ops->dma_2d == 0 ||
        !multiply(command->source_stride_2, command->repetitions_2 - 1u, &source_bytes) ||
        source_bytes > 0xffffffffu - command->length ||
        !multiply(command->destination_stride_2, command->repetitions_2 - 1u, &destination_bytes) ||
        destination_bytes > 0xffffffffu - command->length) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    source_bytes += command->length;
    destination_bytes += command->length;
    if (resolve(view, resolver, &command->source, source_bytes, NAI_ALIGNMENT_BYTES, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, destination_bytes, NAI_ALIGNMENT_BYTES, &destination) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return ops->dma_2d(ops->context, source, destination, command->length,
        command->source_stride_2, command->destination_stride_2, command->repetitions_2,
        command->direction) == 0u ? NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_dma_3d(const nai_cmd_dma_3d_v2_t *command,
                                           const nai_model_view_v1_t *view,
                                           const nai_resolver_v1_t *resolver,
                                           const nai_runtime_ops_v2_t *ops)
{
    uint32_t source_2;
    uint32_t destination_2;
    uint32_t source_3;
    uint32_t destination_3;
    uint32_t source_bytes;
    uint32_t destination_bytes;
    if (command->length == 0u || command->repetitions_2 == 0u || command->repetitions_3 == 0u || ops->dma_3d == 0 ||
        !multiply(command->source_stride_2, command->repetitions_2 - 1u, &source_2) ||
        !multiply(command->destination_stride_2, command->repetitions_2 - 1u, &destination_2) ||
        !multiply(command->source_stride_3, command->repetitions_3 - 1u, &source_3) ||
        !multiply(command->destination_stride_3, command->repetitions_3 - 1u, &destination_3) ||
        source_2 > 0xffffffffu - source_3 || destination_2 > 0xffffffffu - destination_3) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    source_bytes = source_2 + source_3;
    destination_bytes = destination_2 + destination_3;
    if (source_bytes > 0xffffffffu - command->length || destination_bytes > 0xffffffffu - command->length) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    source_bytes += command->length;
    destination_bytes += command->length;
    if (resolve(view, resolver, &command->source, source_bytes, NAI_ALIGNMENT_BYTES, &source_2) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, destination_bytes, NAI_ALIGNMENT_BYTES, &destination_2) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return ops->dma_3d(ops->context, source_2, destination_2, command->length,
        command->source_stride_2, command->destination_stride_2, command->repetitions_2,
        command->source_stride_3, command->destination_stride_3, command->repetitions_3,
        command->direction) == 0u ? NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_gemm(const nai_cmd_gemm32_v2_t *command,
                                         const nai_model_view_v1_t *view,
                                         const nai_resolver_v1_t *resolver,
                                         const nai_runtime_ops_v2_t *ops)
{
    uint32_t weights;
    uint32_t ifm;
    uint32_t partial_sums = 0u;
    uint32_t ofm;
    uint32_t ifm_bytes;
    uint32_t ofm_bytes;
    uint32_t partial_bytes;
    uint32_t needs_partial = command->header.type != NAI_CMD_GEMM32;
    uint32_t requant = command->header.type == NAI_CMD_GEMM32_REQUANT;

    if (command->dim_m == 0u || command->dim_m > 256u || !all_zero(command->reserved, 8u) ||
        ops->gemm32 == 0 || !multiply(command->dim_m, 32u, &ifm_bytes) ||
        !multiply(command->dim_m, requant ? 32u : 128u, &ofm_bytes) ||
        !multiply(command->dim_m, 128u, &partial_bytes)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->weights, 1024u, NAI_ALIGNMENT_BYTES, &weights) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ifm, ifm_bytes, NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, ofm_bytes, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (needs_partial && resolve(view, resolver, &command->partial_sums, partial_bytes,
        NAI_ALIGNMENT_BYTES, &partial_sums) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return ops->gemm32(ops->context, command, weights, ifm, partial_sums, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_copy(const nai_cmd_copy_layout_v2_t *command,
                                         const nai_model_view_v1_t *view,
                                         const nai_resolver_v1_t *resolver,
                                         const nai_runtime_ops_v2_t *ops)
{
    uint32_t elements = 1u;
    uint32_t compact_bytes;
    uint32_t native_bytes;
    uint32_t rows = 1u;
    uint32_t source_bytes;
    uint32_t destination_bytes;
    uint32_t source;
    uint32_t destination;
    uint32_t element_bytes = command->data_type == NAI_DTYPE_I8 ? 1u :
        command->data_type == NAI_DTYPE_I32 ? 4u : 0u;

    if (element_bytes == 0u || command->valid_channels == 0u || command->valid_channels > 0xffffffe0u ||
        !all_zero(command->reserved, 7u) ||
        ops->copy_layout == 0) return NAI_DISPATCH_BAD_COMMAND;
    for (uint32_t axis = 0; axis < 4u; axis++) {
        if (command->dimensions[axis] == 0u || !multiply(elements, command->dimensions[axis], &elements))
            return NAI_DISPATCH_BAD_COMMAND;
        if (axis < 3u && !multiply(rows, command->dimensions[axis], &rows))
            return NAI_DISPATCH_BAD_COMMAND;
    }
    if (command->valid_channels != command->dimensions[3] ||
        !multiply(elements, element_bytes, &compact_bytes) ||
        !multiply(rows, (command->valid_channels + 31u) & ~31u, &native_bytes) ||
        !multiply(native_bytes, element_bytes, &native_bytes)) return NAI_DISPATCH_BAD_COMMAND;
    if (command->mode == NAI_COPY_NHWC_TO_ROW32 || command->mode == NAI_COPY_NHWC_TO_C32) {
        source_bytes = compact_bytes;
        destination_bytes = native_bytes;
    } else if (command->mode == NAI_COPY_ROW32_TO_NHWC || command->mode == NAI_COPY_C32_TO_NHWC) {
        source_bytes = native_bytes;
        destination_bytes = compact_bytes;
    } else return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->source, source_bytes, 1u, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, destination_bytes, 1u, &destination) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_REFERENCE;
    return ops->copy_layout(ops->context, command, source, destination) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

nai_dispatch_status_v2_t nai_cmd_dispatch_v2(const nai_model_view_v1_t *view,
                                             const nai_resolver_v1_t *resolver,
                                             const nai_runtime_ops_v2_t *ops,
                                             uint32_t *completed_commands,
                                             uint32_t *failure_command_offset)
{
    uint32_t offset;
    uint32_t completed = 0u;

    if (completed_commands != 0) *completed_commands = 0u;
    if (failure_command_offset != 0) *failure_command_offset = 0u;
    if (view == 0 || view->header == 0 || view->commands == 0 || resolver == 0 || ops == 0 ||
        view->header->entry_command_off < view->commands->offset) return NAI_DISPATCH_BAD_STREAM;
    offset = view->header->entry_command_off - view->commands->offset;

    while (offset < view->commands->size && completed <= view->header->command_count) {
        const nai_cmd_header_v2_t *header;
        nai_dispatch_status_v2_t status = NAI_DISPATCH_OK;
        if (!valid_range(offset, sizeof(nai_cmd_header_v2_t), view->commands->size)) return NAI_DISPATCH_BAD_STREAM;
        header = (const nai_cmd_header_v2_t *)(view->model + view->commands->offset + offset);
        if (header->size_bytes < 32u || (header->size_bytes & 31u) != 0u ||
            !valid_range(offset, header->size_bytes, view->commands->size) ||
            (header->flags & ~(NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) != 0u) {
            status = NAI_DISPATCH_BAD_COMMAND;
        } else if (header->type == NAI_CMD_END) {
            if (header->size_bytes != sizeof(nai_cmd_control_v2_t) ||
                !all_zero(((const nai_cmd_control_v2_t *)header)->reserved, 4u) ||
                completed != view->header->command_count) status = NAI_DISPATCH_BAD_STREAM;
            else {
                if (completed_commands != 0) *completed_commands = completed;
                return NAI_DISPATCH_OK;
            }
        } else if (header->type == NAI_CMD_BARRIER && header->size_bytes == sizeof(nai_cmd_control_v2_t)) {
            status = all_zero(((const nai_cmd_control_v2_t *)header)->reserved, 4u) &&
                ops->barrier != 0 && ops->barrier(ops->context) == 0u ?
                NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
        } else if (header->type == NAI_CMD_DMA_1D && header->size_bytes == sizeof(nai_cmd_dma_1d_v2_t)) {
            status = run_dma_1d((const nai_cmd_dma_1d_v2_t *)header, view, resolver, ops);
        } else if (header->type == NAI_CMD_DMA_2D && header->size_bytes == sizeof(nai_cmd_dma_2d_v2_t)) {
            status = run_dma_2d((const nai_cmd_dma_2d_v2_t *)header, view, resolver, ops);
        } else if (header->type == NAI_CMD_DMA_3D && header->size_bytes == sizeof(nai_cmd_dma_3d_v2_t)) {
            status = run_dma_3d((const nai_cmd_dma_3d_v2_t *)header, view, resolver, ops);
        } else if ((header->type == NAI_CMD_GEMM32 || header->type == NAI_CMD_GEMM32_ACCUM ||
                    header->type == NAI_CMD_GEMM32_REQUANT) && header->size_bytes == sizeof(nai_cmd_gemm32_v2_t)) {
            status = run_gemm((const nai_cmd_gemm32_v2_t *)header, view, resolver, ops);
        } else if (header->type == NAI_CMD_COPY_LAYOUT && header->size_bytes == sizeof(nai_cmd_copy_layout_v2_t)) {
            status = run_copy((const nai_cmd_copy_layout_v2_t *)header, view, resolver, ops);
        } else if ((header->flags & (NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) ==
                   (NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) {
            status = NAI_DISPATCH_OK;
        } else {
            status = NAI_DISPATCH_UNSUPPORTED;
        }
        if (status != NAI_DISPATCH_OK) {
            if (completed_commands != 0) *completed_commands = completed;
            if (failure_command_offset != 0) *failure_command_offset = view->commands->offset + offset;
            return status;
        }
        completed++;
        offset += header->size_bytes;
    }
    return NAI_DISPATCH_BAD_STREAM;
}
