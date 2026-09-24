#include "npu_cmd_desc_v2.h"

#if defined(NAI_TRUSTED_FIRMWARE)
#define NAI_TRUSTED_INVALID(condition) 0u
#else
#define NAI_TRUSTED_INVALID(condition) (condition)
#endif

static uint32_t all_zero(const uint32_t *words, uint32_t count)
{
    for (uint32_t index = 0; index < count; index++) {
        if (words[index] != 0u) return 0u;
    }
    return 1u;
}

static uint32_t all_zero_bytes(const uint8_t *bytes, uint32_t count)
{
    for (uint32_t index = 0; index < count; index++) {
        if (bytes[index] != 0u) return 0u;
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
#if defined(NAI_TRUSTED_FIRMWARE)
    uint32_t base;
    (void)bytes;
    (void)alignment;
    switch (ref->region) {
        case NAI_REGION_MODEL_CONSTANTS:
            base = resolver->model_base + view->constants->offset;
            break;
        case NAI_REGION_MODEL_COMMANDS:
            base = resolver->model_base + view->commands->offset;
            break;
        case NAI_REGION_INPUT_BINDING:
            base = resolver->bindings[ref->index].base;
            break;
        case NAI_REGION_OUTPUT_BINDING:
            base = resolver->bindings[view->header->input_count + ref->index].base;
            break;
        case NAI_REGION_L2_TEMP_BINDING:
            base = resolver->bindings[view->header->input_count +
                view->header->output_count].base;
            break;
        case NAI_REGION_TCDM_SCRATCH:
            base = resolver->tcdm_scratch_base;
            break;
        default:
            return NAI_DISPATCH_BAD_REFERENCE;
    }
    *address = base + ref->offset;
    return NAI_DISPATCH_OK;
#else
    nai_loader_status_t status = nai_resolve_ref_v1(view, resolver, ref, bytes, alignment, address);
    return status == NAI_LOADER_OK ? NAI_DISPATCH_OK : NAI_DISPATCH_BAD_REFERENCE;
#endif
}

static uint32_t multiply(uint32_t lhs, uint32_t rhs, uint32_t *result)
{
    if (lhs != 0u && rhs > 0xffffffffu / lhs) return 0u;
    *result = lhs * rhs;
    return 1u;
}

static uint32_t ranges_overlap(uint32_t lhs, uint32_t rhs, uint32_t bytes)
{
    return lhs < rhs ? bytes > rhs - lhs : bytes > lhs - rhs;
}

static uint32_t ranges_overlap_sized(
    uint32_t lhs, uint32_t lhs_bytes, uint32_t rhs, uint32_t rhs_bytes)
{
    return lhs < rhs ? lhs_bytes > rhs - lhs : rhs_bytes > lhs - rhs;
}

static uint32_t dma_region_is_local(uint16_t region, uint32_t *is_local)
{
    switch (region) {
        case NAI_REGION_MODEL_CONSTANTS:
        case NAI_REGION_MODEL_COMMANDS:
        case NAI_REGION_INPUT_BINDING:
        case NAI_REGION_OUTPUT_BINDING:
        case NAI_REGION_L2_TEMP_BINDING:
            *is_local = 0u;
            return 1u;
        case NAI_REGION_TCDM_SCRATCH:
        case NAI_REGION_DTCM_RUNTIME:
            *is_local = 1u;
            return 1u;
        default:
            return 0u;
    }
}

static nai_dispatch_status_v2_t validate_dma_direction(
    const nai_ref_v1_t *source, const nai_ref_v1_t *destination, uint32_t direction)
{
    uint32_t source_local;
    uint32_t destination_local;
    uint32_t expected;
    if (!dma_region_is_local(source->region, &source_local) ||
        !dma_region_is_local(destination->region, &destination_local) ||
        (!source_local && !destination_local)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    expected = source_local ?
        (destination_local ? NAI_DMA_LOCAL_TO_LOCAL : NAI_DMA_LOCAL_TO_EXTERNAL) :
        NAI_DMA_EXTERNAL_TO_LOCAL;
    return direction == expected ? NAI_DISPATCH_OK : NAI_DISPATCH_BAD_COMMAND;
}

static nai_dispatch_status_v2_t run_dma_1d(const nai_cmd_dma_1d_v2_t *command,
                                           const nai_model_view_v1_t *view,
                                           const nai_resolver_v1_t *resolver,
                                           const nai_runtime_ops_v2_t *ops)
{
    uint32_t (*execute)(void *, uint32_t, uint32_t, uint32_t, uint32_t) =
        command->header.type == NAI_CMD_DMA_SUBMIT_1D ? ops->dma_submit_1d : ops->dma_1d;
    uint32_t source;
    uint32_t destination;
    if (execute == 0 ||
        NAI_TRUSTED_INVALID(command->length == 0u || !all_zero(command->reserved, 6u)))
        return NAI_DISPATCH_BAD_COMMAND;
    if (NAI_TRUSTED_INVALID(validate_dma_direction(&command->source, &command->destination,
            command->direction) != NAI_DISPATCH_OK))
        return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->source, command->length, 1u, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, command->length, 1u, &destination) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return execute(ops->context, source, destination, command->length, command->direction) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_dma_wait(const nai_cmd_dma_wait_v2_t *command,
                                              const nai_runtime_ops_v2_t *ops)
{
    if (ops->dma_wait == 0 ||
        NAI_TRUSTED_INVALID(command->direction > NAI_DMA_LOCAL_TO_EXTERNAL ||
                            !all_zero(command->reserved, 3u)))
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->dma_wait(ops->context, command->direction) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_rq_load(const nai_cmd_rq_load_v2_t *command,
                                             const nai_model_view_v1_t *view,
                                             const nai_resolver_v1_t *resolver,
                                             const nai_runtime_ops_v2_t *ops)
{
    uint32_t qparam_offset;
#if !defined(NAI_TRUSTED_FIRMWARE)
    uint32_t qparam_bytes;
#endif
    uint32_t address;
    if (ops->rq_load == 0 || view->qparams == 0) return NAI_DISPATCH_BAD_COMMAND;
#if defined(NAI_TRUSTED_FIRMWARE)
    qparam_offset = command->qparam_index * sizeof(nai_qparam_v1_t);
#else
    if (command->qparam_count != 32u || command->reserved != 0u ||
        command->qparam_index > view->qparams->element_count ||
        command->qparam_count > view->qparams->element_count - command->qparam_index ||
        !multiply(command->qparam_index, sizeof(nai_qparam_v1_t), &qparam_offset) ||
        !multiply(command->qparam_count, sizeof(nai_qparam_v1_t), &qparam_bytes) ||
        !valid_range(qparam_offset, qparam_bytes, view->qparams->size)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolver->model_bytes < view->model_bytes ||
        resolver->model_base > 0xffffffffu - view->qparams->offset ||
        resolver->model_base + view->qparams->offset > 0xffffffffu - qparam_offset) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
#endif
    address = resolver->model_base + view->qparams->offset + qparam_offset;
#if !defined(NAI_TRUSTED_FIRMWARE)
    if ((address & (NAI_ALIGNMENT_BYTES - 1u)) != 0u) return NAI_DISPATCH_BAD_REFERENCE;
#endif
    return ops->rq_load(ops->context, address, command->qparam_count,
        command->qparam_block) == 0u ? NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_dma_2d(const nai_cmd_dma_2d_v2_t *command,
                                           const nai_model_view_v1_t *view,
                                           const nai_resolver_v1_t *resolver,
                                           const nai_runtime_ops_v2_t *ops)
{
    uint32_t (*execute)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t) =
        command->header.type == NAI_CMD_DMA_SUBMIT_2D ? ops->dma_submit_2d : ops->dma_2d;
    uint32_t source;
    uint32_t destination;
#if !defined(NAI_TRUSTED_FIRMWARE)
    uint32_t source_bytes;
    uint32_t destination_bytes;
#endif
    if (execute == 0) return NAI_DISPATCH_BAD_COMMAND;
#if !defined(NAI_TRUSTED_FIRMWARE)
    if (command->length == 0u || command->repetitions_2 == 0u ||
        !all_zero(command->reserved, 3u) ||
        !multiply(command->source_stride_2, command->repetitions_2 - 1u, &source_bytes) ||
        source_bytes > 0xffffffffu - command->length ||
        !multiply(command->destination_stride_2, command->repetitions_2 - 1u, &destination_bytes) ||
        destination_bytes > 0xffffffffu - command->length) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    source_bytes += command->length;
    destination_bytes += command->length;
    if (validate_dma_direction(&command->source, &command->destination,
            command->direction) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->source, source_bytes, 1u, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, destination_bytes, 1u, &destination) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
#else
    if (resolve(view, resolver, &command->source, 0u, 1u, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, 0u, 1u, &destination) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
#endif
    return execute(ops->context, source, destination, command->length,
        command->source_stride_2, command->destination_stride_2, command->repetitions_2,
        command->direction) == 0u ? NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_dma_3d(const nai_cmd_dma_3d_v2_t *command,
                                           const nai_model_view_v1_t *view,
                                           const nai_resolver_v1_t *resolver,
                                           const nai_runtime_ops_v2_t *ops)
{
    uint32_t (*execute)(void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) =
        command->header.type == NAI_CMD_DMA_SUBMIT_3D ? ops->dma_submit_3d : ops->dma_3d;
    uint32_t source_2;
    uint32_t destination_2;
#if !defined(NAI_TRUSTED_FIRMWARE)
    uint32_t source_3;
    uint32_t destination_3;
    uint32_t source_bytes;
    uint32_t destination_bytes;
#endif
    if (execute == 0) return NAI_DISPATCH_BAD_COMMAND;
#if !defined(NAI_TRUSTED_FIRMWARE)
    if (command->length == 0u || command->repetitions_2 == 0u ||
        command->repetitions_3 == 0u ||
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
    if (validate_dma_direction(&command->source, &command->destination,
            command->direction) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->source, source_bytes, 1u, &source_2) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, destination_bytes, 1u, &destination_2) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
#else
    if (resolve(view, resolver, &command->source, 0u, 1u, &source_2) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, 0u, 1u, &destination_2) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
#endif
    return execute(ops->context, source_2, destination_2, command->length,
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
    uint32_t requant = command->header.type == NAI_CMD_GEMM32_REQUANT;
    uint32_t direct_requant = requant && command->partial_sums.region == 0u;
    uint32_t needs_partial = command->header.type == NAI_CMD_GEMM32_ACCUM ||
        (requant && !direct_requant);
    uint32_t ofm_row_bytes = requant ? 32u : 128u;
    uint32_t ofm_stride = command->ofm_row_stride != 0u ? command->ofm_row_stride : ofm_row_bytes;
    uint32_t partial_stride = command->partial_sum_row_stride != 0u ?
        command->partial_sum_row_stride : 128u;

    if (ops->gemm32 == 0 ||
        NAI_TRUSTED_INVALID(command->dim_m == 0u || command->dim_m > 256u ||
            !all_zero(command->reserved, 8u) || ofm_stride < ofm_row_bytes ||
            (ofm_stride & 31u) != 0u || partial_stride < 128u ||
            (partial_stride & 31u) != 0u ||
            (direct_requant && (command->partial_sums.index != 0u ||
                                command->partial_sums.offset != 0u))) ||
        !multiply(command->dim_m, 32u, &ifm_bytes) ||
        !multiply(command->dim_m - 1u, ofm_stride, &ofm_bytes) ||
        ofm_bytes > 0xffffffffu - ofm_row_bytes ||
        !multiply(command->dim_m - 1u, partial_stride, &partial_bytes) ||
        partial_bytes > 0xffffffffu - 128u) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    ofm_bytes += ofm_row_bytes;
    partial_bytes += 128u;
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

static nai_dispatch_status_v2_t run_pointwise_c32(
    const nai_cmd_pointwise_c32_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t weights;
    uint32_t ifm;
    uint32_t partial_sums = 0u;
    uint32_t ofm;
    uint32_t weight_tiles;
    uint32_t weight_bytes;
    uint32_t ifm_groups_bytes;
    uint32_t ofm_groups_bytes;
    uint32_t ifm_bytes;
    uint32_t ofm_bytes;
    uint32_t partial_bytes;

    if (ops->pointwise_c32 == 0 ||
        NAI_TRUSTED_INVALID(command->rows == 0u || command->input_c32_groups == 0u ||
            command->output_c32_groups != 1u || !all_zero(command->reserved, 6u) ||
            command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
            command->ofm.region != NAI_REGION_TCDM_SCRATCH ||
            (command->input_group_stride_bytes & (NAI_ALIGNMENT_BYTES - 1u)) != 0u ||
            (command->output_group_stride_bytes & (NAI_ALIGNMENT_BYTES - 1u)) != 0u) ||
        !multiply(command->input_c32_groups, command->output_c32_groups, &weight_tiles) ||
        !multiply(weight_tiles, 32u * 32u, &weight_bytes) ||
        !multiply(command->rows, 32u, &ifm_groups_bytes) ||
        !multiply(command->rows, 32u, &ofm_groups_bytes) ||
        NAI_TRUSTED_INVALID(command->input_group_stride_bytes < ifm_groups_bytes ||
                            command->output_group_stride_bytes < ofm_groups_bytes) ||
        !multiply(command->input_group_stride_bytes, command->input_c32_groups - 1u, &ifm_bytes) ||
        ifm_bytes > 0xffffffffu - ifm_groups_bytes ||
        !multiply(command->output_group_stride_bytes, command->output_c32_groups - 1u, &ofm_bytes) ||
        ofm_bytes > 0xffffffffu - ofm_groups_bytes ||
        !multiply(command->rows < 256u ? command->rows : 256u, 32u * 4u, &partial_bytes)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    ifm_bytes += ifm_groups_bytes;
    ofm_bytes += ofm_groups_bytes;
    if (NAI_TRUSTED_INVALID(
        (command->input_c32_groups == 1u &&
         (command->partial_sums.region != 0u || command->partial_sums.index != 0u ||
          command->partial_sums.offset != 0u)) ||
        (command->input_c32_groups != 1u &&
         command->partial_sums.region != NAI_REGION_TCDM_SCRATCH))) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->weights, weight_bytes, NAI_ALIGNMENT_BYTES, &weights) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ifm, ifm_bytes, NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, ofm_bytes, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    /* nai_resolve_ref_v1 validates the section-relative range.  Also guard
       the resolved physical address arithmetic here: a resolver base near
       UINT32_MAX must not wrap when the runtime walks the complete tensor. */
    if (weights > 0xffffffffu - weight_bytes ||
        ifm > 0xffffffffu - ifm_bytes || ofm > 0xffffffffu - ofm_bytes) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (command->input_c32_groups > 1u &&
        resolve(view, resolver, &command->partial_sums, partial_bytes,
                NAI_ALIGNMENT_BYTES, &partial_sums) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (command->input_c32_groups > 1u &&
        partial_sums > 0xffffffffu - partial_bytes) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return ops->pointwise_c32(ops->context, command, weights, ifm, partial_sums, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_depthwise_c32(
    const nai_cmd_depthwise_c32_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t weights;
    uint32_t ifm;
    uint32_t ofm;
    uint32_t groups;
    uint32_t input_pixels;
    uint32_t output_pixels;
    uint32_t input_bytes;
    uint32_t output_bytes;
    uint32_t weight_bytes;
    uint32_t expected_output_h;
    uint32_t expected_output_w;
    uint32_t expected_valid_h;
    uint32_t expected_valid_w;
    uint32_t total_pad_h;
    uint32_t total_pad_w;
    uint32_t same_shape;
    uint32_t valid_shape;

    if (ops->depthwise_c32 == 0 || NAI_TRUSTED_INVALID(
        command->input_h == 0u || command->input_w == 0u ||
        command->output_h == 0u || command->output_w == 0u ||
        command->channels == 0u || !all_zero(command->reserved, 4u) ||
        command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH ||
        (command->stride_h != 1u && command->stride_h != 2u) ||
        (command->stride_w != 1u && command->stride_w != 2u) ||
        command->pad_h > 1u || command->pad_w > 1u ||
        command->input_h > 0xffffu || command->input_w > 0xffffu ||
        command->output_h > 0xffffu || command->output_w > 0xffffu ||
        command->channels > 32u)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    expected_output_h = (command->input_h + command->stride_h - 1u) / command->stride_h;
    expected_output_w = (command->input_w + command->stride_w - 1u) / command->stride_w;
    expected_valid_h = command->input_h >= 3u ?
        (command->input_h - 3u) / command->stride_h + 1u : 0u;
    expected_valid_w = command->input_w >= 3u ?
        (command->input_w - 3u) / command->stride_w + 1u : 0u;
    total_pad_h = (expected_output_h - 1u) * command->stride_h + 3u - command->input_h;
    total_pad_w = (expected_output_w - 1u) * command->stride_w + 3u - command->input_w;
    same_shape = command->output_h == expected_output_h &&
        command->output_w == expected_output_w &&
        command->pad_h == total_pad_h / 2u && command->pad_w == total_pad_w / 2u;
    valid_shape = command->output_h == expected_valid_h &&
        command->output_w == expected_valid_w && command->pad_h == 0u && command->pad_w == 0u;
    groups = 1u;
    if (NAI_TRUSTED_INVALID(!same_shape && !valid_shape) ||
        !multiply(command->input_h, command->input_w, &input_pixels) ||
        !multiply(command->output_h, command->output_w, &output_pixels) ||
        !multiply(input_pixels, groups, &input_bytes) ||
        !multiply(input_bytes, 32u, &input_bytes) ||
        !multiply(output_pixels, groups, &output_bytes) ||
        !multiply(output_bytes, 32u, &output_bytes) ||
        !multiply(groups, 3u * 3u * 32u, &weight_bytes)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->weights, weight_bytes, NAI_ALIGNMENT_BYTES, &weights) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ifm, input_bytes, NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, output_bytes, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return ops->depthwise_c32(ops->context, command, weights, ifm, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_afu_binary(
    const nai_cmd_afu_binary_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t lhs;
    uint32_t rhs;
    uint32_t ofm;
    const uint32_t biased = command->mode == NAI_AFU_BINARY_ADD_I8_BIAS;
    if (ops->afu_binary == 0 || NAI_TRUSTED_INVALID(
        command->length == 0u ||
        (command->mode != NAI_AFU_BINARY_ADD_I8 && !biased) ||
        (!biased && command->bias != 0) || command->bias < -382 || command->bias > 383 ||
        !all_zero(command->reserved, 3u) ||
        command->lhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->rhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->lhs, command->length, NAI_ALIGNMENT_BYTES, &lhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->rhs, command->length, NAI_ALIGNMENT_BYTES, &rhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, command->length, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (NAI_TRUSTED_INVALID(ranges_overlap(lhs, ofm, command->length) ||
        ranges_overlap(rhs, ofm, command->length))) return NAI_DISPATCH_BAD_COMMAND;
    return ops->afu_binary(ops->context, command, lhs, rhs, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_spatz_add(
    const nai_cmd_spatz_add_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t lhs;
    uint32_t rhs;
    uint32_t ofm;
    if (ops->spatz_add == 0 || NAI_TRUSTED_INVALID(
        command->length == 0u || command->lhs_scale <= 0 || command->rhs_scale <= 0 ||
        command->output_scale <= 0 || command->lhs_shift > 63u || command->rhs_shift > 63u ||
        command->output_shift > 63u ||
        (command->double_round_shift != 0u && command->double_round_shift != 20u) ||
        command->mode > NAI_SPATZ_BINARY_SUBTRACT ||
        command->lhs_zero_point < -128 || command->lhs_zero_point > 127 ||
        command->rhs_zero_point < -128 || command->rhs_zero_point > 127 ||
        command->output_zero_point < -128 || command->output_zero_point > 127 ||
        command->clamp_min < -128 || command->clamp_max > 127 ||
        command->clamp_min > command->clamp_max ||
        command->lhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->rhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH)) return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->lhs, command->length, NAI_ALIGNMENT_BYTES, &lhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->rhs, command->length, NAI_ALIGNMENT_BYTES, &rhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, command->length, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_REFERENCE;
    if (NAI_TRUSTED_INVALID(ranges_overlap(lhs, ofm, command->length) ||
        ranges_overlap(rhs, ofm, command->length))) return NAI_DISPATCH_BAD_COMMAND;
    return ops->spatz_add(ops->context, command, lhs, rhs, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_afu_binary_quant(
    const nai_cmd_afu_binary_quant_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t lhs;
    uint32_t rhs;
    uint32_t ofm;
    if (ops->afu_binary_quant == 0 || NAI_TRUSTED_INVALID(
        command->length == 0u || command->lhs_scale <= 0 || command->rhs_scale <= 0 ||
        command->output_scale <= 0 || command->lhs_shift > 63u || command->rhs_shift > 63u ||
        command->output_shift > 63u ||
        (command->double_round_shift != 0u && command->double_round_shift != 20u) ||
        command->mode > NAI_SPATZ_BINARY_MULTIPLY ||
        command->lhs_zero_point < -128 || command->lhs_zero_point > 127 ||
        command->rhs_zero_point < -128 || command->rhs_zero_point > 127 ||
        command->output_zero_point < -128 || command->output_zero_point > 127 ||
        command->clamp_min < -128 || command->clamp_max > 127 ||
        command->clamp_min > command->clamp_max ||
        command->lhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->rhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH)) return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->lhs, command->length, NAI_ALIGNMENT_BYTES, &lhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->rhs, command->length, NAI_ALIGNMENT_BYTES, &rhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, command->length, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_REFERENCE;
    if (NAI_TRUSTED_INVALID(ranges_overlap(lhs, ofm, command->length) ||
        ranges_overlap(rhs, ofm, command->length))) return NAI_DISPATCH_BAD_COMMAND;
    return ops->afu_binary_quant(ops->context, command, lhs, rhs, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_afu_lut(
    const nai_cmd_afu_lut_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t ifm;
    uint32_t ofm;
    uint32_t lut;
    if (ops->afu_lut == 0 || NAI_TRUSTED_INVALID(
        command->length == 0u || !all_zero(command->reserved, 5u) ||
        command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH ||
        (command->lut.region != NAI_REGION_MODEL_CONSTANTS &&
         command->lut.region != NAI_REGION_TCDM_SCRATCH))) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->ifm, command->length,
            NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, command->length,
            NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->lut, 256u, 1u, &lut) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (NAI_TRUSTED_INVALID(ifm != ofm && ranges_overlap(ifm, ofm, command->length)))
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->afu_lut(ops->context, command, ifm, ofm, lut) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_afu_dfl16(
    const nai_cmd_afu_dfl16_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    const uint32_t scratch_bytes = 32u * 34u;
    uint32_t records;
    uint32_t source_bytes;
    uint32_t source;
    uint32_t destination;
    uint32_t scratch;
    uint32_t exp_lut;
    uint32_t recip_lut;
    if (ops->afu_dfl16 == 0 ||
        NAI_TRUSTED_INVALID(command->locations == 0u || command->locations > 2100u ||
            command->source_layout > 1u ||
            command->output_multiplier <= 0 || command->output_multiplier > 65535 ||
            command->output_shift > 31u ||
            command->output_zero_point < -128 || command->output_zero_point > 127 ||
            command->clamp_min < -128 || command->clamp_max > 127 ||
            command->clamp_min > command->clamp_max ||
            !all_zero(command->reserved, 3u) ||
            command->source.region != NAI_REGION_TCDM_SCRATCH ||
            command->destination.region != NAI_REGION_TCDM_SCRATCH ||
            command->scratch.region != NAI_REGION_TCDM_SCRATCH ||
            command->exp_lut.region != NAI_REGION_MODEL_CONSTANTS ||
            command->recip_lut.region != NAI_REGION_MODEL_CONSTANTS ||
            command->exp_lut.index != command->recip_lut.index ||
            command->exp_lut.offset > 0xffffffffu - 1024u ||
            command->recip_lut.offset != command->exp_lut.offset + 1024u) ||
        !multiply(4u, command->locations, &records) ||
        !multiply(command->source_layout == 1u ? 64u : 144u,
            command->locations, &source_bytes))
        return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->source, source_bytes,
            NAI_ALIGNMENT_BYTES, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, records,
            NAI_ALIGNMENT_BYTES, &destination) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->scratch, scratch_bytes,
            NAI_ALIGNMENT_BYTES, &scratch) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->exp_lut, 2048u, 4u, &exp_lut) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_REFERENCE;
    recip_lut = exp_lut + 1024u;
    if (NAI_TRUSTED_INVALID(
        ranges_overlap_sized(source, source_bytes, destination, records) ||
        ranges_overlap_sized(source, source_bytes, scratch, scratch_bytes) ||
        ranges_overlap_sized(destination, records, scratch, scratch_bytes)))
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->afu_dfl16(ops->context, command, source, destination, scratch,
                          exp_lut, recip_lut) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_afu_global_avgpool(
    const nai_cmd_afu_global_avgpool_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    const uint32_t requant =
        command->header.flags & NAI_CMD_FLAG_AFU_GLOBAL_AVGPOOL_REQUANT;
    uint32_t spatial_count;
    uint32_t groups;
    uint32_t input_bytes;
    uint32_t output_bytes;
    uint32_t ifm;
    uint32_t ofm;
    if (ops->afu_global_avgpool == 0 || NAI_TRUSTED_INVALID(
        command->input_h == 0u || command->input_w == 0u ||
        command->channels == 0u ||
        (requant == 0u && (command->output_multiplier != 0 ||
            command->output_shift != 0u || command->input_offset != 0 ||
            command->output_zero_point != 0 || command->double_round_shift != 0u)) ||
        (requant != 0u && (command->output_multiplier <= 0 ||
            command->output_shift > 63u || command->output_zero_point < -128 ||
            command->output_zero_point > 127 || command->double_round_shift > 30u)) ||
        command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH) ||
        !multiply(command->input_h, command->input_w, &spatial_count) ||
        command->channels > 0xffffffffu - 31u) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    groups = (command->channels + 31u) / 32u;
    if (requant != 0u && NAI_TRUSTED_INVALID(
        (int64_t)command->input_offset - (int64_t)128 * spatial_count <
            (-2147483647LL - 1LL) ||
        (int64_t)command->input_offset + (int64_t)127 * spatial_count >
            2147483647LL)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (!multiply(spatial_count, groups, &input_bytes) ||
        !multiply(input_bytes, 32u, &input_bytes) ||
        !multiply(groups, 32u, &output_bytes)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->ifm, input_bytes, NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, output_bytes, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (NAI_TRUSTED_INVALID(ranges_overlap_sized(ifm, input_bytes, ofm, output_bytes)))
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->afu_global_avgpool(ops->context, command, ifm, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_upsample_nearest(
    const nai_cmd_upsample_nearest_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t input_pixels;
    uint32_t input_bytes;
    uint32_t output_bytes;
    uint32_t ifm;
    uint32_t ofm;
    if (ops->upsample_nearest == 0 || NAI_TRUSTED_INVALID(
        command->input_h == 0u || command->input_w == 0u ||
        (command->channels != 32u && command->channels != 128u &&
         command->channels != 256u) ||
        command->scale_h != 2u || command->scale_w != 2u ||
        !all_zero(command->reserved, 3u) ||
        command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH) ||
        !multiply(command->input_h, command->input_w, &input_pixels) ||
        !multiply(input_pixels, command->channels, &input_bytes) ||
        !multiply(input_bytes, 4u, &output_bytes)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->ifm, input_bytes, NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, output_bytes, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (NAI_TRUSTED_INVALID(ranges_overlap_sized(ifm, input_bytes, ofm, output_bytes)))
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->upsample_nearest(ops->context, command, ifm, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_maxpool(
    const nai_cmd_maxpool_v2_t *command,
    const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t pixels;
    uint32_t bytes;
    uint32_t ifm;
    uint32_t ofm;
    if (ops->maxpool == 0 || NAI_TRUSTED_INVALID(
        command->input_h == 0u || command->input_w == 0u ||
        (command->channels != 32u && command->channels != 128u) ||
        command->kernel_h != 5u || command->kernel_w != 5u ||
        command->stride_h != 1u || command->stride_w != 1u ||
        command->pad_h != 2u || command->pad_w != 2u ||
        !all_zero(command->reserved, 7u) ||
        command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH) ||
        !multiply(command->input_h, command->input_w, &pixels) ||
        !multiply(pixels, command->channels, &bytes)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->ifm, bytes, NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, bytes, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (NAI_TRUSTED_INVALID(ranges_overlap_sized(ifm, bytes, ofm, bytes)))
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->maxpool(ops->context, command, ifm, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_linebuf_job(
    const nai_cmd_linebuf_job_v2_t *command,
    const nai_runtime_ops_v2_t *ops)
{
    const uint32_t binary = command->header.type == NAI_CMD_LINEBUF_BINARY ||
                            command->header.type == NAI_CMD_LINEBUF_BINARY_SUBMIT;
    uint32_t (*execute)(void *, const nai_cmd_linebuf_job_v2_t *) =
        command->header.type == NAI_CMD_LINEBUF_SUBMIT ?
            ops->linebuf_submit : ops->linebuf_job;
    uint32_t (*execute_binary)(void *, const nai_cmd_linebuf_binary_v2_t *) =
        command->header.type == NAI_CMD_LINEBUF_BINARY_SUBMIT ?
            ops->linebuf_binary_submit : ops->linebuf_binary_job;
#if !defined(NAI_TRUSTED_FIRMWARE)
    const uint32_t max_rows = command->job.gemm.accum_en == 0u ? 1024u : 256u;
    uint32_t expected_k_tiles;
    uint32_t kernel_elements;
    uint32_t expected_kgen_schedule;
    if (command->job.rows == 0u || command->job.rows > max_rows ||
        command->job.k_tiles == 0u || command->job.k_tiles > 0xffffu ||
        command->job.linebuf.kernel_h == 0u || command->job.linebuf.kernel_h > 5u ||
        command->job.linebuf.kernel_w == 0u || command->job.linebuf.kernel_w > 5u ||
        ((!binary && !all_zero_bytes(command->reserved, sizeof(command->reserved))) ||
         (binary && !all_zero_bytes(
             ((const nai_cmd_linebuf_binary_v2_t *)command)->reserved,
             sizeof(((const nai_cmd_linebuf_binary_v2_t *)command)->reserved)))) ||
        (binary ? execute_binary == 0 : execute == 0)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (command->job.linebuf.input_c != 0u &&
        multiply(command->job.linebuf.input_c, command->job.linebuf.kernel_h, &kernel_elements) &&
        multiply(kernel_elements, command->job.linebuf.kernel_w, &expected_k_tiles)) {
        expected_k_tiles = (expected_k_tiles + 31u) / 32u;
    } else {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (command->job.gemm.dim_m != command->job.rows ||
        command->job.linebuf.k_tiles != command->job.k_tiles ||
        command->job.k_tiles != expected_k_tiles ||
        command->job.linebuf.spatial_m != command->job.rows ||
        command->job.linebuf.input_h == 0u || command->job.linebuf.input_w == 0u ||
        command->job.linebuf.input_c == 0u || command->job.linebuf.output_w == 0u ||
        command->job.linebuf.block_valid_bytes == 0u ||
        command->job.linebuf.block_valid_bytes > 32u ||
        command->job.gemm.accum_en > 3u ||
        command->job.linebuf.coalesce > 1u || command->job.linebuf.kgen > 1u ||
        command->job.linebuf.pool > 1u || command->job.linebuf.c32_fast > 1u ||
        command->job.linebuf.depthwise > 1u || command->job.linebuf.c32_group_stationary > 2u ||
        command->job.linebuf.stride_h == 0u || command->job.linebuf.stride_h > 2u ||
        command->job.linebuf.stride_w == 0u || command->job.linebuf.stride_w > 2u ||
        command->job.linebuf.kernel_h == 0u || command->job.linebuf.kernel_h > 5u ||
        command->job.linebuf.kernel_w == 0u || command->job.linebuf.kernel_w > 5u ||
        command->job.linebuf.pad_h >= command->job.linebuf.kernel_h ||
        command->job.linebuf.pad_w >= command->job.linebuf.kernel_w ||
        command->job.linebuf.row_stride_bytes == 0u ||
        command->job.linebuf.pixel_stride_bytes == 0u ||
        command->job.linebuf.ow_step_bytes == 0u ||
        command->job.linebuf.oh_step_bytes == 0u ||
        command->job.gemm.ofm_tile_cols == 0u ||
        (command->job.gemm.ofm_row_stride_bytes & 31u) != 0u ||
        (command->job.gemm.accum_en != 0u && command->job.gemm.psum_row_stride_bytes == 0u)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    expected_kgen_schedule = SYSTOLIC_LINEBUF_SCHEDULE_NONE;
    if (command->job.linebuf.coalesce == 1u &&
        command->job.linebuf.kgen == 1u && command->job.k_tiles > 1u) {
        if (command->job.linebuf.c32_fast == 1u &&
            command->job.linebuf.lane_base == 0u &&
            command->job.linebuf.block_valid_bytes == 32u &&
            command->job.linebuf.input_c >= 32u &&
            (command->job.linebuf.input_c & 31u) == 0u) {
            expected_kgen_schedule = SYSTOLIC_LINEBUF_SCHEDULE_C32_GROUP_STATIONARY;
        } else if (command->job.linebuf.c32_fast == 0u &&
                   command->job.linebuf.input_c == 32u &&
                   command->job.linebuf.pixel_stride_bytes == 32u) {
            expected_kgen_schedule = SYSTOLIC_LINEBUF_SCHEDULE_GENERIC_LINEAR_K32;
        }
    }
    if (command->job.linebuf.c32_group_stationary != expected_kgen_schedule)
        return NAI_DISPATCH_BAD_COMMAND;
    if (binary) {
        const systolic_binary_cfg_t *cfg =
            &((const nai_cmd_linebuf_binary_v2_t *)command)->binary;
        if ((command->job.gemm.accum_en != 0u && command->job.gemm.accum_en != 2u) ||
            (cfg->rhs_addr & 31u) != 0u || cfg->rhs_tile_cols == 0u ||
            (cfg->rhs_row_stride_bytes & 31u) != 0u || cfg->mode > SYSTOLIC_BINARY_MUL ||
            (cfg->mode != SYSTOLIC_BINARY_MUL &&
             (cfg->lhs_multiplier <= 0 || cfg->rhs_multiplier <= 0)) ||
            cfg->output_multiplier <= 0 || cfg->lhs_shift > 63u ||
            cfg->rhs_shift > 63u || cfg->output_shift > 63u ||
            cfg->double_round_shift > 30u ||
            cfg->lhs_zero_point < -128 || cfg->lhs_zero_point > 127 ||
            cfg->rhs_zero_point < -128 || cfg->rhs_zero_point > 127 ||
            cfg->output_zero_point < -128 || cfg->output_zero_point > 127 ||
            cfg->clamp_min < -128 || cfg->clamp_min > 127 ||
            cfg->clamp_max < -128 || cfg->clamp_max > 127 ||
            cfg->clamp_min > cfg->clamp_max)
            return NAI_DISPATCH_BAD_COMMAND;
    }
#else
    if (binary ? execute_binary == 0 : execute == 0) return NAI_DISPATCH_BAD_COMMAND;
#endif
    return (binary ?
        execute_binary(ops->context, (const nai_cmd_linebuf_binary_v2_t *)command) :
        execute(ops->context, command)) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_systolic_wait(
    const nai_cmd_control_v2_t *command, const nai_runtime_ops_v2_t *ops)
{
    if (ops->systolic_wait == 0 || !all_zero(command->reserved, 4u))
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->systolic_wait(ops->context) == 0u ?
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

    if (ops->copy_layout == 0 || NAI_TRUSTED_INVALID(element_bytes == 0u ||
        command->valid_channels == 0u || command->valid_channels > 0xffffffe0u ||
        !all_zero(command->reserved, 7u))) return NAI_DISPATCH_BAD_COMMAND;
    for (uint32_t axis = 0; axis < 4u; axis++) {
        if (NAI_TRUSTED_INVALID(command->dimensions[axis] == 0u) ||
            !multiply(elements, command->dimensions[axis], &elements))
            return NAI_DISPATCH_BAD_COMMAND;
        if (axis < 3u && !multiply(rows, command->dimensions[axis], &rows))
            return NAI_DISPATCH_BAD_COMMAND;
    }
    if (NAI_TRUSTED_INVALID(command->valid_channels != command->dimensions[3]) ||
        !multiply(elements, element_bytes, &compact_bytes) ||
        !multiply(rows, (command->valid_channels + 31u) & ~31u, &native_bytes) ||
        !multiply(native_bytes, element_bytes, &native_bytes)) return NAI_DISPATCH_BAD_COMMAND;
    if (command->mode == NAI_COPY_C32_TO_CHW) {
        uint32_t expected_source_stride;
        if (NAI_TRUSTED_INVALID(command->data_type != NAI_DTYPE_I8 || command->dimensions[0] != 1u ||
            command->dimensions[1] != command->dimensions[2] ||
            (command->dimensions[1] != 10u && command->dimensions[1] != 20u &&
             command->dimensions[1] != 40u) || command->dimensions[3] > 144u ||
            command->source_layout != NAI_LAYOUT_C32_BLOCKED ||
            command->destination_layout != NAI_LAYOUT_NHWC ||
            command->source.region != NAI_REGION_TCDM_SCRATCH ||
            command->destination.region != NAI_REGION_TCDM_SCRATCH) ||
            !multiply(rows, 32u, &expected_source_stride) ||
            command->source_row_stride != expected_source_stride ||
            command->destination_row_stride != rows)
            return NAI_DISPATCH_BAD_COMMAND;
        source_bytes = native_bytes;
        destination_bytes = compact_bytes;
    } else if (command->mode == NAI_COPY_NHWC_TO_ROW32 || command->mode == NAI_COPY_NHWC_TO_C32) {
        source_bytes = compact_bytes;
        destination_bytes = native_bytes;
    } else if (command->mode == NAI_COPY_ROW32_TO_NHWC || command->mode == NAI_COPY_C32_TO_NHWC) {
        source_bytes = native_bytes;
        destination_bytes = compact_bytes;
    } else return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->source, source_bytes, 1u, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, destination_bytes, 1u, &destination) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_REFERENCE;
    if (NAI_TRUSTED_INVALID(command->mode == NAI_COPY_C32_TO_CHW &&
        source < destination + destination_bytes && destination < source + source_bytes)
        )
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->copy_layout(ops->context, command, source, destination) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static uint32_t valid_executable_header(const nai_cmd_header_v2_t *header,
                                        uint32_t available)
{
    return header->size_bytes >= 32u && (header->size_bytes & 31u) == 0u &&
        header->size_bytes <= available &&
        !NAI_TRUSTED_INVALID(
            (header->flags & ~(NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE |
                               NAI_CMD_FLAG_AFU_LUT_REUSE |
                               NAI_CMD_FLAG_AFU_LUT_CHAIN |
                               NAI_CMD_FLAG_AFU_GLOBAL_AVGPOOL_REQUANT)) != 0u ||
            ((header->flags & NAI_CMD_FLAG_AFU_LUT_REUSE) != 0u &&
             header->type != NAI_CMD_AFU_LUT) ||
            ((header->flags & NAI_CMD_FLAG_AFU_LUT_CHAIN) != 0u &&
             header->type != NAI_CMD_AFU_BINARY_QUANT) ||
            ((header->flags & NAI_CMD_FLAG_AFU_GLOBAL_AVGPOOL_REQUANT) != 0u &&
             header->type != NAI_CMD_AFU_GLOBAL_AVGPOOL));
}

static nai_dispatch_status_v2_t run_executable_command(
    const nai_cmd_header_v2_t *header, const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver, const nai_runtime_ops_v2_t *ops)
{
    if (header->type == NAI_CMD_BARRIER && header->size_bytes == sizeof(nai_cmd_control_v2_t)) {
        return !NAI_TRUSTED_INVALID(
            !all_zero(((const nai_cmd_control_v2_t *)header)->reserved, 4u)) &&
            ops->barrier != 0 && ops->barrier(ops->context) == 0u ?
            NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
    }
    if (header->type == NAI_CMD_RQ_LOAD && header->size_bytes == sizeof(nai_cmd_rq_load_v2_t))
        return run_rq_load((const nai_cmd_rq_load_v2_t *)header, view, resolver, ops);
    if ((header->type == NAI_CMD_DMA_1D || header->type == NAI_CMD_DMA_SUBMIT_1D) &&
        header->size_bytes == sizeof(nai_cmd_dma_1d_v2_t))
        return run_dma_1d((const nai_cmd_dma_1d_v2_t *)header, view, resolver, ops);
    if ((header->type == NAI_CMD_DMA_2D || header->type == NAI_CMD_DMA_SUBMIT_2D) &&
        header->size_bytes == sizeof(nai_cmd_dma_2d_v2_t))
        return run_dma_2d((const nai_cmd_dma_2d_v2_t *)header, view, resolver, ops);
    if ((header->type == NAI_CMD_DMA_3D || header->type == NAI_CMD_DMA_SUBMIT_3D) &&
        header->size_bytes == sizeof(nai_cmd_dma_3d_v2_t))
        return run_dma_3d((const nai_cmd_dma_3d_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_DMA_WAIT && header->size_bytes == sizeof(nai_cmd_dma_wait_v2_t))
        return run_dma_wait((const nai_cmd_dma_wait_v2_t *)header, ops);
    if ((header->type == NAI_CMD_GEMM32 || header->type == NAI_CMD_GEMM32_ACCUM ||
         header->type == NAI_CMD_GEMM32_REQUANT) &&
        header->size_bytes == sizeof(nai_cmd_gemm32_v2_t))
        return run_gemm((const nai_cmd_gemm32_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_POINTWISE_C32 &&
        header->size_bytes == sizeof(nai_cmd_pointwise_c32_v2_t))
        return run_pointwise_c32((const nai_cmd_pointwise_c32_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_DEPTHWISE_C32 &&
        header->size_bytes == sizeof(nai_cmd_depthwise_c32_v2_t))
        return run_depthwise_c32((const nai_cmd_depthwise_c32_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_AFU_LUT && header->size_bytes == sizeof(nai_cmd_afu_lut_v2_t))
        return run_afu_lut((const nai_cmd_afu_lut_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_AFU_BINARY &&
        header->size_bytes == sizeof(nai_cmd_afu_binary_v2_t))
        return run_afu_binary((const nai_cmd_afu_binary_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_SPATZ_ADD &&
        header->size_bytes == sizeof(nai_cmd_spatz_add_v2_t))
        return run_spatz_add((const nai_cmd_spatz_add_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_AFU_BINARY_QUANT &&
        header->size_bytes == sizeof(nai_cmd_afu_binary_quant_v2_t))
        return run_afu_binary_quant(
            (const nai_cmd_afu_binary_quant_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_AFU_GLOBAL_AVGPOOL &&
        header->size_bytes == sizeof(nai_cmd_afu_global_avgpool_v2_t))
        return run_afu_global_avgpool(
            (const nai_cmd_afu_global_avgpool_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_UPSAMPLE_NEAREST &&
        header->size_bytes == sizeof(nai_cmd_upsample_nearest_v2_t))
        return run_upsample_nearest(
            (const nai_cmd_upsample_nearest_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_MAXPOOL && header->size_bytes == sizeof(nai_cmd_maxpool_v2_t))
        return run_maxpool((const nai_cmd_maxpool_v2_t *)header, view, resolver, ops);
    if ((header->type == NAI_CMD_LINEBUF_JOB || header->type == NAI_CMD_LINEBUF_SUBMIT) &&
        header->size_bytes == sizeof(nai_cmd_linebuf_job_v2_t))
        return run_linebuf_job((const nai_cmd_linebuf_job_v2_t *)header, ops);
    if ((header->type == NAI_CMD_LINEBUF_BINARY ||
         header->type == NAI_CMD_LINEBUF_BINARY_SUBMIT) &&
        header->size_bytes == sizeof(nai_cmd_linebuf_binary_v2_t))
        return run_linebuf_job((const nai_cmd_linebuf_job_v2_t *)header, ops);
    if (header->type == NAI_CMD_SYSTOLIC_WAIT &&
        header->size_bytes == sizeof(nai_cmd_control_v2_t))
        return run_systolic_wait((const nai_cmd_control_v2_t *)header, ops);
    if (header->type == NAI_CMD_COPY_LAYOUT &&
        header->size_bytes == sizeof(nai_cmd_copy_layout_v2_t))
        return run_copy((const nai_cmd_copy_layout_v2_t *)header, view, resolver, ops);
    if (header->type == NAI_CMD_AFU_DFL16 &&
        header->size_bytes == sizeof(nai_cmd_afu_dfl16_v2_t))
        return run_afu_dfl16((const nai_cmd_afu_dfl16_v2_t *)header, view, resolver, ops);
    if ((header->flags & (NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) ==
        (NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) return NAI_DISPATCH_OK;
    return NAI_DISPATCH_UNSUPPORTED;
}

static nai_dispatch_status_v2_t run_profiled_command(
    const nai_cmd_header_v2_t *header, const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver, const nai_runtime_ops_v2_t *ops,
    uint32_t command_id)
{
#if defined(NAI_PMU_PROFILE) && NAI_PMU_PROFILE
    extern void nai_pmu_command_begin(uint32_t command_id);
    extern void nai_pmu_command_end(uint32_t command_id);
    nai_pmu_command_begin(command_id);
#else
    (void)command_id;
#endif
    nai_dispatch_status_v2_t status = run_executable_command(header, view, resolver, ops);
#if defined(NAI_PMU_PROFILE) && NAI_PMU_PROFILE
    nai_pmu_command_end(command_id);
#endif
    return status;
}

static uint32_t affine_loop_descriptor_bytes(uint32_t patch_count)
{
    return (sizeof(nai_cmd_affine_loop_v2_t) +
        patch_count * sizeof(nai_cmd_affine_patch_v2_t) + 31u) & ~31u;
}

static uint32_t validate_affine_loop(
    const uint8_t *record, uint32_t available, uint32_t child_offsets[NAI_AFFINE_LOOP_MAX_BODY_COMMANDS],
    uint32_t child_sizes[NAI_AFFINE_LOOP_MAX_BODY_COMMANDS])
{
    const nai_cmd_affine_loop_v2_t *loop = (const nai_cmd_affine_loop_v2_t *)record;
    uint32_t body_offset;
    uint32_t cursor;
    uint32_t previous_patch = 0u;
    if (available < sizeof(*loop) || loop->header.type != NAI_CMD_AFFINE_LOOP ||
        loop->header.flags != 0u || loop->iteration_count < 2u ||
        loop->body_command_count == 0u ||
        loop->body_command_count > NAI_AFFINE_LOOP_MAX_BODY_COMMANDS ||
        loop->patch_count > NAI_AFFINE_LOOP_MAX_PATCHES ||
        loop->header.size_bytes != affine_loop_descriptor_bytes(loop->patch_count) ||
        loop->header.size_bytes > available || loop->body_bytes > available - loop->header.size_bytes ||
        loop->header.size_bytes + loop->body_bytes > NAI_AFFINE_LOOP_MAX_RECORD_BYTES)
        return 0u;
    if (NAI_TRUSTED_INVALID(!all_zero_bytes(
            record + sizeof(*loop) + loop->patch_count * sizeof(nai_cmd_affine_patch_v2_t),
            loop->header.size_bytes - sizeof(*loop) -
                loop->patch_count * sizeof(nai_cmd_affine_patch_v2_t)))) return 0u;

    body_offset = loop->header.size_bytes;
    cursor = body_offset;
    for (uint32_t child = 0u; child < loop->body_command_count; child++) {
        const nai_cmd_header_v2_t *header;
        if (cursor > body_offset + loop->body_bytes ||
            body_offset + loop->body_bytes - cursor < sizeof(nai_cmd_header_v2_t)) return 0u;
        header = (const nai_cmd_header_v2_t *)(record + cursor);
        if (!valid_executable_header(header, body_offset + loop->body_bytes - cursor) ||
            header->type == NAI_CMD_END || header->type == NAI_CMD_AFFINE_LOOP) return 0u;
        child_offsets[child] = cursor - body_offset;
        child_sizes[child] = header->size_bytes;
        cursor += header->size_bytes;
    }
    if (cursor != body_offset + loop->body_bytes) return 0u;

    const nai_cmd_affine_patch_v2_t *patches =
        (const nai_cmd_affine_patch_v2_t *)(record + sizeof(*loop));
    for (uint32_t patch = 0u; patch < loop->patch_count; patch++) {
        uint32_t byte_offset;
        uint32_t patchable = 0u;
        if (patches[patch].body_word_offset > 0x3fffffffu) return 0u;
        byte_offset = patches[patch].body_word_offset * 4u;
        if (byte_offset > loop->body_bytes || loop->body_bytes - byte_offset < 4u ||
            (patch != 0u && patches[patch].body_word_offset <= previous_patch)) return 0u;
        for (uint32_t child = 0u; child < loop->body_command_count; child++) {
            if (byte_offset >= child_offsets[child] + 8u &&
                byte_offset + 4u <= child_offsets[child] + child_sizes[child]) {
                patchable = 1u;
                break;
            }
        }
        if (!patchable) return 0u;
        previous_patch = patches[patch].body_word_offset;
    }
    return 1u;
}

static nai_dispatch_status_v2_t run_affine_loop_direct(
    const uint8_t *record, uint32_t available, const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver, const nai_runtime_ops_v2_t *ops,
    uint32_t command_limit, uint32_t *completed, uint32_t *consumed,
    uint32_t *failure_relative)
{
    uint32_t child_offsets[NAI_AFFINE_LOOP_MAX_BODY_COMMANDS];
    uint32_t child_sizes[NAI_AFFINE_LOOP_MAX_BODY_COMMANDS];
    uint32_t child_words[sizeof(nai_cmd_linebuf_binary_v2_t) / 4u];
    const nai_cmd_affine_loop_v2_t *loop = (const nai_cmd_affine_loop_v2_t *)record;
    if (!validate_affine_loop(record, available, child_offsets, child_sizes) ||
        loop->iteration_count > (command_limit - *completed) / loop->body_command_count)
        return NAI_DISPATCH_BAD_STREAM;
    const nai_cmd_affine_patch_v2_t *patches =
        (const nai_cmd_affine_patch_v2_t *)(record + sizeof(*loop));
    const uint32_t body_offset = loop->header.size_bytes;
    for (uint32_t iteration = 0u; iteration < loop->iteration_count; iteration++) {
        for (uint32_t child = 0u; child < loop->body_command_count; child++) {
            const uint32_t child_offset = child_offsets[child];
            const uint32_t child_size = child_sizes[child];
            if (child_size > sizeof(child_words)) return NAI_DISPATCH_BAD_STREAM;
            __builtin_memcpy(child_words, record + body_offset + child_offset, child_size);
            for (uint32_t patch = 0u; patch < loop->patch_count; patch++) {
                const uint32_t byte_offset = patches[patch].body_word_offset * 4u;
                if (byte_offset >= child_offset && byte_offset < child_offset + child_size) {
                    const uint32_t word = (byte_offset - child_offset) / 4u;
                    child_words[word] += iteration * patches[patch].delta;
                }
            }
            const nai_cmd_header_v2_t *header = (const nai_cmd_header_v2_t *)child_words;
            nai_dispatch_status_v2_t status = run_profiled_command(
                header, view, resolver, ops, *completed);
            if (status != NAI_DISPATCH_OK) {
                *failure_relative = body_offset + child_offset;
                return status;
            }
            ++*completed;
        }
    }
    *consumed = loop->header.size_bytes + loop->body_bytes;
    return NAI_DISPATCH_OK;
}

static nai_dispatch_status_v2_t run_affine_loop_buffer(
    uint8_t *record, uint32_t available, const nai_model_view_v1_t *view,
    const nai_resolver_v1_t *resolver, const nai_runtime_ops_v2_t *ops,
    uint32_t command_limit, uint32_t *completed, uint32_t *consumed,
    uint32_t *failure_relative)
{
    uint32_t child_offsets[NAI_AFFINE_LOOP_MAX_BODY_COMMANDS];
    uint32_t child_sizes[NAI_AFFINE_LOOP_MAX_BODY_COMMANDS];
    nai_cmd_affine_loop_v2_t loop;
    nai_cmd_affine_patch_v2_t patches[NAI_AFFINE_LOOP_MAX_PATCHES];
    __builtin_memcpy(&loop, record, sizeof(loop));
    if (!validate_affine_loop(record, available, child_offsets, child_sizes) ||
        loop.iteration_count > (command_limit - *completed) / loop.body_command_count)
        return NAI_DISPATCH_BAD_STREAM;
    __builtin_memcpy(patches, record + sizeof(loop),
        loop.patch_count * sizeof(nai_cmd_affine_patch_v2_t));
    const uint32_t body_offset = loop.header.size_bytes;
    for (uint32_t iteration = 0u; iteration < loop.iteration_count; iteration++) {
        if (iteration != 0u) {
            for (uint32_t patch = 0u; patch < loop.patch_count; patch++) {
                uint32_t *word = (uint32_t *)(record + body_offset +
                    patches[patch].body_word_offset * 4u);
                *word += patches[patch].delta;
            }
        }
        for (uint32_t child = 0u; child < loop.body_command_count; child++) {
            nai_cmd_header_v2_t *header = (nai_cmd_header_v2_t *)(record + body_offset +
                child_offsets[child]);
            __builtin_memcpy(record, header, sizeof(*header));
            nai_dispatch_status_v2_t status = run_profiled_command(
                header, view, resolver, ops, *completed);
            if (status != NAI_DISPATCH_OK) {
                *failure_relative = body_offset + child_offsets[child];
                return status;
            }
            ++*completed;
        }
    }
    *consumed = loop.header.size_bytes + loop.body_bytes;
    return NAI_DISPATCH_OK;
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
        if (!valid_executable_header(header, view->commands->size - offset)) {
            status = NAI_DISPATCH_BAD_COMMAND;
        } else if (header->type == NAI_CMD_END) {
            if (header->size_bytes != sizeof(nai_cmd_control_v2_t) ||
                !all_zero(((const nai_cmd_control_v2_t *)header)->reserved, 4u) ||
                completed != view->header->command_count) status = NAI_DISPATCH_BAD_STREAM;
            else {
                if (completed_commands != 0) *completed_commands = completed;
                return NAI_DISPATCH_OK;
            }
        } else if (header->type == NAI_CMD_AFFINE_LOOP) {
            uint32_t consumed = 0u;
            uint32_t failure_relative = 0u;
            status = run_affine_loop_direct((const uint8_t *)header,
                view->commands->size - offset, view, resolver, ops,
                view->header->command_count, &completed, &consumed, &failure_relative);
            if (status == NAI_DISPATCH_OK) {
                offset += consumed;
                continue;
            }
            if (failure_command_offset != 0)
                *failure_command_offset = view->commands->offset + offset + failure_relative;
        } else {
            status = run_profiled_command(header, view, resolver, ops, completed);
        }
        if (status != NAI_DISPATCH_OK) {
            if (completed_commands != 0) *completed_commands = completed;
            if (failure_command_offset != 0 && *failure_command_offset == 0u)
                *failure_command_offset = view->commands->offset + offset;
            return status;
        }
        completed++;
        offset += header->size_bytes;
    }
    return NAI_DISPATCH_BAD_STREAM;
}

nai_dispatch_status_v2_t nai_cmd_dispatch_stream_v2(const nai_model_view_v1_t *view,
                                                    const nai_resolver_v1_t *resolver,
                                                    const nai_runtime_ops_v2_t *ops,
                                                    const nai_model_reader_v1_t *reader,
                                                    void *command_buffer,
                                                    uint32_t command_buffer_bytes,
                                                    uint32_t *completed_commands,
                                                    uint32_t *failure_command_offset)
{
    uint32_t offset;
    uint32_t completed = 0u;

    if (completed_commands != 0) *completed_commands = 0u;
    if (failure_command_offset != 0) *failure_command_offset = 0u;
    if (view == 0 || view->header == 0 || view->commands == 0 || resolver == 0 || ops == 0 ||
        reader == 0 || reader->read == 0 || command_buffer == 0 ||
        command_buffer_bytes < sizeof(nai_cmd_gemm32_v2_t) ||
        view->header->entry_command_off < view->commands->offset) return NAI_DISPATCH_BAD_STREAM;
    offset = view->header->entry_command_off - view->commands->offset;

    while (offset < view->commands->size && completed <= view->header->command_count) {
        nai_cmd_header_v2_t header;
        nai_dispatch_status_v2_t status = NAI_DISPATCH_OK;
        uint32_t model_offset = view->commands->offset + offset;
        uint32_t prefetched_bytes = view->commands->size - offset;
        if (prefetched_bytes > sizeof(nai_cmd_linebuf_binary_v2_t))
            prefetched_bytes = sizeof(nai_cmd_linebuf_binary_v2_t);
        if (prefetched_bytes > command_buffer_bytes) prefetched_bytes = command_buffer_bytes;
        if (!valid_range(offset, sizeof(header), view->commands->size) ||
            reader->read(reader->context, model_offset, command_buffer, prefetched_bytes) != 0u)
            return NAI_DISPATCH_BAD_STREAM;
        __builtin_memcpy(&header, command_buffer, sizeof(header));
        if (!valid_executable_header(&header, view->commands->size - offset)) {
            status = NAI_DISPATCH_BAD_COMMAND;
        } else if (header.type == NAI_CMD_END) {
            if (header.size_bytes > prefetched_bytes)
                status = NAI_DISPATCH_BAD_STREAM;
            else if (header.size_bytes != sizeof(nai_cmd_control_v2_t) ||
                NAI_TRUSTED_INVALID(
                    !all_zero(((const nai_cmd_control_v2_t *)command_buffer)->reserved, 4u) ||
                    completed != view->header->command_count)) status = NAI_DISPATCH_BAD_STREAM;
            else {
                if (completed_commands != 0) *completed_commands = completed;
                return NAI_DISPATCH_OK;
            }
        } else if (header.type == NAI_CMD_AFFINE_LOOP) {
            const nai_cmd_affine_loop_v2_t *loop =
                (const nai_cmd_affine_loop_v2_t *)command_buffer;
            uint32_t record_bytes = 0u;
            uint32_t failure_relative = 0u;
            if (prefetched_bytes < sizeof(*loop) || loop->patch_count > NAI_AFFINE_LOOP_MAX_PATCHES ||
                header.size_bytes > NAI_AFFINE_LOOP_MAX_RECORD_BYTES ||
                header.size_bytes != affine_loop_descriptor_bytes(loop->patch_count) ||
                loop->body_bytes > NAI_AFFINE_LOOP_MAX_RECORD_BYTES - header.size_bytes) {
                status = NAI_DISPATCH_BAD_STREAM;
            } else {
                record_bytes = header.size_bytes + loop->body_bytes;
                if (record_bytes > command_buffer_bytes ||
                    !valid_range(offset, record_bytes, view->commands->size)) {
                    status = NAI_DISPATCH_BAD_STREAM;
                } else {
                    if (record_bytes > prefetched_bytes &&
                        reader->read(reader->context, model_offset + prefetched_bytes,
                            (uint8_t *)command_buffer + prefetched_bytes,
                            record_bytes - prefetched_bytes) != 0u)
                        status = NAI_DISPATCH_BAD_STREAM;
                    else status = run_affine_loop_buffer((uint8_t *)command_buffer,
                        record_bytes, view, resolver, ops, view->header->command_count,
                        &completed, &record_bytes, &failure_relative);
                }
                if (status == NAI_DISPATCH_OK) {
                    offset += record_bytes;
                    continue;
                }
                if (failure_command_offset != 0)
                    *failure_command_offset = model_offset + failure_relative;
            }
        } else {
            if (header.size_bytes > prefetched_bytes) status = NAI_DISPATCH_BAD_STREAM;
            else status = run_profiled_command(
                (const nai_cmd_header_v2_t *)command_buffer, view, resolver, ops, completed);
        }
        if (status != NAI_DISPATCH_OK) {
            if (completed_commands != 0) *completed_commands = completed;
            if (failure_command_offset != 0 && *failure_command_offset == 0u)
                *failure_command_offset = model_offset;
            return status;
        }
        completed++;
        offset += header.size_bytes;
    }
    return NAI_DISPATCH_BAD_STREAM;
}
