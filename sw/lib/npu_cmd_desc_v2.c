#include "npu_cmd_desc_v2.h"

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
    nai_loader_status_t status = nai_resolve_ref_v1(view, resolver, ref, bytes, alignment, address);
    return status == NAI_LOADER_OK ? NAI_DISPATCH_OK : NAI_DISPATCH_BAD_REFERENCE;
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
    uint32_t source;
    uint32_t destination;
    if (command->length == 0u || !all_zero(command->reserved, 6u) || ops->dma_1d == 0)
        return NAI_DISPATCH_BAD_COMMAND;
    if (validate_dma_direction(&command->source, &command->destination,
            command->direction) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_COMMAND;
    if (resolve(view, resolver, &command->source, command->length, NAI_ALIGNMENT_BYTES, &source) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->destination, command->length, NAI_ALIGNMENT_BYTES, &destination) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    return ops->dma_1d(ops->context, source, destination, command->length, command->direction) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_rq_load(const nai_cmd_rq_load_v2_t *command,
                                             const nai_model_view_v1_t *view,
                                             const nai_resolver_v1_t *resolver,
                                             const nai_runtime_ops_v2_t *ops)
{
    uint32_t qparam_offset;
    uint32_t qparam_bytes;
    uint32_t address;
    if (command->qparam_count != 32u || command->reserved != 0u || ops->rq_load == 0 ||
        view->qparams == 0 || command->qparam_index > view->qparams->element_count ||
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
    address = resolver->model_base + view->qparams->offset + qparam_offset;
    if ((address & (NAI_ALIGNMENT_BYTES - 1u)) != 0u) return NAI_DISPATCH_BAD_REFERENCE;
    return ops->rq_load(ops->context, address, command->qparam_count,
        command->qparam_block) == 0u ? NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
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
    if (validate_dma_direction(&command->source, &command->destination,
            command->direction) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_COMMAND;
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
    if (validate_dma_direction(&command->source, &command->destination,
            command->direction) != NAI_DISPATCH_OK)
        return NAI_DISPATCH_BAD_COMMAND;
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
    uint32_t requant = command->header.type == NAI_CMD_GEMM32_REQUANT;
    uint32_t direct_requant = requant && command->partial_sums.region == 0u;
    uint32_t needs_partial = command->header.type == NAI_CMD_GEMM32_ACCUM ||
        (requant && !direct_requant);
    uint32_t ofm_row_bytes = requant ? 32u : 128u;
    uint32_t ofm_stride = command->ofm_row_stride != 0u ? command->ofm_row_stride : ofm_row_bytes;
    uint32_t partial_stride = command->partial_sum_row_stride != 0u ?
        command->partial_sum_row_stride : 128u;

    if (command->dim_m == 0u || command->dim_m > 256u || !all_zero(command->reserved, 8u) ||
        ops->gemm32 == 0 || ofm_stride < ofm_row_bytes || (ofm_stride & 31u) != 0u ||
        partial_stride < 128u || (partial_stride & 31u) != 0u ||
        (direct_requant && (command->partial_sums.index != 0u || command->partial_sums.offset != 0u)) ||
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

    if (command->rows == 0u || command->rows > 256u || command->input_c32_groups == 0u ||
        command->output_c32_groups != 1u || ops->pointwise_c32 == 0 ||
        !all_zero(command->reserved, 6u) ||
        command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH ||
        !multiply(command->input_c32_groups, command->output_c32_groups, &weight_tiles) ||
        !multiply(weight_tiles, 32u * 32u, &weight_bytes) ||
        !multiply(command->rows, 32u, &ifm_groups_bytes) ||
        !multiply(command->rows, 32u, &ofm_groups_bytes) ||
        command->input_group_stride_bytes < ifm_groups_bytes ||
        command->output_group_stride_bytes < ofm_groups_bytes ||
        (command->input_group_stride_bytes & (NAI_ALIGNMENT_BYTES - 1u)) != 0u ||
        (command->output_group_stride_bytes & (NAI_ALIGNMENT_BYTES - 1u)) != 0u ||
        !multiply(command->input_group_stride_bytes, command->input_c32_groups - 1u, &ifm_bytes) ||
        ifm_bytes > 0xffffffffu - ifm_groups_bytes ||
        !multiply(command->output_group_stride_bytes, command->output_c32_groups - 1u, &ofm_bytes) ||
        ofm_bytes > 0xffffffffu - ofm_groups_bytes ||
        !multiply(command->rows, 32u * 4u, &partial_bytes)) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    ifm_bytes += ifm_groups_bytes;
    ofm_bytes += ofm_groups_bytes;
    if (command->input_c32_groups == 1u) {
        if (command->partial_sums.region != 0u || command->partial_sums.index != 0u ||
            command->partial_sums.offset != 0u) return NAI_DISPATCH_BAD_COMMAND;
    } else if (command->partial_sums.region != NAI_REGION_TCDM_SCRATCH) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->weights, weight_bytes, NAI_ALIGNMENT_BYTES, &weights) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ifm, ifm_bytes, NAI_ALIGNMENT_BYTES, &ifm) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, ofm_bytes, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (command->input_c32_groups > 1u &&
        resolve(view, resolver, &command->partial_sums, partial_bytes,
                NAI_ALIGNMENT_BYTES, &partial_sums) != NAI_DISPATCH_OK) {
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
    uint32_t padded_input_h;
    uint32_t padded_input_w;

    if (command->input_h == 0u || command->input_w == 0u ||
        command->output_h == 0u || command->output_w == 0u ||
        command->channels == 0u || ops->depthwise_c32 == 0 ||
        !all_zero(command->reserved, 4u) ||
        command->ifm.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH ||
        (command->stride_h != 1u && command->stride_h != 2u) ||
        (command->stride_w != 1u && command->stride_w != 2u) ||
        command->pad_h != 1u || command->pad_w != 1u ||
        command->input_h > 0xffffu || command->input_w > 0xffffu ||
        command->output_h > 0xffffu || command->output_w > 0xffffu ||
        command->channels > 32u ||
        command->input_h > 0xffffffffu - 2u * command->pad_h ||
        command->input_w > 0xffffffffu - 2u * command->pad_w) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    padded_input_h = command->input_h + 2u * command->pad_h;
    padded_input_w = command->input_w + 2u * command->pad_w;
    if (padded_input_h < 3u || padded_input_w < 3u) return NAI_DISPATCH_BAD_COMMAND;
    expected_output_h = ((padded_input_h - 3u) / command->stride_h) + 1u;
    expected_output_w = ((padded_input_w - 3u) / command->stride_w) + 1u;
    groups = 1u;
    if (command->output_h != expected_output_h || command->output_w != expected_output_w ||
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
    if (command->length == 0u || command->mode != NAI_AFU_BINARY_ADD_I8 ||
        !all_zero(command->reserved, 4u) || ops->afu_binary == 0 ||
        command->lhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->rhs.region != NAI_REGION_TCDM_SCRATCH ||
        command->ofm.region != NAI_REGION_TCDM_SCRATCH) {
        return NAI_DISPATCH_BAD_COMMAND;
    }
    if (resolve(view, resolver, &command->lhs, command->length, NAI_ALIGNMENT_BYTES, &lhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->rhs, command->length, NAI_ALIGNMENT_BYTES, &rhs) != NAI_DISPATCH_OK ||
        resolve(view, resolver, &command->ofm, command->length, NAI_ALIGNMENT_BYTES, &ofm) != NAI_DISPATCH_OK) {
        return NAI_DISPATCH_BAD_REFERENCE;
    }
    if (ranges_overlap(lhs, ofm, command->length) ||
        ranges_overlap(rhs, ofm, command->length)) return NAI_DISPATCH_BAD_COMMAND;
    return ops->afu_binary(ops->context, command, lhs, rhs, ofm) == 0u ?
        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
}

static nai_dispatch_status_v2_t run_linebuf_job(
    const nai_cmd_linebuf_job_v2_t *command,
    const nai_runtime_ops_v2_t *ops)
{
    uint32_t expected_k_tiles;
    uint32_t kernel_elements;
    uint32_t expected_group_stationary;
    if (command->job.rows == 0u || command->job.rows > 256u ||
        command->job.k_tiles == 0u || command->job.k_tiles > 0xffffu ||
        command->job.linebuf.kernel_h == 0u || command->job.linebuf.kernel_h > 5u ||
        command->job.linebuf.kernel_w == 0u || command->job.linebuf.kernel_w > 5u ||
        !all_zero_bytes(command->reserved, sizeof(command->reserved)) || ops->linebuf_job == 0) {
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
        command->job.gemm.accum_en > 2u ||
        command->job.linebuf.coalesce > 1u || command->job.linebuf.kgen > 1u ||
        command->job.linebuf.pool > 1u || command->job.linebuf.c32_fast > 1u ||
        command->job.linebuf.depthwise > 1u || command->job.linebuf.c32_group_stationary > 1u ||
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
    expected_group_stationary =
        command->job.linebuf.coalesce == 1u &&
        command->job.linebuf.kgen == 1u &&
        command->job.linebuf.c32_fast == 1u &&
        command->job.linebuf.lane_base == 0u &&
        command->job.linebuf.block_valid_bytes == 32u &&
        command->job.linebuf.input_c >= 32u &&
        (command->job.linebuf.input_c & 31u) == 0u &&
        command->job.k_tiles > 1u;
    if (command->job.linebuf.c32_group_stationary != expected_group_stationary)
        return NAI_DISPATCH_BAD_COMMAND;
    return ops->linebuf_job(ops->context, command) == 0u ?
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
        } else if (header->type == NAI_CMD_RQ_LOAD && header->size_bytes == sizeof(nai_cmd_rq_load_v2_t)) {
            status = run_rq_load((const nai_cmd_rq_load_v2_t *)header, view, resolver, ops);
        } else if (header->type == NAI_CMD_DMA_1D && header->size_bytes == sizeof(nai_cmd_dma_1d_v2_t)) {
            status = run_dma_1d((const nai_cmd_dma_1d_v2_t *)header, view, resolver, ops);
        } else if (header->type == NAI_CMD_DMA_2D && header->size_bytes == sizeof(nai_cmd_dma_2d_v2_t)) {
            status = run_dma_2d((const nai_cmd_dma_2d_v2_t *)header, view, resolver, ops);
        } else if (header->type == NAI_CMD_DMA_3D && header->size_bytes == sizeof(nai_cmd_dma_3d_v2_t)) {
            status = run_dma_3d((const nai_cmd_dma_3d_v2_t *)header, view, resolver, ops);
        } else if ((header->type == NAI_CMD_GEMM32 || header->type == NAI_CMD_GEMM32_ACCUM ||
                    header->type == NAI_CMD_GEMM32_REQUANT) && header->size_bytes == sizeof(nai_cmd_gemm32_v2_t)) {
            status = run_gemm((const nai_cmd_gemm32_v2_t *)header, view, resolver, ops);
        } else if (header->type == NAI_CMD_POINTWISE_C32 &&
                   header->size_bytes == sizeof(nai_cmd_pointwise_c32_v2_t)) {
            status = run_pointwise_c32((const nai_cmd_pointwise_c32_v2_t *)header,
                view, resolver, ops);
        } else if (header->type == NAI_CMD_DEPTHWISE_C32 &&
                   header->size_bytes == sizeof(nai_cmd_depthwise_c32_v2_t)) {
            status = run_depthwise_c32((const nai_cmd_depthwise_c32_v2_t *)header,
                view, resolver, ops);
        } else if (header->type == NAI_CMD_AFU_BINARY &&
                   header->size_bytes == sizeof(nai_cmd_afu_binary_v2_t)) {
            status = run_afu_binary((const nai_cmd_afu_binary_v2_t *)header,
                view, resolver, ops);
        } else if (header->type == NAI_CMD_LINEBUF_JOB &&
                   header->size_bytes == sizeof(nai_cmd_linebuf_job_v2_t)) {
            status = run_linebuf_job((const nai_cmd_linebuf_job_v2_t *)header, ops);
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
        if (!valid_range(offset, sizeof(header), view->commands->size) ||
            reader->read(reader->context, model_offset, &header, sizeof(header)) != 0u)
            return NAI_DISPATCH_BAD_STREAM;
        if (header.size_bytes < 32u || (header.size_bytes & 31u) != 0u ||
            !valid_range(offset, header.size_bytes, view->commands->size) ||
            (header.flags & ~(NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) != 0u) {
            status = NAI_DISPATCH_BAD_COMMAND;
        } else if (header.type == NAI_CMD_END || header.type == NAI_CMD_BARRIER ||
                   header.type == NAI_CMD_RQ_LOAD ||
                   header.type == NAI_CMD_DMA_1D || header.type == NAI_CMD_DMA_2D ||
                   header.type == NAI_CMD_DMA_3D || header.type == NAI_CMD_GEMM32 ||
                   header.type == NAI_CMD_GEMM32_ACCUM || header.type == NAI_CMD_GEMM32_REQUANT ||
                   header.type == NAI_CMD_POINTWISE_C32 || header.type == NAI_CMD_DEPTHWISE_C32 ||
                   header.type == NAI_CMD_AFU_BINARY ||
                   header.type == NAI_CMD_LINEBUF_JOB ||
                   header.type == NAI_CMD_COPY_LAYOUT) {
            if (header.size_bytes > command_buffer_bytes ||
                reader->read(reader->context, model_offset, command_buffer, header.size_bytes) != 0u)
                status = NAI_DISPATCH_BAD_STREAM;
            else {
                if (header.type == NAI_CMD_END) {
                    if (header.size_bytes != sizeof(nai_cmd_control_v2_t) ||
                        !all_zero(((const nai_cmd_control_v2_t *)command_buffer)->reserved, 4u) ||
                        completed != view->header->command_count) status = NAI_DISPATCH_BAD_STREAM;
                    else {
                        if (completed_commands != 0) *completed_commands = completed;
                        return NAI_DISPATCH_OK;
                    }
                } else if (header.type == NAI_CMD_BARRIER && header.size_bytes == sizeof(nai_cmd_control_v2_t)) {
                    status = all_zero(((const nai_cmd_control_v2_t *)command_buffer)->reserved, 4u) &&
                        ops->barrier != 0 && ops->barrier(ops->context) == 0u ?
                        NAI_DISPATCH_OK : NAI_DISPATCH_OPERATION_FAILED;
                } else if (header.type == NAI_CMD_RQ_LOAD &&
                           header.size_bytes == sizeof(nai_cmd_rq_load_v2_t)) {
                    status = run_rq_load((const nai_cmd_rq_load_v2_t *)command_buffer,
                        view, resolver, ops);
                } else if (header.type == NAI_CMD_DMA_1D && header.size_bytes == sizeof(nai_cmd_dma_1d_v2_t)) {
                    status = run_dma_1d((const nai_cmd_dma_1d_v2_t *)command_buffer, view, resolver, ops);
                } else if (header.type == NAI_CMD_DMA_2D && header.size_bytes == sizeof(nai_cmd_dma_2d_v2_t)) {
                    status = run_dma_2d((const nai_cmd_dma_2d_v2_t *)command_buffer, view, resolver, ops);
                } else if (header.type == NAI_CMD_DMA_3D && header.size_bytes == sizeof(nai_cmd_dma_3d_v2_t)) {
                    status = run_dma_3d((const nai_cmd_dma_3d_v2_t *)command_buffer, view, resolver, ops);
                } else if ((header.type == NAI_CMD_GEMM32 || header.type == NAI_CMD_GEMM32_ACCUM ||
                            header.type == NAI_CMD_GEMM32_REQUANT) &&
                           header.size_bytes == sizeof(nai_cmd_gemm32_v2_t)) {
                    status = run_gemm((const nai_cmd_gemm32_v2_t *)command_buffer, view, resolver, ops);
                } else if (header.type == NAI_CMD_POINTWISE_C32 &&
                           header.size_bytes == sizeof(nai_cmd_pointwise_c32_v2_t)) {
                    status = run_pointwise_c32((const nai_cmd_pointwise_c32_v2_t *)command_buffer,
                        view, resolver, ops);
                } else if (header.type == NAI_CMD_DEPTHWISE_C32 &&
                           header.size_bytes == sizeof(nai_cmd_depthwise_c32_v2_t)) {
                    status = run_depthwise_c32((const nai_cmd_depthwise_c32_v2_t *)command_buffer,
                        view, resolver, ops);
                } else if (header.type == NAI_CMD_AFU_BINARY &&
                           header.size_bytes == sizeof(nai_cmd_afu_binary_v2_t)) {
                    status = run_afu_binary((const nai_cmd_afu_binary_v2_t *)command_buffer,
                        view, resolver, ops);
                } else if (header.type == NAI_CMD_LINEBUF_JOB &&
                           header.size_bytes == sizeof(nai_cmd_linebuf_job_v2_t)) {
                    status = run_linebuf_job((const nai_cmd_linebuf_job_v2_t *)command_buffer, ops);
                } else if (header.type == NAI_CMD_COPY_LAYOUT &&
                           header.size_bytes == sizeof(nai_cmd_copy_layout_v2_t)) {
                    status = run_copy((const nai_cmd_copy_layout_v2_t *)command_buffer, view, resolver, ops);
                } else {
                    status = NAI_DISPATCH_BAD_COMMAND;
                }
            }
        } else if ((header.flags & (NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) ==
                   (NAI_CMD_FLAG_OPTIONAL | NAI_CMD_FLAG_SKIPPABLE)) {
            status = NAI_DISPATCH_OK;
        } else {
            status = NAI_DISPATCH_UNSUPPORTED;
        }
        if (status != NAI_DISPATCH_OK) {
            if (completed_commands != 0) *completed_commands = completed;
            if (failure_command_offset != 0) *failure_command_offset = model_offset;
            return status;
        }
        completed++;
        offset += header.size_bytes;
    }
    return NAI_DISPATCH_BAD_STREAM;
}
