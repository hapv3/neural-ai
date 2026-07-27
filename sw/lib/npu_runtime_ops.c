#include "npu_runtime_ops.h"

#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "npu_layout_ops.h"
#include "npu_memory_map.h"
#include "npu_quant_buffer.h"

static uint32_t wait_transfer(uint32_t direction, int transfer_id)
{
    return transfer_id > 0 && idma_mm_wait_for_completion(direction, (uint32_t)transfer_id) ? 0u : 1u;
}

static uint32_t dma_direction(uint32_t source, uint32_t destination, uint32_t requested)
{
    uint32_t source_l1 = idma_mm_is_l1_addr(source);
    uint32_t destination_l1 = idma_mm_is_l1_addr(destination);
    if (source_l1 && destination_l1) return 2u;
    if (!source_l1 && destination_l1) return IDMA_DIR_L2_TO_L1;
    if (source_l1 && !destination_l1) return IDMA_DIR_L1_TO_L2;
    return requested;
}

static uint32_t runtime_dma_1d(void *context, uint32_t source, uint32_t destination,
                               uint32_t length, uint32_t requested)
{
    uint32_t direction = dma_direction(source, destination, requested);
    (void)context;
    if (direction == IDMA_DIR_L2_TO_L1)
        return wait_transfer(direction, idma_L2ToL1(source, destination, length));
    if (direction == IDMA_DIR_L1_TO_L2)
        return wait_transfer(direction, idma_L1ToL2(source, destination, length));
    if (direction == 2u) {
        idma_L1ToL1(source, destination, length);
        return 0u;
    }
    return 1u;
}

static uint32_t runtime_dma_2d(void *context, uint32_t source, uint32_t destination,
                               uint32_t length, uint32_t source_stride,
                               uint32_t destination_stride, uint32_t repetitions,
                               uint32_t requested)
{
    uint32_t direction = dma_direction(source, destination, requested);
    (void)context;
    if (direction == IDMA_DIR_L2_TO_L1)
        return wait_transfer(direction, idma_L2ToL1_2d(source, destination, length,
            source_stride, destination_stride, repetitions));
    if (direction == IDMA_DIR_L1_TO_L2)
        return wait_transfer(direction, idma_L1ToL2_2d(source, destination, length,
            source_stride, destination_stride, repetitions));
    if (direction == 2u) {
        idma_L1ToL1_2d(source, destination, length, source_stride, destination_stride, repetitions);
        return 0u;
    }
    return 1u;
}

static uint32_t runtime_dma_3d(void *context, uint32_t source, uint32_t destination,
                               uint32_t length, uint32_t source_stride_2,
                               uint32_t destination_stride_2, uint32_t repetitions_2,
                               uint32_t source_stride_3, uint32_t destination_stride_3,
                               uint32_t repetitions_3, uint32_t requested)
{
    uint32_t direction = dma_direction(source, destination, requested);
    (void)context;
    if (direction == IDMA_DIR_L2_TO_L1)
        return wait_transfer(direction, idma_L2ToL1_3d(source, destination, length,
            source_stride_2, destination_stride_2, repetitions_2,
            source_stride_3, destination_stride_3, repetitions_3));
    if (direction == IDMA_DIR_L1_TO_L2)
        return wait_transfer(direction, idma_L1ToL2_3d(source, destination, length,
            source_stride_2, destination_stride_2, repetitions_2,
            source_stride_3, destination_stride_3, repetitions_3));
    if (direction == 2u) {
        idma_L1ToL1_3d(source, destination, length,
            source_stride_2, destination_stride_2, repetitions_2,
            source_stride_3, destination_stride_3, repetitions_3);
        return 0u;
    }
    return 1u;
}

static uint32_t runtime_gemm32(void *context, const nai_cmd_gemm32_v2_t *command,
                               uint32_t weights, uint32_t ifm,
                               uint32_t partial_sums, uint32_t ofm)
{
    (void)context;
    if (command->header.type == NAI_CMD_GEMM32_REQUANT &&
        !nai_quant_buffer_is_loaded_v1(command->qparam_block)) return 1u;
    /* The systolic engine reads weights from local TCDM.  Model constants are
       resolved in L2 by the command ABI, so stage each 1 KiB GEMM tile in the
       reserved command window before programming the engine. */
    if (!idma_memcpy_blocking(weights, NPU_CMD_TCDM_BASE, 32u * 32u)) return 1u;
    weights = NPU_CMD_TCDM_BASE;
    if (command->header.type == NAI_CMD_GEMM32) {
        systolic_gemm32_strided(weights, ifm, ofm, command->dim_m,
                                command->ofm_row_stride);
    } else if (command->header.type == NAI_CMD_GEMM32_ACCUM) {
        systolic_gemm32_accumulate_strided(weights, ifm, partial_sums, ofm, command->dim_m,
                                           command->ofm_row_stride,
                                           command->partial_sum_row_stride);
    } else if (command->header.type == NAI_CMD_GEMM32_REQUANT) {
        if (command->partial_sums.region == 0u) {
            systolic_gemm32_requant_strided(weights, ifm, ofm, command->dim_m,
                                            command->ofm_row_stride);
        } else {
            systolic_gemm32_accumulate_requant_strided(weights, ifm, partial_sums, ofm,
                                                       command->dim_m,
                                                       command->ofm_row_stride,
                                                       command->partial_sum_row_stride);
        }
    } else {
        return 1u;
    }
    return 0u;
}

static uint32_t runtime_pointwise_c32(void *context,
                                      const nai_cmd_pointwise_c32_v2_t *command,
                                      uint32_t weights, uint32_t ifm,
                                      uint32_t partial_sums, uint32_t ofm)
{
    const uint32_t weight_tile_bytes = 32u * 32u;
    const uint32_t input_group_stride = command->input_group_stride_bytes;
    const uint32_t output_group_stride = command->output_group_stride_bytes;
    (void)context;
    if (!nai_quant_buffer_is_loaded_v1(command->qparam_block) ||
        command->rows == 0u || command->input_c32_groups == 0u ||
        command->output_c32_groups == 0u) return 1u;

    for (uint32_t output_group = 0u; output_group < command->output_c32_groups; output_group++) {
        const uint32_t output_address = ofm + output_group * output_group_stride;
        const uint32_t output_weight_base = weights +
            output_group * command->input_c32_groups * weight_tile_bytes;
        for (uint32_t input_group = 0u; input_group < command->input_c32_groups; input_group++) {
            const uint32_t weight_address = NPU_CMD_TCDM_BASE;
            const uint32_t input_address = ifm + input_group * input_group_stride;
            if (!idma_memcpy_blocking(output_weight_base + input_group * weight_tile_bytes,
                                      weight_address, weight_tile_bytes)) return 1u;
            if (command->input_c32_groups == 1u) {
                systolic_gemm32_requant(weight_address, input_address, output_address,
                                        command->rows);
            } else if (input_group == 0u) {
                systolic_gemm32(weight_address, input_address, partial_sums, command->rows);
            } else if (input_group + 1u == command->input_c32_groups) {
                systolic_gemm32_accumulate_requant(weight_address, input_address,
                                                   partial_sums, output_address,
                                                   command->rows);
            } else {
                systolic_gemm32_accumulate(weight_address, input_address, partial_sums,
                                           partial_sums, command->rows);
            }
        }
    }
    systolic_requant_disable();
    return 0u;
}

static void runtime_zero(uint32_t address, uint32_t bytes)
{
    volatile uint8_t *destination = (volatile uint8_t *)(unsigned long)address;
    for (uint32_t byte = 0; byte < bytes; byte++) destination[byte] = 0u;
}

static uint32_t runtime_layout_dma(uint32_t source, uint32_t destination, uint32_t length,
                                   uint32_t source_stride, uint32_t destination_stride,
                                   uint32_t repetitions, uint32_t direction)
{
    if (direction == IDMA_DIR_L2_TO_L1)
        return wait_transfer(direction, idma_L2ToL1_2d(source, destination, length,
            source_stride, destination_stride, repetitions));
    if (direction == IDMA_DIR_L1_TO_L2)
        return wait_transfer(direction, idma_L1ToL2_2d(source, destination, length,
            source_stride, destination_stride, repetitions));
    if (direction == 2u) {
        idma_L1ToL1_2d(source, destination, length, source_stride, destination_stride, repetitions);
        return 0u;
    }
    return 1u;
}

static uint32_t runtime_copy_layout(void *context, const nai_cmd_copy_layout_v2_t *command,
                                    uint32_t source_address, uint32_t destination_address)
{
    uint32_t source_l1;
    uint32_t destination_l1;
    uint32_t direction;
    uint32_t element_bytes;
    uint32_t pixels;
    uint32_t channels;
    uint32_t padded_channels;
    uint32_t native_bytes;

    (void)context;
    if (command == 0 || source_address == 0u || destination_address == 0u) return 1u;
    source_l1 = idma_mm_is_l1_addr(source_address);
    destination_l1 = idma_mm_is_l1_addr(destination_address);
    if (source_l1 && destination_l1)
        return nai_copy_layout_v2(command, (const void *)(unsigned long)source_address,
                                  (void *)(unsigned long)destination_address);
    if (!source_l1 && !destination_l1) return 1u;

    if (command->data_type == NAI_DTYPE_I8) element_bytes = 1u;
    else if (command->data_type == NAI_DTYPE_I32) element_bytes = 4u;
    else return 1u;
    pixels = command->dimensions[0] * command->dimensions[1] * command->dimensions[2];
    channels = command->valid_channels;
    padded_channels = (channels + 31u) & ~31u;
    native_bytes = pixels * padded_channels * element_bytes;
    direction = dma_direction(source_address, destination_address, 0u);

    if (command->mode == NAI_COPY_NHWC_TO_ROW32 || command->mode == NAI_COPY_ROW32_TO_NHWC) {
        uint32_t compact_stride = channels * element_bytes;
        uint32_t native_stride = padded_channels * element_bytes;
        uint32_t length = compact_stride;
        if (command->source_row_stride == 0u || command->destination_row_stride == 0u)
            return 1u;
        if (command->mode == NAI_COPY_NHWC_TO_ROW32) {
            if (command->source_row_stride != compact_stride ||
                command->destination_row_stride != native_stride) return 1u;
            if (destination_l1) runtime_zero(destination_address, native_bytes);
        } else {
            if (command->source_row_stride != native_stride ||
                command->destination_row_stride != compact_stride) return 1u;
            length = compact_stride;
        }
        return runtime_layout_dma(source_address, destination_address, length,
                                  command->source_row_stride, command->destination_row_stride,
                                  pixels, direction);
    }

    if (command->mode == NAI_COPY_NHWC_TO_C32 || command->mode == NAI_COPY_C32_TO_NHWC) {
        if (command->mode == NAI_COPY_NHWC_TO_C32 && destination_l1)
            runtime_zero(destination_address, native_bytes);
        for (uint32_t group = 0u; group < padded_channels / 32u; group++) {
            uint32_t group_channels = channels - group * 32u;
            uint32_t source_stride;
            uint32_t destination_stride;
            uint32_t source_offset;
            uint32_t destination_offset;
            if (group_channels > 32u) group_channels = 32u;
            if (command->mode == NAI_COPY_NHWC_TO_C32) {
                source_stride = channels * element_bytes;
                destination_stride = 32u * element_bytes;
                source_offset = group * 32u * element_bytes;
                destination_offset = group * pixels * 32u * element_bytes;
            } else {
                source_stride = 32u * element_bytes;
                destination_stride = channels * element_bytes;
                source_offset = group * pixels * 32u * element_bytes;
                destination_offset = group * 32u * element_bytes;
            }
            if (runtime_layout_dma(source_address + source_offset, destination_address + destination_offset,
                                   group_channels * element_bytes, source_stride, destination_stride,
                                   pixels, direction) != 0u) return 1u;
        }
        return 0u;
    }
    return 1u;
}

static uint32_t runtime_barrier(void *context)
{
    (void)context;
    dma_barrier();
    return 0u;
}

static uint32_t runtime_rq_load(void *context, uint32_t qparam_address,
                                uint32_t qparam_count, uint32_t qparam_block)
{
    (void)context;
    return (uint32_t)nai_quant_buffer_load_l2_v1(qparam_address, qparam_count, qparam_block);
}

const nai_runtime_ops_v2_t *nai_default_runtime_ops_v2(void)
{
    static const nai_runtime_ops_v2_t ops = {
        0,
        runtime_dma_1d,
        runtime_dma_2d,
        runtime_dma_3d,
        runtime_gemm32,
        runtime_pointwise_c32,
        runtime_copy_layout,
        runtime_barrier,
        runtime_rq_load
    };
    return &ops;
}
