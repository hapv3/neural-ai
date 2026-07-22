#include "npu_runtime_ops.h"

#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "npu_layout_ops.h"
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

static uint32_t runtime_copy_layout(void *context, const nai_cmd_copy_layout_v2_t *command,
                                    uint32_t source_address, uint32_t destination_address)
{
    (void)context;
    return nai_copy_layout_v2(command, (const void *)(unsigned long)source_address,
                              (void *)(unsigned long)destination_address);
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
        runtime_copy_layout,
        runtime_barrier,
        runtime_rq_load
    };
    return &ops;
}
