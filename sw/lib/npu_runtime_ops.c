#include "npu_runtime_ops.h"

#include "hal_afu.h"
#include "hal_systolic.h"
#include "idma_mm_utils.h"
#include "npu_layout_ops.h"
#include "npu_memory_map.h"
#include "npu_quant_buffer.h"
#include "spatz_ops.h"

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
        spatz_vec_copy_i8((const int8_t *)(unsigned long)source,
                          (int8_t *)(unsigned long)destination, length);
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
    const uint32_t stripe_rows = 256u;
    const uint32_t cached_weight_slots = 3u;
    const uint32_t input_group_stride = command->input_group_stride_bytes;
    const uint32_t output_group_stride = command->output_group_stride_bytes;
    uint32_t cached_weight_sources[3] = {0u, 0u, 0u};
    uint32_t cached_weight_valid[3] = {0u, 0u, 0u};
    (void)context;
    if (!nai_quant_buffer_is_loaded_v1(command->qparam_block) ||
        command->rows == 0u || command->input_c32_groups == 0u ||
        command->output_c32_groups != 1u) return 1u;

    for (uint32_t output_group = 0u; output_group < command->output_c32_groups; output_group++) {
        const uint32_t output_address = ofm + output_group * output_group_stride;
        const uint32_t output_weight_base = weights +
            output_group * command->input_c32_groups * weight_tile_bytes;
        /* Keep the row-stripe loop outside the input-group loop.  The partial
           sum scratch is intentionally only one 256-row stripe, so processing
           all input groups for a stripe must complete before the next stripe.
           Three 1 KiB command-TCDM slots retain the first input-group tiles;
           a fourth group (if present) uses the transient slot and is reloaded
           for each stripe.  The source keys are per invocation, preventing a
           stale tile from a previous command from being reused. */
        for (uint32_t row_base = 0u; row_base < command->rows; row_base += stripe_rows) {
            const uint32_t rows = command->rows - row_base > stripe_rows ?
                stripe_rows : command->rows - row_base;
            for (uint32_t input_group = 0u;
                 input_group < command->input_c32_groups; input_group++) {
                uint32_t weight_address;
                const uint32_t weight_source = output_weight_base +
                    input_group * weight_tile_bytes;
                if (input_group < cached_weight_slots) {
                    weight_address = NPU_CMD_TCDM_BASE +
                        (input_group + 1u) * weight_tile_bytes;
                    if (!cached_weight_valid[input_group] ||
                        cached_weight_sources[input_group] != weight_source) {
                        if (!idma_memcpy_blocking(weight_source, weight_address,
                                                  weight_tile_bytes)) return 1u;
                        cached_weight_sources[input_group] = weight_source;
                        cached_weight_valid[input_group] = 1u;
                    }
                } else {
                    weight_address = NPU_CMD_TCDM_BASE;
                    if (!idma_memcpy_blocking(weight_source, weight_address,
                                              weight_tile_bytes)) return 1u;
                }
                const uint32_t input_address = ifm + input_group * input_group_stride +
                    row_base * 32u;
                const uint32_t output_row_address = output_address + row_base * 32u;
                if (command->input_c32_groups == 1u) {
                    systolic_gemm32_requant(weight_address, input_address,
                                            output_row_address, rows);
                } else if (input_group == 0u) {
                    systolic_gemm32(weight_address, input_address, partial_sums, rows);
                } else if (input_group + 1u == command->input_c32_groups) {
                    systolic_gemm32_accumulate_requant(weight_address, input_address,
                                                       partial_sums, output_row_address,
                                                       rows);
                } else {
                    systolic_gemm32_accumulate(weight_address, input_address, partial_sums,
                                               partial_sums, rows);
                }
            }
        }
    }
    systolic_requant_disable();
    return 0u;
}

static uint32_t runtime_depthwise_c32(void *context,
                                      const nai_cmd_depthwise_c32_v2_t *command,
                                      uint32_t weights, uint32_t ifm,
                                      uint32_t ofm)
{
    const uint32_t weight_bytes = 3u * 3u * 32u;
    (void)context;
    if (command->channels == 0u || command->channels > 32u ||
        !nai_quant_buffer_is_loaded_v1(command->qparam_block) ||
        !idma_memcpy_blocking(weights, NPU_CMD_TCDM_BASE, weight_bytes)) return 1u;
    systolic_depthwise3x3_c32_requant_channels(ifm, NPU_CMD_TCDM_BASE, ofm,
                                               command->input_h, command->input_w,
                                               command->output_h, command->output_w,
                                               command->channels, command->stride_h,
                                               command->stride_w, command->pad_h,
                                               command->pad_w);
    systolic_requant_disable();
    return 0u;
}

static uint32_t runtime_afu_binary(void *context,
                                   const nai_cmd_afu_binary_v2_t *command,
                                   uint32_t lhs, uint32_t rhs, uint32_t ofm)
{
    (void)context;
    if (command->mode != NAI_AFU_BINARY_ADD_I8) return 1u;
    afu_start_add_i8(lhs, rhs, ofm, command->length);
    return afu_wait_done(1000000u) ? 0u : 1u;
}

static uint32_t runtime_spatz_add(void *context,
                                  const nai_cmd_spatz_add_v2_t *command,
                                  uint32_t lhs, uint32_t rhs, uint32_t ofm)
{
    const spatz_quantized_add_params_t params = {
        command->lhs_scale, command->lhs_shift,
        command->rhs_scale, command->rhs_shift,
        command->output_scale, command->output_shift,
        command->lhs_zero_point, command->rhs_zero_point,
        command->output_zero_point, command->clamp_min,
        command->clamp_max, command->double_round_shift,
    };
    (void)context;
    spatz_quantized_add_i8((const int8_t *)(unsigned long)lhs,
        (const int8_t *)(unsigned long)rhs, (int8_t *)(unsigned long)ofm,
        command->length, &params);
    return 0u;
}

static uint32_t runtime_afu_lut(void *context,
                                const nai_cmd_afu_lut_v2_t *command,
                                uint32_t ifm, uint32_t ofm, uint32_t lut)
{
    const uint8_t *lut_bytes;
    (void)context;
    if (command->lut.region == NAI_REGION_MODEL_CONSTANTS) {
        if (!idma_memcpy_blocking(lut, NPU_CMD_TCDM_BASE, 256u)) return 1u;
        lut = NPU_CMD_TCDM_BASE;
    }
    lut_bytes = (const uint8_t *)(unsigned long)lut;
    afu_preload(ifm, ofm, command->length, NPU_AFU_MODE_E8);
    for (uint32_t index = 0u; index < 256u; index++) {
        afu_load_lut_entry(index, (uint32_t)lut_bytes[index]);
    }
    afu_start_preloaded();
    return afu_wait_done(1000000u) ? 0u : 1u;
}

static uint32_t runtime_afu_global_avgpool(
    void *context, const nai_cmd_afu_global_avgpool_v2_t *command,
    uint32_t ifm, uint32_t ofm)
{
    const uint32_t spatial_count = command->input_h * command->input_w;
    const uint32_t groups = (command->channels + 31u) / 32u;
    const uint32_t input_bytes = spatial_count * groups * 32u;
    const uint32_t reciprocal_q31 =
        (uint32_t)((1ull << 31) / (uint64_t)spatial_count);
    (void)context;
    afu_load_lut_entry(0u, reciprocal_q31);
    afu_start_global_avgpool_c32(ifm, ofm, input_bytes, spatial_count);
    return afu_wait_done(100000u + input_bytes) ? 0u : 1u;
}

static uint32_t runtime_upsample_nearest(
    void *context, const nai_cmd_upsample_nearest_v2_t *command,
    uint32_t ifm, uint32_t ofm)
{
    (void)context;
    spatz_upsample_nearest2x_c32_i8(
        (const int8_t *)(unsigned long)ifm, (int8_t *)(unsigned long)ofm,
        command->input_h, command->input_w);
    return 0u;
}

static uint32_t runtime_maxpool(
    void *context, const nai_cmd_maxpool_v2_t *command,
    uint32_t ifm, uint32_t ofm)
{
    (void)context;
    systolic_maxpool5x5s1p2_c32_linebuf(
        ifm, ofm, command->input_h, command->input_w);
    return 0u;
}

static uint32_t runtime_linebuf_job(void *context, const nai_cmd_linebuf_job_v2_t *command)
{
    systolic_linebuf_cfg_t linebuf = command->job.linebuf;
    systolic_gemm32_req_t gemm = command->job.gemm;
    (void)context;
    /* The fixed 124-byte linebuffer wire record carries compact TCDM offsets,
       matching RefV1 scratch offsets used by the compiler.  Resolve those
       offsets to the physical TCDM window immediately before programming HAL;
       the input address additionally accounts for the HAL's virtual top
       padding row. */
    if (linebuf.input_base >= NPU_TCDM_SIZE || gemm.weight_addr >= NPU_TCDM_SIZE ||
        gemm.ifm_addr >= NPU_TCDM_SIZE || gemm.psum_addr >= NPU_TCDM_SIZE ||
        gemm.ofm_addr >= NPU_TCDM_SIZE) return 1u;
    /* The HAL linebuffer contract addresses the virtual top-padding row by
       subtracting pad_h*row_stride from input_base.  Keep the wire field a
       compact TCDM offset, then perform that subtraction after relocation so
       an offset of zero does not wrap the 32-bit ABI arithmetic. */
    if (linebuf.pad_h != 0u &&
        linebuf.row_stride_bytes > 0xffffffffu / linebuf.pad_h) return 1u;
    {
        uint32_t pad_bytes = (uint32_t)linebuf.pad_h * linebuf.row_stride_bytes;
        uint32_t physical_input = NPU_TCDM_BASE + linebuf.input_base;
        if (physical_input < pad_bytes) return 1u;
        linebuf.input_base = physical_input - pad_bytes;
    }
    gemm.weight_addr += NPU_TCDM_BASE;
    gemm.ifm_addr += NPU_TCDM_BASE;
    gemm.psum_addr += NPU_TCDM_BASE;
    gemm.ofm_addr += NPU_TCDM_BASE;
    systolic_linebuf_config(&linebuf);
    if (gemm.accum_en == 1u) {
        /* Initialize the external partial-sum tile without reading its old
           contents.  The raw INT32 result is written directly to PSUM. */
        systolic_gemm32_linebuf_ktiles_strided(
            gemm.weight_addr, gemm.psum_addr, gemm.psum_addr, gemm.dim_m,
            gemm.psum_row_stride_bytes, gemm.ofm_tile_cols);
    } else if (gemm.accum_en == 3u) {
        /* Intermediate input groups accumulate in place in the external
           partial-sum tile; only the final group writes the quantized OFM. */
        systolic_gemm32_linebuf_ktiles_accumulate_strided(
            gemm.weight_addr, gemm.psum_addr, gemm.psum_addr, gemm.dim_m,
            gemm.psum_row_stride_bytes, gemm.ofm_tile_cols,
            gemm.psum_row_stride_bytes);
    } else if (command->job.k_tiles > 1u) {
        if (gemm.accum_en == 2u) {
            systolic_gemm32_linebuf_ktiles_accumulate_requant_strided(
                gemm.weight_addr, gemm.psum_addr, gemm.ofm_addr, gemm.dim_m,
                gemm.ofm_row_stride_bytes, gemm.ofm_tile_cols,
                gemm.psum_row_stride_bytes);
        } else {
            systolic_gemm32_linebuf_ktiles_requant_strided(
                gemm.weight_addr, gemm.psum_addr, gemm.ofm_addr, gemm.dim_m,
                gemm.ofm_row_stride_bytes, gemm.ofm_tile_cols);
        }
    } else if (gemm.accum_en == 2u) {
        systolic_gemm32_linebuf_accumulate_requant(
            gemm.weight_addr, gemm.psum_addr, gemm.ofm_addr, gemm.dim_m);
    } else {
        systolic_gemm32_linebuf_requant(gemm.weight_addr, gemm.ofm_addr, gemm.dim_m);
    }
    systolic_linebuf_disable();
    systolic_requant_disable();
    return 0u;
}

static void runtime_zero(uint32_t address, uint32_t bytes)
{
    spatz_vec_zero_i8((int8_t *)(unsigned long)address, bytes);
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

/* Wide 2D rows with a compact, non-beat-aligned stride can leave the current
   RTL iDMA path busy indefinitely. Keep every affine row segment within one
   32-byte data beat while preserving the logical source/destination strides. */
#define NAI_LAYOUT_DMA_CHUNK_BYTES 32u

static uint32_t runtime_layout_dma_repetitions_chunked(
    uint32_t source, uint32_t destination, uint32_t length,
    uint32_t source_stride, uint32_t destination_stride,
    uint32_t repetitions, uint32_t direction)
{
    uint32_t completed = 0u;
    while (completed < repetitions) {
        const uint32_t chunk = (repetitions - completed) > 256u ?
            256u : (repetitions - completed);
        uint32_t source_offset;
        uint32_t destination_offset;
        if (source_stride != 0u && completed > 0xffffffffu / source_stride) return 1u;
        if (destination_stride != 0u && completed > 0xffffffffu / destination_stride) return 1u;
        source_offset = completed * source_stride;
        destination_offset = completed * destination_stride;
        if (source > 0xffffffffu - source_offset ||
            destination > 0xffffffffu - destination_offset ||
            (chunk == 1u ?
                runtime_dma_1d(0, source + source_offset, destination + destination_offset,
                               length, direction) :
                runtime_layout_dma(source + source_offset, destination + destination_offset,
                                   length, source_stride, destination_stride, chunk, direction)) != 0u)
            return 1u;
        completed += chunk;
    }
    return 0u;
}

static uint32_t runtime_layout_dma_chunked(uint32_t source, uint32_t destination,
                                           uint32_t length, uint32_t source_stride,
                                           uint32_t destination_stride, uint32_t repetitions,
                                           uint32_t direction)
{
    uint32_t completed = 0u;
    while (completed < length) {
        const uint32_t chunk = (length - completed) > NAI_LAYOUT_DMA_CHUNK_BYTES ?
            NAI_LAYOUT_DMA_CHUNK_BYTES : (length - completed);
        if (source > 0xffffffffu - completed ||
            destination > 0xffffffffu - completed ||
            runtime_layout_dma_repetitions_chunked(
                source + completed, destination + completed, chunk,
                source_stride, destination_stride, repetitions, direction) != 0u)
            return 1u;
        completed += chunk;
    }
    return 0u;
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
        return runtime_layout_dma_chunked(source_address, destination_address, length,
                                          command->source_row_stride, command->destination_row_stride,
                                          pixels, direction);
    }

    if (command->mode == NAI_COPY_NHWC_TO_C32 || command->mode == NAI_COPY_C32_TO_NHWC) {
        if (command->mode == NAI_COPY_NHWC_TO_C32 && destination_l1 &&
            padded_channels != channels)
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
            if (source_address > 0xffffffffu - source_offset ||
                destination_address > 0xffffffffu - destination_offset ||
                runtime_layout_dma_chunked(source_address + source_offset,
                                            destination_address + destination_offset,
                                            group_channels * element_bytes, source_stride,
                                            destination_stride, pixels, direction) != 0u)
                return 1u;
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
        runtime_depthwise_c32,
        runtime_afu_lut,
        runtime_afu_binary,
        runtime_spatz_add,
        runtime_afu_global_avgpool,
        runtime_upsample_nearest,
        runtime_maxpool,
        runtime_linebuf_job,
        runtime_copy_layout,
        runtime_barrier,
        runtime_rq_load
    };
    return &ops;
}
