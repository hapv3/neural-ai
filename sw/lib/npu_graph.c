#include "hal_systolic.h"
#include "conv2d_packed.h"
#include "idma_mm_utils.h"
#include "npu_graph.h"
#include "spatz_ops.h"

__attribute__((weak)) void npu_graph_trace(uint32_t layer_index, npu_op_type_t op, uint32_t event) {
    (void)layer_index;
    (void)op;
    (void)event;
}

static const npu_tensor_t *get_tensor(const npu_graph_t *graph, uint32_t index) {
    if (index >= graph->num_tensors) {
        return 0;
    }
    return &graph->tensors[index];
}

void npu_graph_scratch_init(npu_graph_scratch_t *scratch, uint32_t base, uint32_t size) {
    scratch->base = base;
    scratch->size = size;
    scratch->offset = 0;
}

uint32_t npu_graph_scratch_alloc(npu_graph_scratch_t *scratch, uint32_t bytes, uint32_t align) {
    uint32_t offset = npu_align_up(scratch->offset, align);
    if (offset + bytes > scratch->size) {
        return 0;
    }

    scratch->offset = offset + bytes;
    return scratch->base + offset;
}

static uint32_t tensor_has_layout(const npu_tensor_t *tensor, npu_layout_t layout, npu_dtype_t dtype) {
    return tensor && tensor->layout == layout && tensor->dtype == dtype;
}

static uint32_t tensor_has_dtype(const npu_tensor_t *tensor, npu_dtype_t dtype) {
    return tensor && tensor->dtype == dtype;
}

static uint32_t npu_graph_dma_copy_wait(uint32_t dst_addr, uint32_t src_addr, uint32_t bytes);

static int32_t rq_bias[32];
static int32_t rq_multiplier[32];
static uint8_t rq_shift[32];
static int32_t rq_zero_point[32];

#define NPU_GRAPH_MAX_CONV_TILES 128u

static void configure_uniform_requant(const npu_layer_t *layer) {
    for (uint32_t ch = 0; ch < 32u; ch++) {
        rq_bias[ch] = 0;
        rq_multiplier[ch] = layer->multiplier;
        rq_shift[ch] = (uint8_t)layer->shift;
        rq_zero_point[ch] = 0;
    }
    systolic_requant_config_per_channel(rq_bias, rq_multiplier, rq_shift, rq_zero_point,
                                        layer->min_val, layer->max_val);
}

static uint32_t run_conv2d3x3s1p1_c32_linebuf(const npu_tensor_t *src,
                                              const npu_tensor_t *dst,
                                              const npu_tensor_t *weight,
                                              const npu_layer_t *layer) {
    const uint32_t input_c = 32u;
    const uint32_t kernel_h = 3u;
    const uint32_t kernel_w = 3u;
    const uint32_t weight_bytes = kernel_h * kernel_w * input_c * 32u;

    if (!tensor_has_layout(src, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I32) ||
        !tensor_has_layout(weight, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        src->c != 32u || dst->c != 32u ||
        src->h != dst->h || src->w != dst->w ||
        weight->bytes < weight_bytes ||
        dst->bytes < npu_tensor_i32_row32_bytes((uint32_t)dst->h * dst->w)) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    npu_conv2d_packed_stats_t conv_stats;

    if (layer->linebuf_jobs && layer->linebuf_job_count) {
        uint32_t conv_status =
            npu_conv2d_packed_run_linebuf_job_descs(layer->linebuf_jobs,
                                                    layer->linebuf_job_count,
                                                    &conv_stats);
        return (conv_status == NPU_CONV2D_PACKED_OK) ? NPU_GRAPH_OK : conv_status;
    }

    npu_conv2d_packed_cfg_t conv_cfg = {0};

    conv_cfg.input_addr = src->addr;
    conv_cfg.weight_addr = weight->addr;
    conv_cfg.im2col_addr = 0u;
    conv_cfg.output_addr = dst->addr;
    conv_cfg.input_h = src->h;
    conv_cfg.input_w = src->w;
    conv_cfg.input_c = input_c;
    conv_cfg.output_h = dst->h;
    conv_cfg.output_w = dst->w;
    conv_cfg.kernel_h = kernel_h;
    conv_cfg.kernel_w = kernel_w;
    conv_cfg.stride_h = 1u;
    conv_cfg.stride_w = 1u;
    conv_cfg.pad_h = 1u;
    conv_cfg.pad_w = 1u;
    conv_cfg.dilation_h = 1u;
    conv_cfg.dilation_w = 1u;
    conv_cfg.input_c_stride = input_c;
    conv_cfg.input_row_stride_bytes = 0u;
    conv_cfg.input_c_base = 0u;
    conv_cfg.accumulate = 0u;

    uint32_t tile_oh = layer->dim_m;
    if (tile_oh == 0u) {
        tile_oh = npu_conv2d_packed_linebuf_default_tile_oh(&conv_cfg);
    }
    if (tile_oh == 0u || tile_oh > dst->h) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    npu_conv2d_spatial_tile_t tiles[NPU_GRAPH_MAX_CONV_TILES];
    uint32_t tile_count = 0u;
    for (uint32_t oh_base = 0; oh_base < dst->h; oh_base += tile_oh) {
        if (tile_count >= NPU_GRAPH_MAX_CONV_TILES) {
            return NPU_GRAPH_ERR_BAD_TENSOR;
        }
        uint32_t this_tile_oh = (uint32_t)dst->h - oh_base;
        if (this_tile_oh > tile_oh) {
            this_tile_oh = tile_oh;
        }
        tiles[tile_count].oh_base = oh_base;
        tiles[tile_count].ow_base = 0u;
        tiles[tile_count].tile_oh = this_tile_oh;
        tiles[tile_count].tile_ow = 0u;
        tile_count++;
    }

    uint32_t conv_status = npu_conv2d_packed_run_oc32_linebuf_tiles(&conv_cfg, tiles,
                                                                    tile_count, 4u,
                                                                    &conv_stats);
    return (conv_status == NPU_CONV2D_PACKED_OK) ? NPU_GRAPH_OK : conv_status;
}

static uint32_t tensor_is_c32_i8(const npu_tensor_t *tensor) {
    return tensor_has_layout(tensor, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
           tensor_has_layout(tensor, NPU_LAYOUT_C32_BLOCKED, NPU_DTYPE_I8);
}

static uint32_t run_conv2d1x1_c32_requant(const npu_tensor_t *src,
                                          const npu_tensor_t *dst,
                                          const npu_tensor_t *weight,
                                          const npu_tensor_t *psum,
                                          const npu_layer_t *layer) {
    uint32_t input_groups;
    uint32_t output_groups;
    uint32_t rows;
    uint32_t weight_bytes;
    uint32_t activation_input_bytes;
    uint32_t activation_output_bytes;
    uint32_t psum_bytes;

    if (!tensor_is_c32_i8(src) ||
        !tensor_is_c32_i8(dst) ||
        !tensor_has_layout(weight, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        src->c == 0u || dst->c == 0u ||
        (src->c & 31u) != 0u || (dst->c & 31u) != 0u ||
        src->h != dst->h || src->w != dst->w ||
        src->h == 0u || src->w == 0u) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    rows = (uint32_t)dst->h * dst->w;
    input_groups = npu_c32_blocks(src->c);
    output_groups = npu_c32_blocks(dst->c);
    weight_bytes = input_groups * output_groups * 32u * 32u;
    activation_input_bytes = npu_tensor_c32_bytes(src->h, src->w, src->c);
    activation_output_bytes = npu_tensor_c32_bytes(dst->h, dst->w, dst->c);
    psum_bytes = rows * 32u * 4u;

    if (weight->bytes < weight_bytes ||
        src->bytes < activation_input_bytes ||
        dst->bytes < activation_output_bytes) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    if (input_groups > 1u) {
        if (!psum ||
            !tensor_has_layout(psum, NPU_LAYOUT_ROW32, NPU_DTYPE_I32) ||
            psum->bytes < psum_bytes) {
            return NPU_GRAPH_ERR_BAD_TENSOR;
        }
    }

    configure_uniform_requant(layer);
    systolic_pointwise1x1_c32_multi_requant(src->addr,
                                            weight->addr,
                                            psum ? psum->addr : 0u,
                                            dst->addr,
                                            rows,
                                            input_groups,
                                            output_groups);
    systolic_requant_disable();
    return NPU_GRAPH_OK;
}

static uint32_t conv3x3p1_output_dim(uint32_t input, uint32_t stride) {
    if (input == 0u || stride == 0u) {
        return 0u;
    }
    return ((input - 1u) / stride) + 1u;
}

static uint32_t run_depthwise3x3_c32_requant(const npu_tensor_t *src,
                                             const npu_tensor_t *dst,
                                             const npu_tensor_t *weight,
                                             const npu_layer_t *layer,
                                             uint32_t stride_h,
                                             uint32_t stride_w) {
    uint32_t channels;
    uint32_t groups;
    uint32_t weight_bytes;
    uint32_t expected_h;
    uint32_t expected_w;

    channels = src ? src->c : 0u;
    groups = npu_c32_blocks(channels);
    weight_bytes = groups * 3u * 3u * 32u;
    expected_h = conv3x3p1_output_dim(src ? (uint32_t)src->h : 0u, stride_h);
    expected_w = conv3x3p1_output_dim(src ? (uint32_t)src->w : 0u, stride_w);

    if (!tensor_is_c32_i8(src) ||
        !tensor_is_c32_i8(dst) ||
        !tensor_has_layout(weight, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        channels == 0u ||
        dst->c != channels ||
        expected_h == 0u || expected_w == 0u ||
        dst->h != expected_h || dst->w != expected_w ||
        weight->bytes < weight_bytes) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    if (src->bytes < npu_tensor_c32_bytes((uint32_t)src->h, (uint32_t)src->w, channels) ||
        dst->bytes < npu_tensor_c32_bytes((uint32_t)dst->h, (uint32_t)dst->w, channels)) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    configure_uniform_requant(layer);
    systolic_depthwise3x3_c32_requant_channels(src->addr, weight->addr, dst->addr,
                                               (uint32_t)src->h, (uint32_t)src->w,
                                               (uint32_t)dst->h, (uint32_t)dst->w,
                                               channels, stride_h, stride_w, 1u, 1u);
    systolic_requant_disable();
    return NPU_GRAPH_OK;
}

static uint32_t run_conv2d3x3_c32_linebuf_requant(const npu_tensor_t *src,
                                                  const npu_tensor_t *dst,
                                                  const npu_tensor_t *weight,
                                                  const npu_tensor_t *psum,
                                                  const npu_layer_t *layer,
                                                  uint32_t stride_h,
                                                  uint32_t stride_w) {
    const uint32_t input_c = 32u;
    const uint32_t kernel_h = 3u;
    const uint32_t kernel_w = 3u;
    const uint32_t weight_bytes = kernel_h * kernel_w * input_c * 32u;
    uint32_t rows;
    uint32_t expected_h;
    uint32_t expected_w;

    if (stride_h == 0u || stride_w == 0u || src->h == 0u || src->w == 0u) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }
    expected_h = (((uint32_t)src->h - 1u) / stride_h) + 1u;
    expected_w = (((uint32_t)src->w - 1u) / stride_w) + 1u;

    if (!tensor_has_layout(src, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(weight, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(psum, NPU_LAYOUT_ROW32, NPU_DTYPE_I32) ||
        src->c != 32u || dst->c != 32u || psum->c != 32u ||
        dst->h != expected_h || dst->w != expected_w ||
        psum->h != dst->h || psum->w != dst->w ||
        weight->bytes < weight_bytes ||
        dst->bytes < npu_tensor_row32_bytes((uint32_t)dst->h * dst->w) ||
        psum->bytes < npu_tensor_i32_row32_bytes((uint32_t)dst->h * dst->w)) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    rows = (uint32_t)dst->h * dst->w;

    npu_conv2d_packed_stats_t conv_stats;

    if (layer->linebuf_jobs && layer->linebuf_job_count) {
        configure_uniform_requant(layer);
        uint32_t conv_status =
            npu_conv2d_packed_run_linebuf_job_descs(layer->linebuf_jobs,
                                                    layer->linebuf_job_count,
                                                    &conv_stats);
        systolic_requant_disable();
        return (conv_status == NPU_CONV2D_PACKED_OK) ? NPU_GRAPH_OK : conv_status;
    }

    npu_conv2d_packed_cfg_t conv_cfg = {0};

    conv_cfg.input_addr = src->addr;
    conv_cfg.weight_addr = weight->addr;
    conv_cfg.im2col_addr = 0u;
    conv_cfg.output_addr = dst->addr;
    conv_cfg.input_h = src->h;
    conv_cfg.input_w = src->w;
    conv_cfg.input_c = input_c;
    conv_cfg.output_h = dst->h;
    conv_cfg.output_w = dst->w;
    conv_cfg.kernel_h = kernel_h;
    conv_cfg.kernel_w = kernel_w;
    conv_cfg.stride_h = stride_h;
    conv_cfg.stride_w = stride_w;
    conv_cfg.pad_h = 1u;
    conv_cfg.pad_w = 1u;
    conv_cfg.dilation_h = 1u;
    conv_cfg.dilation_w = 1u;
    conv_cfg.input_c_stride = input_c;
    conv_cfg.input_row_stride_bytes = 0u;
    conv_cfg.input_c_base = 0u;
    conv_cfg.accumulate = 0u;

    uint32_t tile_oh = layer->dim_m;
    uint32_t tile_ow = layer->dim_n ? layer->dim_n : dst->w;
    if (tile_oh == 0u) {
        tile_oh = npu_conv2d_packed_linebuf_default_tile_oh(&conv_cfg);
    }
    if (tile_oh == 0u || tile_ow == 0u || tile_oh > dst->h || tile_ow > dst->w || rows == 0u) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    npu_conv2d_spatial_tile_t tiles[NPU_GRAPH_MAX_CONV_TILES];
    uint32_t tile_count = 0u;
    for (uint32_t oh_base = 0; oh_base < dst->h; oh_base += tile_oh) {
        uint32_t this_tile_oh = (uint32_t)dst->h - oh_base;
        if (this_tile_oh > tile_oh) {
            this_tile_oh = tile_oh;
        }
        for (uint32_t ow_base = 0; ow_base < dst->w; ow_base += tile_ow) {
            if (tile_count >= NPU_GRAPH_MAX_CONV_TILES) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            uint32_t this_tile_ow = (uint32_t)dst->w - ow_base;
            if (this_tile_ow > tile_ow) {
                this_tile_ow = tile_ow;
            }
            tiles[tile_count].oh_base = oh_base;
            tiles[tile_count].ow_base = ow_base;
            tiles[tile_count].tile_oh = this_tile_oh;
            tiles[tile_count].tile_ow = this_tile_ow;
            tile_count++;
        }
    }

    configure_uniform_requant(layer);
    uint32_t conv_status = npu_conv2d_packed_run_oc32_linebuf_tiles_requant(&conv_cfg,
                                                                            tiles,
                                                                            tile_count,
                                                                            psum->addr,
                                                                            &conv_stats);
    systolic_requant_disable();
    return (conv_status == NPU_CONV2D_PACKED_OK) ? NPU_GRAPH_OK : conv_status;
}

static uint32_t build_full_width_conv_tiles(uint32_t output_h,
                                            uint32_t output_w,
                                            uint32_t tile_oh,
                                            npu_conv2d_spatial_tile_t *tiles,
                                            uint32_t *tile_count) {
    if (tile_oh == 0u || output_h == 0u || output_w == 0u || tile_oh > output_h) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    *tile_count = 0u;
    for (uint32_t oh_base = 0u; oh_base < output_h; oh_base += tile_oh) {
        if (*tile_count >= NPU_GRAPH_MAX_CONV_TILES) {
            return NPU_GRAPH_ERR_BAD_TENSOR;
        }

        uint32_t this_tile_oh = output_h - oh_base;
        if (this_tile_oh > tile_oh) {
            this_tile_oh = tile_oh;
        }

        tiles[*tile_count].oh_base = oh_base;
        tiles[*tile_count].ow_base = 0u;
        tiles[*tile_count].tile_oh = this_tile_oh;
        tiles[*tile_count].tile_ow = output_w;
        (*tile_count)++;
    }

    return NPU_GRAPH_OK;
}

static uint32_t run_conv2d3x3_c32_multi_linebuf_requant(const npu_tensor_t *src,
                                                        const npu_tensor_t *dst,
                                                        const npu_tensor_t *weight,
                                                        const npu_tensor_t *psum,
                                                        const npu_layer_t *layer,
                                                        uint32_t stride_h,
                                                        uint32_t stride_w) {
    const uint32_t kernel_h = 3u;
    const uint32_t kernel_w = 3u;
    const uint32_t input_group_c = 32u;
    const uint32_t chunk_weight_bytes = kernel_h * kernel_w * input_group_c * 32u;
    uint32_t input_groups;
    uint32_t output_groups;
    uint32_t rows;
    uint32_t expected_h;
    uint32_t expected_w;
    uint32_t tile_oh;
    uint32_t tile_count;
    npu_conv2d_spatial_tile_t tiles[NPU_GRAPH_MAX_CONV_TILES];

    if (!src || !dst || !weight || !psum ||
        stride_h == 0u || stride_w == 0u ||
        src->h == 0u || src->w == 0u ||
        (src->c & 31u) != 0u || (dst->c & 31u) != 0u ||
        src->c == 0u || dst->c == 0u) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    expected_h = (((uint32_t)src->h - 1u) / stride_h) + 1u;
    expected_w = (((uint32_t)src->w - 1u) / stride_w) + 1u;
    rows = expected_h * expected_w;
    input_groups = npu_c32_blocks(src->c);
    output_groups = npu_c32_blocks(dst->c);

    if (!tensor_is_c32_i8(src) ||
        !tensor_is_c32_i8(dst) ||
        !tensor_has_layout(weight, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(psum, NPU_LAYOUT_ROW32, NPU_DTYPE_I32) ||
        dst->h != expected_h || dst->w != expected_w ||
        weight->bytes < input_groups * output_groups * chunk_weight_bytes ||
        src->bytes < npu_tensor_c32_bytes(src->h, src->w, src->c) ||
        dst->bytes < npu_tensor_c32_bytes(dst->h, dst->w, dst->c) ||
        psum->bytes < npu_tensor_i32_row32_bytes(rows)) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    tile_oh = layer->dim_m;
    if (tile_oh == 0u) {
        npu_conv2d_packed_cfg_t default_cfg = {0};
        default_cfg.input_addr = src->addr;
        default_cfg.weight_addr = weight->addr;
        default_cfg.output_addr = dst->addr;
        default_cfg.input_h = src->h;
        default_cfg.input_w = src->w;
        default_cfg.input_c = input_group_c;
        default_cfg.output_h = dst->h;
        default_cfg.output_w = dst->w;
        default_cfg.kernel_h = kernel_h;
        default_cfg.kernel_w = kernel_w;
        default_cfg.stride_h = stride_h;
        default_cfg.stride_w = stride_w;
        default_cfg.pad_h = 1u;
        default_cfg.pad_w = 1u;
        default_cfg.dilation_h = 1u;
        default_cfg.dilation_w = 1u;
        default_cfg.input_c_stride = input_group_c;
        default_cfg.input_row_stride_bytes = 0u;
        tile_oh = npu_conv2d_packed_linebuf_default_tile_oh(&default_cfg);
    }

    uint32_t tile_status = build_full_width_conv_tiles(dst->h, dst->w, tile_oh,
                                                       tiles, &tile_count);
    if (tile_status != NPU_GRAPH_OK) {
        return tile_status;
    }

    for (uint32_t ocg = 0u; ocg < output_groups; ocg++) {
        uint32_t out_group_addr = dst->addr + (ocg * rows * 32u);
        uint32_t oc_weight_base = weight->addr + (ocg * input_groups * chunk_weight_bytes);
        npu_conv2d_packed_stats_t conv_stats;

        if (input_groups == 1u) {
            npu_conv2d_packed_cfg_t cfg = {0};
            cfg.input_addr = src->addr;
            cfg.weight_addr = oc_weight_base;
            cfg.output_addr = out_group_addr;
            cfg.input_h = src->h;
            cfg.input_w = src->w;
            cfg.input_c = input_group_c;
            cfg.output_h = dst->h;
            cfg.output_w = dst->w;
            cfg.kernel_h = kernel_h;
            cfg.kernel_w = kernel_w;
            cfg.stride_h = stride_h;
            cfg.stride_w = stride_w;
            cfg.pad_h = 1u;
            cfg.pad_w = 1u;
            cfg.dilation_h = 1u;
            cfg.dilation_w = 1u;
            cfg.input_c_stride = input_group_c;
            cfg.input_row_stride_bytes = 0u;
            cfg.input_c_base = 0u;
            cfg.accumulate = 0u;

            configure_uniform_requant(layer);
            uint32_t conv_status =
                npu_conv2d_packed_run_oc32_linebuf_tiles_requant(&cfg, tiles,
                                                                  tile_count,
                                                                  psum->addr,
                                                                  &conv_stats);
            systolic_requant_disable();
            if (conv_status != NPU_CONV2D_PACKED_OK) {
                return conv_status;
            }
            continue;
        }

        systolic_requant_disable();
        for (uint32_t icg = 0u; icg < input_groups; icg++) {
            uint32_t in_group_addr = src->addr + (icg * src->h * src->w * 32u);
            npu_conv2d_packed_cfg_t cfg = {0};
            cfg.input_addr = in_group_addr;
            cfg.weight_addr = oc_weight_base + (icg * chunk_weight_bytes);
            cfg.output_addr = (icg + 1u == input_groups) ? out_group_addr : psum->addr;
            cfg.input_h = src->h;
            cfg.input_w = src->w;
            cfg.input_c = input_group_c;
            cfg.output_h = dst->h;
            cfg.output_w = dst->w;
            cfg.kernel_h = kernel_h;
            cfg.kernel_w = kernel_w;
            cfg.stride_h = stride_h;
            cfg.stride_w = stride_w;
            cfg.pad_h = 1u;
            cfg.pad_w = 1u;
            cfg.dilation_h = 1u;
            cfg.dilation_w = 1u;
            cfg.input_c_stride = input_group_c;
            cfg.input_row_stride_bytes = 0u;
            cfg.input_c_base = 0u;
            cfg.accumulate = (icg == 0u) ? 0u : 1u;

            if (icg + 1u == input_groups) {
                configure_uniform_requant(layer);
                for (uint32_t tile_idx = 0u; tile_idx < tile_count; tile_idx++) {
                    const npu_conv2d_spatial_tile_t *tile = &tiles[tile_idx];
                    uint32_t output_tile_addr =
                        out_group_addr + ((tile->oh_base * dst->w + tile->ow_base) * 32u);
                    uint32_t conv_status =
                        npu_conv2d_packed_run_oc32_linebuf_tile_accumulate_requant(&cfg,
                                                                                   tile,
                                                                                   psum->addr,
                                                                                   output_tile_addr,
                                                                                   &conv_stats);
                    if (conv_status != NPU_CONV2D_PACKED_OK) {
                        systolic_requant_disable();
                        return conv_status;
                    }
                }
                systolic_requant_disable();
            } else {
                uint32_t conv_status =
                    npu_conv2d_packed_run_oc32_linebuf_tiles(&cfg, tiles,
                                                             tile_count, 4u,
                                                             &conv_stats);
                if (conv_status != NPU_CONV2D_PACKED_OK) {
                    return conv_status;
                }
            }
        }
    }

    return NPU_GRAPH_OK;
}

static uint32_t run_conv2d3x3s1p1_c32x2_linebuf_requant_l2(const npu_tensor_t *src0,
                                                           const npu_tensor_t *src1,
                                                           const npu_tensor_t *dst_tile,
                                                           const npu_tensor_t *weight,
                                                           const npu_tensor_t *psum,
                                                           const npu_layer_t *layer) {
    const uint32_t input_c = 32u;
    const uint32_t kernel_h = 3u;
    const uint32_t kernel_w = 3u;
    const uint32_t chunk_weight_bytes = kernel_h * kernel_w * input_c * 32u;
    const uint32_t output_h = src0 ? src0->h : 0u;
    const uint32_t output_w = src0 ? src0->w : 0u;
    const uint32_t rows = output_h * output_w;
    const uint32_t full_psum_bytes = rows * 32u * 4u;
    uint32_t tile_oh = layer->dim_m;
    uint32_t tile_ow = layer->dim_n;

    if (!src0 || !src1 || !dst_tile || !weight || !psum ||
        tile_oh == 0u || tile_ow == 0u || layer->l2_addr == 0u) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    if (!tensor_has_layout(src0, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(src1, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(dst_tile, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(weight, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
        !tensor_has_layout(psum, NPU_LAYOUT_ROW32, NPU_DTYPE_I32) ||
        src0->h != src1->h || src0->w != src1->w ||
        src0->c != 32u || src1->c != 32u ||
        src0->bytes < npu_tensor_row32_bytes(rows) ||
        src1->bytes < npu_tensor_row32_bytes(rows) ||
        weight->bytes < (2u * chunk_weight_bytes) ||
        psum->bytes < full_psum_bytes ||
        dst_tile->bytes < (tile_oh * tile_ow * 32u)) {
        return NPU_GRAPH_ERR_BAD_TENSOR;
    }

    npu_conv2d_packed_stats_t stats0;
    npu_conv2d_packed_stats_t stats1;

    if (layer->linebuf_jobs && layer->linebuf_job_count &&
        layer->linebuf_l2_jobs && layer->linebuf_l2_job_count) {
        uint32_t conv_status =
            npu_conv2d_packed_run_linebuf_job_descs(layer->linebuf_jobs,
                                                    layer->linebuf_job_count,
                                                    &stats0);
        if (conv_status != NPU_CONV2D_PACKED_OK) {
            return conv_status;
        }

        configure_uniform_requant(layer);
        for (uint32_t job_idx = 0; job_idx < layer->linebuf_l2_job_count; job_idx++) {
            const npu_conv2d_l2_copy_job_desc_t *copy_job = &layer->linebuf_l2_jobs[job_idx];
            conv_status = npu_conv2d_packed_run_linebuf_job_descs(&copy_job->job, 1u, &stats1);
            if (conv_status != NPU_CONV2D_PACKED_OK) {
                systolic_requant_disable();
                return conv_status;
            }

            for (uint32_t th = 0; th < copy_job->tile_oh; th++) {
                uint32_t src_offset = th * copy_job->tile_ow * 32u;
                uint32_t dst_offset = th * output_w * 32u;
                uint32_t copy_status =
                    npu_graph_dma_copy_wait(copy_job->l2_addr + dst_offset,
                                            copy_job->tile_output_addr + src_offset,
                                            copy_job->tile_ow * 32u);
                if (copy_status != NPU_GRAPH_OK) {
                    systolic_requant_disable();
                    return copy_status;
                }
            }
        }
        systolic_requant_disable();
        return NPU_GRAPH_OK;
    }

    npu_conv2d_packed_cfg_t cfg0 = {0};
    npu_conv2d_packed_cfg_t cfg1 = {0};
    npu_conv2d_spatial_tile_t tiles[NPU_GRAPH_MAX_CONV_TILES];
    uint32_t tile_count = 0u;

    cfg0.input_addr = src0->addr;
    cfg0.weight_addr = weight->addr;
    cfg0.im2col_addr = 0u;
    cfg0.output_addr = psum->addr;
    cfg0.input_h = output_h;
    cfg0.input_w = output_w;
    cfg0.input_c = input_c;
    cfg0.output_h = output_h;
    cfg0.output_w = output_w;
    cfg0.kernel_h = kernel_h;
    cfg0.kernel_w = kernel_w;
    cfg0.stride_h = 1u;
    cfg0.stride_w = 1u;
    cfg0.pad_h = 1u;
    cfg0.pad_w = 1u;
    cfg0.dilation_h = 1u;
    cfg0.dilation_w = 1u;
    cfg0.input_c_stride = input_c;
    cfg0.input_row_stride_bytes = 0u;
    cfg0.input_c_base = 0u;
    cfg0.accumulate = 0u;
    cfg1.input_addr = src1->addr;
    cfg1.weight_addr = weight->addr + chunk_weight_bytes;
    cfg1.im2col_addr = cfg0.im2col_addr;
    cfg1.output_addr = dst_tile->addr;
    cfg1.input_h = cfg0.input_h;
    cfg1.input_w = cfg0.input_w;
    cfg1.input_c = cfg0.input_c;
    cfg1.output_h = cfg0.output_h;
    cfg1.output_w = cfg0.output_w;
    cfg1.kernel_h = cfg0.kernel_h;
    cfg1.kernel_w = cfg0.kernel_w;
    cfg1.stride_h = cfg0.stride_h;
    cfg1.stride_w = cfg0.stride_w;
    cfg1.pad_h = cfg0.pad_h;
    cfg1.pad_w = cfg0.pad_w;
    cfg1.dilation_h = cfg0.dilation_h;
    cfg1.dilation_w = cfg0.dilation_w;
    cfg1.input_c_stride = cfg0.input_c_stride;
    cfg1.input_row_stride_bytes = cfg0.input_row_stride_bytes;
    cfg1.input_c_base = cfg0.input_c_base;
    cfg1.accumulate = 1u;

    for (uint32_t oh_base = 0; oh_base < output_h; oh_base += tile_oh) {
        uint32_t this_tile_oh = output_h - oh_base;
        if (this_tile_oh > tile_oh) {
            this_tile_oh = tile_oh;
        }
        for (uint32_t ow_base = 0; ow_base < output_w; ow_base += tile_ow) {
            if (tile_count >= NPU_GRAPH_MAX_CONV_TILES) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            uint32_t this_tile_ow = output_w - ow_base;
            if (this_tile_ow > tile_ow) {
                this_tile_ow = tile_ow;
            }
            tiles[tile_count].oh_base = oh_base;
            tiles[tile_count].ow_base = ow_base;
            tiles[tile_count].tile_oh = this_tile_oh;
            tiles[tile_count].tile_ow = this_tile_ow;
            tile_count++;
        }
    }

    uint32_t conv_status = npu_conv2d_packed_run_oc32_linebuf_tiles(&cfg0, tiles,
                                                                    tile_count, 4u,
                                                                    &stats0);
    if (conv_status != NPU_CONV2D_PACKED_OK) {
        return conv_status;
    }

    configure_uniform_requant(layer);
    for (uint32_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        const npu_conv2d_spatial_tile_t *tile = &tiles[tile_idx];
        uint32_t this_tile_ow = tile->tile_ow ? tile->tile_ow : output_w;
        conv_status = npu_conv2d_packed_run_oc32_linebuf_tile_accumulate_requant(&cfg1,
                                                                                 tile,
                                                                                 psum->addr,
                                                                                 dst_tile->addr,
                                                                                 &stats1);
        if (conv_status != NPU_CONV2D_PACKED_OK) {
            systolic_requant_disable();
            return conv_status;
        }

        for (uint32_t th = 0; th < tile->tile_oh; th++) {
            uint32_t src_offset = th * this_tile_ow * 32u;
            uint32_t dst_offset = (((tile->oh_base + th) * output_w) + tile->ow_base) * 32u;
            uint32_t copy_status = npu_graph_dma_copy_wait(layer->l2_addr + dst_offset,
                                                           dst_tile->addr + src_offset,
                                                           this_tile_ow * 32u);
            if (copy_status != NPU_GRAPH_OK) {
                systolic_requant_disable();
                return copy_status;
            }
        }
    }
    systolic_requant_disable();

    return NPU_GRAPH_OK;
}

static uint32_t npu_graph_dma_copy_wait(uint32_t dst_addr, uint32_t src_addr, uint32_t bytes) {
    if (!idma_memcpy_blocking(src_addr, dst_addr, bytes)) {
        return NPU_GRAPH_ERR_DMA;
    }

    return NPU_GRAPH_OK;
}

void npu_im2col3x3s1p1_c3_pad32(const int8_t *input_hwc, int8_t *output_row32) {
    for (uint32_t y = 0; y < 32; y++) {
        for (uint32_t x = 0; x < 32; x++) {
            int8_t *row = output_row32 + ((y * 32 + x) * 32);
            uint32_t k = 0;

            for (int32_t ky = -1; ky <= 1; ky++) {
                int32_t iy = (int32_t)y + ky;
                for (int32_t kx = -1; kx <= 1; kx++) {
                    int32_t ix = (int32_t)x + kx;
                    for (uint32_t c = 0; c < 3; c++) {
                        int8_t val = 0;
                        if (iy >= 0 && iy < 32 && ix >= 0 && ix < 32) {
                            val = input_hwc[((uint32_t)iy * 32 + (uint32_t)ix) * 3 + c];
                        }
                        row[k++] = val;
                    }
                }
            }

            while (k < 32) {
                row[k++] = 0;
            }
        }
    }
}

void npu_im2col3x3s2p1_c3_pad32(const int8_t *input_hwc, uint32_t input_h, uint32_t input_w,
                                int8_t *output_row32) {
    const uint32_t output_h = (input_h + 1u) / 2u;
    const uint32_t output_w = (input_w + 1u) / 2u;

    for (uint32_t oy = 0; oy < output_h; oy++) {
        for (uint32_t ox = 0; ox < output_w; ox++) {
            int8_t *row = output_row32 + ((oy * output_w + ox) * 32u);
            uint32_t k = 0;

            for (int32_t ky = -1; ky <= 1; ky++) {
                int32_t iy = (int32_t)(oy * 2u) + ky;
                for (int32_t kx = -1; kx <= 1; kx++) {
                    int32_t ix = (int32_t)(ox * 2u) + kx;
                    for (uint32_t c = 0; c < 3u; c++) {
                        int8_t val = 0;
                        if (iy >= 0 && iy < (int32_t)input_h &&
                            ix >= 0 && ix < (int32_t)input_w) {
                            val = input_hwc[((uint32_t)iy * input_w + (uint32_t)ix) * 3u + c];
                        }
                        row[k++] = val;
                    }
                }
            }

            while (k < 32u) {
                row[k++] = 0;
            }
        }
    }
}

uint32_t npu_graph_run(const npu_graph_t *graph) {
    for (uint32_t i = 0; i < graph->num_layers; i++) {
        const npu_layer_t *layer = &graph->layers[i];
        const npu_tensor_t *src = get_tensor(graph, layer->src);
        const npu_tensor_t *src2 = get_tensor(graph, layer->src2);
        const npu_tensor_t *dst = get_tensor(graph, layer->dst);
        const npu_tensor_t *aux = get_tensor(graph, layer->aux);
        const npu_tensor_t *aux2 = get_tensor(graph, layer->aux2);

        npu_graph_trace(i, layer->op, 1);

        switch (layer->op) {
        case DMA_IN:
            if (!dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (layer->bytes > dst->bytes) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t dma_status = npu_graph_dma_copy_wait(dst->addr, layer->l2_addr, layer->bytes);
                if (dma_status != NPU_GRAPH_OK) return dma_status;
            }
            break;

        case DMA_OUT:
            if (!src) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (layer->bytes > src->bytes) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t dma_status = npu_graph_dma_copy_wait(layer->l2_addr, src->addr, layer->bytes);
                if (dma_status != NPU_GRAPH_OK) return dma_status;
            }
            break;

        case LEGACY_IM2COL_C3_PAD32:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_layout(src, NPU_LAYOUT_HWC, NPU_DTYPE_I8) ||
                !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                src->h != 32u || src->w != 32u || src->c != 3u ||
                dst->bytes < npu_tensor_row32_bytes(32u * 32u)) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            npu_im2col3x3s1p1_c3_pad32((const int8_t *)src->addr, (int8_t *)dst->addr);
            break;

        case LEGACY_IM2COL_C3_DOWNSAMPLE_PAD32:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t output_h = ((uint32_t)src->h + 1u) / 2u;
                uint32_t output_w = ((uint32_t)src->w + 1u) / 2u;
                if (!tensor_has_layout(src, NPU_LAYOUT_HWC, NPU_DTYPE_I8) ||
                    !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                    src->c != 3u || dst->h != output_h || dst->w != output_w ||
                    dst->c != 32u ||
                    dst->bytes < npu_tensor_row32_bytes(output_h * output_w)) {
                    return NPU_GRAPH_ERR_BAD_TENSOR;
                }
            }
            npu_im2col3x3s2p1_c3_pad32((const int8_t *)src->addr, src->h, src->w,
                                        (int8_t *)dst->addr);
            break;

        case SYSTOLIC_GEMM32:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_layout(src, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                !tensor_has_layout(aux, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I32) ||
                layer->dim_m == 0u ||
                src->bytes < npu_tensor_row32_bytes(layer->dim_m) ||
                aux->bytes < npu_tensor_row32_bytes(32u) ||
                dst->bytes < npu_tensor_i32_row32_bytes(layer->dim_m)) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            systolic_gemm32(aux->addr, src->addr, dst->addr, layer->dim_m);
            break;

        case SYSTOLIC_GEMM32_REQUANT:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_layout(src, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                !tensor_has_layout(aux, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                layer->dim_m == 0u ||
                src->bytes < npu_tensor_row32_bytes(layer->dim_m) ||
                aux->bytes < npu_tensor_row32_bytes(32u) ||
                dst->bytes < npu_tensor_row32_bytes(layer->dim_m)) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            configure_uniform_requant(layer);
            systolic_gemm32_requant(aux->addr, src->addr, dst->addr, layer->dim_m);
            systolic_requant_disable();
            break;

        case CONV2D_RGB_LINEBUF_REQUANT:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_layout(src, NPU_LAYOUT_HWC, NPU_DTYPE_I8) ||
                !tensor_has_layout(aux, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                src->c != 3u ||
                dst->h != ((uint32_t)src->h + 1u) / 2u ||
                dst->w != ((uint32_t)src->w + 1u) / 2u ||
                dst->c != 32u ||
                aux->bytes < npu_tensor_row32_bytes(32u) ||
                dst->bytes < npu_tensor_row32_bytes((uint32_t)dst->h * dst->w)) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            {
                npu_conv2d_packed_stats_t conv_stats;

                if (layer->linebuf_jobs && layer->linebuf_job_count) {
                    configure_uniform_requant(layer);
                    uint32_t conv_status =
                        npu_conv2d_packed_run_linebuf_job_descs(layer->linebuf_jobs,
                                                                layer->linebuf_job_count,
                                                                &conv_stats);
                    systolic_requant_disable();
                    if (conv_status != NPU_CONV2D_PACKED_OK) return conv_status;
                    break;
                }

                npu_conv2d_packed_cfg_t conv_cfg = {0};

                conv_cfg.input_addr = src->addr;
                conv_cfg.weight_addr = aux->addr;
                conv_cfg.im2col_addr = 0u;
                conv_cfg.output_addr = dst->addr;
                conv_cfg.input_h = src->h;
                conv_cfg.input_w = src->w;
                conv_cfg.input_c = src->c;
                conv_cfg.output_h = dst->h;
                conv_cfg.output_w = dst->w;
                conv_cfg.kernel_h = 3u;
                conv_cfg.kernel_w = 3u;
                conv_cfg.stride_h = 2u;
                conv_cfg.stride_w = 2u;
                conv_cfg.pad_h = 1u;
                conv_cfg.pad_w = 1u;
                conv_cfg.dilation_h = 1u;
                conv_cfg.dilation_w = 1u;
                conv_cfg.input_c_stride = src->c;
                conv_cfg.input_row_stride_bytes = 0u;
                conv_cfg.input_c_base = 0u;
                conv_cfg.accumulate = 0u;

                configure_uniform_requant(layer);
                uint32_t conv_status = npu_conv2d_packed_run_oc32_linebuf_requant(&conv_cfg, &conv_stats);
                systolic_requant_disable();
                if (conv_status != NPU_CONV2D_PACKED_OK) return conv_status;
            }
            break;

        case LOGISTIC_LUT_I8:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t count = layer->bytes ? layer->bytes : dst->bytes;
                if (!tensor_has_dtype(src, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(dst, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(aux, NPU_DTYPE_I8) ||
                    src->bytes < count || dst->bytes < count || aux->bytes < 256u) {
                    return NPU_GRAPH_ERR_BAD_TENSOR;
                }
                if (!npu_logistic_i8((const int8_t *)src->addr, (int8_t *)dst->addr,
                                     count, (const uint8_t *)aux->addr)) {
                    return NPU_GRAPH_ERR_ACCEL;
                }
            }
            break;

        case CLAMP_I8:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t count = layer->bytes ? layer->bytes : dst->bytes;
                if (!tensor_has_dtype(src, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(dst, NPU_DTYPE_I8) ||
                    src->layout != dst->layout ||
                    src->h != dst->h || src->w != dst->w || src->c != dst->c ||
                    src->bytes < count || dst->bytes < count ||
                    count == 0u || layer->min_val > layer->max_val ||
                    layer->min_val < -128 || layer->max_val > 127 ||
                    src->addr == dst->addr) {
                    return NPU_GRAPH_ERR_BAD_TENSOR;
                }
                if (!npu_clamp_i8((const int8_t *)src->addr,
                                  (int8_t *)dst->addr,
                                  count,
                                  layer->min_val,
                                  layer->max_val)) {
                    return NPU_GRAPH_ERR_ACCEL;
                }
            }
            break;

        case MUL_I8:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t count = layer->bytes ? layer->bytes : dst->bytes;
                if (!tensor_has_dtype(src, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(dst, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(aux, NPU_DTYPE_I8) ||
                    src->bytes < count || dst->bytes < count || aux->bytes < count) {
                    return NPU_GRAPH_ERR_BAD_TENSOR;
                }
                if (layer->multiplier == 1 && layer->shift == 7u &&
                    layer->min_val == -128 && layer->max_val == 127) {
                    if (!npu_mul_q7_i8((const int8_t *)src->addr, (const int8_t *)aux->addr,
                                       (int8_t *)dst->addr, count)) {
                        return NPU_GRAPH_ERR_ACCEL;
                    }
                } else {
                    spatz_mul_i8((const int8_t *)src->addr, (const int8_t *)aux->addr,
                                 (int8_t *)dst->addr, count, layer->multiplier, layer->shift,
                                 layer->min_val, layer->max_val);
                }
            }
            break;

        case ADD_I8:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t count = layer->bytes ? layer->bytes : dst->bytes;
                if (!tensor_has_dtype(src, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(dst, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(aux, NPU_DTYPE_I8) ||
                    src->bytes < count || dst->bytes < count || aux->bytes < count ||
                    src->h != dst->h || src->w != dst->w || src->c != dst->c ||
                    aux->h != dst->h || aux->w != dst->w || aux->c != dst->c) {
                    return NPU_GRAPH_ERR_BAD_TENSOR;
                }
                spatz_add_i8((const int8_t *)src->addr, (const int8_t *)aux->addr,
                             (int8_t *)dst->addr, count,
                             layer->min_val, layer->max_val);
            }
            break;

        case MAXPOOL2D_I8:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_dtype(src, NPU_DTYPE_I8) ||
                !tensor_has_dtype(dst, NPU_DTYPE_I8) ||
                src->h != dst->h || src->w != dst->w || src->c != dst->c ||
                src->bytes < dst->bytes || dst->c == 0u ||
                src->addr == dst->addr) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            if (src->c == 32u && src->bytes >= ((uint32_t)src->h * src->w * 32u) &&
                dst->bytes >= ((uint32_t)dst->h * dst->w * 32u)) {
                systolic_maxpool5x5s1p2_c32_linebuf(src->addr, dst->addr, src->h, src->w);
            } else {
                spatz_maxpool2d_i8((const int8_t *)src->addr, (int8_t *)dst->addr,
                                   src->h, src->w, src->c,
                                   5u, 5u, 1u, 1u, 2u, 2u);
            }
            break;

        case GLOBAL_AVGPOOL_C32_REDUCE:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_layout(src, NPU_LAYOUT_C32_BLOCKED, NPU_DTYPE_I8) ||
                !tensor_has_layout(dst, NPU_LAYOUT_C32_BLOCKED, NPU_DTYPE_I8) ||
                src->h == 0u || src->w == 0u || src->c == 0u ||
                dst->h != 1u || dst->w != 1u || dst->c != src->c ||
                src->bytes < npu_tensor_c32_bytes(src->h, src->w, src->c) ||
                dst->bytes < npu_tensor_c32_bytes(1u, 1u, dst->c) ||
                src->addr == dst->addr) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            if (!npu_global_avgpool_c32_i8((const int8_t *)src->addr,
                                           (int8_t *)dst->addr,
                                           src->h,
                                           src->w,
                                           src->c)) {
                return NPU_GRAPH_ERR_ACCEL;
            }
            break;

        case UPSAMPLE_NEAREST_I8:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_dtype(src, NPU_DTYPE_I8) ||
                !tensor_has_dtype(dst, NPU_DTYPE_I8) ||
                src->c != dst->c ||
                dst->h != ((uint32_t)src->h * 2u) ||
                dst->w != ((uint32_t)src->w * 2u) ||
                src->bytes < ((uint32_t)src->h * src->w * src->c) ||
                dst->bytes < ((uint32_t)dst->h * dst->w * dst->c) ||
                src->addr == dst->addr) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            spatz_upsample_nearest_i8((const int8_t *)src->addr, (int8_t *)dst->addr,
                                      src->h, src->w, src->c, 2u, 2u);
            break;

        case CONV2D_DUAL_SOURCE_C32_LINEBUF_REQUANT_L2:
            if (!src || !src2 || !dst || !aux || !aux2) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status =
                    run_conv2d3x3s1p1_c32x2_linebuf_requant_l2(src, src2, dst,
                                                               aux, aux2, layer);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case DFL_SOFTMAX_I8_Q8:
            if (!src || !dst || !aux || !aux2) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t locations = (uint32_t)src->h * src->w;
                uint32_t dst_bytes = locations * 4u * 2u;

                if (!tensor_has_layout(src, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                    src->c != 32u ||
                    src->bytes < npu_tensor_row32_bytes(locations) ||
                    dst->bytes < dst_bytes ||
                    aux->bytes < 1024u ||
                    aux2->bytes < 1024u) {
                    return NPU_GRAPH_ERR_BAD_TENSOR;
                }
                if (!npu_dfl_softmax4_row32_i8_q8((const int8_t *)src->addr,
                                                  (uint16_t *)dst->addr,
                                                  locations,
                                                  (const uint32_t *)aux->addr,
                                                  (const uint32_t *)aux2->addr)) {
                    return NPU_GRAPH_ERR_ACCEL;
                }
            }
            break;

        case CLASS_SIGMOID_ROW32_HIGH16_I8:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t locations = (uint32_t)src->h * src->w;
                uint32_t dst_bytes = locations * 16u;

                if (!tensor_has_layout(src, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                    !tensor_has_dtype(dst, NPU_DTYPE_I8) ||
                    src->c != 32u ||
                    src->bytes < npu_tensor_row32_bytes(locations) ||
                    dst->bytes < dst_bytes ||
                    aux->bytes < 256u) {
                    return NPU_GRAPH_ERR_BAD_TENSOR;
                }
                if (!npu_class_sigmoid_row32_high16_i8((const int8_t *)src->addr,
                                                       (int8_t *)dst->addr,
                                                       locations,
                                                       (const uint8_t *)aux->addr)) {
                    return NPU_GRAPH_ERR_ACCEL;
                }
            }
            break;

        case CONV2D_C32_LINEBUF:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status =
                    run_conv2d3x3s1p1_c32_linebuf(src, dst, aux, layer);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case CONV2D_C32_LINEBUF_REQUANT:
            if (!src || !dst || !aux || !aux2) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status =
                    run_conv2d3x3_c32_linebuf_requant(src, dst, aux, aux2, layer, 1u, 1u);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case CONV2D_C32_DOWNSAMPLE_LINEBUF_REQUANT:
            if (!src || !dst || !aux || !aux2) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status =
                    run_conv2d3x3_c32_linebuf_requant(src, dst, aux, aux2, layer, 2u, 2u);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case CONV2D_C32_MULTI_LINEBUF_REQUANT:
            if (!src || !dst || !aux || !aux2) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status =
                    run_conv2d3x3_c32_multi_linebuf_requant(src, dst, aux, aux2, layer, 1u, 1u);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case CONV2D_POINTWISE_C32_REQUANT:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status = run_conv2d1x1_c32_requant(src, dst, aux, aux2, layer);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case DEPTHWISE_CONV2D_C32_REQUANT:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status =
                    run_depthwise3x3_c32_requant(src, dst, aux, layer, 1u, 1u);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case DEPTHWISE_CONV2D_C32_DOWNSAMPLE_REQUANT:
            if (!src || !dst || !aux) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t conv_status =
                    run_depthwise3x3_c32_requant(src, dst, aux, layer, 2u, 2u);
                if (conv_status != NPU_GRAPH_OK) return conv_status;
            }
            break;

        case SPATZ_REQUANT:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_layout(src, NPU_LAYOUT_ROW32, NPU_DTYPE_I32) ||
                !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                src->bytes < dst->bytes * 4u) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            spatz_requant_i32_to_i8((const int32_t *)src->addr, (int8_t *)dst->addr,
                                    dst->bytes, layer->multiplier, layer->shift,
                                    layer->min_val, layer->max_val);
            break;

        default:
            return NPU_GRAPH_ERR_BAD_OP;
        }

        npu_graph_trace(i, layer->op, 2);
    }

    return NPU_GRAPH_OK;
}
