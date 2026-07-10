#include "hal_systolic.h"
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

uint32_t npu_graph_run(const npu_graph_t *graph) {
    for (uint32_t i = 0; i < graph->num_layers; i++) {
        const npu_layer_t *layer = &graph->layers[i];
        const npu_tensor_t *src = get_tensor(graph, layer->src);
        const npu_tensor_t *dst = get_tensor(graph, layer->dst);
        const npu_tensor_t *aux = get_tensor(graph, layer->aux);

        npu_graph_trace(i, layer->op, 1);

        switch (layer->op) {
        case NPU_OP_DMA_IN:
            if (!dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (layer->bytes > dst->bytes) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t dma_status = npu_graph_dma_copy_wait(dst->addr, layer->l2_addr, layer->bytes);
                if (dma_status != NPU_GRAPH_OK) return dma_status;
            }
            break;

        case NPU_OP_DMA_OUT:
            if (!src) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (layer->bytes > src->bytes) return NPU_GRAPH_ERR_BAD_TENSOR;
            {
                uint32_t dma_status = npu_graph_dma_copy_wait(layer->l2_addr, src->addr, layer->bytes);
                if (dma_status != NPU_GRAPH_OK) return dma_status;
            }
            break;

        case NPU_OP_IM2COL3X3S1P1_C3_PAD32:
            if (!src || !dst) return NPU_GRAPH_ERR_BAD_TENSOR;
            if (!tensor_has_layout(src, NPU_LAYOUT_HWC, NPU_DTYPE_I8) ||
                !tensor_has_layout(dst, NPU_LAYOUT_ROW32, NPU_DTYPE_I8) ||
                src->h != 32u || src->w != 32u || src->c != 3u ||
                dst->bytes < npu_tensor_row32_bytes(32u * 32u)) {
                return NPU_GRAPH_ERR_BAD_TENSOR;
            }
            npu_im2col3x3s1p1_c3_pad32((const int8_t *)src->addr, (int8_t *)dst->addr);
            break;

        case NPU_OP_SYSTOLIC_GEMM32:
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

        case NPU_OP_SPATZ_REQUANT:
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
