#include "npu_model_loader.h"

static uint32_t is_power_of_two(uint32_t value)
{
    return value != 0u && (value & (value - 1u)) == 0u;
}

static uint32_t is_aligned(uint32_t value, uint32_t alignment)
{
    return is_power_of_two(alignment) && (value & (alignment - 1u)) == 0u;
}

static uint32_t valid_range(uint32_t offset, uint32_t size, uint32_t total)
{
    return offset <= total && size <= total - offset;
}

static uint32_t all_zero(const uint32_t *words, uint32_t count)
{
    for (uint32_t index = 0; index < count; index++) {
        if (words[index] != 0u) {
            return 0u;
        }
    }
    return 1u;
}

static nai_loader_status_t set_section(const nai_section_v1_t *section,
                                       nai_model_view_v1_t *view)
{
    const nai_section_v1_t **destination = 0;

    switch (section->type) {
        case NAI_SECTION_COMMANDS: destination = &view->commands; break;
        case NAI_SECTION_CONSTANTS: destination = &view->constants; break;
        case NAI_SECTION_TENSORS: destination = &view->tensors; break;
        case NAI_SECTION_BINDINGS: destination = &view->bindings; break;
        case NAI_SECTION_QPARAMS: destination = &view->qparams; break;
        case NAI_SECTION_DEBUG_MAP: destination = &view->debug_map; break;
        default: return NAI_LOADER_BAD_SECTION;
    }
    if (*destination != 0) {
        return NAI_LOADER_BAD_SECTION;
    }
    *destination = section;
    return NAI_LOADER_OK;
}

nai_loader_status_t nai_model_open_v1(const void *model, uint32_t available_bytes,
                                      uint32_t expected_target, nai_model_view_v1_t *view)
{
    const nai_model_header_v1_t *header;
    uint32_t section_bytes;

    if (model == 0 || view == 0 || available_bytes < sizeof(nai_model_header_v1_t)) {
        return NAI_LOADER_BAD_ARGUMENT;
    }
    *view = (nai_model_view_v1_t){0};
    header = (const nai_model_header_v1_t *)model;
    if (header->magic != NAI_MODEL_MAGIC) {
        return NAI_LOADER_BAD_MAGIC;
    }
    if (header->abi_major != NAI_ABI_MAJOR || header->abi_minor > NAI_ABI_MINOR) {
        return NAI_LOADER_BAD_VERSION;
    }
    if (header->target_id != expected_target) {
        return NAI_LOADER_BAD_TARGET;
    }
    if (!all_zero(header->reserved, 3u)) {
        return NAI_LOADER_BAD_RESERVED;
    }
    if (header->total_bytes < sizeof(*header) || header->total_bytes > available_bytes ||
        !is_aligned(header->total_bytes, NAI_ALIGNMENT_BYTES)) {
        return NAI_LOADER_BAD_SIZE;
    }
    if (header->required_tcdm_align != NAI_ALIGNMENT_BYTES ||
        header->required_tcdm_bytes > 0x0007f000u) {
        return NAI_LOADER_BAD_ALIGNMENT;
    }
    if (header->section_count > 64u) {
        return NAI_LOADER_BAD_SECTION;
    }
    if (header->section_count > 0xffffffffu / (uint32_t)sizeof(nai_section_v1_t)) {
        return NAI_LOADER_BAD_SIZE;
    }
    section_bytes = header->section_count * (uint32_t)sizeof(nai_section_v1_t);
    if (!is_aligned(header->section_table_off, NAI_ALIGNMENT_BYTES) ||
        !valid_range(header->section_table_off, section_bytes, header->total_bytes)) {
        return NAI_LOADER_BAD_SECTION;
    }

    view->model = (const uint8_t *)model;
    view->model_bytes = header->total_bytes;
    view->header = header;
    view->sections = (const nai_section_v1_t *)(view->model + header->section_table_off);

    for (uint32_t index = 0; index < header->section_count; index++) {
        const nai_section_v1_t *section = &view->sections[index];
        nai_loader_status_t status;
        if (!all_zero(section->reserved, 2u)) {
            return NAI_LOADER_BAD_RESERVED;
        }
        if (section->alignment < NAI_ALIGNMENT_BYTES || !is_power_of_two(section->alignment) ||
            !is_aligned(section->offset, section->alignment) ||
            !is_aligned(section->size, NAI_ALIGNMENT_BYTES)) {
            return NAI_LOADER_BAD_ALIGNMENT;
        }
        if (!valid_range(section->offset, section->size, header->total_bytes)) {
            return NAI_LOADER_BAD_SECTION;
        }
        status = set_section(section, view);
        if (status != NAI_LOADER_OK) {
            return status;
        }
    }
    if (view->commands == 0 || view->constants == 0 || view->tensors == 0 ||
        view->bindings == 0 || view->qparams == 0) {
        return NAI_LOADER_MISSING_SECTION;
    }
    if (header->entry_command_off < view->commands->offset ||
        header->entry_command_off >= view->commands->offset + view->commands->size ||
        !is_aligned(header->entry_command_off, NAI_ALIGNMENT_BYTES)) {
        return NAI_LOADER_BAD_SECTION;
    }
    view->public_bindings = (const nai_binding_v1_t *)(view->model + view->bindings->offset);
    return nai_model_validate_bindings_v1(view);
}

static uint32_t multiply_checked(uint32_t lhs, uint32_t rhs, uint32_t *result)
{
    if (lhs != 0u && rhs > 0xffffffffu / lhs) {
        return 0u;
    }
    *result = lhs * rhs;
    return 1u;
}

nai_loader_status_t nai_model_validate_bindings_v1(const nai_model_view_v1_t *view)
{
    uint32_t expected_count;
    uint32_t required_bytes;

    if (view == 0 || view->header == 0 || view->bindings == 0) {
        return NAI_LOADER_BAD_ARGUMENT;
    }
    expected_count = view->header->input_count + view->header->output_count;
    if (expected_count < view->header->input_count ||
        view->bindings->element_count != expected_count ||
        !multiply_checked(expected_count, sizeof(nai_binding_v1_t), &required_bytes) ||
        required_bytes > view->bindings->size) {
        return NAI_LOADER_BAD_BINDING;
    }

    const nai_binding_v1_t *bindings = view->public_bindings;
    if (bindings == 0) return NAI_LOADER_BAD_ARGUMENT;
    for (uint32_t index = 0; index < expected_count; index++) {
        const nai_binding_v1_t *binding = &bindings[index];
        uint32_t elements = 1u;
        uint32_t element_bytes;
        if (binding->rank == 0u || binding->rank > 4u || binding->reserved0 != 0u ||
            !all_zero(binding->reserved, 4u)) {
            return NAI_LOADER_BAD_BINDING;
        }
        if ((binding->direction != NAI_BINDING_INPUT && binding->direction != NAI_BINDING_OUTPUT) ||
            (binding->rank == 4u && binding->layout != NAI_LAYOUT_NHWC)) {
            return NAI_LOADER_BAD_BINDING;
        }
        if (binding->data_type == NAI_DTYPE_I8) {
            element_bytes = 1u;
        } else if (binding->data_type == NAI_DTYPE_I32) {
            element_bytes = 4u;
        } else {
            return NAI_LOADER_BAD_BINDING;
        }
        for (uint32_t axis = 0; axis < binding->rank; axis++) {
            if (binding->dimensions[axis] == 0u ||
                !multiply_checked(elements, binding->dimensions[axis], &elements)) {
                return NAI_LOADER_BAD_BINDING;
            }
        }
        if (!multiply_checked(elements, element_bytes, &required_bytes) ||
            binding->byte_size != required_bytes) {
            return NAI_LOADER_BAD_BINDING;
        }
    }
    return NAI_LOADER_OK;
}

nai_loader_status_t nai_model_open_stream_v1(const nai_model_reader_v1_t *reader,
                                             uint32_t available_bytes, uint32_t expected_target,
                                             nai_model_stream_storage_v1_t *storage,
                                             nai_model_view_v1_t *view)
{
    uint32_t section_bytes;

    if (reader == 0 || reader->read == 0 || storage == 0 || view == 0 ||
        available_bytes < sizeof(nai_model_header_v1_t))
        return NAI_LOADER_BAD_ARGUMENT;
    *storage = (nai_model_stream_storage_v1_t){0};
    *view = (nai_model_view_v1_t){0};
    if (reader->read(reader->context, 0u, &storage->header, sizeof(storage->header)) != 0u)
        return NAI_LOADER_BAD_SIZE;

    const nai_model_header_v1_t *header = &storage->header;
    if (header->magic != NAI_MODEL_MAGIC) return NAI_LOADER_BAD_MAGIC;
    if (header->abi_major != NAI_ABI_MAJOR || header->abi_minor > NAI_ABI_MINOR)
        return NAI_LOADER_BAD_VERSION;
    if (header->target_id != expected_target) return NAI_LOADER_BAD_TARGET;
    if (!all_zero(header->reserved, 3u)) return NAI_LOADER_BAD_RESERVED;
    if (header->total_bytes < sizeof(*header) || header->total_bytes > available_bytes ||
        !is_aligned(header->total_bytes, NAI_ALIGNMENT_BYTES)) return NAI_LOADER_BAD_SIZE;
    if (header->required_tcdm_align != NAI_ALIGNMENT_BYTES ||
        header->required_tcdm_bytes > 0x0007f000u) return NAI_LOADER_BAD_ALIGNMENT;
    if (header->section_count == 0u || header->section_count > NAI_MAX_SECTIONS_V1)
        return NAI_LOADER_BAD_SECTION;
    section_bytes = header->section_count * sizeof(nai_section_v1_t);
    if (!is_aligned(header->section_table_off, NAI_ALIGNMENT_BYTES) ||
        !valid_range(header->section_table_off, section_bytes, header->total_bytes))
        return NAI_LOADER_BAD_SECTION;
    if (reader->read(reader->context, header->section_table_off, storage->sections, section_bytes) != 0u)
        return NAI_LOADER_BAD_SIZE;

    view->model_bytes = header->total_bytes;
    view->header = header;
    view->sections = storage->sections;
    for (uint32_t index = 0; index < header->section_count; index++) {
        const nai_section_v1_t *section = &storage->sections[index];
        nai_loader_status_t status;
        if (!all_zero(section->reserved, 2u)) return NAI_LOADER_BAD_RESERVED;
        if (section->alignment < NAI_ALIGNMENT_BYTES || !is_power_of_two(section->alignment) ||
            !is_aligned(section->offset, section->alignment) ||
            !is_aligned(section->size, NAI_ALIGNMENT_BYTES)) return NAI_LOADER_BAD_ALIGNMENT;
        if (!valid_range(section->offset, section->size, header->total_bytes)) return NAI_LOADER_BAD_SECTION;
        status = set_section(section, view);
        if (status != NAI_LOADER_OK) return status;
    }
    if (view->commands == 0 || view->constants == 0 || view->tensors == 0 ||
        view->bindings == 0 || view->qparams == 0) return NAI_LOADER_MISSING_SECTION;
    if (header->entry_command_off < view->commands->offset ||
        header->entry_command_off >= view->commands->offset + view->commands->size ||
        !is_aligned(header->entry_command_off, NAI_ALIGNMENT_BYTES)) return NAI_LOADER_BAD_SECTION;

    uint32_t binding_count = header->input_count + header->output_count;
    uint32_t binding_bytes;
    if (binding_count < header->input_count || binding_count > NAI_MAX_PUBLIC_BINDINGS_V1 ||
        !multiply_checked(binding_count, sizeof(nai_binding_v1_t), &binding_bytes) ||
        binding_bytes > view->bindings->size) return NAI_LOADER_BAD_BINDING;
    if (binding_bytes != 0u && reader->read(reader->context, view->bindings->offset,
        storage->bindings, binding_bytes) != 0u) return NAI_LOADER_BAD_SIZE;
    view->public_bindings = storage->bindings;
    return nai_model_validate_bindings_v1(view);
}

static nai_loader_status_t resolve_binding(const nai_resolver_v1_t *resolver,
                                           uint16_t direction, uint16_t index,
                                           uint32_t *base, uint32_t *available)
{
    for (uint32_t item = 0; item < resolver->binding_count; item++) {
        const nai_binding_address_v1_t *binding = &resolver->bindings[item];
        if (binding->direction == direction && binding->index == index) {
            *base = binding->base;
            *available = binding->byte_size;
            return NAI_LOADER_OK;
        }
    }
    return NAI_LOADER_BAD_REFERENCE;
}

nai_loader_status_t nai_resolve_ref_v1(const nai_model_view_v1_t *view,
                                       const nai_resolver_v1_t *resolver,
                                       const nai_ref_v1_t *ref,
                                       uint32_t bytes, uint32_t alignment,
                                       uint32_t *address)
{
    uint32_t base;
    uint32_t available;
    nai_loader_status_t status;

    if (view == 0 || resolver == 0 || ref == 0 || address == 0 ||
        !is_power_of_two(alignment)) {
        return NAI_LOADER_BAD_ARGUMENT;
    }
    switch (ref->region) {
        case NAI_REGION_MODEL_CONSTANTS:
            if (ref->index != 0u || resolver->model_bytes < view->model_bytes ||
                resolver->model_base > 0xffffffffu - view->constants->offset) {
                return NAI_LOADER_BAD_REFERENCE;
            }
            base = resolver->model_base + view->constants->offset;
            available = view->constants->size;
            break;
        case NAI_REGION_MODEL_COMMANDS:
            if (ref->index != 0u || resolver->model_bytes < view->model_bytes ||
                resolver->model_base > 0xffffffffu - view->commands->offset) {
                return NAI_LOADER_BAD_REFERENCE;
            }
            base = resolver->model_base + view->commands->offset;
            available = view->commands->size;
            break;
        case NAI_REGION_INPUT_BINDING:
            status = resolve_binding(resolver, NAI_BINDING_INPUT, ref->index, &base, &available);
            if (status != NAI_LOADER_OK) return status;
            break;
        case NAI_REGION_OUTPUT_BINDING:
            status = resolve_binding(resolver, NAI_BINDING_OUTPUT, ref->index, &base, &available);
            if (status != NAI_LOADER_OK) return status;
            break;
        case NAI_REGION_L2_TEMP_BINDING:
            status = resolve_binding(resolver, NAI_BINDING_L2_TEMPORARY, ref->index, &base, &available);
            if (status != NAI_LOADER_OK) return status;
            break;
        case NAI_REGION_TCDM_SCRATCH:
            if (ref->index != 0u) return NAI_LOADER_BAD_REFERENCE;
            base = resolver->tcdm_scratch_base;
            available = resolver->tcdm_scratch_bytes;
            break;
        case NAI_REGION_DTCM_RUNTIME:
            return NAI_LOADER_BAD_REFERENCE;
        default:
            return NAI_LOADER_BAD_REFERENCE;
    }
    if (!valid_range(ref->offset, bytes, available) || base > 0xffffffffu - ref->offset) {
        return NAI_LOADER_BAD_REFERENCE;
    }
    *address = base + ref->offset;
    if (!is_aligned(*address, alignment)) {
        return NAI_LOADER_BAD_ALIGNMENT;
    }
    return NAI_LOADER_OK;
}
