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

static const uint32_t crc32_byte_table[256] = {
    0x00000000u, 0x77073096u, 0xee0e612cu, 0x990951bau,
    0x076dc419u, 0x706af48fu, 0xe963a535u, 0x9e6495a3u,
    0x0edb8832u, 0x79dcb8a4u, 0xe0d5e91eu, 0x97d2d988u,
    0x09b64c2bu, 0x7eb17cbdu, 0xe7b82d07u, 0x90bf1d91u,
    0x1db71064u, 0x6ab020f2u, 0xf3b97148u, 0x84be41deu,
    0x1adad47du, 0x6ddde4ebu, 0xf4d4b551u, 0x83d385c7u,
    0x136c9856u, 0x646ba8c0u, 0xfd62f97au, 0x8a65c9ecu,
    0x14015c4fu, 0x63066cd9u, 0xfa0f3d63u, 0x8d080df5u,
    0x3b6e20c8u, 0x4c69105eu, 0xd56041e4u, 0xa2677172u,
    0x3c03e4d1u, 0x4b04d447u, 0xd20d85fdu, 0xa50ab56bu,
    0x35b5a8fau, 0x42b2986cu, 0xdbbbc9d6u, 0xacbcf940u,
    0x32d86ce3u, 0x45df5c75u, 0xdcd60dcfu, 0xabd13d59u,
    0x26d930acu, 0x51de003au, 0xc8d75180u, 0xbfd06116u,
    0x21b4f4b5u, 0x56b3c423u, 0xcfba9599u, 0xb8bda50fu,
    0x2802b89eu, 0x5f058808u, 0xc60cd9b2u, 0xb10be924u,
    0x2f6f7c87u, 0x58684c11u, 0xc1611dabu, 0xb6662d3du,
    0x76dc4190u, 0x01db7106u, 0x98d220bcu, 0xefd5102au,
    0x71b18589u, 0x06b6b51fu, 0x9fbfe4a5u, 0xe8b8d433u,
    0x7807c9a2u, 0x0f00f934u, 0x9609a88eu, 0xe10e9818u,
    0x7f6a0dbbu, 0x086d3d2du, 0x91646c97u, 0xe6635c01u,
    0x6b6b51f4u, 0x1c6c6162u, 0x856530d8u, 0xf262004eu,
    0x6c0695edu, 0x1b01a57bu, 0x8208f4c1u, 0xf50fc457u,
    0x65b0d9c6u, 0x12b7e950u, 0x8bbeb8eau, 0xfcb9887cu,
    0x62dd1ddfu, 0x15da2d49u, 0x8cd37cf3u, 0xfbd44c65u,
    0x4db26158u, 0x3ab551ceu, 0xa3bc0074u, 0xd4bb30e2u,
    0x4adfa541u, 0x3dd895d7u, 0xa4d1c46du, 0xd3d6f4fbu,
    0x4369e96au, 0x346ed9fcu, 0xad678846u, 0xda60b8d0u,
    0x44042d73u, 0x33031de5u, 0xaa0a4c5fu, 0xdd0d7cc9u,
    0x5005713cu, 0x270241aau, 0xbe0b1010u, 0xc90c2086u,
    0x5768b525u, 0x206f85b3u, 0xb966d409u, 0xce61e49fu,
    0x5edef90eu, 0x29d9c998u, 0xb0d09822u, 0xc7d7a8b4u,
    0x59b33d17u, 0x2eb40d81u, 0xb7bd5c3bu, 0xc0ba6cadu,
    0xedb88320u, 0x9abfb3b6u, 0x03b6e20cu, 0x74b1d29au,
    0xead54739u, 0x9dd277afu, 0x04db2615u, 0x73dc1683u,
    0xe3630b12u, 0x94643b84u, 0x0d6d6a3eu, 0x7a6a5aa8u,
    0xe40ecf0bu, 0x9309ff9du, 0x0a00ae27u, 0x7d079eb1u,
    0xf00f9344u, 0x8708a3d2u, 0x1e01f268u, 0x6906c2feu,
    0xf762575du, 0x806567cbu, 0x196c3671u, 0x6e6b06e7u,
    0xfed41b76u, 0x89d32be0u, 0x10da7a5au, 0x67dd4accu,
    0xf9b9df6fu, 0x8ebeeff9u, 0x17b7be43u, 0x60b08ed5u,
    0xd6d6a3e8u, 0xa1d1937eu, 0x38d8c2c4u, 0x4fdff252u,
    0xd1bb67f1u, 0xa6bc5767u, 0x3fb506ddu, 0x48b2364bu,
    0xd80d2bdau, 0xaf0a1b4cu, 0x36034af6u, 0x41047a60u,
    0xdf60efc3u, 0xa867df55u, 0x316e8eefu, 0x4669be79u,
    0xcb61b38cu, 0xbc66831au, 0x256fd2a0u, 0x5268e236u,
    0xcc0c7795u, 0xbb0b4703u, 0x220216b9u, 0x5505262fu,
    0xc5ba3bbeu, 0xb2bd0b28u, 0x2bb45a92u, 0x5cb36a04u,
    0xc2d7ffa7u, 0xb5d0cf31u, 0x2cd99e8bu, 0x5bdeae1du,
    0x9b64c2b0u, 0xec63f226u, 0x756aa39cu, 0x026d930au,
    0x9c0906a9u, 0xeb0e363fu, 0x72076785u, 0x05005713u,
    0x95bf4a82u, 0xe2b87a14u, 0x7bb12baeu, 0x0cb61b38u,
    0x92d28e9bu, 0xe5d5be0du, 0x7cdcefb7u, 0x0bdbdf21u,
    0x86d3d2d4u, 0xf1d4e242u, 0x68ddb3f8u, 0x1fda836eu,
    0x81be16cdu, 0xf6b9265bu, 0x6fb077e1u, 0x18b74777u,
    0x88085ae6u, 0xff0f6a70u, 0x66063bcau, 0x11010b5cu,
    0x8f659effu, 0xf862ae69u, 0x616bffd3u, 0x166ccf45u,
    0xa00ae278u, 0xd70dd2eeu, 0x4e048354u, 0x3903b3c2u,
    0xa7672661u, 0xd06016f7u, 0x4969474du, 0x3e6e77dbu,
    0xaed16a4au, 0xd9d65adcu, 0x40df0b66u, 0x37d83bf0u,
    0xa9bcae53u, 0xdebb9ec5u, 0x47b2cf7fu, 0x30b5ffe9u,
    0xbdbdf21cu, 0xcabac28au, 0x53b39330u, 0x24b4a3a6u,
    0xbad03605u, 0xcdd70693u, 0x54de5729u, 0x23d967bfu,
    0xb3667a2eu, 0xc4614ab8u, 0x5d681b02u, 0x2a6f2b94u,
    0xb40bbe37u, 0xc30c8ea1u, 0x5a05df1bu, 0x2d02ef8du
};

static uint32_t crc32_update(uint32_t crc, const uint8_t *data, uint32_t bytes)
{
    for (uint32_t index = 0; index < bytes; index++) {
        crc = (crc >> 8) ^ crc32_byte_table[(crc ^ data[index]) & 0xffu];
    }
    return crc;
}

uint32_t nai_crc32(const void *data, uint32_t bytes)
{
    return ~crc32_update(0xffffffffu, (const uint8_t *)data, bytes);
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
        if (section->reserved != 0u) {
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
        if (nai_crc32(view->model + section->offset, section->size) != section->crc32) {
            return NAI_LOADER_BAD_CRC;
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

static nai_loader_status_t stream_crc(const nai_model_reader_v1_t *reader,
                                      const nai_section_v1_t *section,
                                      uint8_t *scratch, uint32_t scratch_bytes)
{
    uint32_t crc = 0xffffffffu;
    uint32_t offset = 0u;
    while (offset < section->size) {
        uint32_t chunk = section->size - offset;
        if (chunk > scratch_bytes) chunk = scratch_bytes;
        if (reader->read(reader->context, section->offset + offset, scratch, chunk) != 0u)
            return NAI_LOADER_BAD_SIZE;
        crc = crc32_update(crc, scratch, chunk);
        offset += chunk;
    }
    return ~crc == section->crc32 ? NAI_LOADER_OK : NAI_LOADER_BAD_CRC;
}

nai_loader_status_t nai_model_open_stream_v1(const nai_model_reader_v1_t *reader,
                                             uint32_t available_bytes, uint32_t expected_target,
                                             void *scratch_pointer, uint32_t scratch_bytes,
                                             nai_model_stream_storage_v1_t *storage,
                                             nai_model_view_v1_t *view)
{
    uint8_t *scratch = (uint8_t *)scratch_pointer;
    uint32_t section_bytes;

    if (reader == 0 || reader->read == 0 || scratch == 0 || scratch_bytes == 0u ||
        storage == 0 || view == 0 || available_bytes < sizeof(nai_model_header_v1_t))
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
        if (section->reserved != 0u) return NAI_LOADER_BAD_RESERVED;
        if (section->alignment < NAI_ALIGNMENT_BYTES || !is_power_of_two(section->alignment) ||
            !is_aligned(section->offset, section->alignment) ||
            !is_aligned(section->size, NAI_ALIGNMENT_BYTES)) return NAI_LOADER_BAD_ALIGNMENT;
        if (!valid_range(section->offset, section->size, header->total_bytes)) return NAI_LOADER_BAD_SECTION;
        status = stream_crc(reader, section, scratch, scratch_bytes);
        if (status != NAI_LOADER_OK) return status;
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
