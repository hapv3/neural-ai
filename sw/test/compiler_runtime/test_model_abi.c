#include "npu_memory_map.h"
#include "npu_cmd_desc_v2.h"
#include "npu_model_abi.h"
#include "npu_model_loader.h"

#include <assert.h>
#include <string.h>

static int valid_range(uint32_t offset, uint32_t size, uint32_t total)
{
    return offset <= total && size <= total - offset;
}

static void make_valid_model(uint8_t model[320])
{
    nai_model_header_v1_t *header;
    nai_section_v1_t *sections;
    nai_binding_v1_t *binding;

    memset(model, 0, 320);
    header = (nai_model_header_v1_t *)model;
    sections = (nai_section_v1_t *)(model + 64);
    binding = (nai_binding_v1_t *)(model + 256);

    header->magic = NAI_MODEL_MAGIC;
    header->abi_major = NAI_ABI_MAJOR;
    header->abi_minor = NAI_ABI_MINOR;
    header->target_id = NAI_TARGET_ID;
    header->total_bytes = 320;
    header->section_count = 5;
    header->section_table_off = 64;
    header->entry_command_off = 224;
    header->required_tcdm_align = NAI_ALIGNMENT_BYTES;
    header->input_count = 1;

    sections[0] = (nai_section_v1_t){NAI_SECTION_COMMANDS, 0, 224, 32, 32, 1, {0, 0}};
    sections[1] = (nai_section_v1_t){NAI_SECTION_CONSTANTS, 0, 256, 0, 32, 0, {0, 0}};
    sections[2] = (nai_section_v1_t){NAI_SECTION_TENSORS, 0, 256, 0, 32, 0, {0, 0}};
    sections[3] = (nai_section_v1_t){NAI_SECTION_BINDINGS, 0, 256, 64, 32, 1, {0, 0}};
    sections[4] = (nai_section_v1_t){NAI_SECTION_QPARAMS, 0, 320, 0, 32, 0, {0, 0}};

    binding->direction = NAI_BINDING_INPUT;
    binding->data_type = NAI_DTYPE_I8;
    binding->layout = NAI_LAYOUT_NHWC;
    binding->rank = 4;
    binding->dimensions[0] = 1;
    binding->dimensions[1] = 2;
    binding->dimensions[2] = 3;
    binding->dimensions[3] = 5;
    binding->byte_size = 30;

}

int main(void)
{
    static const uint8_t expected_header[64] = {
        0x4e, 0x41, 0x49, 0x4d, 0x01, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x04, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, 0x40, 0x23, 0x01, 0x00,
        0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    nai_model_header_v1_t header = {0};
    uint8_t model[320];
    nai_model_view_v1_t view;

    header.magic = NAI_MODEL_MAGIC;
    header.abi_major = NAI_ABI_MAJOR;
    header.abi_minor = NAI_ABI_MINOR;
    header.target_id = NAI_TARGET_ID;
    header.total_bytes = 0x400;
    header.section_count = 5;
    header.section_table_off = 0x40;
    header.entry_command_off = 0x100;
    header.command_count = 3;
    header.required_tcdm_bytes = 0x12340;
    header.required_tcdm_align = NAI_ALIGNMENT_BYTES;
    header.input_count = 1;
    header.output_count = 2;

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    assert(memcmp(&header, expected_header, sizeof(expected_header)) == 0);
#endif
    assert(NPU_DTCM_SIZE == 0x00008000u);
    assert(NPU_CMD_TCDM_BASE + NPU_CMD_TCDM_SIZE == NPU_TCDM_BASE + NPU_TCDM_SIZE);
    assert(valid_range(64, 32, 96));
    assert(valid_range(96, 0, 96));
    assert(!valid_range(0xffffffffu - 15u, 32u, 0xffffffffu));

    make_valid_model(model);
    assert(nai_model_open_v1(model, sizeof(model), NAI_TARGET_ID, &view) == NAI_LOADER_OK);
    assert(view.bindings->element_count == 1u);

    ((nai_model_header_v1_t *)model)->total_bytes = 319;
    assert(nai_model_open_v1(model, sizeof(model), NAI_TARGET_ID, &view) == NAI_LOADER_BAD_SIZE);
    make_valid_model(model);
    ((nai_section_v1_t *)(model + 64))[0].offset = 225;
    assert(nai_model_open_v1(model, sizeof(model), NAI_TARGET_ID, &view) == NAI_LOADER_BAD_ALIGNMENT);
    make_valid_model(model);
    ((nai_binding_v1_t *)(model + 256))->layout = NAI_LAYOUT_C32_BLOCKED;
    assert(nai_model_open_v1(model, sizeof(model), NAI_TARGET_ID, &view) == NAI_LOADER_BAD_BINDING);
    return 0;
}
