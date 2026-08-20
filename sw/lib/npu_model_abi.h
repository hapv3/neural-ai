#ifndef NPU_MODEL_ABI_H
#define NPU_MODEL_ABI_H

#include "npu_types.h"

#include <stddef.h>

#define NAI_MODEL_MAGIC       0x4D49414Eu
#define NAI_INVOCATION_MAGIC  0x5649414Eu
#define NAI_ABI_MAJOR         1u
#define NAI_ABI_MINOR         2u
#define NAI_TARGET_ID         1u
#define NAI_ALIGNMENT_BYTES   32u

typedef enum {
    NAI_SECTION_COMMANDS = 1,
    NAI_SECTION_CONSTANTS = 2,
    NAI_SECTION_TENSORS = 3,
    NAI_SECTION_BINDINGS = 4,
    NAI_SECTION_QPARAMS = 5,
    NAI_SECTION_DEBUG_MAP = 6
} nai_section_type_t;

typedef enum {
    NAI_DTYPE_I8 = 1,
    NAI_DTYPE_I32 = 4
} nai_dtype_t;

typedef enum {
    NAI_LAYOUT_NHWC = 1,
    NAI_LAYOUT_ROW32 = 2,
    NAI_LAYOUT_C32_BLOCKED = 3
} nai_layout_t;

typedef enum {
    NAI_BINDING_INPUT = 1,
    NAI_BINDING_OUTPUT = 2,
    NAI_BINDING_L2_TEMPORARY = 3
} nai_binding_direction_t;

typedef enum {
    NAI_REGION_MODEL_CONSTANTS = 1,
    NAI_REGION_MODEL_COMMANDS = 2,
    NAI_REGION_INPUT_BINDING = 3,
    NAI_REGION_OUTPUT_BINDING = 4,
    NAI_REGION_L2_TEMP_BINDING = 5,
    NAI_REGION_TCDM_SCRATCH = 6,
    NAI_REGION_DTCM_RUNTIME = 7
} nai_region_t;

typedef struct {
    uint32_t magic;
    uint16_t abi_major;
    uint16_t abi_minor;
    uint32_t target_id;
    uint32_t flags;
    uint32_t total_bytes;
    uint32_t section_count;
    uint32_t section_table_off;
    uint32_t entry_command_off;
    uint32_t command_count;
    uint32_t required_tcdm_bytes;
    uint32_t required_tcdm_align;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t reserved[3];
} nai_model_header_v1_t;

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint32_t offset;
    uint32_t size;
    uint32_t alignment;
    uint32_t element_count;
    uint32_t reserved[2];
} nai_section_v1_t;

typedef struct {
    uint32_t magic;
    uint16_t abi_major;
    uint16_t abi_minor;
    uint32_t total_bytes;
    uint32_t model_base;
    uint32_t model_bytes;
    uint32_t binding_table_base;
    uint32_t binding_count;
    uint32_t flags;
    uint32_t reserved[8];
} nai_invocation_v1_t;

typedef struct {
    uint16_t region;
    uint16_t index;
    uint32_t offset;
} nai_ref_v1_t;

typedef struct {
    uint32_t tensor_id;
    uint32_t flags;
    uint16_t data_type;
    uint16_t layout;
    uint16_t rank;
    uint16_t reserved0;
    uint32_t dimensions[4];
    uint32_t byte_size;
    uint32_t alignment;
    uint32_t scratch_offset;
    uint32_t qparam_index;
    uint32_t reserved[4];
} nai_tensor_v1_t;

typedef struct {
    uint16_t direction;
    uint16_t index;
    uint16_t data_type;
    uint16_t layout;
    uint16_t rank;
    uint16_t reserved0;
    uint32_t tensor_id;
    uint32_t dimensions[4];
    uint32_t byte_size;
    uint32_t scale_bits;
    int32_t zero_point;
    uint32_t flags;
    uint32_t reserved[4];
} nai_binding_v1_t;

typedef struct {
    uint16_t direction;
    uint16_t index;
    uint32_t base;
    uint32_t byte_size;
    uint32_t flags;
} nai_binding_address_v1_t;

typedef struct {
    int32_t bias;
    int32_t multiplier;
    uint32_t shift;
    int32_t zero_point;
    int32_t clamp_min;
    int32_t clamp_max;
    uint32_t reserved[2];
} nai_qparam_v1_t;

_Static_assert(sizeof(nai_model_header_v1_t) == 64, "nai_model_header_v1_t ABI size");
_Static_assert(sizeof(nai_section_v1_t) == 32, "nai_section_v1_t ABI size");
_Static_assert(sizeof(nai_invocation_v1_t) == 64, "nai_invocation_v1_t ABI size");
_Static_assert(sizeof(nai_ref_v1_t) == 8, "nai_ref_v1_t ABI size");
_Static_assert(sizeof(nai_tensor_v1_t) == 64, "nai_tensor_v1_t ABI size");
_Static_assert(sizeof(nai_binding_v1_t) == 64, "nai_binding_v1_t ABI size");
_Static_assert(sizeof(nai_binding_address_v1_t) == 16, "nai_binding_address_v1_t ABI size");
_Static_assert(sizeof(nai_qparam_v1_t) == 32, "nai_qparam_v1_t ABI size");

_Static_assert(offsetof(nai_model_header_v1_t, section_table_off) == 24, "model section table offset field");
_Static_assert(offsetof(nai_model_header_v1_t, required_tcdm_bytes) == 36, "model TCDM field");
_Static_assert(offsetof(nai_invocation_v1_t, model_base) == 12, "invocation model base field");
_Static_assert(offsetof(nai_binding_v1_t, dimensions) == 16, "binding dimensions field");

#endif
