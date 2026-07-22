#ifndef NPU_MODEL_LOADER_H
#define NPU_MODEL_LOADER_H

#include "npu_model_abi.h"

typedef enum {
    NAI_LOADER_OK = 0,
    NAI_LOADER_BAD_ARGUMENT = 1,
    NAI_LOADER_BAD_MAGIC = 2,
    NAI_LOADER_BAD_VERSION = 3,
    NAI_LOADER_BAD_TARGET = 4,
    NAI_LOADER_BAD_SIZE = 5,
    NAI_LOADER_BAD_ALIGNMENT = 6,
    NAI_LOADER_BAD_RESERVED = 7,
    NAI_LOADER_BAD_SECTION = 8,
    NAI_LOADER_MISSING_SECTION = 9,
    NAI_LOADER_BAD_CRC = 10,
    NAI_LOADER_BAD_BINDING = 11,
    NAI_LOADER_BAD_REFERENCE = 12
} nai_loader_status_t;

typedef struct {
    const uint8_t *model;
    uint32_t model_bytes;
    const nai_model_header_v1_t *header;
    const nai_section_v1_t *sections;
    const nai_section_v1_t *commands;
    const nai_section_v1_t *constants;
    const nai_section_v1_t *tensors;
    const nai_section_v1_t *bindings;
    const nai_section_v1_t *qparams;
    const nai_section_v1_t *debug_map;
    const nai_binding_v1_t *public_bindings;
} nai_model_view_v1_t;

#define NAI_MAX_SECTIONS_V1 6u
#define NAI_MAX_PUBLIC_BINDINGS_V1 64u

typedef struct {
    void *context;
    uint32_t (*read)(void *context, uint32_t offset, void *destination, uint32_t bytes);
} nai_model_reader_v1_t;

typedef struct {
    nai_model_header_v1_t header;
    nai_section_v1_t sections[NAI_MAX_SECTIONS_V1];
    nai_binding_v1_t bindings[NAI_MAX_PUBLIC_BINDINGS_V1];
} nai_model_stream_storage_v1_t;

typedef struct {
    uint32_t model_base;
    uint32_t model_bytes;
    const nai_binding_address_v1_t *bindings;
    uint32_t binding_count;
    uint32_t tcdm_scratch_base;
    uint32_t tcdm_scratch_bytes;
    uint32_t dtcm_runtime_base;
    uint32_t dtcm_runtime_bytes;
} nai_resolver_v1_t;

uint32_t nai_crc32(const void *data, uint32_t bytes);
nai_loader_status_t nai_model_open_v1(const void *model, uint32_t available_bytes,
                                      uint32_t expected_target, nai_model_view_v1_t *view);
nai_loader_status_t nai_model_open_stream_v1(const nai_model_reader_v1_t *reader,
                                             uint32_t available_bytes, uint32_t expected_target,
                                             void *scratch, uint32_t scratch_bytes,
                                             nai_model_stream_storage_v1_t *storage,
                                             nai_model_view_v1_t *view);
nai_loader_status_t nai_model_validate_bindings_v1(const nai_model_view_v1_t *view);
nai_loader_status_t nai_resolve_ref_v1(const nai_model_view_v1_t *view,
                                       const nai_resolver_v1_t *resolver,
                                       const nai_ref_v1_t *ref,
                                       uint32_t bytes, uint32_t alignment,
                                       uint32_t *address);

#endif
