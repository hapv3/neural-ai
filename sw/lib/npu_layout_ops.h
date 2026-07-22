#ifndef NPU_LAYOUT_OPS_H
#define NPU_LAYOUT_OPS_H

#include "npu_cmd_desc_v2.h"

uint32_t nai_copy_layout_v2(const nai_cmd_copy_layout_v2_t *command,
                            const void *source, void *destination);

#endif
