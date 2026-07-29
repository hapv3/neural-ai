#!/usr/bin/env python3

import os
import re
from pathlib import Path


COMPILER_CONSTANTS = {
    "ModelMagic": "NAI_MODEL_MAGIC",
    "InvocationMagic": "NAI_INVOCATION_MAGIC",
    "AbiMajor": "NAI_ABI_MAJOR",
    "AbiMinor": "NAI_ABI_MINOR",
    "TargetId": "NAI_TARGET_ID",
    "Alignment": "NAI_ALIGNMENT_BYTES",
}

ENUMS = {
    "Commands": "NAI_SECTION_COMMANDS",
    "Constants": "NAI_SECTION_CONSTANTS",
    "Tensors": "NAI_SECTION_TENSORS",
    "Bindings": "NAI_SECTION_BINDINGS",
    "QParams": "NAI_SECTION_QPARAMS",
    "DebugMap": "NAI_SECTION_DEBUG_MAP",
    "Int8": "NAI_DTYPE_I8",
    "Int32": "NAI_DTYPE_I32",
    "NHWC": "NAI_LAYOUT_NHWC",
    "Row32": "NAI_LAYOUT_ROW32",
    "C32Blocked": "NAI_LAYOUT_C32_BLOCKED",
    "Input": "NAI_BINDING_INPUT",
    "Output": "NAI_BINDING_OUTPUT",
    "L2Temporary": "NAI_BINDING_L2_TEMPORARY",
    "ModelConstants": "NAI_REGION_MODEL_CONSTANTS",
    "ModelCommands": "NAI_REGION_MODEL_COMMANDS",
    "InputBinding": "NAI_REGION_INPUT_BINDING",
    "OutputBinding": "NAI_REGION_OUTPUT_BINDING",
    "L2TemporaryBinding": "NAI_REGION_L2_TEMP_BINDING",
    "TCDMScratch": "NAI_REGION_TCDM_SCRATCH",
    "DTCMRuntime": "NAI_REGION_DTCM_RUNTIME",
    "ExternalToLocal": "NAI_DMA_EXTERNAL_TO_LOCAL",
    "LocalToExternal": "NAI_DMA_LOCAL_TO_EXTERNAL",
    "LocalToLocal": "NAI_DMA_LOCAL_TO_LOCAL",
    "End": "NAI_CMD_END",
    "Barrier": "NAI_CMD_BARRIER",
    "DMA1D": "NAI_CMD_DMA_1D",
    "DMA2D": "NAI_CMD_DMA_2D",
    "DMA3D": "NAI_CMD_DMA_3D",
    "RQLoad": "NAI_CMD_RQ_LOAD",
    "Gemm32": "NAI_CMD_GEMM32",
    "Gemm32Accum": "NAI_CMD_GEMM32_ACCUM",
    "Gemm32Requant": "NAI_CMD_GEMM32_REQUANT",
    "LineBufferJob": "NAI_CMD_LINEBUF_JOB",
    "PointwiseC32": "NAI_CMD_POINTWISE_C32",
    "DepthwiseC32": "NAI_CMD_DEPTHWISE_C32",
    "AFULut": "NAI_CMD_AFU_LUT",
    "AFUBinary": "NAI_CMD_AFU_BINARY",
    "AFUGlobalAvgPool": "NAI_CMD_AFU_GLOBAL_AVGPOOL",
    "SpatzRequant": "NAI_CMD_SPATZ_REQUANT",
    "SpatzAdd": "NAI_CMD_SPATZ_ADD",
    "SpatzMul": "NAI_CMD_SPATZ_MUL",
    "CopyLayout": "NAI_CMD_COPY_LAYOUT",
    "MaxPool": "NAI_CMD_MAXPOOL",
    "UpsampleNearest": "NAI_CMD_UPSAMPLE_NEAREST",
    "RollingReset": "NAI_CMD_ROLLING_RESET",
    "RollingProduce": "NAI_CMD_ROLLING_PRODUCE",
    "RollingConsumeRelease": "NAI_CMD_ROLLING_CONSUME_RELEASE",
    "DMASubmit1D": "NAI_CMD_DMA_SUBMIT_1D",
    "DMASubmit2D": "NAI_CMD_DMA_SUBMIT_2D",
    "DMASubmit3D": "NAI_CMD_DMA_SUBMIT_3D",
    "DMAWait": "NAI_CMD_DMA_WAIT",
    "AddI8": "NAI_AFU_BINARY_ADD_I8",
    "NHWCToRow32": "NAI_COPY_NHWC_TO_ROW32",
    "Row32ToNHWC": "NAI_COPY_ROW32_TO_NHWC",
    "NHWCToC32": "NAI_COPY_NHWC_TO_C32",
    "C32ToNHWC": "NAI_COPY_C32_TO_NHWC",
}

STRUCTS = {
    "ModelHeaderV1": "nai_model_header_v1_t",
    "SectionV1": "nai_section_v1_t",
    "InvocationV1": "nai_invocation_v1_t",
    "RefV1": "nai_ref_v1_t",
    "TensorV1": "nai_tensor_v1_t",
    "BindingV1": "nai_binding_v1_t",
    "BindingAddressV1": "nai_binding_address_v1_t",
    "QParamV1": "nai_qparam_v1_t",
    "CommandHeaderV2": "nai_cmd_header_v2_t",
    "CommandRQLoadV2": "nai_cmd_rq_load_v2_t",
    "CommandDMA1DV2": "nai_cmd_dma_1d_v2_t",
    "CommandDMA2DV2": "nai_cmd_dma_2d_v2_t",
    "CommandDMA3DV2": "nai_cmd_dma_3d_v2_t",
    "CommandGemm32V2": "nai_cmd_gemm32_v2_t",
    "CommandPointwiseC32V2": "nai_cmd_pointwise_c32_v2_t",
    "CommandDepthwiseC32V2": "nai_cmd_depthwise_c32_v2_t",
    "CommandAFUBinaryV2": "nai_cmd_afu_binary_v2_t",
    "LinebufJobWireV1": "nai_linebuf_job_wire_v1_t",
    "CommandLineBufferJobV2": "nai_cmd_linebuf_job_v2_t",
    "CommandCopyLayoutV2": "nai_cmd_copy_layout_v2_t",
}


def integer(text, name):
    patterns = (
        rf"\b{name}\s*=\s*(0x[0-9A-Fa-f]+|\d+)[uUlL]*",
        rf"#define\s+{name}\s+(0x[0-9A-Fa-f]+|\d+)[uUlL]*",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1), 0)
    raise AssertionError(f"missing ABI integer {name}")


def struct_size(text, name):
    match = re.search(
        rf"(?:static_assert|_Static_assert)\s*\(\s*sizeof\s*\(\s*{name}\s*\)"
        rf"\s*==\s*(\d+)",
        text,
    )
    if not match:
        raise AssertionError(f"missing ABI size assertion for {name}")
    return int(match.group(1))


def main():
    default_compiler = Path(__file__).resolve().parents[4] / "neural-compiler"
    compiler_root = Path(
        os.environ.get("NEURAL_COMPILER_ROOT", default_compiler)
    ).resolve()
    runtime_root = Path(__file__).resolve().parents[2]
    compiler_abi = (
        compiler_root
        / "ethosu/regor/architecture/neuralai/neural_ai_abi.hpp"
    ).read_text(encoding="utf-8")
    runtime_model = (runtime_root / "lib/npu_model_abi.h").read_text(
        encoding="utf-8"
    )
    runtime_commands = (runtime_root / "lib/npu_cmd_desc_v2.h").read_text(
        encoding="utf-8"
    )
    runtime_abi = runtime_model + "\n" + runtime_commands

    for compiler_name, runtime_name in COMPILER_CONSTANTS.items():
        assert integer(compiler_abi, compiler_name) == integer(
            runtime_abi, runtime_name
        ), f"ABI constant mismatch: {compiler_name}/{runtime_name}"
    for compiler_name, runtime_name in ENUMS.items():
        assert integer(compiler_abi, compiler_name) == integer(
            runtime_abi, runtime_name
        ), f"ABI enum mismatch: {compiler_name}/{runtime_name}"
    for compiler_name, runtime_name in STRUCTS.items():
        assert struct_size(compiler_abi, compiler_name) == struct_size(
            runtime_abi, runtime_name
        ), f"ABI size mismatch: {compiler_name}/{runtime_name}"

    print(
        "cross-repository ABI manifest matches: "
        f"{len(COMPILER_CONSTANTS)} constants, {len(ENUMS)} enums, "
        f"{len(STRUCTS)} structure sizes"
    )


if __name__ == "__main__":
    main()
