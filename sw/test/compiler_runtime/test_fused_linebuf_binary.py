import struct

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

import test_compiled_model as compiled


MODEL_BASE = 0x80050000
BINDING_TABLE_BASE = 0x80055000


@cocotb.test()
async def test_compiler_generated_fused_linebuf_binary_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await compiled.reset_dut(dut)

    height, width, channels = 4, 4, 32
    input_data = bytearray(height * width * channels)
    for pixel in range(height * width):
        input_data[pixel * channels] = 1
    expected = bytes([127] * (height * width * channels))

    model = compiled._compile_tflite_fixture_model(
        "fused_linebuf_binary_h4w4_c32",
        "neural-ai-compiled-fused-linebuf-binary-",
        compressed=True,
    )
    command_offset, command_bytes = struct.unpack_from("<II", model, 64 + 8)
    fused_commands = 0
    general_adds = 0
    offset = command_offset
    while offset < command_offset + command_bytes:
        command_type, size = struct.unpack_from("<HH", model, offset)
        assert size >= 32
        fused_commands += command_type in (31, 32)
        general_adds += command_type == 16
        offset += size
    assert offset == command_offset + command_bytes
    assert fused_commands == 1
    assert general_adds == 0

    runtime_bindings = [
        (1, 0, compiled.INPUT_BASE, len(input_data)),
        (2, 0, compiled.OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = compiled.build_invocation_with_bindings(
        model,
        runtime_bindings,
        model_base=MODEL_BASE,
        binding_table_base=BINDING_TABLE_BASE,
    )
    await compiled.write_l2_bytes(dut, compiled.INPUT_BASE, bytes(input_data))
    await compiled.write_l2_bytes(dut, compiled.OUTPUT_BASE, bytes(len(expected)))
    await compiled.write_l2_bytes(dut, MODEL_BASE, model)
    await compiled.write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await compiled.write_l2_bytes(dut, compiled.INVOCATION_BASE, invocation)

    await compiled._load_and_run(dut, axi_master, invocation, model=model)

    assert await compiled._axi_read32(
        axi_master, compiled.NPU_CMD_STATUS
    ) == compiled.NPU_CMD_STATUS_PASS
    assert await compiled._axi_read32(axi_master, compiled.NPU_CMD_FAIL_CODE) == 0
    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await compiled._axi_read32(
        axi_master, compiled.NPU_CMD_DONE_COUNT
    ) == command_count
    assert bytes(
        await compiled.read_l2_bytes(dut, compiled.OUTPUT_BASE, len(expected))
    ) == expected
