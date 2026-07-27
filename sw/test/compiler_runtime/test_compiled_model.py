import base64
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    NPU_CMD_DONE_COUNT,
    NPU_CMD_FAIL_CODE,
    NPU_CMD_FAIL_PTR,
    NPU_CMD_STATUS,
    NPU_CMD_STATUS_FAIL,
    NPU_CMD_STATUS_PASS,
    _axi_read32,
    load_firmware_elf_axi,
    program_command_queue,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


INPUT_BASE = 0x80000000
OUTPUT_BASE = 0x80001000
INVOCATION_BASE = 0x80040000
MODEL_BASE = 0x80041000
BINDING_TABLE_BASE = 0x80044000
WEIGHT_SCRATCH_OFFSET = 0x00010000
IFM_SCRATCH_OFFSET = 0x00020000
OFM_SCRATCH_OFFSET = 0x00030000

NAI_MODEL_MAGIC = 0x4D49414E
NAI_INVOCATION_MAGIC = 0x5649414E
NPU_CMD_FAIL_BAD_MODEL = 0xBADCD00B
NPU_CMD_FAIL_BAD_BINDING = 0xBADCD00C


def _ref(region, index=0, offset=0):
    return struct.pack("<HHI", region, index, offset)


def _command_header(command_type, size, layer=0, tile=0):
    return struct.pack("<HHIII", command_type, size, 0, layer, tile)


def _dma_1d(source, destination, length, direction, tile):
    command = _command_header(2, 64, tile=tile)
    command += source + destination
    command += struct.pack("<II6I", length, direction, 0, 0, 0, 0, 0, 0)
    assert len(command) == 64
    return command


def _binding(direction, index, data_type=1, dimensions=(1, 1, 1, 32)):
    element_bytes = 1 if data_type == 1 else 4
    byte_size = element_bytes
    for dimension in dimensions:
        byte_size *= dimension
    descriptor = struct.pack(
        "<6HI4IIIiI4I",
        direction,
        index,
        data_type,
        1,
        4,
        0,
        index,
        *dimensions,
        byte_size,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert len(descriptor) == 64
    return descriptor


def _gemm(weights, ifm, ofm, dim_m, tile):
    command = _command_header(6, 96, tile=tile)
    command += weights + ifm + _ref(6) + ofm
    command += struct.pack("<4I8I", dim_m, 128, 128, 0, *([0] * 8))
    assert len(command) == 96
    return command


def _pointwise_c32(weights, ifm, partial, ofm, rows, input_groups, output_groups, qparam_block, tile):
    command = _command_header(10, 64, tile=tile)
    command += weights + ifm + partial + ofm
    command += struct.pack("<4I", rows, input_groups, output_groups, qparam_block)
    assert len(command) == 64
    return command


def _copy_layout(source, destination, mode, dimensions, tile):
    command = _command_header(18, 96, tile=tile)
    command += source + destination
    source_layout, destination_layout = {
        1: (1, 2),
        2: (2, 1),
        3: (1, 3),
        4: (3, 1),
    }[mode]
    channels = dimensions[3]
    compact_stride = channels
    native_stride = ((channels + 31) // 32) * 32
    command += struct.pack(
        "<4H4I3I7I",
        mode,
        source_layout,
        destination_layout,
        1,
        *dimensions,
        channels,
        compact_stride if mode == 1 else native_stride,
        native_stride if mode == 1 else compact_stride,
        *([0] * 7),
    )
    assert len(command) == 96
    return command


def _package(commands, constants, bindings, command_count, required_tcdm_bytes,
             input_count, output_count, qparams=b""):
    payloads = [commands, constants, b"", bindings, qparams]
    section_types = [1, 2, 3, 4, 5]
    element_counts = [command_count + 1, len(constants), 0,
                      input_count + output_count, len(qparams) // 32]
    section_table_offset = 64
    offset = section_table_offset + 5 * 32
    sections = []
    for section_type, element_count, payload in zip(section_types, element_counts, payloads):
        assert offset % 32 == 0 and len(payload) % 32 == 0
        sections.append(
            struct.pack(
                "<8I",
                section_type,
                0,
                offset,
                len(payload),
                32,
                element_count,
                zlib.crc32(payload) & 0xFFFFFFFF,
                0,
            )
        )
        offset += len(payload)

    header = struct.pack(
        "<IHH11I3I",
        NAI_MODEL_MAGIC,
        1,
        0,
        1,
        0,
        offset,
        5,
        section_table_offset,
        224,
        command_count,
        required_tcdm_bytes,
        32,
        input_count,
        output_count,
        0,
        0,
        0,
    )
    model = header + b"".join(sections) + b"".join(payloads)
    assert len(header) == 64 and len(model) == offset
    return model


def build_model():
    commands = b"".join(
        [
            _dma_1d(_ref(3), _ref(6), 32, 0, 0),
            _dma_1d(_ref(6), _ref(4), 32, 1, 1),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0) + _binding(2, 0)
    return _package(commands, b"", bindings, 2, 32, 1, 1)


def build_layout_model():
    dimensions = (1, 2, 2, 33)
    commands = b"".join(
        [
            _copy_layout(_ref(3), _ref(6), 1, dimensions, 0),
            _copy_layout(_ref(6), _ref(4), 2, dimensions, 1),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0, dimensions=dimensions) + _binding(2, 0, dimensions=dimensions)
    return _package(commands, b"", bindings, 2, 256, 1, 1)


def build_c32_layout_model():
    dimensions = (1, 2, 2, 33)
    commands = b"".join(
        [
            _copy_layout(_ref(3), _ref(6), 3, dimensions, 0),
            _copy_layout(_ref(6), _ref(4), 4, dimensions, 1),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0, dimensions=dimensions) + _binding(2, 0, dimensions=dimensions)
    return _package(commands, b"", bindings, 2, 256, 1, 1)


def build_pointwise_c32_model(height=2, width=2, channels=64):
    rows = height * width
    groups = channels // 32
    weights = bytes([1]) * (groups * groups * 32 * 32)
    qparams = b"".join(
        struct.pack("<iiIiiiII", 0, 1 << 30, 30, 0, -128, 127, 0, 0)
        for _ in range(32)
    )
    dimensions = (1, height, width, channels)
    commands = b"".join(
        [
            _copy_layout(_ref(3), _ref(6), 3, dimensions, 0),
            _command_header(5, 32, tile=1) + struct.pack("<4I", 0, 32, 0, 0),
            _pointwise_c32(_ref(1), _ref(6), _ref(6, offset=0x1000),
                           _ref(6, offset=0x2000), rows, groups, groups, 0, 2),
            _copy_layout(_ref(6, offset=0x2000), _ref(4), 4, dimensions, 3),
            _command_header(0, 32, tile=4).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0, dimensions=dimensions) + _binding(2, 0, dimensions=dimensions)
    return _package(commands, weights, bindings, 4, 0x2200, 1, 1, qparams)


def build_gemm_model(dimensions):
    weights = bytes(1 if row == column else 0 for row in range(32) for column in range(32))
    commands = [
        _dma_1d(_ref(1), _ref(6, offset=WEIGHT_SCRATCH_OFFSET), len(weights), 0, 0)
    ]
    bindings = []
    for index, dim_m in enumerate(dimensions):
        commands.extend(
            [
                _dma_1d(_ref(3, index), _ref(6, offset=IFM_SCRATCH_OFFSET), dim_m * 32, 0, index * 3 + 1),
                _gemm(
                    _ref(6, offset=WEIGHT_SCRATCH_OFFSET),
                    _ref(6, offset=IFM_SCRATCH_OFFSET),
                    _ref(6, offset=OFM_SCRATCH_OFFSET),
                    dim_m,
                    index * 3 + 2,
                ),
                _dma_1d(
                    _ref(6, offset=OFM_SCRATCH_OFFSET),
                    _ref(4, index),
                    dim_m * 32 * 4,
                    1,
                    index * 3 + 3,
                ),
            ]
        )
        bindings.append(_binding(1, index, dimensions=(1, 1, dim_m, 32)))
    for index, dim_m in enumerate(dimensions):
        bindings.append(_binding(2, index, data_type=4, dimensions=(1, 1, dim_m, 32)))
    commands.append(_command_header(0, 32, tile=len(commands)).ljust(32, b"\x00"))
    required_tcdm_bytes = OFM_SCRATCH_OFFSET + max(dimensions) * 32 * 4
    return _package(
        b"".join(commands),
        weights,
        b"".join(bindings),
        len(commands) - 1,
        required_tcdm_bytes,
        len(dimensions),
        len(dimensions),
    )


def build_invocation(model):
    binding_addresses = struct.pack(
        "<HHIIIHHIII",
        1,
        0,
        INPUT_BASE,
        32,
        0,
        2,
        0,
        OUTPUT_BASE,
        32,
        0,
    )
    invocation = struct.pack(
        "<IHH6I8I",
        NAI_INVOCATION_MAGIC,
        1,
        0,
        64,
        MODEL_BASE,
        len(model),
        BINDING_TABLE_BASE,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert len(invocation) == 64 and len(binding_addresses) == 32
    return invocation, binding_addresses


def build_invocation_with_bindings(model, bindings):
    binding_addresses = b"".join(
        struct.pack("<HHIII", direction, index, base, byte_size, 0)
        for direction, index, base, byte_size in bindings
    )
    invocation = struct.pack(
        "<IHH6I8I",
        NAI_INVOCATION_MAGIC,
        1,
        0,
        64,
        MODEL_BASE,
        len(model),
        BINDING_TABLE_BASE,
        len(bindings),
        0,
        *([0] * 8),
    )
    return invocation, binding_addresses


def _signed_input(dim_m, seed):
    values = [((row * 3 + column + seed) % 15) - 7 for row in range(dim_m) for column in range(32)]
    return bytes(value & 0xFF for value in values), values


def _compile_fully_connected_model():
    default_root = Path(__file__).resolve().parents[4] / "neural-compiler"
    compiler_root = Path(os.environ.get("NEURAL_COMPILER_ROOT", default_root)).resolve()
    extension_modules = list((compiler_root / "ethosu").glob("regor*.so"))
    if not extension_modules:
        raise RuntimeError(
            f"Neural compiler Python extension is not built under {compiler_root}/ethosu"
        )

    fixture = Path(__file__).with_name("fully_connected_k33_n34.tflite.b64")
    model_data = base64.b64decode(fixture.read_text(encoding="ascii"))
    with tempfile.TemporaryDirectory(prefix="neural-ai-compiled-model-") as temporary_dir:
        temporary_path = Path(temporary_dir)
        input_path = temporary_path / "fully_connected_k33_n34.tflite"
        output_path = temporary_path / "output"
        input_path.write_bytes(model_data)
        command = [
            sys.executable,
            "-c",
            "import sys; from ethosu.vela.vela import main; raise SystemExit(main(sys.argv[1:]))",
            "--accelerator-config=neural-ai",
            "--output-format=nai",
            f"--output-dir={output_path}",
            str(input_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(compiler_root), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        result = subprocess.run(
            command,
            cwd=compiler_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Neural compiler failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
            )
        package = (output_path / "fully_connected_k33_n34.nai").read_bytes()
    assert package[:4] == b"NAIM"
    return package


async def _load_and_run(dut, axi_master, invocation):
    await load_firmware_elf_axi(
        dut,
        axi_master,
        Path(__file__).resolve().parents[3] / "sw/runtime/neural_ai/neural_ai.elf",
    )
    await program_command_queue(axi_master, INVOCATION_BASE, len(invocation))
    await release_fetch(dut, axi_master=axi_master)
    try:
        await wait_for_host_irq(
            dut,
            timeout_cycles=600000,
            axi_master=axi_master,
            report_name="test_compiler_runtime",
        )
    except AssertionError as error:
        status = await _axi_read32(axi_master, NPU_CMD_STATUS)
        fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
        fail_pointer = await _axi_read32(axi_master, NPU_CMD_FAIL_PTR)
        done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
        pc_signal = dut.u_npu_cluster.u_snitch_core.i_snitch.pc_q
        program_counter = (
            pc_signal.value.to_unsigned() if pc_signal.value.is_resolvable else 0
        )
        raise AssertionError(
            f"{error}; status={status} fail=0x{fail_code:08x} "
            f"pointer=0x{fail_pointer:08x} done={done_count} "
            f"pc=0x{program_counter:08x}"
        ) from error


@cocotb.test()
async def test_compiler_runtime_dma_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_data = bytes((index * 7 + 3) & 0xFF for index in range(32))
    model = build_model()
    invocation, binding_addresses = build_invocation(model)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(32))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, 32)) == input_data


@cocotb.test()
async def test_compiler_runtime_layout_round_trip(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_data = bytes((index * 11 + 5) & 0xFF for index in range(132))
    model = build_layout_model()
    invocation, binding_addresses = build_invocation(model)
    binding_addresses = bytearray(binding_addresses)
    struct.pack_into("<I", binding_addresses, 8, len(input_data))
    struct.pack_into("<I", binding_addresses, 24, len(input_data))
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(input_data)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(input_data))) == input_data


@cocotb.test()
async def test_compiler_runtime_c32_layout_round_trip(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    input_data = bytes((index * 13 + 7) & 0xFF for index in range(132))
    model = build_c32_layout_model()
    invocation, binding_addresses = build_invocation(model)
    binding_addresses = bytearray(binding_addresses)
    struct.pack_into("<I", binding_addresses, 8, len(input_data))
    struct.pack_into("<I", binding_addresses, 24, len(input_data))
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(input_data)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(input_data))) == input_data


@cocotb.test()
async def test_compiler_runtime_pointwise_c32_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    height, width, channels = 2, 2, 64
    input_values = [((pixel + channel) % 5) - 2
                    for pixel in range(height * width)
                    for channel in range(channels)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytearray()
    for pixel in range(height * width):
        value = sum(input_values[pixel * channels:(pixel + 1) * channels]) & 0xFF
        expected.extend([value] * channels)
    model = build_pointwise_c32_model(height, width, channels)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)

    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 4
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == bytes(expected)


@cocotb.test()
async def test_compiler_runtime_gemm_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    dimensions = (1, 31, 32, 33, 256)
    input_bases = (0x80000000, 0x80001000, 0x80002000, 0x80004000, 0x80006000)
    output_bases = (0x80010000, 0x80011000, 0x80015000, 0x80019000, 0x80020000)
    model = build_gemm_model(dimensions)
    runtime_bindings = []
    expected_outputs = []
    for index, (dim_m, base) in enumerate(zip(dimensions, input_bases)):
        input_data, values = _signed_input(dim_m, index)
        await write_l2_bytes(dut, base, input_data)
        runtime_bindings.append((1, index, base, len(input_data)))
        expected_outputs.append(b"".join(struct.pack("<i", value) for value in values))
    for index, (base, output) in enumerate(zip(output_bases, expected_outputs)):
        runtime_bindings.append((2, index, base, len(output)))

    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 16
    for base, expected in zip(output_bases, expected_outputs):
        assert bytes(await read_l2_bytes(dut, base, len(expected))) == expected


@cocotb.test()
async def test_compiler_generated_fully_connected_package(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)

    model = _compile_fully_connected_model()
    input_values = [(index % 5) - 2 for index in range(33)]
    input_data = bytes(value & 0xFF for value in input_values)
    expected = bytes([sum(input_values) & 0xFF] * 34)
    runtime_bindings = [
        (1, 0, INPUT_BASE, len(input_data)),
        (2, 0, OUTPUT_BASE, len(expected)),
    ]
    invocation, binding_addresses = build_invocation_with_bindings(model, runtime_bindings)
    await write_l2_bytes(dut, INPUT_BASE, input_data)
    await write_l2_bytes(dut, OUTPUT_BASE, bytes(len(expected)))
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    await _load_and_run(dut, axi_master, invocation)

    command_count = struct.unpack_from("<I", model, 32)[0]
    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == command_count
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, len(expected))) == expected


@cocotb.test()
async def test_compiler_runtime_rejects_invalid_bindings(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    model = build_model()
    invocation, original_addresses = build_invocation(model)
    invalid_records = []
    misaligned = bytearray(original_addresses)
    struct.pack_into("<I", misaligned, 20, OUTPUT_BASE + 4)
    invalid_records.append(misaligned)
    undersized = bytearray(original_addresses)
    struct.pack_into("<I", undersized, 24, 16)
    invalid_records.append(undersized)

    for binding_addresses in invalid_records:
        await reset_dut(dut)
        await write_l2_bytes(dut, MODEL_BASE, model)
        await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
        await write_l2_bytes(dut, INVOCATION_BASE, invocation)
        await _load_and_run(dut, axi_master, invocation)
        assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_FAIL
        assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == NPU_CMD_FAIL_BAD_BINDING
        assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 0


@cocotb.test()
async def test_compiler_runtime_rejects_bad_model_crc(dut):
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )
    await reset_dut(dut)
    model = bytearray(build_model())
    model[240] ^= 0x01
    invocation, binding_addresses = build_invocation(model)
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    await _load_and_run(dut, axi_master, invocation)

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_FAIL
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == NPU_CMD_FAIL_BAD_MODEL
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 0
