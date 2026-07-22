import struct
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
BINDING_TABLE_BASE = 0x80042000
TCDM_SCRATCH_BASE = 0x10100000

NAI_MODEL_MAGIC = 0x4D49414E
NAI_INVOCATION_MAGIC = 0x5649414E


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


def _binding(direction, index):
    descriptor = struct.pack(
        "<6HI4IIIiI4I",
        direction,
        index,
        1,
        1,
        4,
        0,
        index,
        1,
        1,
        1,
        32,
        32,
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


def build_model():
    commands = b"".join(
        [
            _dma_1d(_ref(3), _ref(6), 32, 0, 0),
            _dma_1d(_ref(6), _ref(4), 32, 1, 1),
            _command_header(0, 32, tile=2).ljust(32, b"\x00"),
        ]
    )
    bindings = _binding(1, 0) + _binding(2, 0)
    payloads = [commands, b"", b"", bindings, b""]
    section_types = [1, 2, 3, 4, 5]
    element_counts = [3, 0, 0, 2, 0]
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
        2,
        32,
        32,
        1,
        1,
        0,
        0,
        0,
    )
    model = header + b"".join(sections) + b"".join(payloads)
    assert len(header) == 64 and len(model) == 512
    return model


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
            timeout_cycles=160000,
            axi_master=axi_master,
            report_name="test_compiler_runtime_dma_package",
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

    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_FAIL_CODE) == 0
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == 2
    assert bytes(await read_l2_bytes(dut, OUTPUT_BASE, 32)) == input_data
