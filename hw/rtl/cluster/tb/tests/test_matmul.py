import random
import struct

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    NPU_CMD_DONE_COUNT,
    NPU_CMD_FAIL_CODE,
    NPU_CMD_FAIL_PTR,
    NPU_CMD_STATUS,
    NPU_CMD_STATUS_PASS,
    NPU_CMD_TCDM_BASE,
    NPU_DTCM_BASE,
    format_pmu_report,
    firmware_path,
    load_firmware_elf_axi,
    pmu_snapshot_report,
    program_command_queue,
    read_l2_bytes,
    read_dtcm_word,
    read_tcdm_word32,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
    _axi_read32,
)


EXT_MEM_IFM = 0x80001000
EXT_MEM_OFM = 0x80002000
INVOCATION_BASE = 0x80040000
MODEL_BASE = 0x80041000
BINDING_TABLE_BASE = 0x80044000
WEIGHT_PING_ADDR = 0x10110000
IFM_PING_ADDR = 0x10120000
OFM_PING_ADDR = 0x10130000
TCDM_BASE = 0x10100000
DIM_M = 64
NAI_MODEL_MAGIC = 0x4D49414E
NAI_INVOCATION_MAGIC = 0x5649414E


def pack_i8_words(matrix):
    return bytes(
        int(value).to_bytes(1, "little", signed=False)[0]
        for value in matrix.flatten()
    )


def unpack_i32_words(data):
    words = []
    for offset in range(0, len(data), 4):
        words.append(
            int.from_bytes(bytes(data[offset : offset + 4]), "little", signed=True)
        )
    return words


def ref(region, index=0, offset=0):
    return struct.pack("<HHI", region, index, offset)


def command_header(command_type, size, tile):
    return struct.pack("<HHIII", command_type, size, 0, 0, tile)


def dma_1d(source, destination, length, direction, tile):
    command = command_header(2, 64, tile) + source + destination
    return command + struct.pack("<II6I", length, direction, *([0] * 6))


def gemm32(weights, ifm, ofm, dim_m, tile):
    command = command_header(6, 96, tile)
    command += weights + ifm + ref(6) + ofm
    return command + struct.pack("<4I8I", dim_m, 128, 128, 0, *([0] * 8))


def binding(direction, index, data_type, byte_size):
    dimensions = (1, 1, DIM_M, 32)
    descriptor = struct.pack(
        "<6HI4IIIiI4I",
        direction, index, data_type, 1, 4, 0, index,
        *dimensions, byte_size, 0, 0, 0, 0, 0, 0, 0,
    )
    assert len(descriptor) == 64
    return descriptor


def build_v2_model(weights):
    commands = [
        dma_1d(ref(1), ref(6, offset=WEIGHT_PING_ADDR - TCDM_BASE),
               len(weights), 0, 0),
        dma_1d(ref(3), ref(6, offset=IFM_PING_ADDR - TCDM_BASE),
               DIM_M * 32, 0, 1),
        gemm32(ref(6, offset=WEIGHT_PING_ADDR - TCDM_BASE),
               ref(6, offset=IFM_PING_ADDR - TCDM_BASE),
               ref(6, offset=OFM_PING_ADDR - TCDM_BASE),
               DIM_M, 2),
        dma_1d(ref(6, offset=OFM_PING_ADDR - TCDM_BASE), ref(4),
               DIM_M * 32 * 4, 1, 3),
    ]
    command_count = len(commands)
    commands.append(command_header(0, 32, command_count).ljust(32, b"\x00"))
    command_bytes = b"".join(commands)
    bindings = (
        binding(1, 0, 1, DIM_M * 32) +
        binding(2, 0, 4, DIM_M * 32 * 4)
    )
    payloads = [command_bytes, weights, b"", bindings, b""]
    section_types = [1, 2, 3, 4, 5]
    element_counts = [command_count + 1, len(weights), 0, 2, 0]
    offset = 64 + len(payloads) * 32
    sections = []
    for section_type, element_count, payload in zip(section_types, element_counts, payloads):
        assert offset % 32 == 0 and len(payload) % 32 == 0
        sections.append(struct.pack("<8I", section_type, 0, offset, len(payload),
                                    32, element_count, 0, 0))
        offset += len(payload)
    required_tcdm_bytes = OFM_PING_ADDR - TCDM_BASE + DIM_M * 32 * 4
    assert required_tcdm_bytes <= 0x7F000
    header = struct.pack(
        "<IHH11I3I", NAI_MODEL_MAGIC, 1, 1, 1, 0, offset, 5, 64, 224,
        command_count, required_tcdm_bytes, 32, 1, 1, 0, 0, 0,
    )
    return header + b"".join(sections) + b"".join(payloads)


def build_v2_invocation(model):
    invocation = struct.pack(
        "<IHH6I8I", NAI_INVOCATION_MAGIC, 1, 0, 64, MODEL_BASE, len(model),
        BINDING_TABLE_BASE, 2, 0, *([0] * 8),
    )
    addresses = struct.pack(
        "<HHIIIHHIII", 1, 0, EXT_MEM_IFM, DIM_M * 32, 0,
        2, 0, EXT_MEM_OFM, DIM_M * 32 * 4, 0,
    )
    return invocation, addresses


@cocotb.test()
async def test_matmul(dut):
    clock = Clock(dut.clk_i, 1, unit="ns")
    cocotb.start_soon(clock.start())

    axi_master = AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "s_axi"),
        dut.clk_i,
        dut.rst_ni,
        reset_active_level=False,
    )

    await reset_dut(dut)

    rng = random.Random(0x4D41544D)
    np_rng = np.random.default_rng(rng.randrange(1 << 32))
    weights = np_rng.integers(0, 6, size=(32, 32), dtype=np.int32)
    ifm = np_rng.integers(0, 6, size=(DIM_M, 32), dtype=np.int32)
    golden = np.dot(ifm, weights).flatten()

    weight_bytes = pack_i8_words(weights)
    await write_l2_bytes(dut, EXT_MEM_IFM, pack_i8_words(ifm))
    model = build_v2_model(weight_bytes)
    invocation, binding_addresses = build_v2_invocation(model)
    await write_l2_bytes(dut, INVOCATION_BASE, invocation)
    await write_l2_bytes(dut, MODEL_BASE, model)
    await write_l2_bytes(dut, BINDING_TABLE_BASE, binding_addresses)
    await load_firmware_elf_axi(
        dut,
        axi_master,
        firmware_path(__file__, "sw/runtime/neural_ai/neural_ai.elf"),
    )
    await program_command_queue(axi_master, INVOCATION_BASE, len(invocation))
    await release_fetch(dut, axi_master=axi_master)

    try:
        await wait_for_host_irq(dut, timeout_cycles=120000, axi_master=axi_master, report_name="test_matmul")
    except AssertionError:
        status = await _axi_read32(axi_master, NPU_CMD_STATUS)
        fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
        fail_ptr = await _axi_read32(axi_master, NPU_CMD_FAIL_PTR)
        done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
        cmd_words = [
            read_tcdm_word32(dut, NPU_CMD_TCDM_BASE + word_idx * 4)
            for word_idx in range(24)
        ]
        weight_words = [
            read_tcdm_word32(dut, WEIGHT_PING_ADDR + word_idx * 4)
            for word_idx in range(8)
        ]
        phase = read_dtcm_word(dut, NPU_DTCM_BASE + 0x18)
        op = read_dtcm_word(dut, NPU_DTCM_BASE + 0x1C)
        pmu = await pmu_snapshot_report(axi_master)
        dut._log.error(
            "command queue timeout: status=0x%08x fail=0x%08x ptr=0x%08x done=%d phase=0x%08x op=0x%08x cmd_words=%s weight_words=%s\n%s",
            status,
            fail_code,
            fail_ptr,
            done_count,
            phase,
            op,
            " ".join(f"{word:08x}" for word in cmd_words),
            " ".join(f"{word:08x}" for word in weight_words),
            format_pmu_report(pmu),
        )
        raise
    status = await _axi_read32(axi_master, NPU_CMD_STATUS)
    fail_code = await _axi_read32(axi_master, NPU_CMD_FAIL_CODE)
    fail_ptr = await _axi_read32(axi_master, NPU_CMD_FAIL_PTR)
    done_count = await _axi_read32(axi_master, NPU_CMD_DONE_COUNT)
    assert status == NPU_CMD_STATUS_PASS, (
        f"matmul V2 dispatch failed: status={status} fail=0x{fail_code:08x} "
        f"ptr=0x{fail_ptr:08x} done={done_count}"
    )
    assert done_count == 4

    ofm_data = await read_l2_bytes(dut, EXT_MEM_OFM, DIM_M * 32 * 4)
    ofm_words = unpack_i32_words(ofm_data)

    for idx, (got, expected) in enumerate(zip(ofm_words, golden)):
        assert got == int(expected), (
            f"OFM[{idx // 32},{idx % 32}] got={got} expected={int(expected)}"
        )
