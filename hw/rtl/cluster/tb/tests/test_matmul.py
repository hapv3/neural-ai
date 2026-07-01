import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    NPU_CMD_DIR_L1_TO_L2,
    NPU_CMD_DIR_L2_TO_L1,
    NPU_CMD_DONE_COUNT,
    NPU_CMD_FAIL_CODE,
    NPU_CMD_FAIL_PTR,
    NPU_CMD_STATUS,
    NPU_CMD_STATUS_PASS,
    NPU_CMD_TCDM_BASE,
    NPU_DTCM_BASE,
    build_command_table,
    cmd_barrier,
    cmd_end,
    cmd_idma_1d,
    cmd_systolic_gemm32,
    format_pmu_report,
    firmware_path,
    load_firmware_axi,
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


EXT_MEM_WEIGHT = 0x80000000
EXT_MEM_IFM = 0x80001000
EXT_MEM_OFM = 0x80002000
EXT_MEM_CMD = 0x80040000
WEIGHT_PING_ADDR = 0x10110000
IFM_PING_ADDR = 0x10120000
OFM_PING_ADDR = 0x10200000
DIM_M = 64
STREAMING_BARRIER_COUNT = 128


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

    await write_l2_bytes(dut, EXT_MEM_WEIGHT, pack_i8_words(weights))
    await write_l2_bytes(dut, EXT_MEM_IFM, pack_i8_words(ifm))
    command_stream = build_command_table(
        ([cmd_barrier(layer_id=0, tile_id=tile_id) for tile_id in range(STREAMING_BARRIER_COUNT)] + [
            cmd_idma_1d(EXT_MEM_WEIGHT, WEIGHT_PING_ADDR, 32 * 32, NPU_CMD_DIR_L2_TO_L1, layer_id=0, tile_id=0),
            cmd_idma_1d(EXT_MEM_IFM, IFM_PING_ADDR, DIM_M * 32, NPU_CMD_DIR_L2_TO_L1, layer_id=0, tile_id=1),
            cmd_systolic_gemm32(WEIGHT_PING_ADDR, IFM_PING_ADDR, OFM_PING_ADDR, DIM_M, layer_id=0, tile_id=2),
            cmd_idma_1d(OFM_PING_ADDR, EXT_MEM_OFM, DIM_M * 32 * 4, NPU_CMD_DIR_L1_TO_L2, layer_id=0, tile_id=3),
            cmd_end(layer_id=0, tile_id=4),
        ])
    )
    await write_l2_bytes(dut, EXT_MEM_CMD, command_stream)
    await load_firmware_axi(
        axi_master,
        firmware_path(__file__, "sw/test/matmul/matmul.bin"),
    )
    await program_command_queue(axi_master, EXT_MEM_CMD, len(command_stream))
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
    assert await _axi_read32(axi_master, NPU_CMD_STATUS) == NPU_CMD_STATUS_PASS
    assert await _axi_read32(axi_master, NPU_CMD_DONE_COUNT) == STREAMING_BARRIER_COUNT + 4

    ofm_data = await read_l2_bytes(dut, EXT_MEM_OFM, DIM_M * 32 * 4)
    ofm_words = unpack_i32_words(ofm_data)

    for idx, (got, expected) in enumerate(zip(ofm_words, golden)):
        assert got == int(expected), (
            f"OFM[{idx // 32},{idx % 32}] got={got} expected={int(expected)}"
        )
