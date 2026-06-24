import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster

from npu_test_utils import (
    firmware_path,
    load_firmware_axi,
    read_l2_bytes,
    release_fetch,
    reset_dut,
    wait_for_host_irq,
    write_l2_bytes,
)


EXT_MEM_WEIGHT = 0x80000000
EXT_MEM_IFM = 0x80001000
EXT_MEM_OFM = 0x80002000
DIM_M = 64


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
    await load_firmware_axi(
        axi_master,
        firmware_path(__file__, "sw/test/matmul/matmul.bin"),
    )
    await release_fetch(dut)

    await wait_for_host_irq(dut, timeout_cycles=120000)

    ofm_data = await read_l2_bytes(dut, EXT_MEM_OFM, DIM_M * 32 * 4)
    ofm_words = unpack_i32_words(ofm_data)

    for idx, (got, expected) in enumerate(zip(ofm_words, golden)):
        assert got == int(expected), (
            f"OFM[{idx // 32},{idx % 32}] got={got} expected={int(expected)}"
        )
