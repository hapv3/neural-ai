import cocotb
from cocotb.utils import get_sim_time

from npu_test_utils import read_l2_bytes, write_l2_bytes


@cocotb.test()
async def test_l2_block_backdoor_round_trip(dut):
    base = 0x80123457
    payload = bytes((index * 29 + 7) & 0xFF for index in range(77))

    start_ps = get_sim_time("ps")
    await write_l2_bytes(dut, base, payload)
    actual = bytes(await read_l2_bytes(dut, base, len(payload)))
    elapsed_ps = get_sim_time("ps") - start_ps

    assert actual == payload
    assert elapsed_ps <= 8
