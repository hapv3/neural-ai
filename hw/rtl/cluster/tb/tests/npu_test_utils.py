import os

from cocotb.triggers import ClockCycles, RisingEdge, Timer


PASS_SIGNATURE = 0xDEADBEEF
FAIL_SIGNATURE_MASK = 0xFFF00000
FAIL_SIGNATURE_PREFIX = 0xBAD00000

ITCM_BASE = 0x10000000
DTCM_BASE = 0x10008000
TCM_SIZE_BYTES = 32 * 1024
ITCM_WORD_BYTES = 4
DTCM_WORD_BYTES = 4

TCDM_NUM_BANKS = 16
TCDM_BANK_WORDS = 1024
TCDM_WORD_BYTES = 32


async def reset_dut(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, unit="ns")
    dut.rst_ni.value = 1
    await Timer(20, unit="ns")


async def hold_reset(dut):
    dut.rst_ni.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, unit="ns")


async def release_reset(dut):
    dut.rst_ni.value = 1
    await Timer(20, unit="ns")


async def load_firmware_axi(axi_master, filename, base_addr=ITCM_BASE, width=4):
    with open(filename, "rb") as firmware_file:
        firmware = firmware_file.read()

    if len(firmware) % width != 0:
        firmware += b"\x00" * (width - (len(firmware) % width))

    for offset in range(0, len(firmware), width):
        await axi_master.write(base_addr + offset, firmware[offset : offset + width])


def load_firmware_tcm_backdoor(dut, filename, base_addr=ITCM_BASE):
    with open(filename, "rb") as firmware_file:
        firmware = firmware_file.read()

    firmware_word_bytes = ITCM_WORD_BYTES

    if len(firmware) % firmware_word_bytes != 0:
        firmware += b"\x00" * (firmware_word_bytes - (len(firmware) % firmware_word_bytes))

    for offset in range(0, len(firmware), firmware_word_bytes):
        addr = base_addr + offset
        word = int.from_bytes(firmware[offset : offset + firmware_word_bytes], "little")
        if ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES:
            dut.u_npu_cluster.u_sram_i_tcm.mem[(addr - ITCM_BASE) // ITCM_WORD_BYTES].value = word
        elif DTCM_BASE <= addr < DTCM_BASE + TCM_SIZE_BYTES:
            dut.u_npu_cluster.u_sram_d_tcm.mem[(addr - DTCM_BASE) // DTCM_WORD_BYTES].value = word
        else:
            raise AssertionError(f"Firmware byte outside I/D-TCM range at 0x{addr:08x}")


def read_dtcm_word(dut, word_index):
    val_32 = dut.u_npu_cluster.u_sram_d_tcm.mem[word_index].value
    if not val_32.is_resolvable:
        return 0
    return val_32.to_unsigned() & 0xFFFFFFFF


def write_dtcm_word(dut, word_index, value):
    dut.u_npu_cluster.u_sram_d_tcm.mem[word_index].value = value & 0xFFFFFFFF


def read_status_debug(dut):
    return {
        "status": read_dtcm_word(dut, 0),
        "pass_count": read_dtcm_word(dut, 1),
        "fail_test": read_dtcm_word(dut, 2),
        "fail_index": read_dtcm_word(dut, 3),
        "got": read_dtcm_word(dut, 4),
        "expected": read_dtcm_word(dut, 5),
        "phase": read_dtcm_word(dut, 6),
        "op": read_dtcm_word(dut, 7),
    }


async def wait_for_status(dut, expected_pass_count=None, timeout_cycles=50000):
    for _ in range(timeout_cycles):
        status = read_dtcm_word(dut, 0)
        if status == PASS_SIGNATURE:
            debug = read_status_debug(dut)
            if expected_pass_count is not None:
                assert debug["pass_count"] == expected_pass_count, debug
            return debug
        if (status & FAIL_SIGNATURE_MASK) == FAIL_SIGNATURE_PREFIX:
            raise AssertionError(f"firmware failed: {read_status_debug(dut)}")
        await RisingEdge(dut.clk_i)
    raise AssertionError(f"timeout waiting for firmware: {read_status_debug(dut)}")


async def write_l2_bytes(dut, base_addr, data):
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)
    for offset, byte_val in enumerate(data):
        dut.backdoor_we_i.value = 1
        dut.backdoor_addr_i.value = base_addr + offset
        dut.backdoor_data_i.value = byte_val & 0xFF
        await RisingEdge(dut.clk_i)
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)


async def read_l2_bytes(dut, base_addr, length):
    data = []
    dut.backdoor_we_i.value = 0
    await RisingEdge(dut.clk_i)
    for offset in range(length):
        dut.backdoor_addr_i.value = base_addr + offset
        await Timer(1, unit="ps")
        data.append(dut.backdoor_rdata_o.value.to_unsigned() & 0xFF)
        if (offset & 0x1F) == 0x1F:
            await ClockCycles(dut.clk_i, 1)
    return data


def read_tcdm_byte(dut, addr):
    bank_idx = (addr >> 5) % TCDM_NUM_BANKS
    word_index = ((addr >> 5) // TCDM_NUM_BANKS) & (TCDM_BANK_WORDS - 1)
    bit_offset = (addr % TCDM_WORD_BYTES) * 8
    val_256 = dut.u_npu_cluster.gen_sram_banks[bank_idx].u_sram_bank.mem[word_index].value
    if not val_256.is_resolvable:
        return 0
    return (val_256.to_unsigned() >> bit_offset) & 0xFF


def read_tcdm_word32(dut, addr):
    word = 0
    for byte_idx in range(4):
        word |= read_tcdm_byte(dut, addr + byte_idx) << (byte_idx * 8)
    return word


def firmware_path(test_file, relative_fw):
    return os.path.join(os.path.dirname(test_file), "../../../../../", relative_fw)
