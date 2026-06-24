import os

from cocotb.triggers import ClockCycles, RisingEdge, Timer


PASS_SIGNATURE = 0xDEADBEEF
FAIL_SIGNATURE_MASK = 0xFFF00000
FAIL_SIGNATURE_PREFIX = 0xBAD00000

ITCM_BASE = 0x10000000
TCM_SIZE_BYTES = 32 * 1024

TCDM_NUM_BANKS = 16
TCDM_BANK_WORDS = 1024
TCDM_WORD_BYTES = 32


async def reset_dut(dut):
    dut.rst_ni.value = 0
    if hasattr(dut, "fetch_enable_i"):
        dut.fetch_enable_i.value = 0
    dut.backdoor_we_i.value = 0
    await Timer(20, unit="ns")
    dut.rst_ni.value = 1
    await Timer(20, unit="ns")


async def hold_reset(dut):
    dut.rst_ni.value = 0
    if hasattr(dut, "fetch_enable_i"):
        dut.fetch_enable_i.value = 0
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
        addr = base_addr + offset
        if not (ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES):
            raise AssertionError(f"AXI boot image exceeds I-TCM at 0x{addr:08x}")
        await axi_master.write(base_addr + offset, firmware[offset : offset + width])

async def release_fetch(dut):
    dut.fetch_enable_i.value = 1
    await Timer(1, unit="ns")


async def wait_for_host_irq(dut, timeout_cycles=50000):
    for _ in range(timeout_cycles):
        irq_value = dut.irq_o.value
        if irq_value.is_resolvable and int(irq_value) == 1:
            return
        await RisingEdge(dut.clk_i)
    raise AssertionError("timeout waiting for host irq")

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
