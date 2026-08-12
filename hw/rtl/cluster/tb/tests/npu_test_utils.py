import os
import struct

from cocotb.triggers import ClockCycles, RisingEdge, Timer


PASS_SIGNATURE = 0xDEADBEEF
FAIL_SIGNATURE_MASK = 0xFFF00000
FAIL_SIGNATURE_PREFIX = 0xBAD00000

ITCM_BASE = 0x10000000
TCM_SIZE_BYTES = 32 * 1024
PMU_BASE = 0x20004000
PMU_CTRL = PMU_BASE + 0x0000
PMU_STATUS = PMU_BASE + 0x0004
PMU_NUM_COUNTERS = PMU_BASE + 0x0008
PMU_COUNTER_BASE = PMU_BASE + 0x0100
PMU_CTRL_ENABLE = 0x00000001
PMU_CTRL_CLEAR = 0x00000002
PMU_CTRL_SNAPSHOT = 0x00000004

PMU_COUNTER_NAMES = [
    "cycle",
    "snitch_retired_instr",
    "snitch_retired_load",
    "snitch_retired_int",
    "snitch_retired_acc",
    "snitch_tcdm_req",
    "snitch_tcdm_stall",
    "spatz_issue",
    "spatz_rsp",
    "spatz_tcdm_req",
    "spatz_tcdm_stall",
    "idma_busy",
    "idma_start",
    "idma_done",
    "idma_tcdm_req",
    "idma_tcdm_stall",
    "afu_done",
    "afu_tcdm_req",
    "afu_tcdm_stall",
    "sys_compute",
    "sys_weight_load",
    "sys_ofm_valid",
    "sys_ifm_req",
    "sys_ifm_stall",
    "sys_ofm_req",
    "sys_ofm_stall",
    "tcdm_req",
    "tcdm_gnt",
    "tcdm_stall",
    "tcdm_bank_req",
    "tcdm_read_req",
    "tcdm_write_req",
]

TCDM_NUM_BANKS = 16
TCDM_BANK_WORDS = 1024
TCDM_WORD_BYTES = 32
NPU_DTCM_BASE = 0x10008000
NPU_CMD_CTRL_BASE = 0x20005000
NPU_CMD_L2_BASE = NPU_CMD_CTRL_BASE + 0x00
NPU_CMD_TOTAL_BYTES = NPU_CMD_CTRL_BASE + 0x04
NPU_CMD_TCDM_BASE_REG = NPU_CMD_CTRL_BASE + 0x08
NPU_CMD_TCDM_BYTES = NPU_CMD_CTRL_BASE + 0x0C
NPU_CMD_START = NPU_CMD_CTRL_BASE + 0x10
NPU_CMD_STATUS = NPU_CMD_CTRL_BASE + 0x14
NPU_CMD_FAIL_CODE = NPU_CMD_CTRL_BASE + 0x18
NPU_CMD_FAIL_PTR = NPU_CMD_CTRL_BASE + 0x1C
NPU_CMD_DONE_COUNT = NPU_CMD_CTRL_BASE + 0x20
NPU_CMD_TCDM_BASE = 0x1017F000
NPU_CMD_TCDM_SIZE = 0x00001000
NPU_CMD_STATUS_IDLE = 0
NPU_CMD_STATUS_LOADING = 1
NPU_CMD_STATUS_RUNNING = 2
NPU_CMD_STATUS_PASS = 3
NPU_CMD_STATUS_FAIL = 4


async def reset_dut(dut):
    dut.rst_ni.value = 0
    if hasattr(dut, "fetch_enable_i"):
        dut.fetch_enable_i.value = 0
    dut.backdoor_we_i.value = 0
    if hasattr(dut, "backdoor_block_toggle_i"):
        dut.backdoor_block_bytes_i.value = 0
        dut.backdoor_block_addr_i.value = 0
        dut.backdoor_block_data_i.value = 0
        dut.backdoor_block_toggle_i.value = 0
    await Timer(20, unit="ns")
    dut.rst_ni.value = 1
    await Timer(20, unit="ns")


async def hold_reset(dut):
    dut.rst_ni.value = 0
    if hasattr(dut, "fetch_enable_i"):
        dut.fetch_enable_i.value = 0
    dut.backdoor_we_i.value = 0
    if hasattr(dut, "backdoor_block_toggle_i"):
        dut.backdoor_block_bytes_i.value = 0
        dut.backdoor_block_addr_i.value = 0
        dut.backdoor_block_data_i.value = 0
        dut.backdoor_block_toggle_i.value = 0
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


def _is_tcm_addr(addr):
    return (
        ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES
        or NPU_DTCM_BASE <= addr < NPU_DTCM_BASE + TCM_SIZE_BYTES
    )


def _write_dtcm_bytes(dut, addr, data, width=4):
    if addr < NPU_DTCM_BASE:
        raise AssertionError(f"D-TCM write below base at 0x{addr:08x}")
    if len(data) % width != 0:
        data += b"\x00" * (width - (len(data) % width))
    for offset in range(0, len(data), width):
        index = (addr + offset - NPU_DTCM_BASE) >> 2
        word = int.from_bytes(data[offset : offset + width], "little")
        dut.u_npu_cluster.u_sram_d_tcm.mem[index].value = word


async def load_firmware_elf_axi(dut, axi_master, filename, width=4):
    with open(filename, "rb") as firmware_file:
        elf = firmware_file.read()

    if len(elf) < 52 or elf[:4] != b"\x7fELF" or elf[4] != 1 or elf[5] != 1:
        raise AssertionError("expected little-endian ELF32 firmware")

    header = struct.unpack_from("<16sHHIIIIIHHHHHH", elf, 0)
    e_phoff = header[5]
    e_phentsize = header[9]
    e_phnum = header[10]
    if e_phentsize < 32:
        raise AssertionError("unsupported ELF program header size")

    for ph_idx in range(e_phnum):
        ph_off = e_phoff + ph_idx * e_phentsize
        if ph_off + 32 > len(elf):
            raise AssertionError("ELF program header exceeds file size")
        p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, _p_flags, _p_align = struct.unpack_from(
            "<IIIIIIII", elf, ph_off
        )
        if p_type != 1 or p_memsz == 0:
            continue

        addr = p_paddr or p_vaddr
        end_addr = addr + p_memsz
        if not _is_tcm_addr(addr) or not _is_tcm_addr(end_addr - 1):
            raise AssertionError(f"ELF load segment outside TCM at 0x{addr:08x}")
        if p_offset + p_filesz > len(elf):
            raise AssertionError("ELF load segment exceeds file size")

        payload = elf[p_offset : p_offset + p_filesz]
        if p_memsz > p_filesz:
            payload += b"\x00" * (p_memsz - p_filesz)
        if len(payload) % width != 0:
            payload += b"\x00" * (width - (len(payload) % width))

        if ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES:
            for offset in range(0, len(payload), width):
                await axi_master.write(addr + offset, payload[offset : offset + width])
        else:
            _write_dtcm_bytes(dut, addr, payload, width)

async def _axi_read32(axi_master, addr):
    resp = await axi_master.read(addr, 4)
    data = resp.data if hasattr(resp, "data") else resp
    return int.from_bytes(bytes(data), "little")


async def _axi_write32(axi_master, addr, value):
    await axi_master.write(addr, int(value & 0xFFFFFFFF).to_bytes(4, "little"))


async def program_command_queue(axi_master, l2_base, total_bytes, tcdm_base=NPU_CMD_TCDM_BASE, tcdm_size=NPU_CMD_TCDM_SIZE):
    await _axi_write32(axi_master, NPU_CMD_L2_BASE, l2_base)
    await _axi_write32(axi_master, NPU_CMD_TOTAL_BYTES, total_bytes)
    await _axi_write32(axi_master, NPU_CMD_TCDM_BASE_REG, tcdm_base)
    await _axi_write32(axi_master, NPU_CMD_TCDM_BYTES, tcdm_size)
    await _axi_write32(axi_master, NPU_CMD_FAIL_CODE, 0)
    await _axi_write32(axi_master, NPU_CMD_FAIL_PTR, 0)
    await _axi_write32(axi_master, NPU_CMD_DONE_COUNT, 0)
    await _axi_write32(axi_master, NPU_CMD_STATUS, 0)
    await _axi_write32(axi_master, NPU_CMD_START, 1)


async def pmu_start(axi_master):
    await _axi_write32(axi_master, PMU_CTRL, PMU_CTRL_CLEAR)
    await _axi_write32(axi_master, PMU_CTRL, PMU_CTRL_ENABLE)


async def pmu_snapshot_report(axi_master):
    await _axi_write32(axi_master, PMU_CTRL, PMU_CTRL_ENABLE | PMU_CTRL_SNAPSHOT)
    await _axi_write32(axi_master, PMU_CTRL, 0)

    num_counters = min(await _axi_read32(axi_master, PMU_NUM_COUNTERS), len(PMU_COUNTER_NAMES))
    counters = {}
    for counter_id in range(num_counters):
        lo = await _axi_read32(axi_master, PMU_COUNTER_BASE + counter_id * 8)
        hi = await _axi_read32(axi_master, PMU_COUNTER_BASE + counter_id * 8 + 4)
        counters[PMU_COUNTER_NAMES[counter_id]] = (hi << 32) | lo

    counters["overflow_status"] = await _axi_read32(axi_master, PMU_STATUS)
    return counters


def format_pmu_report(counters):
    cycles = counters.get("cycle", 0)

    def pct(name):
        return (100.0 * counters.get(name, 0) / cycles) if cycles else 0.0

    lines = [
        "PMU performance report:",
        f"  cycles={cycles}",
        (
            "  snitch: "
            f"instr={counters.get('snitch_retired_instr', 0)} "
            f"load={counters.get('snitch_retired_load', 0)} "
            f"tcdm_req={counters.get('snitch_tcdm_req', 0)} "
            f"stall={counters.get('snitch_tcdm_stall', 0)}"
        ),
        (
            "  systolic: "
            f"compute={counters.get('sys_compute', 0)} ({pct('sys_compute'):.2f}%) "
            f"ifm_req={counters.get('sys_ifm_req', 0)} "
            f"ofm_req={counters.get('sys_ofm_req', 0)} "
            f"ofm_stall={counters.get('sys_ofm_stall', 0)}"
        ),
        (
            "  spatz: "
            f"issue={counters.get('spatz_issue', 0)} "
            f"rsp={counters.get('spatz_rsp', 0)} "
            f"tcdm_req={counters.get('spatz_tcdm_req', 0)} "
            f"stall={counters.get('spatz_tcdm_stall', 0)}"
        ),
        (
            "  idma: "
            f"busy={counters.get('idma_busy', 0)} ({pct('idma_busy'):.2f}%) "
            f"start={counters.get('idma_start', 0)} "
            f"done={counters.get('idma_done', 0)} "
            f"tcdm_stall={counters.get('idma_tcdm_stall', 0)}"
        ),
        (
            "  afu: "
            f"done={counters.get('afu_done', 0)} "
            f"tcdm_req={counters.get('afu_tcdm_req', 0)} "
            f"stall={counters.get('afu_tcdm_stall', 0)}"
        ),
        (
            "  tcdm: "
            f"req={counters.get('tcdm_req', 0)} "
            f"gnt={counters.get('tcdm_gnt', 0)} "
            f"stall={counters.get('tcdm_stall', 0)} "
            f"read={counters.get('tcdm_read_req', 0)} "
            f"write={counters.get('tcdm_write_req', 0)}"
        ),
    ]
    if counters.get("overflow_status", 0):
        lines.append(f"  overflow_status=0x{counters['overflow_status']:08x}")
    return "\n".join(lines)


async def release_fetch(dut, axi_master=None, enable_pmu=True):
    if axi_master is not None and enable_pmu:
        await pmu_start(axi_master)
    dut.fetch_enable_i.value = 1
    await Timer(1, unit="ns")


async def wait_for_host_irq(dut, timeout_cycles=50000, axi_master=None, report_name=None):
    for _ in range(timeout_cycles):
        irq_value = dut.irq_o.value
        if irq_value.is_resolvable and int(irq_value) == 1:
            if axi_master is not None:
                report = await pmu_snapshot_report(axi_master)
                prefix = f"{report_name}: " if report_name else ""
                dut._log.info("%s%s", prefix, format_pmu_report(report))
                return report
            return None
        await RisingEdge(dut.clk_i)
    raise AssertionError("timeout waiting for host irq")

async def write_l2_bytes(dut, base_addr, data):
    if hasattr(dut, "backdoor_block_toggle_i"):
        dut.backdoor_we_i.value = 0
        toggle = dut.backdoor_block_toggle_i.value
        toggle_value = int(toggle) if toggle.is_resolvable else 0
        for offset in range(0, len(data), 32):
            block = bytes(data[offset : offset + 32])
            dut.backdoor_block_addr_i.value = base_addr + offset
            dut.backdoor_block_bytes_i.value = len(block)
            dut.backdoor_block_data_i.value = int.from_bytes(block, "little")
            toggle_value ^= 1
            dut.backdoor_block_toggle_i.value = toggle_value
            await Timer(1, unit="ps")
        dut.backdoor_block_bytes_i.value = 0
        return

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
    if hasattr(dut, "backdoor_block_rdata_o"):
        data = bytearray()
        dut.backdoor_we_i.value = 0
        for offset in range(0, length, 32):
            block_bytes = min(32, length - offset)
            dut.backdoor_block_addr_i.value = base_addr + offset
            await Timer(1, unit="ps")
            block = dut.backdoor_block_rdata_o.value.to_unsigned().to_bytes(32, "little")
            data.extend(block[:block_bytes])
        return data

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


def read_dtcm_word(dut, addr):
    index = (addr - NPU_DTCM_BASE) >> 2
    value = dut.u_npu_cluster.u_sram_d_tcm.mem[index].value
    return value.to_unsigned() if value.is_resolvable else 0


def firmware_path(test_file, relative_fw):
    return os.path.join(os.path.dirname(test_file), "../../../../../", relative_fw)
