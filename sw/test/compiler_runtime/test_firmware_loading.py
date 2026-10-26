"""Offline boot-loader checks; no simulator is started by these tests."""

import ast
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

AI_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AI_ROOT / "hw/rtl/cluster/tb/tests"))

from npu_test_utils import (  # noqa: E402
    ITCM_BASE,
    NPU_DTCM_BASE,
    TCM_SIZE_BYTES,
    load_firmware_axi,
    load_firmware_elf_axi,
)


class FakeAXI:
    def __init__(self):
        self.writes = []

    async def write(self, address, data):
        self.writes.append((address, bytes(data)))


def fake_dut():
    memory = [SimpleNamespace(value=0xA5A5A5A5) for _ in range(TCM_SIZE_BYTES // 4)]
    return SimpleNamespace(u_npu_cluster=SimpleNamespace(u_sram_d_tcm=SimpleNamespace(mem=memory)))


def make_elf(segments):
    """Each segment is (physical address, file bytes, in-memory size)."""
    payload_offset = 52 + 32 * len(segments)
    ident = b"\x7fELF\x01\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIIIIIHHHHHH", ident, 2, 243, 1, ITCM_BASE,
                         52, 0, 0, 52, 32, len(segments), 0, 0, 0)
    headers = bytearray()
    payload = bytearray()
    for address, data, memory_size in segments:
        headers.extend(struct.pack("<8I", 1, payload_offset + len(payload), address,
                                   address, len(data), memory_size, 7, 4))
        payload.extend(data)
    return header + headers + payload


class FirmwareLoadingTests(unittest.IsolatedAsyncioTestCase):
    async def load_elf(self, data):
        dut, axi = fake_dut(), FakeAXI()
        with tempfile.TemporaryDirectory(prefix="neural-ai-elf-loader-") as folder:
            path = Path(folder) / "firmware.elf"
            path.write_bytes(data)
            await load_firmware_elf_axi(dut, axi, path)
        return dut, axi

    async def test_routes_text_rodata_and_zero_initialized_data(self):
        image = make_elf([
            (ITCM_BASE, b"TEXT", 8),
            (NPU_DTCM_BASE, b"", 0x100),
            (NPU_DTCM_BASE + 0x100, b"\x51\x62\x73\x84", 12),
        ])
        dut, axi = await self.load_elf(image)
        self.assertEqual(axi.writes, [(ITCM_BASE, b"TEXT"), (ITCM_BASE + 4, bytes(4))])
        memory = dut.u_npu_cluster.u_sram_d_tcm.mem
        self.assertTrue(all(word.value == 0 for word in memory[:64]))
        self.assertEqual(memory[64].value, 0x84736251)
        self.assertEqual([memory[65].value, memory[66].value], [0, 0])
        self.assertEqual(memory[67].value, 0xA5A5A5A5)

    async def test_supports_short_text_tail(self):
        _, axi = await self.load_elf(make_elf([(ITCM_BASE, b"abc", 3)]))
        self.assertEqual(axi.writes, [(ITCM_BASE, b"abc\x00")])

    async def test_oversized_raw_image_fails_before_any_bus_write(self):
        axi = FakeAXI()
        with tempfile.TemporaryDirectory(prefix="neural-ai-bin-loader-") as folder:
            path = Path(folder) / "firmware.bin"
            path.write_bytes(bytes(TCM_SIZE_BYTES + 4))
            with self.assertRaisesRegex(AssertionError, "exceeds I-TCM"):
                await load_firmware_axi(axi, path)
        self.assertEqual(axi.writes, [])

    async def test_raw_tail_fits_exactly_at_itcm_boundary(self):
        axi = FakeAXI()
        with tempfile.TemporaryDirectory(prefix="neural-ai-bin-loader-") as folder:
            path = Path(folder) / "firmware.bin"
            path.write_bytes(b"abc")
            await load_firmware_axi(axi, path, ITCM_BASE + TCM_SIZE_BYTES - 4)
        self.assertEqual(axi.writes, [(ITCM_BASE + TCM_SIZE_BYTES - 4, b"abc\x00")])

    async def test_rejects_segment_outside_tcm(self):
        with self.assertRaisesRegex(AssertionError, "outside TCM"):
            await self.load_elf(make_elf([(0x10100000, b"data", 4)]))

    async def test_rejects_invalid_elf_header(self):
        with self.assertRaisesRegex(AssertionError, "ELF32"):
            await self.load_elf(b"\x7fELF\x02\x01" + bytes(64))

    async def test_real_batch_test_elf_including_dtcm_geometry_table(self):
        path = AI_ROOT / "sw/test/spatz_ops/spatz_ops_copy_batch.elf"
        if not path.exists():
            self.skipTest("build sw/test/spatz_ops/spatz_ops_copy_batch.elf first")
        image = path.read_bytes()
        dut, axi = await self.load_elf(image)
        self.assertTrue(axi.writes)
        self.assertTrue(all(ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES for addr, _ in axi.writes))
        header = struct.unpack_from("<16sHHIIIIIHHHHHH", image)
        checked = 0
        memory = dut.u_npu_cluster.u_sram_d_tcm.mem
        for index in range(header[10]):
            kind, offset, virtual, physical, file_size, _, _, _ = struct.unpack_from(
                "<8I", image, header[5] + index * header[9])
            address = physical or virtual
            if kind != 1 or address < NPU_DTCM_BASE or not file_size:
                continue
            actual = b"".join(memory[(address - NPU_DTCM_BASE) // 4 + word].value.to_bytes(4, "little")
                              for word in range((file_size + 3) // 4))[:file_size]
            self.assertEqual(actual, image[offset:offset + file_size])
            checked += 1
        self.assertGreater(checked, 0)

    async def test_all_operator_suite_images_use_segment_aware_loading(self):
        suite = AI_ROOT / "hw/rtl/cluster/tb/tests/test_spatz_operator_library.py"
        paths = []
        for call in ast.walk(ast.parse(suite.read_text())):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id != "run_firmware_case":
                continue
            name = ast.literal_eval(call.args[1])
            self.assertEqual(Path(name).suffix, ".elf")
            folder = "sw/test/spatz_ops"
            for keyword in call.keywords:
                if keyword.arg == "firmware_dir":
                    folder = ast.literal_eval(keyword.value)
            paths.append(AI_ROOT / folder / name)
        self.assertTrue(paths)
        if any(not path.exists() for path in paths):
            self.skipTest("build sw/test/spatz_ops all and sw/test/afu_ops all first")
        for path in paths:
            with self.subTest(firmware=path.name):
                _, axi = await self.load_elf(path.read_bytes())
                self.assertTrue(axi.writes)
                self.assertTrue(all(ITCM_BASE <= addr < ITCM_BASE + TCM_SIZE_BYTES
                                    for addr, _ in axi.writes))


if __name__ == "__main__":
    unittest.main()
