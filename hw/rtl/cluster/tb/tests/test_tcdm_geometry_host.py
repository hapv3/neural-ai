"""Host-only snapshot mapping tests for baseline and experimental TCDM."""
import unittest
from types import SimpleNamespace
from npu_test_utils import (_write_dtcm_bytes, read_tcdm_byte, read_tcdm_bytes, read_tcdm_word32,
                            tcdm_geometry, write_tcdm_bytes)


class Word:
    def __init__(self, bits):
        self.bits = bits
        self.raw = 0

    @property
    def value(self):
        return self

    @value.setter
    def value(self, value):
        self.raw = value

    def __len__(self):
        return self.bits

    @property
    def is_resolvable(self):
        return True

    def to_unsigned(self):
        return self.raw


class GeometryTest(unittest.TestCase):
    def test_private_dtcm_signature_clear_backdoor(self):
        words = [Word(32) for _ in range(8)]
        for word in words:
            word.value = 0xdeadbeef
        dut = SimpleNamespace(u_npu_cluster=SimpleNamespace(
            u_sram_d_tcm=SimpleNamespace(mem=words)))
        _write_dtcm_bytes(dut, 0x10008000, bytes(24))
        self.assertEqual([word.raw for word in words], [0]*6 + [0xdeadbeef]*2)

    def test_baseline_and_candidate_roundtrip(self):
        data = bytes((i * 7 + (i >> 8)) & 255 for i in range(524288))
        for banks, words, word_bytes in ((16, 1024, 32), (64, 2048, 4)):
            with self.subTest(banks=banks, word_bytes=word_bytes):
                dut = SimpleNamespace(u_npu_cluster=SimpleNamespace(gen_sram_banks=[
                    SimpleNamespace(u_sram_bank=SimpleNamespace(mem=[Word(word_bytes*8)
                                    for _ in range(words)])) for _ in range(banks)]))
                self.assertEqual(tcdm_geometry(dut), (banks, words, word_bytes))
                write_tcdm_bytes(dut, data)
                self.assertEqual(read_tcdm_bytes(dut), data)
                for offset in (0, 3, 4, 31, 32, 255, 256, 383999, 393216, 524287):
                    self.assertEqual(read_tcdm_byte(dut, 0x10100000 + offset), data[offset])
                self.assertEqual(read_tcdm_word32(dut, 0x1010001f), int.from_bytes(data[31:35], "little"))
                with self.assertRaises(ValueError):
                    write_tcdm_bytes(dut, data[:-1])


if __name__ == "__main__":
    unittest.main()
