"""Offline validation gates for command streams consumed by trusted firmware."""

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "hw/rtl/cluster/tb/tests"))

import test_compiled_model as runtime


def model(stream, count=6):
    data = bytearray(96) + stream
    struct.pack_into("<2I", data, 20, 1, 64)
    struct.pack_into("<I", data, 32, count)
    struct.pack_into("<8I", data, 64, 1, 0, 96, len(stream), 32, count, 0, 0)
    return data


def affine_stream():
    stream = bytearray(64)
    struct.pack_into("<HHIII4I", stream, 0, 33, 64, 0, 0, 0, 3, 2, 96, 3)
    struct.pack_into("<6I", stream, 32, 3, 1, 5, 32, 19, 1)
    stream += runtime._command_header(2, 64) + bytes(48)
    stream += runtime._command_header(5, 32) + bytes(16)
    stream += runtime._command_header(0, 32) + bytes(16)
    return stream


class CommandStreamValidationTests(unittest.TestCase):
    def test_valid_affine_and_modular_patching(self):
        stream = affine_stream()
        struct.pack_into("<I", stream, 64 + 12, 0xffffffff)
        commands, end = runtime._logical_command_records(model(stream))
        self.assertEqual(len(commands), 6)
        self.assertEqual([struct.unpack_from("<I", commands[index], 12)[0]
                          for index in (0, 2, 4)], [0xffffffff, 0, 1])
        self.assertEqual(len(end), 32)

    def test_malformed_affine_semantics_and_bounds(self):
        mutations = [(32, 1), (48, 16), (40, 3), (40, 2), (48, 24),
                     (48, 0x40000000), (24, 0xfffffff0), (56, 1), (4, 1),
                     (68, 0x80000000), (68, 4), (64, 33 | (64 << 16)),
                     (64, 64 << 16), (64, 2), (28, 25), (20, 17), (20, 0),
                     (16, 0xffffffff), (188, 1)]
        for offset, value in mutations:
            with self.subTest(offset=offset, value=value):
                stream = affine_stream()
                struct.pack_into("<I", stream, offset, value)
                with self.assertRaises(ValueError):
                    runtime._logical_command_records(model(stream))

    def test_count_truncation_and_trailing_bytes(self):
        for stream, count in ((affine_stream(), 5), (affine_stream(), 7),
                              (affine_stream()[:-1], 6), (affine_stream() + bytes(32), 6)):
            with self.subTest(bytes=len(stream), count=count):
                with self.assertRaises(ValueError):
                    runtime._logical_command_records(model(stream, count))


if __name__ == "__main__":
    unittest.main()
