"""Offline regression tests for reference-checkpoint preparation."""

from pathlib import Path
import struct
import tempfile
import unittest

from yolo_reference_replay import artifact, checked_read, plan_checkpoints, plan_lut_restore, plan_restore_preamble, reset_buffers, runtime


def linebuffer(layer, address, rows, width, accum=0):
    command = bytearray(160)
    struct.pack_into("<HHIII", command, 0, 9, 160, 0, layer, 0)
    struct.pack_into("<5I", command, 108, address, rows, accum, width * 32, width)
    return bytes(command)


def lut(src, dst, length, reuse=False):
    command = bytearray(64)
    struct.pack_into("<HHIII", command, 0, 12, 64, 4 if reuse else 0, 1, 0)
    struct.pack_into("<HHI", command, 16, 6, 0, src)
    struct.pack_into("<HHI", command, 24, 6, 0, dst)
    struct.pack_into("<HHI", command, 32, 1, 0, 0)
    struct.pack_into("<I", command, 40, length)
    return bytes(command)


class ReferenceReplayTest(unittest.TestCase):
    def test_fresh_reset_uses_binding_sizes_and_no_snapshot(self):
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        bindings = b"".join(runtime._binding(direction, 0, dimensions=(1, 1, 1, size))
                            for direction, size in ((1, 64), (2, 96), (3, 128)))
        model = bytearray(runtime._package(end, b"", bindings, 0, 1024, 1, 1))
        # The lightweight package helper counts only public bindings. This test
        # exercises descriptor decoding, including the explicit temporary one.
        struct.pack_into("<I", model, 64 + 3 * 32 + 20, 3)
        memory, temporary, output, input_bytes = reset_buffers(model)
        self.assertEqual((len(memory), len(temporary), len(output), input_bytes), (524288, 128, 96, 64))
        self.assertFalse(any(memory + temporary + output))
        invalid = runtime._package(end, b"", bindings[:128], 0, 1024, 1, 1)
        with self.assertRaisesRegex(ValueError, "one input, output and temporary"):
            reset_buffers(invalid)

    def commands(self):
        return [linebuffer(0, 128, 4, 2), lut(128, 0, 128),
                linebuffer(2, 0, 1, 2), linebuffer(2, 32, 1, 2)]

    def test_plan_derives_command_numbers_addresses_and_rows(self):
        checks, end = plan_checkpoints(self.commands(), 1, (1, 2, 2, 2), (1, 1, 2, 3))
        self.assertEqual(end, 4)
        self.assertEqual([c["command"] for c in checks], [2, 3, 4])
        self.assertEqual([c["reference_first_row"] for c in checks], [0, 0, 1])
        self.assertEqual([c["channels"] for c in checks], [2, 3, 3])
        self.assertEqual(checks[-1]["address"], 32)

    def test_rejects_noncontiguous_or_partial_outputs(self):
        for bad in (linebuffer(2, 64, 1, 2), linebuffer(2, 32, 1, 2, accum=1)):
            commands = self.commands()
            commands[-1] = bad
            with self.assertRaises(ValueError):
                plan_checkpoints(commands, 1, (1, 2, 2, 2), (1, 1, 2, 3))

    def test_rejects_async_mapping_or_multigroup_tensor(self):
        commands = self.commands()
        async_command = bytearray(commands[2])
        struct.pack_into("<H", async_command, 0, 29)
        commands[2] = bytes(async_command)
        with self.assertRaisesRegex(ValueError, "Async"):
            plan_checkpoints(commands, 1, (1, 2, 2, 2), (1, 1, 2, 3))
        with self.assertRaises(ValueError):
            plan_checkpoints(self.commands(), 1, (1, 2, 2, 33), (1, 1, 2, 3))

    def test_lut_source_offset_maps_reference_rows(self):
        commands = self.commands()
        commands[1] = lut(192, 0, 64)
        checks, _ = plan_checkpoints(commands, 1, (1, 2, 2, 2), (1, 1, 2, 3))
        self.assertEqual((checks[0]["reference_first_row"], checks[0]["rows"]), (2, 2))
        commands[1] = lut(224, 0, 64)
        with self.assertRaises(ValueError):
            plan_checkpoints(commands, 1, (1, 2, 2, 2), (1, 1, 2, 3))

    def test_changed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.bin"
            path.write_bytes(bytes([1, 2, 3]))
            record = artifact(path)
            self.assertEqual(checked_read(record), bytes([1, 2, 3]))
            path.write_bytes(bytes([1, 2, 4]))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                checked_read(record)

    def test_lut_restore_is_derived_from_prior_load(self):
        constants = bytes(range(256))
        commands = lut(128, 0, 128) + lut(128, 0, 128, reuse=True)
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        model = runtime._package(commands + end, constants, b"", 2, 1024, 0, 0)
        restore = plan_lut_restore(model, 1, 2)
        self.assertEqual((restore["source_command"], restore["reuse_command"]), (1, 2))
        self.assertEqual(bytes.fromhex(restore["data_hex"]), constants)
        self.assertIsNone(plan_lut_restore(model, 0, 2))
        # A memory snapshot alone cannot satisfy a first-command reuse flag.
        missing = runtime._package(lut(128, 0, 128, reuse=True) + end, constants, b"", 1, 1024, 0, 0)
        with self.assertRaisesRegex(ValueError, "known pre-snapshot load"):
            plan_lut_restore(missing, 0, 1)

    def test_restore_preamble_requires_byte_idempotence(self):
        constants = bytes(reversed(range(256)))
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        model = runtime._package(lut(128, 0, 32) + lut(160, 32, 32, reuse=True) + end,
                                 constants, b"", 2, 1024, 0, 0)
        restore = plan_lut_restore(model, 1, 2)
        memory = bytearray(1024)
        memory[128:160] = bytes(range(32))
        memory[:32] = constants[:32]
        preamble = plan_restore_preamble(model, 1, restore, memory)
        self.assertEqual((preamble["command"], preamble["destination"], preamble["bytes"]), (1, 0, 32))
        memory[0] ^= 1
        with self.assertRaisesRegex(ValueError, "byte-idempotent"):
            plan_restore_preamble(model, 1, restore, memory)
        with self.assertRaisesRegex(ValueError, "immediately precede"):
            plan_restore_preamble(model, 2, restore, memory)
        self.assertIsNone(plan_restore_preamble(model, 1, None, memory))

    def test_restore_preamble_rejects_overlap_and_scratch(self):
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        for src, dst in ((128, 128), (0x7F000, 0), (128, 1)):
            model = runtime._package(lut(src, dst, 32) + lut(160, 32, 32, reuse=True) + end,
                                     bytes(range(256)), b"", 2, 1024, 0, 0)
            restore = plan_lut_restore(model, 1, 2)
            with self.assertRaisesRegex(ValueError, "Unsafe LUT preamble"):
                plan_restore_preamble(model, 1, restore, bytes(524288))


if __name__ == "__main__":
    unittest.main()
