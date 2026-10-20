"""Snapshot reuse permits only changes strictly after an identical prefix."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import yolo_reference_replay as replay
from snapshot_prefix_compatibility import prove_snapshot_prefix, runtime


class SnapshotPrefixCompatibilityTest(unittest.TestCase):
    def model(self, extra=False, constants=bytes(32), tcdm=1024):
        barrier = runtime._command_header(1, 32) + bytes(16)
        rq = runtime._command_header(5, 32) + bytes(16)
        end = runtime._command_header(0, 32) + bytes(16)
        commands = [barrier, rq, barrier] + ([barrier] if extra else [])
        bindings = b"".join(runtime._binding(d, 0, dimensions=(1, 1, 1, 32)) for d in (1, 2, 3))
        return runtime._package(b"".join(commands) + end, constants, bindings, len(commands), tcdm, 1, 1)

    def test_future_commands_can_change_but_snapshot_identity_is_not_rewritten(self):
        old, new = self.model(), self.model(extra=True)
        proof = prove_snapshot_prefix(old, new, 1)
        self.assertEqual(proof["boundary"], 1)
        self.assertNotEqual(proof["previous_model_sha256"], proof["current_model_sha256"])
        snapshot = runtime._encode_yolo320_snapshot(old, 1, bytes(1024), bytes(32), bytes(32))
        with self.assertRaisesRegex(ValueError, "model CRC"):
            runtime._verify_yolo320_snapshot(new, snapshot)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "old.nai").write_bytes(old)
            (folder / "case.json").write_text(json.dumps({"model": replay.artifact(folder / "old.nai")}))
            case = dict(remaining_graph=True, start_mode="snapshot", resumed_pointwise_checkpoints=True,
                        first_command=1, snapshot_case=replay.artifact(folder / "case.json"),
                        snapshot_prefix_compatibility=proof)
            identity = replay.snapshot_identity_model(case, new)
            self.assertEqual(identity, old)
            self.assertEqual(runtime._verify_yolo320_snapshot(identity, snapshot)[0], 1)
            for change in (dict(start_mode="reset"), dict(remaining_graph=False),
                           dict(resumed_pointwise_checkpoints=False), dict(first_command=2),
                           dict(snapshot_prefix_compatibility={})):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    replay.snapshot_identity_model({**case, **change}, new)

    def test_executed_command_cannot_change(self):
        old, new = self.model(), bytearray(self.model(extra=True))
        struct.pack_into("<I", new, 224 + 8, 9)
        with self.assertRaisesRegex(ValueError, "executed command prefix"):
            prove_snapshot_prefix(old, bytes(new), 1)

    def test_memory_contract_constants_bindings_and_qparams_cannot_change(self):
        old = self.model()
        for new in (self.model(extra=True, constants=bytes([1]) * 32), self.model(extra=True, tcdm=2048)):
            with self.assertRaises(ValueError):
                prove_snapshot_prefix(old, new, 1)
        for section in (2, 3, 4, 5):
            new = bytearray(self.model(extra=True))
            if section in (2, 4):
                offset = struct.unpack_from("<I", new, 64 + (section - 1) * 32 + 8)[0]
                new[offset] ^= 1
            else:
                struct.pack_into("<I", new, 64 + (section - 1) * 32 + 20, 7)
            with self.subTest(section=section), self.assertRaises(ValueError):
                prove_snapshot_prefix(old, bytes(new), 1)

    def test_invalid_boundary_and_section_layout_are_rejected(self):
        old, new = self.model(), self.model(extra=True)
        for boundary in (0, -1, True, 3, 999):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                prove_snapshot_prefix(old, new, boundary)
        for offset, value in ((16, 1), (20, 4), (64, 2), (72, len(new)), (76, len(new))):
            invalid = bytearray(new)
            struct.pack_into("<I", invalid, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                prove_snapshot_prefix(old, bytes(invalid), 1)
        with self.assertRaises(ValueError):
            prove_snapshot_prefix(old, new[:100], 1)

    def test_proof_cannot_cross_an_outstanding_dma(self):
        load = bytearray(runtime._dma_1d(runtime._ref(1), runtime._ref(6), 32, 0, 0))
        struct.pack_into("<H", load, 0, 24)
        end = runtime._command_header(0, 32) + bytes(16)
        commands = bytes(load) + runtime._dma_wait(0, 0) + end
        model = runtime._package(commands, bytes(32), bytes(64), 2, 1024, 1, 1)
        with self.assertRaisesRegex(ValueError, "not restart-safe"):
            prove_snapshot_prefix(model, model, 1)


if __name__ == "__main__":
    unittest.main()
