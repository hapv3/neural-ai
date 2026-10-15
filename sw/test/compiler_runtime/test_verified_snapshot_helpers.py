"""Verified parent provenance and self-configuring grouped suffix regressions."""

import copy
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import yolo_reference_replay as replay
from grouped_linebuffer_checkpoints import plan_grouped_pointwise_checkpoints
from test_yolo_reference_helpers import lut
from yolo_c2f_checkpoints import checkpoint_indices


class GroupedPointwiseResumeTest(unittest.TestCase):
    def commands(self):
        runtime = replay.runtime

        def rq(first, group):
            return struct.pack("<HHIII4I", 5, 32, 0, 5, 0, first, 32, group, 0)

        def job(address, group):
            command = bytearray(runtime._pointwise_c32(runtime._ref(1), runtime._ref(6),
                runtime._ref(6, offset=512), runtime._ref(6, offset=address), 2, 2, 1, group, 64, 64, 0))
            struct.pack_into("<I", command, 8, 5)
            return bytes(command)

        return [runtime._command_header(1, 32) + bytes(16), rq(96, 0), job(128, 0),
                rq(128, 1), job(192, 1), lut(128, 0, 128)]

    def plan(self, commands):
        parent = dict(command=1, address=0, rows=2, channels=64, tile_cols=2, row_stride=64, group_stride=64)
        shapes = {r: [1, 1, 2, 64] for r in
                  ("downsample_activation", "resumed_pointwise_conv", "resumed_pointwise_activation")}
        return plan_grouped_pointwise_checkpoints(commands, parent, shapes, 1024, 128)

    def test_both_input_output_groups_and_full_lut(self):
        checks, end = self.plan(self.commands())
        self.assertEqual(end, 6)
        self.assertEqual([c["command"] for c in checks], [3, 5, 6])
        self.assertEqual([c["reference_first_channel"] for c in checks], [0, 32, 0])

    def test_requires_fresh_rq_and_lut_configuration(self):
        commands = self.commands()
        del commands[1]
        with self.assertRaisesRegex(ValueError, "explicit matching RQ"):
            self.plan(commands)
        commands = self.commands()
        command = bytearray(commands[-1])
        struct.pack_into("<I", command, 4, 4)
        commands[-1] = bytes(command)
        with self.assertRaisesRegex(ValueError, "explicitly load"):
            self.plan(commands)

    def test_rejects_wrong_selector_stride_partial_alias_and_missing_group(self):
        for command_index, offset, value, message in ((3, 24, 0, "Stale.*RQ selector"),
                                                     (2, 64, 96, "matching RQ loads"),
                                                     (2, 36, 192, "partial overlaps output")):
            commands = self.commands()
            command = bytearray(commands[command_index])
            struct.pack_into("<I", command, offset, value)
            commands[command_index] = bytes(command)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, message):
                self.plan(commands)
        commands = self.commands()
        del commands[4]
        with self.assertRaisesRegex(ValueError, "consume all groups"):
            self.plan(commands)

    def test_rejects_overwritten_parent_between_output_groups(self):
        commands = self.commands()
        commands.insert(3, replay.runtime._dma_1d(replay.runtime._ref(1), replay.runtime._ref(6), 32, 0, 0))
        with self.assertRaisesRegex(ValueError, "Stale resumed.*input"):
            self.plan(commands)

    def test_explicit_future_lut_load_does_not_depend_on_historical_afu_state(self):
        runtime = replay.runtime
        opaque = runtime._command_header(13, 64) + bytes(48)
        end = runtime._command_header(0, 32) + bytes(16)
        for reuse in (False, True):
            model = runtime._package(opaque + lut(0, 128, 64, reuse) + end, bytes(256), b"", 2, 1024, 0, 0)
            if reuse:
                with self.assertRaisesRegex(ValueError, "non-LUT AFU modes"):
                    replay.plan_lut_restore(model, 1, 2)
            else:
                self.assertIsNone(replay.plan_lut_restore(model, 1, 2))


class VerifiedSnapshotParentTest(unittest.TestCase):
    def fixture(self, folder):
        expected = np.arange(128, dtype=np.uint8).tobytes()
        path = folder / "expected.bin"
        path.write_bytes(expected)
        parent = dict(command=3, tensor_role="downsample_activation", address=0, rows=2, channels=64,
                      tile_cols=2, row_stride=64, group_stride=64, reference_first_row=0,
                      reference_first_channel=0, tensor=5, expected=replay.artifact(path))
        memory = np.zeros(512, np.uint8)
        memory[checkpoint_indices(parent, len(memory))] = np.frombuffer(expected, np.uint8).reshape(2, 64)
        identity = {k: {"path": str(folder / k), "sha256": k + "-identity"}
                    for k in ("model", "source", "firmware", "input")}
        build = {"library_sha256": "reference-build"}
        previous = dict(end_command=3, checkpoints=[parent], reference_build=build, **identity)
        snapshot = folder / "command-3.snapshot"
        report = dict(status="PASS", end_command=3, snapshot=str(snapshot),
                      checkpoints=[{**parent, "mismatches": 0}], **identity)
        return previous, report, snapshot, identity, build, memory.tobytes()

    def test_full_retained_parent_is_independently_recompared(self):
        with tempfile.TemporaryDirectory() as directory:
            previous, report, snapshot, identity, build, memory = self.fixture(Path(directory))
            parent, stats = replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, build, memory)
            self.assertEqual(parent["channels"], 64)
            self.assertEqual((stats["bytes"], stats["mismatches"]), (128, 0))

    def test_rejects_running_failed_wrong_boundary_or_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            previous, report, snapshot, identity, build, memory = self.fixture(Path(directory))
            for changes in (dict(status="running"), dict(status="FAIL"), dict(end_command=4),
                            dict(snapshot=str(snapshot.parent / "other.snapshot")),
                            dict(firmware={"path": "other", "sha256": "other"})):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    replay.validate_snapshot_parent(previous, {**report, **changes}, snapshot, 3, identity, build, memory)
            with self.assertRaisesRegex(ValueError, "reference build changed"):
                replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, {}, memory)

    def test_rejects_corrupted_data_and_partial_or_unverified_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            previous, report, snapshot, identity, build, memory = self.fixture(Path(directory))
            corrupt = bytearray(memory)
            corrupt[0] ^= 1
            with self.assertRaisesRegex(ValueError, "differs from its pinned reference"):
                replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, build, bytes(corrupt))
            for field, value in (("mismatches", 1), ("reference_first_channel", 32)):
                changed = copy.deepcopy(report); changed["checkpoints"][0][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "partial or lacks"):
                    replay.validate_snapshot_parent(previous, changed, snapshot, 3, identity, build, memory)


if __name__ == "__main__":
    unittest.main()
