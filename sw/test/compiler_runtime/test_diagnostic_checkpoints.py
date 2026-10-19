"""Pinned extra Conv observations must remain separate from numerical proof."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from diagnostic_checkpoints import load_diagnostic_checkpoints
from yolo_reference_replay import artifact, checked_read, runtime, save_checkpoint_result


class DiagnosticCheckpointTest(unittest.TestCase):
    def fixture(self, folder):
        command = runtime._pointwise_c32(runtime._ref(1), runtime._ref(6), runtime._ref(6, offset=512),
            runtime._ref(6, offset=128), 6, 2, 1, 0, 192, 192, 0)
        oracle = folder / "expected.bin"
        oracle.write_bytes(bytes(24))
        check = dict(command=2, tensor=7, tensor_role="head_conv_diagnostic", address=128, rows=6, channels=4,
                     tile_cols=3, row_stride=96, reference_first_row=0, reference_first_channel=0,
                     reference_shape=[1, 2, 3, 4], expected=artifact(oracle),
                     command_sha256=hashlib.sha256(command).hexdigest())
        case = dict(remaining_graph=True, first_command=1, end_command=3, checkpoints=[],
                    model="model", source="source", input="input", reference_build="build")
        manifest = {key: case[key] for key in ("model", "source", "input", "reference_build")}
        manifest["checkpoints"] = [check]
        return case, [bytes(32), command, bytes(32)], manifest

    def load(self, folder, case, commands, manifest):
        path = folder / "observations.json"
        path.write_text(json.dumps(manifest))
        case["diagnostic_checkpoints"] = artifact(path)
        return load_diagnostic_checkpoints(case, commands, 4096, checked_read)

    def test_optional_and_valid_observation(self):
        self.assertEqual(load_diagnostic_checkpoints({}, [], 0, checked_read), [])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, commands, manifest = self.fixture(folder)
            self.assertEqual(self.load(folder, case, commands, manifest), manifest["checkpoints"])

    def test_provenance_range_duplicate_and_command_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for mutation in ("source", "range", "duplicate", "command", "prefix", "scope"):
                case, commands, manifest = self.fixture(folder)
                check = manifest["checkpoints"][0]
                if mutation == "source": manifest["source"] = "other"
                elif mutation == "range": check["command"] = 4
                elif mutation == "duplicate": manifest["checkpoints"].append(copy.deepcopy(check))
                elif mutation == "command": check["command_sha256"] = "0" * 64
                elif mutation == "prefix": case["checkpoints"] = [dict(command=2)]
                else: case["remaining_graph"] = False
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.load(folder, case, commands, manifest)

    def test_geometry_and_oracle_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for change in (dict(address=160), dict(rows=5), dict(row_stride=128), dict(tile_cols=None),
                           dict(channels=33), dict(reference_first_row=1), dict(group_stride=64),
                           dict(reference_first_channel=1), dict(reference_shape=[1, True, 3, 4])):
                case, commands, manifest = self.fixture(folder)
                manifest["checkpoints"][0].update(change)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.load(folder, case, commands, manifest)
            case, commands, manifest = self.fixture(folder)
            (folder / "expected.bin").write_bytes(bytes(23))
            manifest["checkpoints"][0]["expected"] = artifact(folder / "expected.bin")
            with self.assertRaisesRegex(ValueError, "oracle size"):
                self.load(folder, case, commands, manifest)

    def test_async_and_nonlocal_pointwise_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for offset, value in ((4, 1), (40, 4)):
                case, commands, manifest = self.fixture(folder)
                command = bytearray(commands[1])
                struct.pack_into("<I", command, offset, value)
                commands[1] = bytes(command)
                manifest["checkpoints"][0]["command_sha256"] = hashlib.sha256(command).hexdigest()
                with self.subTest(offset=offset), self.assertRaises(ValueError):
                    self.load(folder, case, commands, manifest)

    def test_linebuffer_observation_requires_final_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for mode in (0, 1, 2, 3):
                case, commands, manifest = self.fixture(folder)
                command = bytearray(160)
                struct.pack_into("<HHIII", command, 0, 9, 160, 0, 0, 0)
                struct.pack_into("<5I", command, 108, 128, 6, mode, 96, 3)
                commands[1] = bytes(command)
                manifest["checkpoints"][0]["command_sha256"] = hashlib.sha256(command).hexdigest()
                if mode in (0, 2):
                    self.assertEqual(len(self.load(folder, case, commands, manifest)), 1)
                else:
                    with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "final requant"):
                        self.load(folder, case, commands, manifest)

    def test_async_observation_waits_for_matching_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for mutation in (None, "submit", "wait_hash", "wrong_tile", "intervening_compute"):
                case, commands, manifest = self.fixture(folder)
                command = bytearray(160)
                struct.pack_into("<HHIII", command, 0, 29, 160, 0, 5, 7)
                struct.pack_into("<5I", command, 108, 128, 6, 2, 96, 3)
                wait = runtime._command_header(30, 32, layer=5, tile=7) + bytes(16)
                commands[1:] = [bytes(command), wait]
                check = manifest["checkpoints"][0]
                check.update(command=3, submit_command=2, command_sha256=hashlib.sha256(command).hexdigest(),
                             completion_sha256=hashlib.sha256(wait).hexdigest())
                if mutation == "submit": check["command"] = 2
                elif mutation == "wait_hash": check["completion_sha256"] = "0" * 64
                elif mutation == "wrong_tile": commands[-1] = runtime._command_header(30, 32, layer=5, tile=8) + bytes(16)
                elif mutation == "intervening_compute":
                    commands.insert(2, commands[1])
                    check["command"] = case["end_command"] = 4
                if mutation is None:
                    self.assertEqual(len(self.load(folder, case, commands, manifest)), 1)
                else:
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        self.load(folder, case, commands, manifest)

    def test_failure_is_recorded_separately_before_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _, _, manifest = self.fixture(folder)
            report = dict(checkpoints=[], diagnostic_checkpoints=[], status="running")
            actual = np.zeros((6, 4), np.int8)
            actual[0, 0] = 1
            stats = save_checkpoint_result(folder, report, manifest["checkpoints"][0], actual, True,
                                           "diagnostic_checkpoints")
            self.assertEqual((stats["mismatches"], report["status"], report["failed_command"]), (1, "FAIL", 2))
            self.assertEqual(report["checkpoints"], [])
            self.assertEqual(len(report["diagnostic_checkpoints"]), 1)
            self.assertEqual(json.loads((folder / "byte_exact.json").read_text())["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
