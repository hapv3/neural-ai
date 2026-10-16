"""Cross-group streamed oracles, exact coverage and bounded workspace tests."""

import copy
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import yolo_reference_replay as replay
from grouped_linebuffer_checkpoints import reference_tensor_slice, validate_grouped_coverage
from resident_pointwise_checkpoints import OUTPUT_ROLES, plan_resident_pointwise_checkpoints
import test_verified_snapshot_helpers as verified
from test_yolo_reference_helpers import lut
from yolo_c2f_checkpoints import checkpoint_indices


class ResidentOutputMappingTest(unittest.TestCase):
    def fixture(self, rows=3, input_channels=128, output_channels=64):
        runtime = replay.runtime
        commands = [runtime._command_header(1, 32) + bytes(16)]
        for group in range(output_channels // 32):
            commands.append(struct.pack("<HHIII4I", 5, 32, 0, 5, 0, 96 + group * 32, 32, group, 0))
            job = bytearray(runtime._pointwise_c32(runtime._ref(1), runtime._ref(6, offset=65536),
                runtime._ref(6, offset=32768), runtime._ref(6, offset=group * rows * 32),
                rows, input_channels // 32, 1, group, rows * 32, rows * 32, 0))
            struct.pack_into("<I", job, 8, 5)
            commands.append(bytes(job))
        for offset in range(0, rows * output_channels, 64):
            length = min(64, rows * output_channels - offset)
            commands.append(lut(offset, 20000, length, reuse=offset != 0))
            store = bytearray(runtime._dma_1d(runtime._ref(6, offset=20000),
                runtime._ref(5, offset=128 + offset), length, 1, 0))
            struct.pack_into("<H", store, 0, 24)
            commands.append(bytes(store))
            if offset == 64:
                prefetch = bytearray(runtime._dma_1d(runtime._ref(1), runtime._ref(6), 96, 0, 0))
                struct.pack_into("<H", prefetch, 0, 24)
                commands.append(bytes(prefetch))
            commands.append(runtime._dma_wait(1, 0))
        commands.append(runtime._dma_wait(0, 0))
        parent = dict(command=1, tensor_role="resumed_c2f_concat", address=65536, rows=rows, channels=input_channels,
                      tile_cols=rows, row_stride=rows * 32, group_stride=rows * 32,
                      reference_first_row=0, reference_first_channel=0)
        shapes = {role: [1, 1, rows, output_channels] for role in OUTPUT_ROLES}
        shapes["parent"] = [1, 1, rows, input_channels]
        return commands, parent, shapes

    def plan(self, fixture):
        return plan_resident_pointwise_checkpoints(*fixture, 131072, 32768)

    def mutate(self, fixture, kind, offset, value):
        index = next(i for i, command in enumerate(fixture[0]) if struct.unpack_from("<H", command)[0] == kind)
        command = bytearray(fixture[0][index])
        struct.pack_into("<I", command, offset, value)
        fixture[0][index] = bytes(command)

    def test_four_input_groups_two_outputs_crossing_lut_and_complete_l2(self):
        checks, end, retained = self.plan(self.fixture())
        self.assertEqual(len(checks), 5)
        self.assertEqual(end, len(self.fixture()[0]) - 1)
        self.assertEqual((retained["address"], retained["rows"], retained["channels"]), (128, 3, 64))
        crossing, = [c for c in checks if "reference_spans" in c]
        self.assertEqual(crossing["reference_spans"], [
            dict(rows=1, channels=32, reference_first_row=2, reference_first_channel=0),
            dict(rows=1, channels=32, reference_first_row=0, reference_first_channel=32)])
        coverage = validate_grouped_coverage(checks, self.fixture()[2], OUTPUT_ROLES)
        self.assertEqual([c["bytes"] for c in coverage.values()], [192, 192])

    def test_workspace_uses_256_row_stripe_not_total_m(self):
        checks, _, _ = self.plan(self.fixture(257))
        self.assertEqual(checks[0]["rows"], 257)
        fixture = self.fixture(257)
        self.mutate(fixture, 10, 36, 32800)
        with self.assertRaisesRegex(ValueError, "workspace overlaps parent"):
            self.plan(fixture)

    def test_eight_input_groups_four_outputs_and_complete_native_l2(self):
        fixture = self.fixture(input_channels=256, output_channels=128)
        checks, end, retained = self.plan(fixture)
        self.assertEqual(end, len(fixture[0]) - 1)
        self.assertEqual(len(checks), 10)
        self.assertEqual([c["reference_first_channel"] for c in checks[:4]], [0, 32, 64, 96])
        self.assertEqual((retained["rows"], retained["channels"], retained["group_stride"]), (3, 128, 96))
        coverage = validate_grouped_coverage(checks, fixture[2], OUTPUT_ROLES)
        self.assertEqual([c["bytes"] for c in coverage.values()], [384, 384])
        tensor = np.arange(384, dtype=np.int16).astype(np.int8).reshape(1, 1, 3, 128)
        actual = np.concatenate([reference_tensor_slice(tensor, c).ravel() for c in checks[4:]])
        matrix = tensor.reshape(3, 128)
        expected = np.concatenate([matrix[:, group * 32:(group + 1) * 32].ravel() for group in range(4)])
        self.assertTrue(np.array_equal(actual, expected))
        self.assertFalse(np.array_equal(actual, tensor.ravel()))

    def test_eighth_parent_group_corruption_and_missing_fourth_output_are_rejected(self):
        fixture = self.fixture(input_channels=256, output_channels=128)
        fixture[0].insert(1, replay.runtime._dma_1d(replay.runtime._ref(1),
            replay.runtime._ref(6, offset=65536 + 7 * 96), 32, 0, 0))
        with self.assertRaisesRegex(ValueError, "Stale resident pointwise parent"):
            self.plan(fixture)
        fixture = self.fixture(input_channels=256, output_channels=128)
        index = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 10 and
                     struct.unpack_from("<I", c, 60)[0] == 3)
        del fixture[0][index]
        with self.assertRaisesRegex(ValueError, "completed groups"):
            self.plan(fixture)

    def test_rejects_missing_rq_wrong_group_and_partial_output_alias(self):
        for kind, offset, value, message in ((5, 24, 1, "RQ selector"),
                                           (10, 64, 128, "matching RQ"),
                                           (10, 36, 64, "partial overlaps output")):
            fixture = self.fixture()
            self.mutate(fixture, kind, offset, value)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, message):
                self.plan(fixture)
        fixture = self.fixture()
        del fixture[0][1]
        with self.assertRaisesRegex(ValueError, "fresh matching RQ"):
            self.plan(fixture)

    def test_rejects_reused_first_lut_and_duplicate_cells(self):
        fixture = self.fixture()
        self.mutate(fixture, 12, 4, 4)
        with self.assertRaisesRegex(ValueError, "explicit first configuration"):
            self.plan(fixture)
        fixture = self.fixture()
        index = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 12)
        fixture[0].insert(index + 1, fixture[0][index])
        with self.assertRaisesRegex(ValueError, "Duplicate.*activation cells"):
            self.plan(fixture)

    def test_rejects_early_prefetch_clobber_and_missing_store_completion(self):
        fixture = self.fixture()
        index = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 24 and
                     struct.unpack_from("<H", c, 16)[0] == 1)
        command = bytearray(fixture[0][index])
        struct.pack_into("<I", command, 32, 160)
        fixture[0][index] = bytes(command)
        fixture[0].insert(index + 1, replay.runtime._dma_wait(0, 0))
        with self.assertRaisesRegex(ValueError, "Stale.*LUT source"):
            self.plan(fixture)
        fixture = self.fixture()
        del fixture[0][next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 27)]
        with self.assertRaisesRegex(ValueError, "before DMA completion"):
            self.plan(fixture)
        fixture = self.fixture()
        del fixture[0][-2]
        with self.assertRaisesRegex(ValueError, "Incomplete.*L2 output"):
            self.plan(fixture)

    def test_cross_group_oracle_is_not_nhwc_flattened_or_channel_swapped(self):
        checks, _, _ = self.plan(self.fixture())
        check, = [c for c in checks if "reference_spans" in c]
        tensor = np.arange(3 * 64, dtype=np.uint8).view(np.int8).reshape(1, 1, 3, 64)
        expected = np.concatenate([tensor.reshape(3, 64)[2:3, :32], tensor.reshape(3, 64)[:1, 32:]])
        self.assertTrue(np.array_equal(reference_tensor_slice(tensor, check), expected))
        self.assertFalse(np.array_equal(expected, tensor.reshape(-1, 32)[2:4]))

    def test_malformed_missing_duplicate_spans_fail_closed(self):
        checks, _, _ = self.plan(self.fixture())
        check, = [c for c in checks if "reference_spans" in c]
        tensor = np.zeros((1, 1, 3, 64), np.int8)
        for spans in ([], check["reference_spans"][:1],
                      [dict(check["reference_spans"][0], rows=True), check["reference_spans"][1]]):
            with self.subTest(spans=spans), self.assertRaisesRegex(ValueError, "Invalid reference spans"):
                reference_tensor_slice(tensor, {**check, "reference_spans": spans})
        broken = copy.deepcopy(checks)
        span = next(c for c in broken if "reference_spans" in c)["reference_spans"][1]
        span["reference_first_row"] = 2
        span["reference_first_channel"] = 0
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_grouped_coverage(broken, self.fixture()[2], OUTPUT_ROLES)


class ConcatSnapshotParentTest(unittest.TestCase):
    def test_complete_c256_snapshot_parent_and_eighth_group_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = verified.VerifiedSnapshotParentTest().fixture(Path(directory))
            previous, report, snapshot, identity, build, _ = fixture
            parent = previous["checkpoints"][0]
            parent.update(tensor_role="resumed_c2f_concat", channels=256)
            tensor = np.arange(512, dtype=np.int16).astype(np.int8).reshape(2, 256)
            path = Path(directory) / "concat-c256.bin"
            path.write_bytes(tensor.tobytes())
            parent["expected"] = replay.artifact(path)
            report["checkpoints"] = [{**parent, "mismatches": 0}]
            memory = np.zeros(512, np.uint8)
            indices = checkpoint_indices(parent, memory.size)
            memory[indices] = tensor.view(np.uint8)
            _, stats = replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, build,
                memory.tobytes(), "resumed_c2f_concat")
            self.assertEqual(stats["bytes"], 512)
            memory[indices[1, 255]] ^= 1
            with self.assertRaisesRegex(ValueError, "differs from its pinned reference"):
                replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, build,
                    memory.tobytes(), "resumed_c2f_concat")

    def test_explicit_concat_parent_role_recompares_all_four_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = verified.VerifiedSnapshotParentTest().fixture(Path(directory))
            previous, report, snapshot, identity, build, _ = fixture
            parent = previous["checkpoints"][0]
            parent.update(tensor_role="resumed_c2f_concat", channels=128)
            expected = np.arange(256, dtype=np.uint8).tobytes()
            path = Path(directory) / "concat.bin"
            path.write_bytes(expected)
            parent["expected"] = replay.artifact(path)
            report["checkpoints"] = [{**parent, "mismatches": 0}]
            memory = np.zeros(512, np.uint8)
            memory[checkpoint_indices(parent, 512)] = np.frombuffer(expected, np.uint8).reshape(2, 128)
            _, stats = replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, build,
                memory.tobytes(), "resumed_c2f_concat")
            self.assertEqual(stats["bytes"], 256)
            with self.assertRaisesRegex(ValueError, "no complete retained"):
                replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, build, memory.tobytes())
            memory[192] ^= 1
            with self.assertRaisesRegex(ValueError, "differs from its pinned reference"):
                replay.validate_snapshot_parent(previous, report, snapshot, 3, identity, build,
                    memory.tobytes(), "resumed_c2f_concat")


if __name__ == "__main__":
    unittest.main()
