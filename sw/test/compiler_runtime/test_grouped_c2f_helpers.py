"""Multi-group residual chains, staged pixels, pSUM and native concat gates."""

import struct
import unittest

import numpy as np

import yolo_reference_replay as replay
from grouped_c2f_checkpoints import native_values
from resumed_c2f_checkpoints import CONCAT_ROLE, PARENT_ROLE, chain_roles, plan_resumed_c2f_checkpoints
from test_yolo_reference_helpers import lut


class GroupedC2fTest(unittest.TestCase):
    def fixture(self, queued_concat=False, channels=64):
        runtime = replay.runtime
        groups = channels // 32
        tensor_bytes = 4 * channels
        commands = [lut(0, 0, tensor_bytes * 2)]
        previous = tensor_bytes
        residuals = []
        for block in range(2):
            block_base = block * 5120
            for conv in range(2):
                layer = 10 + block * 4 + conv * 2
                source = previous if conv == 0 else block_base + 2048
                output = block_base + 1024 + conv * 2048
                activation = output + 1024
                for group in range(groups):
                    commands.append(struct.pack("<HHIII4I", 5, 32, 0, layer, 0, layer * 32 + group * 32, 32, group, 0))
                    for first in (0, 2):
                        origin = first // 2 - 1
                        start, stop = max(0, origin), min(2, origin + 3)
                        for input_group in range(groups):
                            reference = runtime._ref(6, offset=source + input_group * 128 + start * 64)
                            commands.append(runtime._dma_2d(reference,
                                runtime._ref(6, offset=40960 + (start - origin) * 128 + 32),
                                64, 64, 128, stop - start, 2, 0))
                            command = bytearray(160)
                            struct.pack_into("<HHIII", command, 0, 9, 160, 0, layer, 0)
                            struct.pack_into("<I8H4I", command, 16, 40960, 3, 4, 32, 2, 1, 1, 0, 0,
                                             128, 32, 32, 128)
                            struct.pack_into("<14H4I", command, 52, 3, 3, 0, 0, 1, 1, 0, 1, 0, 1,
                                             32, 0, 0, 0, 9, 2, 0, 288)
                            mode = 1 if input_group == 0 else 2 if input_group == groups - 1 else 3
                            struct.pack_into("<9I", command, 96, 16384 + input_group * 9216, 0, 32768,
                                             output + group * 128 + first * 32, 2, mode,
                                             64, 2, 256)
                            commands.append(bytes(command))
                commands.append(lut(output, activation, tensor_bytes))
            destination = block_base + 5120
            binary = runtime._afu_binary_quant if block == 0 else runtime._afu_binary
            commands.append(binary(runtime._ref(6, offset=previous), runtime._ref(6, offset=block_base + 4096),
                                   runtime._ref(6, offset=destination), tensor_bytes, 0 if block == 0 else 2, 0))
            previous = destination
            residuals.append(destination)

        def copy_tensor(source, destination, src_region=6, dst_region=6, queued=False):
            direction = 1 if dst_region == 5 else 0 if src_region == 5 else 2
            command = bytearray(runtime._dma_1d(runtime._ref(src_region, offset=source),
                runtime._ref(dst_region, offset=destination), tensor_bytes, direction, 0))
            if queued:
                struct.pack_into("<H", command, 0, 24)
            return bytes(command)

        if queued_concat:
            commands += [copy_tensor(0, 0, dst_region=5, queued=True), runtime._dma_wait(1, 0),
                         copy_tensor(0, 12288, src_region=5, queued=True), runtime._dma_wait(0, 0)]
        else:
            commands.append(copy_tensor(0, 12288))
        for slot, address in enumerate([tensor_bytes] + residuals, 1):
            commands.append(copy_tensor(address, 12288 + slot * tensor_bytes))
        commands.append(runtime._pointwise_c32(runtime._ref(1), runtime._ref(6, offset=12288),
            runtime._ref(6, offset=34816), runtime._ref(6, offset=14336), 4, groups * 4, 1, 0, 128, 128, 0))
        shapes = {role: [1, 2, 2, channels] for role in chain_roles(2)}
        shapes.update({PARENT_ROLE: [1, 2, 2, channels * 2], CONCAT_ROLE: [1, 2, 2, channels * 4]})
        parent = dict(command=1, address=0, rows=4, channels=channels * 2,
                      tile_cols=2, row_stride=64, group_stride=128)
        layout = dict(blocks=2, branch_start=channels, bypass_start=0,
                      concat_order=["bypass", "branch", "residual_0", "residual_1"])
        return commands, parent, shapes, layout

    def plan(self, fixture):
        return plan_resumed_c2f_checkpoints(*fixture, 65536, 4096)

    def first_kind(self, fixture, kind):
        return next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == kind)

    def mutate(self, fixture, kind, offset, value):
        index = self.first_kind(fixture, kind)
        command = bytearray(fixture[0][index])
        struct.pack_into("<I", command, offset, value)
        fixture[0][index] = bytes(command)

    def test_complete_two_bottlenecks_and_eight_group_concat(self):
        fixture = self.fixture()
        checks, end = self.plan(fixture)
        self.assertEqual(end, len(fixture[0]) - 1)
        self.assertEqual(len(checks), 23)
        self.assertEqual(sum(c["rows"] * c["channels"] for c in checks), 3584)
        self.assertEqual(checks[-1]["channels"], 256)
        self.assertEqual([c["reference_first_channel"] for c in checks[:4]], [0, 0, 32, 32])

    def test_four_input_groups_require_middle_accumulation(self):
        fixture = self.fixture(channels=128)
        checks, _ = self.plan(fixture)
        self.assertEqual(len(checks), 39)
        self.assertEqual(checks[-1]["channels"], 512)
        index = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 9 and
                     struct.unpack_from("<I", c, 116)[0] == 3)
        command = bytearray(fixture[0][index])
        struct.pack_into("<I", command, 116, 2)
        fixture[0][index] = bytes(command)
        with self.assertRaisesRegex(ValueError, "partial accumulation"):
            self.plan(fixture)

    def test_requires_init_final_order_and_current_rq(self):
        for kind, offset, value, message in ((9, 116, 2, "partial accumulation"),
                                           (5, 20, 16, "current RQ"), (9, 120, 96, "geometry")):
            fixture = self.fixture()
            self.mutate(fixture, kind, offset, value)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, message):
                self.plan(fixture)
        fixture = self.fixture()
        index = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 9 and
                     struct.unpack_from("<I", c, 116)[0] == 2)
        del fixture[0][index]
        with self.assertRaisesRegex(ValueError, "partial accumulation|Noncontiguous"):
            self.plan(fixture)

    def test_staged_channel_and_partial_corruption_are_rejected(self):
        fixture = self.fixture()
        index = self.first_kind(fixture, 3)
        command = bytearray(fixture[0][index])
        source = struct.unpack_from("<I", command, 20)[0]
        struct.pack_into("<I", command, 20, source + 128)
        fixture[0][index] = bytes(command)
        with self.assertRaisesRegex(ValueError, "staged input pixels/channels"):
            self.plan(fixture)
        fixture = self.fixture()
        fixture[0].insert(self.first_kind(fixture, 9) + 1, replay.runtime._dma_1d(
            replay.runtime._ref(1), replay.runtime._ref(6, offset=32768), 32, 0, 0))
        with self.assertRaisesRegex(ValueError, "partial sums were overwritten"):
            self.plan(fixture)

    def test_missing_last_group_cannot_reach_lut(self):
        fixture = self.fixture()
        index = self.first_kind(fixture, 12)  # Parent LUT precedes the mapped suffix.
        index = next(i for i in range(index + 1, len(fixture[0])) if struct.unpack_from("<H", fixture[0][i])[0] == 12)
        del fixture[0][index - 1]
        with self.assertRaisesRegex(ValueError, "complete native tensor"):
            self.plan(fixture)

    def test_residual_native_channel_order_and_modes_are_strict(self):
        for kind, offset, value in ((34, 92, 2), (13, 44, 1)):
            fixture = self.fixture()
            self.mutate(fixture, kind, offset, value)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Add mode/layout"):
                self.plan(fixture)
        fixture = self.fixture()
        index = self.first_kind(fixture, 34)
        fixture[0].insert(index, replay.runtime._dma_1d(replay.runtime._ref(6, offset=4224),
            replay.runtime._ref(6, offset=4096), 128, 2, 0))
        with self.assertRaisesRegex(ValueError, "no completed activation"):
            self.plan(fixture)

    def test_concat_requires_dma_waits_and_source_channel_order(self):
        self.plan(self.fixture(queued_concat=True))
        for direction in (0, 1):
            fixture = self.fixture(queued_concat=True)
            index = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 27 and
                         struct.unpack_from("<I", c, 16)[0] == direction)
            del fixture[0][index]
            with self.subTest(direction=direction), self.assertRaisesRegex(ValueError, "before DMA completion"):
                self.plan(fixture)
        fixture = self.fixture()
        fixture[3]["concat_order"][:2] = ["branch", "bypass"]
        with self.assertRaisesRegex(ValueError, "before complete grouped C2f concat"):
            self.plan(fixture)

    def test_native_labels_are_group_major_not_nhwc_flattened(self):
        matrix = np.arange(4 * 64).reshape(4, 64)
        expected = np.concatenate([matrix[:, :32].ravel(), matrix[:, 32:].ravel()])
        self.assertTrue(np.array_equal(native_values(matrix), expected))
        self.assertFalse(np.array_equal(native_values(matrix), matrix.ravel()))

    def test_rejects_async_compute_and_workspace_alias(self):
        fixture = self.fixture()
        index = self.first_kind(fixture, 9)
        command = bytearray(fixture[0][index])
        struct.pack_into("<H", command, 0, 29)
        fixture[0][index] = bytes(command)
        with self.assertRaisesRegex(ValueError, "Unsupported command 29"):
            self.plan(fixture)
        fixture = self.fixture()
        self.mutate(fixture, 9, 104, 40960)
        with self.assertRaisesRegex(ValueError, "workspace overlaps"):
            self.plan(fixture)


if __name__ == "__main__":
    unittest.main()
