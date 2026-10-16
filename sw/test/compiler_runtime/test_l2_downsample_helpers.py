"""Staged pixels, grouped accumulation and complete downsample coverage tests."""

import copy
import struct
import unittest

import numpy as np

import yolo_reference_replay as replay
from l2_downsample_checkpoints import (
    L2_DOWNSAMPLE_ROLES, plan_l2_downsample_checkpoints, resolve_l2_downsample_source,
)
from test_yolo_reference_helpers import lut


class L2DownsampleMappingTest(unittest.TestCase):
    def fixture(self, inputs=2, queued=False):
        runtime = replay.runtime

        def rq(group):
            return struct.pack("<HHIII4I", 5, 32, 0, 35, 0, 512 + group * 32, 32, group, 0)

        commands = [rq(0)]
        for group in range(4):
            if group:
                commands.append(rq(group))
            for first in (0, 2):
                origin = first - 1
                for input_group in range(inputs):
                    start, stop = max(0, origin), min(4, origin + 3)
                    load = bytearray(runtime._dma_2d(runtime._ref(5, offset=input_group * 512 + start * 128),
                        runtime._ref(6, offset=32768 + (start - origin) * 160 + 32),
                        128, 128, 160, stop - start, 0, 0))
                    if queued:
                        struct.pack_into("<H", load, 0, 25)
                    commands.append(bytes(load))
                    if queued:
                        commands.append(runtime._dma_wait(0, 0))
                    command = bytearray(160)
                    struct.pack_into("<HHIII", command, 0, 9, 160, 0, 35, 0)
                    struct.pack_into("<I8H4I2H", command, 16, 32768, 3, 5, 32, 2, 2, 2, 0, 0,
                                     160, 32, 64, 320, 3, 3)
                    struct.pack_into("<14H4I", command, 52, 3, 3, 0, 0, 1, 1, 0, 1, 0, 1,
                                     32, 0, 0, 0, 9, 2, 0, 288)
                    mode = 1 if input_group == 0 else 2 if input_group == inputs - 1 else 3
                    struct.pack_into("<9I", command, 96, input_group * 9216, 0, 18432,
                                     20000 + group * 128 + first * 32, 2, mode, 64, 2, 256)
                    commands.append(bytes(command))
        commands.append(lut(20000, 0, 512))
        parent = dict(address=0, rows=16, channels=inputs * 32, tile_cols=4, row_stride=128,
                      group_stride=512, reference_first_row=0, reference_first_channel=0)
        shapes = {role: [1, 2, 2, 128] for role in L2_DOWNSAMPLE_ROLES}
        shapes["parent"] = [1, 4, 4, inputs * 32]
        return commands, parent, 1, 0, shapes, [1, 1, 1, 1], 65536, 4096

    def plan(self, fixture):
        return plan_l2_downsample_checkpoints(*fixture)

    def first_conv(self, fixture):
        return next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 9)

    def mutate(self, fixture, offset, value):
        index = self.first_conv(fixture)
        command = bytearray(fixture[0][index])
        struct.pack_into("<I", command, offset, value)
        fixture[0][index] = bytes(command)

    def test_two_and_three_inputs_finalize_all_four_outputs_and_silu(self):
        for inputs in (2, 3):
            fixture = self.fixture(inputs)
            checks, end = self.plan(fixture)
            self.assertEqual(end, len(fixture[0]))
            self.assertEqual(len(checks), 9)
            self.assertEqual([c["reference_first_channel"] for c in checks[:-1]], [0, 0, 32, 32, 64, 64, 96, 96])
            self.assertEqual([c["reference_first_row"] for c in checks[:-1]], [0, 2] * 4)
            self.assertEqual(sum(c["rows"] * c["channels"] for c in checks), 1024)

    def test_rejects_missing_init_final_and_wrong_weight_input_group(self):
        for offset, value in ((116, 2), (96, 9216)):
            fixture = self.fixture()
            self.mutate(fixture, offset, value)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "partial accumulation group"):
                self.plan(fixture)
        fixture = self.fixture()
        first_final = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 9 and
                           struct.unpack_from("<I", c, 116)[0] == 2)
        del fixture[0][first_final]
        with self.assertRaisesRegex(ValueError, "partial accumulation group|Noncontiguous"):
            self.plan(fixture)

    def test_rq_must_be_replayed_not_inferred_from_older_prefix(self):
        fixture = self.fixture()
        fixture[0][0] = replay.runtime._command_header(1, 32) + bytes(16)
        with self.assertRaisesRegex(ValueError, "missing replayed RQ"):
            self.plan(fixture)
        fixture = list(self.fixture())
        fixture[3] = 1
        with self.assertRaisesRegex(ValueError, "complete drained parent"):
            self.plan(fixture)

    def test_rejects_stale_pixel_channel_and_wrong_padding_origin(self):
        fixture = self.fixture()
        command = bytearray(fixture[0][1])
        struct.pack_into("<I", command, 20, 512)
        fixture[0][1] = bytes(command)
        with self.assertRaisesRegex(ValueError, "staged input pixels/channels"):
            self.plan(fixture)
        fixture = list(self.fixture())
        fixture[5] = [0, 1, 1, 1]
        with self.assertRaisesRegex(ValueError, "staged input pixels/channels"):
            self.plan(fixture)

    def test_requires_staged_dma_completion(self):
        fixture = self.fixture(queued=True)
        self.plan(fixture)
        del fixture[0][2]
        with self.assertRaisesRegex(ValueError, "before DMA completion"):
            self.plan(fixture)

    def test_rejects_partial_overwrite_and_workspace_alias(self):
        fixture = self.fixture()
        index = self.first_conv(fixture)
        fixture[0].insert(index + 1, replay.runtime._dma_1d(
            replay.runtime._ref(1), replay.runtime._ref(6, offset=18432), 32, 0, 0))
        with self.assertRaisesRegex(ValueError, "partial sums were overwritten"):
            self.plan(fixture)
        for address in (20000, 32768):
            fixture = self.fixture()
            self.mutate(fixture, 104, address)
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, "workspace overlaps"):
                self.plan(fixture)

    def test_rejects_incomplete_lut_group_reuse_and_parent_l2_clobber(self):
        fixture = self.fixture()
        command = bytearray(fixture[0][-1])
        struct.pack_into("<I", command, 4, 4)
        fixture[0][-1] = bytes(command)
        with self.assertRaisesRegex(ValueError, "every finalized group and fresh"):
            self.plan(fixture)
        fixture = self.fixture()
        fixture[0].insert(-1, replay.runtime._dma_1d(replay.runtime._ref(1), replay.runtime._ref(5), 32, 1, 0))
        with self.assertRaisesRegex(ValueError, "overwrote its retained L2 parent"):
            self.plan(fixture)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.plan((self.fixture()[0][:-1],) + self.fixture()[1:])

    def test_rejects_unsupported_geometry_and_async_compute(self):
        for offset, value in ((120, 96), (128, 128), (112, 3), (80, 8), (84, 3), (56, 32)):
            fixture = self.fixture()
            self.mutate(fixture, offset, value)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "geometry"):
                self.plan(fixture)
        fixture = self.fixture()
        index = self.first_conv(fixture)
        command = bytearray(fixture[0][index])
        struct.pack_into("<H", command, 0, 29)
        fixture[0][index] = bytes(command)
        with self.assertRaisesRegex(ValueError, "Unsupported command 29"):
            self.plan(fixture)


class DownsampleSourceTest(unittest.TestCase):
    def fixture(self):
        class Interpreter:
            ops = [dict(op_name="PAD", inputs=[1, 2], outputs=[3]),
                   dict(op_name="CONV_2D", inputs=[3, 4, 5], outputs=[6])]
            tensors = {2: np.array([[0, 0], [1, 1], [1, 1], [0, 0]], np.int32),
                       4: np.zeros((128, 3, 3, 64), np.int8)}

            def _get_ops_details(self):
                return self.ops

            def get_tensor(self, tensor):
                return self.tensors[tensor]

        return Interpreter()

    def test_resolves_explicit_padding_and_conv_without_tensor_ids(self):
        self.assertEqual(resolve_l2_downsample_source(self.fixture(), 1), (6, [1, 1, 1, 1]))

    def test_rejects_non_k3_batch_padding_and_ambiguous_edges(self):
        for mutation in ("kernel", "padding", "ambiguous", "missing"):
            interpreter = self.fixture()
            if mutation == "kernel":
                interpreter.tensors[4] = np.zeros((128, 1, 1, 64), np.int8)
            elif mutation == "padding":
                interpreter.tensors[2][0, 0] = 1
            elif mutation == "ambiguous":
                interpreter.ops.append(copy.deepcopy(interpreter.ops[0]))
            else:
                interpreter.ops.pop(0)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resolve_l2_downsample_source(interpreter, 1)


if __name__ == "__main__":
    unittest.main()
