"""Offline verification of channel-group oracles and completion-aware mapping."""

import copy
import struct
import unittest

import numpy as np

import yolo_reference_replay as replay
from grouped_linebuffer_checkpoints import (
    DOWNSAMPLE_ROLES, plan_downsample_checkpoints, reference_tensor_slice, validate_grouped_coverage,
)
from test_yolo_reference_helpers import linebuffer, lut


class GroupedLinebufferTest(unittest.TestCase):
    def commands(self):
        def rq(first, group):
            return struct.pack("<HHIII4I", 5, 32, 0, 10, 0, first, 32, group, 0)

        def conv(address, async_job=False):
            command = bytearray(linebuffer(10, address, 4, 2, async_job=async_job))
            struct.pack_into("<2H", command, 28, 2, 2)
            return bytes(command)

        return [rq(96, 0), replay.runtime._command_header(1, 32) + bytes(16), conv(1024),
                rq(128, 1), conv(1152, True), replay.runtime._command_header(30, 32) + bytes(16),
                lut(1024, 0, 256)]

    def plan(self, commands):
        parent = [dict(command=2, tensor_role="block_output_activation", address=0, rows=16,
                       channels=32, tile_cols=4, row_stride=128, reference_first_row=0)]
        shapes = dict(block_output_activation=[1, 4, 4, 32], downsample_conv=[1, 2, 2, 64],
                      downsample_activation=[1, 2, 2, 64])
        return plan_downsample_checkpoints(commands, parent, shapes, 2048, 1024)

    def test_tracks_rq_selectors_and_wait_before_full_tensor_lut(self):
        checks, end = self.plan(self.commands())
        self.assertEqual(end, 7)
        self.assertEqual([c["command"] for c in checks], [3, 6, 7])
        self.assertEqual([c["reference_first_channel"] for c in checks], [0, 32, 0])
        self.assertEqual(checks[1]["submit_command"], 5)
        self.assertEqual(checks[-1]["channels"], 64)
        shapes = {role: [1, 2, 2, 64] for role in DOWNSAMPLE_ROLES}
        self.assertEqual(validate_grouped_coverage(checks, shapes)["downsample_conv"]["bytes"], 256)

    def test_rejects_incomplete_async_output_and_missing_group(self):
        commands = self.commands()
        del commands[5]
        with self.assertRaisesRegex(ValueError, "all completed output groups"):
            self.plan(commands)
        commands = self.commands()
        del commands[3:6]
        with self.assertRaisesRegex(ValueError, "all completed output groups"):
            self.plan(commands)

    def test_rejects_stale_rq_group_and_noncontiguous_output(self):
        for command_index, offset, value, message in ((3, 24, 0, "Stale grouped"),
                                                     (3, 16, 129, "Stale grouped"),
                                                     (3, 24, 2, "Invalid grouped"),
                                                     (4, 108, 1184, "not contiguous")):
            commands = self.commands()
            command = bytearray(commands[command_index])
            struct.pack_into("<I", command, offset, value)
            commands[command_index] = bytes(command)
            with self.subTest(offset=offset, value=value), self.assertRaisesRegex(ValueError, message):
                self.plan(commands)

    def test_coverage_rejects_duplicate_group_and_missing_channel_cells(self):
        checks, _ = self.plan(self.commands())
        shapes = {role: [1, 2, 2, 64] for role in DOWNSAMPLE_ROLES}
        for mutate, message in ((lambda c: c[1].update(reference_first_channel=0), "Overlapping"),
                                (lambda c: c[1].update(channels=31), "Incomplete"),
                                (lambda c: c[1].update(reference_first_channel=True), "out-of-range")):
            broken = copy.deepcopy(checks)
            mutate(broken)
            with self.assertRaisesRegex(ValueError, message):
                validate_grouped_coverage(broken, shapes)

    def test_oracle_slices_channels_without_flattening_them_into_rows(self):
        tensor = np.arange(4 * 64, dtype=np.uint8).view(np.int8).reshape(1, 2, 2, 64)
        check = dict(rows=2, channels=32, reference_first_row=1, reference_first_channel=32)
        expected = tensor.reshape(4, 64)[1:3, 32:64]
        self.assertTrue(np.array_equal(reference_tensor_slice(tensor, check), expected))
        flattened_bug = tensor.reshape(-1, 32)[1:3]
        self.assertFalse(np.array_equal(reference_tensor_slice(tensor, check), flattened_bug))
        for changes in (dict(reference_first_channel=40), dict(reference_first_row=3),
                        dict(reference_first_channel=-1), dict(reference_first_row=True)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "exceeds tensor"):
                reference_tensor_slice(tensor, {**check, **changes})


if __name__ == "__main__":
    unittest.main()
