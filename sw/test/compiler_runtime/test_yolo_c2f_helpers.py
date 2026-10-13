"""Offline layout, DMA completion and residual/concat provenance regressions."""

import struct
import unittest

import numpy as np

import yolo_reference_replay as replay
from npu_test_utils import _emit_command_sample
from test_yolo_reference_helpers import linebuffer, lut
from yolo_c2f_checkpoints import C2F_ROLES, checkpoint_indices, extract_checkpoint, plan_c2f_checkpoints


class C2fMappingTest(unittest.TestCase):
    def commands(self):
        runtime = replay.runtime

        def dma(source, src, destination, dst, length, src_stride, dst_stride, rows, direction, queued=False):
            command = bytearray(runtime._dma_2d(runtime._ref(source, offset=src),
                runtime._ref(destination, offset=dst), length, src_stride, dst_stride, rows, direction, 0))
            if queued:
                struct.pack_into("<H", command, 0, 25)
            return bytes(command)

        add = runtime._command_header(13, 64)
        add += runtime._ref(6, offset=256) + runtime._ref(6, offset=1024) + runtime._ref(6, offset=1152)
        add += struct.pack("<2Ii3I", 32, 2, 124, 0, 0, 0)
        return [lut(0, 64, 64), dma(6, 80, 6, 256, 16, 32, 16, 2, 2),
                linebuffer(10, 512, 2, 2), lut(512, 768, 64),
                dma(6, 768, 5, 0, 64, 64, 64, 1, 1, True), runtime._dma_wait(1, 0),
                linebuffer(12, 896, 2, 2, async_job=True), runtime._command_header(30, 32) + bytes(16),
                dma(6, 896, 6, 1024, 16, 32, 16, 2, 2), lut(1024, 1024, 32), add,
                dma(6, 1152, 5, 128, 32, 32, 32, 1, 1, True), runtime._dma_wait(1, 0),
                dma(6, 64, 5, 160, 16, 32, 16, 2, 1, True), runtime._dma_wait(1, 0),
                dma(5, 160, 6, 0, 16, 16, 32, 2, 0, True),
                dma(5, 128, 6, 64, 16, 16, 32, 2, 0, True), runtime._dma_wait(0, 0),
                dma(6, 256, 6, 16, 16, 16, 32, 2, 2),
                runtime._pointwise_c32(runtime._ref(1), runtime._ref(6), runtime._ref(6, offset=1536),
                    runtime._ref(6, offset=1280), 2, 2, 1, 0, 64, 64, 0)]

    def plan(self, commands):
        seed = [dict(command=1, tensor_role="pointwise_activation", address=64, rows=2, channels=32,
                     tile_cols=2, row_stride=64, reference_first_row=0)]
        shapes = {role: [1, 1, 2, 16] for role in C2F_ROLES}
        shapes.update(pointwise_activation=[1, 1, 2, 32], block_concat=[1, 1, 2, 48])
        return plan_c2f_checkpoints(commands, seed, shapes,
            dict(branch_start=16, bypass_start=0, concat_order=["bypass", "branch", "residual"]), 2048, 1024)

    def test_complete_compact_c32_residual_and_concat_mapping(self):
        checks, end = self.plan(self.commands())
        self.assertEqual(end, 19)
        self.assertEqual([c["command"] for c in checks], [2, 3, 4, 8, 10, 11, 19])
        self.assertEqual([c["tensor_role"] for c in checks], list(C2F_ROLES))
        self.assertEqual(checks[3]["submit_command"], 7)
        self.assertEqual(checks[4]["element_stride"], 16)
        self.assertEqual(checks[-1]["group_stride"], 64)

    def test_systolic_output_is_not_visible_until_wait(self):
        commands = self.commands()
        del commands[7]
        with self.assertRaisesRegex(ValueError, "before systolic completion"):
            self.plan(commands)

    def test_queued_l2_data_is_not_visible_until_wait(self):
        commands = self.commands()
        del commands[14]
        with self.assertRaisesRegex(ValueError, "before DMA completion"):
            self.plan(commands)

    def test_rejects_wrong_residual_inputs_and_unsupported_mode(self):
        for offset, value, message in ((28, 1040, "residual inputs"), (44, 1, "residual Add layout/mode")):
            commands = self.commands()
            add = bytearray(commands[10])
            struct.pack_into("<I", add, offset, value)
            commands[10] = bytes(add)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, message):
                self.plan(commands)

    def test_rejects_wrong_pack_stride_and_concat_channel_order(self):
        commands = self.commands()
        copy = bytearray(commands[8])
        struct.pack_into("<I", copy, 40, 32)
        commands[8] = bytes(copy)
        with self.assertRaisesRegex(ValueError, "residual inputs"):
            self.plan(commands)
        commands = self.commands()
        copy = bytearray(commands[18])
        struct.pack_into("<I", copy, 28, 0)
        commands[18] = bytes(copy)
        with self.assertRaisesRegex(ValueError, "not fully materialized"):
            self.plan(commands)

    def test_rejects_missing_or_duplicate_output_rows(self):
        commands = self.commands()
        commands.insert(4, linebuffer(10, 512, 2, 2))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.plan(commands)
        with self.assertRaisesRegex(ValueError, "no supported two-group pointwise consumer"):
            self.plan(self.commands()[:-2])

    def test_extracts_compact_c16_c32_and_multigroup_with_stride_gaps(self):
        memory = np.arange(512, dtype=np.uint16).astype(np.uint8).tobytes()
        for channels, stride, group_stride in ((16, 16, 0), (16, 32, 0), (48, 32, 160)):
            check = dict(address=8, rows=4, channels=channels, tile_cols=2,
                         element_stride=stride, row_stride=2 * stride + 16, group_stride=group_stride)
            expected = np.array([[memory[8 + row // 2 * (2 * stride + 16) + row % 2 * stride +
                                  channel // 32 * group_stride + channel % 32]
                                  for channel in range(channels)] for row in range(4)], dtype=np.uint8).view(np.int8)
            self.assertTrue(np.array_equal(extract_checkpoint(memory, check), expected))
        for changes in (dict(element_stride=8), dict(address=500), dict(channels=48, group_stride=0)):
            check = dict(address=0, rows=2, channels=16, tile_cols=2, row_stride=64)
            check.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                checkpoint_indices(check, len(memory))


class AsyncCommandCallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_sync_and_async_callbacks_are_both_completed(self):
        seen = []

        def sync(index, counters):
            seen.append((index, counters))

        async def asynchronous(index, counters):
            seen.append((index, counters))

        await _emit_command_sample(sync, 1, {"cycle": 5})
        await _emit_command_sample(asynchronous, 2, {"cycle": 7})
        await _emit_command_sample(None, 3, {})
        self.assertEqual(seen, [(1, {"cycle": 5}), (2, {"cycle": 7})])

    async def test_callback_failure_propagates(self):
        async def failing(index, counters):
            raise AssertionError("checkpoint mismatch")

        with self.assertRaisesRegex(AssertionError, "checkpoint mismatch"):
            await _emit_command_sample(failing, 0, {})


if __name__ == "__main__":
    unittest.main()
