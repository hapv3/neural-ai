"""C32 residual-chain mapping, source edges and strict completion regressions."""

import struct
import unittest

import numpy as np

import yolo_reference_replay as replay
from resumed_c2f_checkpoints import (
    CONCAT_ROLE, PARENT_ROLE, chain_roles, plan_resumed_c2f_checkpoints, resolve_c2f_chain,
)
from test_yolo_reference_helpers import linebuffer, lut


class ResumedC2fMappingTest(unittest.TestCase):
    def fixture(self, blocks=2, async_conv=False, queued_concat=False):
        runtime = replay.runtime
        commands = [lut(0, 0, 128)]

        def rq(layer):
            return struct.pack("<HHIII4I", 5, 32, 0, layer, 0, layer * 32, 32, 0, 0)

        def copy_bytes(src, dst, source=6, destination=6, queued=False):
            command = bytearray(runtime._dma_1d(runtime._ref(source, offset=src),
                runtime._ref(destination, offset=dst), 64, 1 if destination == 5 else 0 if source == 5 else 2, 0))
            if queued:
                struct.pack_into("<H", command, 0, 24)
            return bytes(command)

        previous = 64
        residuals = []
        for block in range(blocks):
            first_layer = 10 + block * 4
            base = 1024 + block * 1280
            commands += [rq(first_layer), linebuffer(first_layer, base, 1, 2)]
            commands.append(linebuffer(first_layer, base + 32, 1, 2, async_job=async_conv))
            if async_conv:
                commands.append(runtime._command_header(30, 32) + bytes(16))
            commands += [lut(base, base + 256, 64), rq(first_layer + 2),
                         linebuffer(first_layer + 2, base + 512, 2, 2), lut(base + 512, base + 768, 64)]
            destination = base + 1024
            if block == 0:
                commands.append(runtime._afu_binary_quant(runtime._ref(6, offset=previous),
                    runtime._ref(6, offset=base + 768), runtime._ref(6, offset=destination), 64, 0, 0))
            else:
                commands.append(runtime._afu_binary(runtime._ref(6, offset=previous),
                    runtime._ref(6, offset=base + 768), runtime._ref(6, offset=destination), 64, 2, 0))
            previous = destination
            residuals.append(destination)
        if queued_concat:
            commands += [copy_bytes(0, 0, destination=5, queued=True), runtime._dma_wait(1, 0),
                         copy_bytes(0, 4096, source=5, queued=True), runtime._dma_wait(0, 0)]
        else:
            commands.append(copy_bytes(0, 4096))
        commands += [copy_bytes(address, 4096 + (group + 1) * 64)
                     for group, address in enumerate([64] + residuals)]
        commands.append(runtime._pointwise_c32(runtime._ref(1), runtime._ref(6, offset=4096),
            runtime._ref(6, offset=6144), runtime._ref(6, offset=5120), 2, blocks + 2, 1, 0, 64, 64, 0))
        shapes = {role: [1, 1, 2, 32] for role in chain_roles(blocks)}
        shapes.update({PARENT_ROLE: [1, 1, 2, 64], CONCAT_ROLE: [1, 1, 2, 32 * (blocks + 2)]})
        parent = dict(command=1, address=0, rows=2, channels=64, tile_cols=2, row_stride=64, group_stride=64)
        layout = dict(blocks=blocks, branch_start=32, bypass_start=0,
                      concat_order=["bypass", "branch"] + [f"residual_{b}" for b in range(blocks)])
        return commands, parent, shapes, layout

    def plan(self, fixture):
        return plan_resumed_c2f_checkpoints(*fixture, 8192, 1024)

    def mutate(self, fixture, kind, offset, value):
        commands = fixture[0]
        index = next(i for i, c in enumerate(commands) if struct.unpack_from("<H", c)[0] == kind)
        command = bytearray(commands[index])
        struct.pack_into("<I", command, offset, value)
        commands[index] = bytes(command)

    def test_one_and_two_bottlenecks_all_rows_and_concat_groups(self):
        for blocks in (1, 2):
            checks, end = self.plan(self.fixture(blocks))
            self.assertEqual(end, len(self.fixture(blocks)[0]) - 1)
            self.assertEqual(checks[-1]["channels"], (blocks + 2) * 32)
            self.assertEqual(set(c["tensor_role"] for c in checks), set(chain_roles(blocks)))

    def test_async_conv_capture_is_at_wait_and_missing_wait_rejected(self):
        fixture = self.fixture(async_conv=True)
        checks, _ = self.plan(fixture)
        self.assertEqual(sum("submit_command" in c for c in checks), 2)
        del fixture[0][next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 30)]
        with self.assertRaisesRegex(ValueError, "before systolic completion"):
            self.plan(fixture)

    def test_queued_l2_concat_requires_both_waits(self):
        self.plan(self.fixture(queued_concat=True))
        for direction in (0, 1):
            fixture = self.fixture(queued_concat=True)
            index = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 27 and
                         struct.unpack_from("<I", c, 16)[0] == direction)
            del fixture[0][index]
            with self.subTest(direction=direction), self.assertRaisesRegex(ValueError, "before DMA completion"):
                self.plan(fixture)

    def test_requires_fresh_c32_rq_and_output_geometry(self):
        for kind, offset, value in ((5, 20, 16), (9, 120, 96), (9, 116, 1)):
            fixture = self.fixture()
            self.mutate(fixture, kind, offset, value)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "fresh C32 RQ"):
                self.plan(fixture)
        fixture = self.fixture()
        del fixture[0][1]
        with self.assertRaisesRegex(ValueError, "fresh C32 RQ"):
            self.plan(fixture)

    def test_general_and_biased_residual_modes_and_inputs(self):
        for kind, offset, value, message in ((34, 92, 2, "Add mode/layout"),
                                           (13, 44, 1, "Add mode/layout"),
                                           (34, 20, 0, "inputs are stale"),
                                           (13, 28, 3104, "inputs are stale")):
            fixture = self.fixture()
            self.mutate(fixture, kind, offset, value)
            with self.subTest(kind=kind, offset=offset), self.assertRaisesRegex(ValueError, message):
                self.plan(fixture)

    def test_rejects_rq_overwritten_by_another_layer(self):
        fixture = self.fixture()
        command = bytearray(fixture[0][1])
        struct.pack_into("<I", command, 8, 12)
        fixture[0].insert(2, bytes(command))
        with self.assertRaisesRegex(ValueError, "fresh C32 RQ"):
            self.plan(fixture)

    def test_rejects_overwritten_lut_data_and_duplicate_conv_rows(self):
        fixture = self.fixture()
        fixture[0].insert(4, replay.runtime._dma_1d(replay.runtime._ref(1),
            replay.runtime._ref(6, offset=1024), 64, 0, 0))
        with self.assertRaisesRegex(ValueError, "completed Conv bytes"):
            self.plan(fixture)
        fixture = self.fixture()
        fixture[0].insert(4, fixture[0][3])
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.plan(fixture)

    def test_rejects_wrong_concat_order_and_consumer_groups(self):
        fixture = self.fixture()
        fixture[3]["concat_order"][:2] = ["branch", "bypass"]
        with self.assertRaisesRegex(ValueError, "before complete C2f chain concat"):
            self.plan(fixture)
        fixture = self.fixture()
        self.mutate(fixture, 10, 52, 3)
        with self.assertRaisesRegex(ValueError, "matching grouped pointwise"):
            self.plan(fixture)

    def test_rejects_lut_reuse_after_non_lut_afu_mode(self):
        fixture = self.fixture()
        binary = next(i for i, c in enumerate(fixture[0]) if struct.unpack_from("<H", c)[0] == 34)
        index = next(i for i in range(binary + 1, len(fixture[0]))
                     if struct.unpack_from("<H", fixture[0][i])[0] == 12)
        command = bytearray(fixture[0][index])
        struct.pack_into("<I", command, 4, 4)
        fixture[0][index] = bytes(command)
        with self.assertRaisesRegex(ValueError, "stale configuration"):
            self.plan(fixture)


class SourceChainTest(unittest.TestCase):
    def fixture(self, channels=32):
        class Interpreter:
            def __init__(self):
                self.ops = []
                self.tensors = {0: np.zeros((1, 1, 2, channels * 2), np.int8),
                                90: np.array([0, 0, 0, 0]), 91: np.array([0, 0, 0, channels]),
                                92: np.array([0, 0, 0, channels * 2]), 93: np.ones(4, np.int32),
                                94: np.zeros((channels, 3, 3, channels), np.int8)}

            def add(self, name, inputs, output, width=channels):
                self.ops.append(dict(op_name=name, inputs=inputs, outputs=[output]))
                self.tensors[output] = np.zeros((1, 1, 2, width), np.int8)

            def _get_ops_details(self):
                return self.ops

            def get_tensor(self, tensor):
                return self.tensors[tensor]

        interpreter = Interpreter()
        interpreter.add("STRIDED_SLICE", [0, 90, 91, 93], 1)
        interpreter.add("STRIDED_SLICE", [0, 91, 92, 93], 2)
        current = 2
        for base in (10, 20):
            interpreter.add("CONV_2D", [current, 94, 94], base)
            interpreter.add("LOGISTIC", [base], base + 1)
            interpreter.add("MUL", [base, base + 1], base + 2)
            interpreter.add("CONV_2D", [base + 2, 94, 94], base + 3)
            interpreter.add("LOGISTIC", [base + 3], base + 4)
            interpreter.add("MUL", [base + 3, base + 4], base + 5)
            interpreter.add("ADD", [current, base + 5], base + 6)
            current = base + 6
        interpreter.add("CONCATENATION", [1, 2, 16, 26], 30, channels * 4)
        return interpreter

    def test_resolves_full_chain_and_source_channel_order(self):
        ids, shapes, layout = resolve_c2f_chain(self.fixture(), 0)
        self.assertEqual(layout, dict(branch_start=32, bypass_start=0, blocks=2,
                                     concat_order=["bypass", "branch", "residual_0", "residual_1"]))
        self.assertEqual(ids[CONCAT_ROLE], 30)
        self.assertEqual(shapes[CONCAT_ROLE], [1, 1, 2, 128])

    def test_resolves_multigroup_branches_and_eight_group_concat(self):
        _, shapes, layout = resolve_c2f_chain(self.fixture(64), 0)
        self.assertEqual((layout["branch_start"], layout["bypass_start"], layout["blocks"]), (64, 0, 2))
        self.assertEqual(shapes[PARENT_ROLE], [1, 1, 2, 128])
        self.assertEqual(shapes[CONCAT_ROLE], [1, 1, 2, 256])

    def test_rejects_non_k3_conv_partial_slice_and_wrong_residual_edge(self):
        for mutation in ("kernel", "slice", "edge", "concat"):
            interpreter = self.fixture()
            if mutation == "kernel":
                interpreter.tensors[94] = np.zeros((32, 1, 1, 32), np.int8)
            elif mutation == "slice":
                interpreter.tensors[91][3] = 16
            elif mutation == "edge":
                interpreter.ops[-2]["inputs"][0] = 1
            else:
                interpreter.ops[-1]["inputs"][0] = 26
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resolve_c2f_chain(interpreter, 0)


if __name__ == "__main__":
    unittest.main()
