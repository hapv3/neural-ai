"""Offline regression tests for reference-checkpoint preparation."""

from pathlib import Path
import json
import struct
import tempfile
import unittest

import numpy as np

from yolo_reference_replay import (
    artifact, cascade_l2_location, checked_read, completed_replay_boundary, consumer_store_location,
    execution_model, invocation_address, load_case, output_comparison,
    plan_checkpoints, plan_lut_restore, plan_post_cascade_checkpoints, plan_restore_preamble, reset_buffers,
    POST_CASCADE_ROLES,
    require_compatible_reference, require_resident_l2_result, runtime, save_checkpoint_result,
    save_resident_l2_result,
    validate_cascade_coverage, validate_output_scope, validate_reference_contract,
)


def linebuffer(layer, address, rows, width, accum=0, async_job=False):
    command = bytearray(160)
    struct.pack_into("<HHIII", command, 0, 29 if async_job else 9, 160, 0, layer, 0)
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


def l2_store(source, destination, length=32, async_job=False):
    command = bytearray(64)
    struct.pack_into("<HHIII", command, 0, 24 if async_job else 2, 64, 0, 2, 0)
    struct.pack_into("<HHIHHI2I", command, 16, 6, 0, source, 5, 0, destination, length, 1)
    return bytes(command)


class ReferenceReplayTest(unittest.TestCase):
    def post_commands(self):
        commands = self.staged_commands()
        for address in (64, 96):
            load = bytearray(runtime._dma_1d(runtime._ref(5, offset=address), runtime._ref(6), 32, 0, 0))
            struct.pack_into("<H", load, 0, 24)
            commands.extend((bytes(load), runtime._dma_wait(0, 0), lut(0, 512, 32),
                             runtime._pointwise_c32(runtime._ref(1), runtime._ref(6, offset=512),
                                 runtime._ref(0), runtime._ref(6, offset=640), 1, 1, 1, 0, 32, 32, 0),
                             lut(640, 768, 32)))
        return commands

    def post_plan(self, commands):
        checks, _ = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), None)
        shapes = {role: [1, 1, 2, 3] for role in POST_CASCADE_ROLES}
        return plan_post_cascade_checkpoints(commands, checks, shapes)

    def test_post_cascade_tracks_reused_staging_and_pointwise_rows(self):
        checks, end = self.post_plan(self.post_commands())
        self.assertEqual(end, 16)
        self.assertEqual([c["command"] for c in checks], [9, 10, 11, 14, 15, 16])
        for role in POST_CASCADE_ROLES:
            selected = [c for c in checks if c["tensor_role"] == role]
            self.assertEqual([c["reference_first_row"] for c in selected], [0, 1])
            self.assertEqual(selected[0]["address"], selected[1]["address"])
        coverage = validate_cascade_coverage(checks, {r: [1, 1, 2, 3] for r in POST_CASCADE_ROLES},
                                             POST_CASCADE_ROLES)
        self.assertEqual([c["bytes"] for c in coverage.values()], [6, 6, 6])

    def test_post_cascade_requires_completed_load_and_store(self):
        commands = self.post_commands()
        del commands[7]
        with self.assertRaisesRegex(ValueError, "before DMA completion"):
            self.post_plan(commands)
        commands = self.post_commands()
        initial, _ = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), None)
        initial[1]["l2_store"]["complete_command"] = 100
        with self.assertRaisesRegex(ValueError, "store is incomplete"):
            plan_post_cascade_checkpoints(commands, initial, {r: [1, 1, 2, 3] for r in POST_CASCADE_ROLES})

    def test_post_cascade_rejects_overwritten_input_and_duplicate_rows(self):
        commands = self.post_commands()
        commands.insert(9, runtime._dma_1d(runtime._ref(1), runtime._ref(6, offset=512), 32, 0, 0))
        with self.assertRaisesRegex(ValueError, "not completed post-cascade activation"):
            self.post_plan(commands)
        commands = self.post_commands()
        load = bytearray(commands[11])
        struct.pack_into("<I", load, 20, 64)
        commands[11] = bytes(load)
        with self.assertRaisesRegex(ValueError, "Duplicate post-cascade"):
            self.post_plan(commands)

    def test_post_cascade_rejects_partial_multigroup_or_flagged_pointwise(self):
        for offset, value in ((32, 6), (52, 2), (56, 2), (4, 1)):
            commands = self.post_commands()
            pointwise = bytearray(commands[9])
            struct.pack_into("<I", pointwise, offset, value)
            commands[9] = bytes(pointwise)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "Unsupported.*pointwise"):
                self.post_plan(commands)

    def test_post_cascade_rejects_missing_rows_and_unsupported_shapes(self):
        commands = self.post_commands()
        with self.assertRaisesRegex(ValueError, "Incomplete post-cascade checkpoints"):
            self.post_plan(commands[:-1])
        initial, _ = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), None)
        for shape in ([1, 1, 2, 33], [2, 1, 2, 3], [1, 1, 2, True], [1, 1, 2, 0]):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "single-C32-group"):
                plan_post_cascade_checkpoints(commands, initial, {r: shape for r in POST_CASCADE_ROLES})

    def test_post_cascade_case_revalidates_mapping_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.full_case(folder)
            load_qparams = struct.pack("<HHIII", 5, 32, 0, 0, 0).ljust(32, b"\0")
            commands = [load_qparams] + self.post_commands()
            initial_shapes = {"producer_activation": [1, 2, 2, 2], "consumer_conv": [1, 1, 2, 3]}
            initial, _ = plan_checkpoints(commands, 0, *initial_shapes.values(), None)
            post_shapes = {role: [1, 1, 2, 3] for role in POST_CASCADE_ROLES}
            post, end = plan_post_cascade_checkpoints(commands, initial, post_shapes)
            bindings = b"".join(runtime._binding(d, 0, dimensions=(1, 1, 1, s))
                                for d, s in ((1, 64), (2, 6), (3, 128)))
            terminal = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
            model = bytearray(runtime._package(b"".join(commands) + terminal, bytes(range(256)),
                                              bindings, len(commands), 1024, 1, 1))
            struct.pack_into("<I", model, 64 + 3 * 32 + 20, 3)
            (folder / "model.nai").write_bytes(model)
            checks = sorted(initial + post, key=lambda c: c["command"])
            for check in checks:
                expected = folder / f"check-{check['command']}.bin"
                expected.write_bytes(bytes(check["rows"] * check["channels"]))
                check["expected"] = artifact(expected)
            expected = folder / "l2.bin"
            expected.write_bytes(bytes(6))
            case.update(full_graph=False, final_output=None, model=artifact(folder / "model.nai"),
                        end_command=end, all_cascade_checkpoints=True, cascade_shapes=initial_shapes,
                        cascade_coverage=validate_cascade_coverage(checks, initial_shapes), checkpoints=checks,
                        cascade_l2_output={**cascade_l2_location(checks, commands, end), "expected": artifact(expected)},
                        post_cascade_checkpoints=True, post_cascade_shapes=post_shapes,
                        post_cascade_coverage=validate_cascade_coverage(post, post_shapes, POST_CASCADE_ROLES))
            path = folder / "case.json"
            path.write_text(json.dumps(case))
            self.assertEqual(len(load_case(path)[0]["checkpoints"]), 9)
            for mutate, message in ((lambda: post[0].update(address=544), "stale post-cascade checkpoint"),
                                    (lambda: case["post_cascade_coverage"]["pointwise_conv"].update(bytes=99),
                                     "Stale post-cascade coverage")):
                original = json.loads(path.read_text())
                mutate()
                path.write_text(json.dumps(case))
                with self.assertRaisesRegex(ValueError, message):
                    load_case(path)
                case = original
                path.write_text(json.dumps(case))

    def staged_commands(self, async_store=False):
        commands = [linebuffer(0, 128, 4, 2), lut(128, 0, 128),
                    linebuffer(2, 256, 1, 2), l2_store(256, 64),
                    linebuffer(2, 256, 1, 2), l2_store(256, 96, async_job=async_store)]
        if async_store:
            commands.append(struct.pack("<HHIII4I", 27, 32, 0, 2, 0, 1, 0, 0, 0))
        return commands

    def test_staged_consumer_uses_l2_offsets_not_reused_tcdm_addresses(self):
        commands = self.staged_commands()
        checks, end = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))
        consumers = [c for c in checks if c["tensor_role"] == "consumer_conv"]
        self.assertEqual((end, [c["address"] for c in consumers]), (6, [256, 256]))
        self.assertEqual([c["l2_store"]["address"] for c in consumers], [64, 96])
        self.assertEqual(cascade_l2_location(checks, commands, end),
                         dict(address=64, rows=2, channels=3, row_stride=64, tile_cols=2))

    def test_staged_consumer_requires_store_completion_before_replay_end(self):
        commands = self.staged_commands(async_store=True)
        checks, end = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))
        self.assertEqual(end, 7)
        with self.assertRaisesRegex(ValueError, "no completion wait"):
            plan_checkpoints(commands[:-1], 0, (1, 2, 2, 2), (1, 1, 2, 3))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            cascade_l2_location(checks, commands, 6)

    def test_staged_consumer_rejects_overwrite_or_noncontiguous_l2_stores(self):
        for overwrite in (linebuffer(2, 256, 1, 2), lut(128, 256, 32)):
            commands = self.staged_commands()
            commands.insert(3, overwrite)
            with self.assertRaisesRegex(ValueError, "overwritten before"):
                plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))
        commands = self.staged_commands()
        commands[-1] = l2_store(256, 128)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))

    def test_staged_consumer_rejects_stale_mapping_and_later_l2_overwrite(self):
        commands = self.staged_commands()
        checks, end = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))
        checks[-1]["l2_store"]["address"] += 32
        with self.assertRaisesRegex(ValueError, "Stale L2"):
            cascade_l2_location(checks, commands, end)
        checks, end = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))
        commands.append(l2_store(256, 64))
        with self.assertRaisesRegex(ValueError, "Unmapped DMA overwrites"):
            cascade_l2_location(checks, commands, end + 1)

    def test_completed_boundary_also_drains_the_other_dma_queue(self):
        commands = self.staged_commands(async_store=True)
        load = bytearray(l2_store(256, 96, async_job=True))
        struct.pack_into("<I", load, 36, 0)
        commands.insert(-1, bytes(load))
        commands.append(struct.pack("<HHIII4I", 27, 32, 0, 2, 0, 0, 0, 0, 0))
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        model = runtime._package(b"".join(commands) + end, b"", b"", len(commands), 1024, 0, 0)
        self.assertEqual(completed_replay_boundary(model, 8), 9)

    def test_fail_fast_saves_first_mismatch_and_complete_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "expected.bin"
            path.write_bytes(bytes((1, 2, 3)))
            check = {"command": 157, "tensor": 161, "address": 61440,
                     "rows": 1, "channels": 3, "reference_first_row": 1920,
                     "expected": artifact(path)}
            actual = np.array([[1, 2, 4]], dtype=np.int8)
            report = {"status": "running", "checkpoints": []}
            stats = save_checkpoint_result(folder, report, check, actual, fail_fast=True)
            saved = json.loads((folder / "byte_exact.json").read_text())
            self.assertEqual((stats["mismatches"], stats["first_mismatch"], stats["max_abs"]), (1, 2, 1))
            self.assertEqual((saved["status"], saved["failed_command"]), ("FAIL", 157))
            self.assertEqual(saved["checkpoints"][0]["reference_first_row"], 1920)
            self.assertEqual((folder / "command-157-actual.bin").read_bytes(), actual.tobytes())

    def test_exact_checkpoint_does_not_claim_whole_case_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "expected.bin"
            path.write_bytes(bytes((1, 2, 3)))
            check = {"command": 1, "tensor": 2, "expected": artifact(path)}
            report = {"status": "running", "checkpoints": []}
            stats = save_checkpoint_result(folder, report, check, np.array([[1, 2, 3]], dtype=np.int8), True)
            self.assertEqual(stats["mismatches"], 0)
            self.assertEqual(report["status"], "running")
            self.assertNotIn("failed_command", report)

    def test_collect_all_mode_retains_mismatch_without_early_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "expected.bin"
            path.write_bytes(bytes((1, 2, 3)))
            check = {"command": 1, "tensor": 2, "expected": artifact(path)}
            report = {"status": "running", "checkpoints": []}
            stats = save_checkpoint_result(folder, report, check, np.array([[1, 2, 4]], dtype=np.int8))
            self.assertEqual(stats["mismatches"], 1)
            self.assertEqual(report["status"], "running")

    def test_case_rejects_invalid_fail_fast_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.full_case(folder)
            path = folder / "case.json"
            for flag in (1, "true", None):
                case["fail_fast"] = flag
                path.write_text(json.dumps(case))
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "fail_fast must be boolean"):
                    load_case(path)

    def test_full_output_only_still_requires_oracle_and_all_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.full_case(folder)
            case["checkpoints"] = []
            path = folder / "case.json"
            path.write_text(json.dumps(case))
            self.assertEqual(load_case(path)[0]["checkpoints"], [])
            case.update(full_graph=False, final_output=None)
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError, "no byte-exact checkpoints"):
                load_case(path)

    def test_full_reset_needs_no_lut_residency_for_mean_only_model(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.full_case(folder)
            end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
            mean = struct.pack("<HHIII", 14, 64, 0, 0, 0).ljust(64, b"\0")
            bindings = b"".join(runtime._binding(direction, 0, dimensions=(1, 1, 1, size))
                                for direction, size in ((1, 64), (2, 6), (3, 128)))
            model = bytearray(runtime._package(mean + end, b"", bindings, 1, 1024, 1, 1))
            struct.pack_into("<I", model, 64 + 3 * 32 + 20, 3)
            (folder / "model.nai").write_bytes(model)
            case.update(model=artifact(folder / "model.nai"), end_command=1, checkpoints=[])
            path = folder / "case.json"
            path.write_text(json.dumps(case))
            self.assertIsNone(load_case(path)[0]["afu_lut_restore"])

    def test_invocation_address_moves_above_large_encoded_model(self):
        self.assertEqual(invocation_address(bytes(0x400000)), 0x81400000)
        self.assertEqual(invocation_address(bytes(0x400001)), 0x81800000)
        self.assertEqual(invocation_address(bytes(0x800001)), 0x81C00000)

    def test_case_rejects_overlapping_or_invalid_invocation_address(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.full_case(folder)
            path = folder / "case.json"
            for address in (0x81000000, 0x81400001, 0xFFFFF000, True):
                case["invocation_base"] = address
                path.write_text(json.dumps(case))
                with self.subTest(address=address), self.assertRaisesRegex(ValueError, "Invocation/control"):
                    load_case(path)

    def full_case(self, folder):
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        load = struct.pack("<HHIII", 5, 32, 0, 0, 0).ljust(32, b"\0")
        bindings = b"".join(runtime._binding(direction, 0, dimensions=(1, 1, 1, size))
                            for direction, size in ((1, 64), (2, 6), (3, 128)))
        model = bytearray(runtime._package(load + b"".join(self.commands()) + end,
                                          bytes(range(256)), bindings, 5, 1024, 1, 1))
        struct.pack_into("<I", model, 64 + 3 * 32 + 20, 3)

        def pinned(name, data):
            path = folder / name
            path.write_bytes(data)
            return artifact(path)

        case = {"version": 1, "start_mode": "reset", "first_command": 0, "end_command": 5,
                "snapshot": None, "model": pinned("model.nai", model),
                "source": pinned("source.tflite", b"source"), "firmware": pinned("firmware.elf", b"elf"),
                "input": pinned("input.bin", bytes(64)), "input_bytes": 64, "output_bytes": 6,
                "temporary_bytes": 128, "full_graph": True, "timeout_cycles": 50000000,
                "afu_lut_restore": None, "restore_preamble": None,
                "final_output": {"dtype": "int8", "shape": [1, 2, 3],
                                 "expected": pinned("output-expected.bin", bytes(range(6)))},
                "checkpoints": [{"command": 5, "address": 32, "rows": 1, "channels": 3,
                                 "tile_cols": 2, "row_stride": 64,
                                 "expected": pinned("checkpoint.bin", bytes(3))}]}
        return case, bytes(model)

    def test_full_graph_case_loads_every_command_and_pinned_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, model = self.full_case(folder)
            path = folder / "case.json"
            path.write_text(json.dumps(case))
            loaded, full_model, *_ = load_case(path)
            self.assertEqual(loaded, case)
            self.assertEqual(full_model, model)
            self.assertIs(execution_model(loaded, full_model), full_model)

    def test_full_graph_rejects_partial_or_snapshot_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.full_case(Path(directory))
            for changes in ({"first_command": 1}, {"end_command": 4}, {"start_mode": "snapshot"},
                            {"snapshot": case["model"]}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "run every command"):
                    validate_output_scope({**case, **changes}, model)

    def test_full_graph_requires_complete_int8_output_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.full_case(Path(directory))
            with self.assertRaisesRegex(ValueError, "pinned final output oracle"):
                validate_output_scope({**case, "final_output": None}, model)
            for changes in ({"dtype": "int16"}, {"shape": [1, 2, 2]}, {"shape": [1, 0, 3]},
                            {"shape": [1, True, 6]}, {"expected": case["checkpoints"][0]["expected"]}):
                invalid = {**case, "final_output": {**case["final_output"], **changes}}
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "shape/dtype/size"):
                    validate_output_scope(invalid, model)

    def test_full_graph_has_no_tolerance_or_completion_only_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            case, _ = self.full_case(Path(directory))
            expected = checked_read(case["final_output"]["expected"])
            self.assertEqual(output_comparison(case["final_output"], expected)["mismatches"], 0)
            actual = bytearray(expected)
            actual[-1] += 1
            stats = output_comparison(case["final_output"], actual)
            self.assertEqual((stats["bytes"], stats["mismatches"], stats["max_abs"]), (6, 1, 1))
            with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
                output_comparison(case["final_output"], expected[:-1])

    def test_bounded_replay_cannot_claim_final_output_and_still_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.full_case(Path(directory))
            case["full_graph"] = False
            with self.assertRaisesRegex(ValueError, "Bounded replay cannot claim"):
                validate_output_scope(case, model)
            case.update({"final_output": None, "end_command": 4})
            validate_output_scope(case, model)
            replay = execution_model(case, model)
            self.assertEqual(len(runtime._logical_command_records(replay)[0]), 4)
            self.assertNotEqual(replay, model)

    def test_full_graph_preserves_encoded_affine_loop(self):
        control = struct.pack("<HHIII", 1, 32, 0, 0, 0).ljust(32, b"\0")
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        loop = struct.pack("<HHIII4I", 33, 32, 0, 0, 0, 3, 1, 32, 0)
        model = runtime._package(loop + control + end, b"", b"", 3, 1024, 0, 0)
        self.assertEqual(len(runtime._logical_command_records(model)[0]), 3)
        self.assertIs(execution_model({"full_graph": True}, model), model)
        flattened = execution_model({"first_command": 0, "end_command": 3, "restore_preamble": None}, model)
        self.assertNotEqual(flattened, model)

    def remaining_case(self, folder):
        case, model = self.full_case(folder)
        case.update(full_graph=False, remaining_graph=True, start_mode="snapshot", first_command=1,
                    snapshot=case["model"], resumed_pointwise_checkpoints=True, remaining_prefix_end_command=3)
        return case, model

    def test_remaining_scope_keeps_reset_regression_distinct_and_runs_to_final_command(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.remaining_case(Path(directory))
            validate_output_scope(case, model)
            suffix = execution_model(case, model)
            original, original_end = runtime._logical_command_records(model)
            selected, selected_end = runtime._logical_command_records(suffix)
            self.assertEqual(selected, original[1:])
            self.assertEqual(selected_end, original_end)
            self.assertFalse(case["full_graph"])

    def test_remaining_scope_rejects_partial_suffix_missing_snapshot_and_unproven_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.remaining_case(Path(directory))
            for changes in ({"end_command": 4}, {"first_command": 0}, {"first_command": True},
                            {"start_mode": "reset"}, {"snapshot": None},
                            {"resumed_pointwise_checkpoints": False},
                            {"remaining_prefix_end_command": None}, {"remaining_prefix_end_command": True},
                            {"remaining_prefix_end_command": 1}, {"remaining_prefix_end_command": 6},
                            {"resumed_c2f_checkpoints": True}, {"resumed_output_checkpoints": True}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "all remaining commands"):
                    validate_output_scope({**case, **changes}, model)

    def test_remaining_scope_accepts_verified_concat_prefix_with_optional_downsample(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.remaining_case(Path(directory))
            case.update(resumed_pointwise_checkpoints=False, resumed_output_checkpoints=True)
            for downsample in (False, True):
                with self.subTest(downsample=downsample):
                    validate_output_scope({**case, "l2_downsample_checkpoints": downsample}, model)
            original, _ = runtime._logical_command_records(model)
            selected, _ = runtime._logical_command_records(execution_model(case, model))
            self.assertEqual(selected, original[case["first_command"]:])
            for changes in ({"resumed_output_checkpoints": False},
                            {"resumed_pointwise_checkpoints": True}, {"end_command": 4},
                            {"resumed_c2f_checkpoints": True}, {"snapshot": None}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "all remaining commands"):
                    validate_output_scope({**case, **changes}, model)

    def test_explicit_afu_reference_pins_kind_sources_input_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            case, _ = self.full_case(Path(directory))
            case.update(reference_build={"library": "pinned"}, reference_kind="neural_ai_afu_dfl_reference")
            proof = dict(description=dict(contract="afu_dfl16_deployed_signed_delta_v1",
                                          source_sha256=case['source']['sha256']),
                         input_sha256=case['input']['sha256'], reference_build=case['reference_build'],
                         output_sha256=case['final_output']['expected']['sha256'],
                         sources={name: case['source'] for name in ('runtime.py', 'afu_dfl.py')})
            case['reference_contract'] = proof
            validate_reference_contract(case)
            for key, value in [('input_sha256', 'wrong'), ('reference_build', {}),
                               ('output_sha256', 'wrong'), ('sources', {}),
                               ('description', {**proof['description'], 'contract': 'wrong'})]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_reference_contract({**case, 'reference_contract': {**proof, key: value}})
            with self.assertRaises(ValueError):
                validate_reference_contract({**case, 'reference_kind': 'unknown'})
            # Old retained cases preserve their original numerical contract.
            del case['reference_contract']
            with self.assertRaisesRegex(ValueError, "Missing explicit"):
                validate_reference_contract(case)
            del case['reference_kind']
            validate_reference_contract(case)

    def test_final_output_preflight_rejects_mac_only_for_deployed_afu(self):
        def model(kinds):
            records = b''.join(runtime._command_header(kind, 96, flags=flags) + bytes(80)
                               for kind, flags in kinds)
            return runtime._package(records + runtime._command_header(0, 32) + bytes(16),
                                    b'', b'', len(kinds), 1024, 1, 1)
        for kinds, selected in [([(28, 0)], 'neural_ai_afu_dfl_reference'),
                                ([(14, 16)], 'neural_ai_afu_mean_reference')]:
            content = model(kinds)
            require_compatible_reference(content, selected, True)
            require_compatible_reference(content, 'neural_ai_reference', False)
            with self.subTest(kinds=kinds), self.assertRaisesRegex(ValueError, '--reference-kind'):
                require_compatible_reference(content, 'neural_ai_reference', True)
        for selected in ('neural_ai_afu_dfl_reference', 'neural_ai_afu_mean_reference'):
            with self.assertRaisesRegex(ValueError, '--reference-kind'):
                require_compatible_reference(model([(28, 0), (14, 16)]), selected, True)

    def test_live_resident_l2_evidence_survives_later_memory_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.remaining_case(folder)
            check = dict(command=3, rows=2, channels=3, expected=case["final_output"]["expected"])
            actual = np.arange(6, dtype=np.int8).reshape(2, 3)
            report = dict(status="running")
            stats = save_resident_l2_result(folder, report, check, actual, True)
            self.assertEqual(stats["mismatches"], 0)
            actual[:] = -1  # A later layer legitimately reuses the same L2 allocation.
            self.assertEqual(require_resident_l2_result(report, check)["mismatches"], 0)
            self.assertEqual((folder / "resident-l2-actual.bin").read_bytes(), bytes(range(6)))
            self.assertEqual(json.loads((folder / "byte_exact.json").read_text()), report)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                save_resident_l2_result(folder, report, check, actual)
            for broken, plan in (({}, check), (report, {**check, "command": 4}),
                                 ({"resumed_output_l2": {**report["resumed_output_l2"], "bytes": 5}}, check)):
                with self.subTest(plan=plan), self.assertRaisesRegex(ValueError, "Missing or stale"):
                    require_resident_l2_result(broken, plan)

    def test_live_resident_l2_mismatch_is_persisted_before_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.remaining_case(folder)
            check = dict(command=3, rows=2, channels=3, expected=case["final_output"]["expected"])
            actual = np.arange(6, dtype=np.int8).reshape(2, 3)
            actual[1, 2] ^= 1
            report = dict(status="running")
            stats = save_resident_l2_result(folder, report, check, actual, True)
            self.assertEqual(stats["mismatches"], 1)
            saved = json.loads((folder / "byte_exact.json").read_text())
            self.assertEqual(saved["status"], "FAIL")
            self.assertEqual(saved["failed_command"], 3)
            self.assertEqual(require_resident_l2_result(saved, check)["mismatches"], 1)

    def test_remaining_scope_rejects_ambiguous_modes_and_nonboolean_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.remaining_case(Path(directory))
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                validate_output_scope({**case, "full_graph": True}, model)
            for value in (1, None, "true"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be boolean"):
                    validate_output_scope({**case, "remaining_graph": value}, model)

    def test_remaining_scope_requires_complete_oracle_and_detects_one_byte_error(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.remaining_case(Path(directory))
            for output in (None, {**case["final_output"], "shape": [1, 2, 2]},
                           {**case["final_output"], "expected": case["checkpoints"][0]["expected"]}):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    validate_output_scope({**case, "final_output": output}, model)
            actual = bytearray(checked_read(case["final_output"]["expected"]))
            actual[-1] ^= 1
            self.assertEqual(output_comparison(case["final_output"], actual)["mismatches"], 1)
            with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
                output_comparison(case["final_output"], actual[:-1])

    def test_invalid_scope_flags_and_timeout_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            case, model = self.full_case(Path(directory))
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                validate_output_scope({**case, "full_graph": 1}, model)
            for timeout in (0, -1, True, "50000000"):
                with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_output_scope({**case, "timeout_cycles": timeout}, model)

    def test_rolling_producer_maps_new_logical_rows_at_reused_addresses(self):
        commands = self.commands() + [linebuffer(0, 128, 4, 2), lut(128, 64, 128),
                                      linebuffer(2, 64, 1, 2)]
        checks, end = plan_checkpoints(commands, 0, (1, 4, 2, 2), (1, 2, 2, 3), 3)
        self.assertEqual(end, 7)
        self.assertEqual([c["reference_first_row"] for c in checks], [0, 0, 1, 4, 2])

    def test_async_producer_allows_disjoint_lut_then_wait(self):
        commands = [linebuffer(0, 128, 2, 2), linebuffer(0, 192, 2, 2, async_job=True),
                    lut(128, 0, 64), struct.pack("<HHIII", 30, 32, 0, 0, 0).ljust(32, b"\0"),
                    lut(192, 64, 64), linebuffer(2, 0, 2, 2)]
        checks, end = plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), 1)
        self.assertEqual(end, 6)
        self.assertEqual([c["reference_first_row"] for c in checks], [0, 2, 0])
        for source, destination in ((192, 64), (128, 192)):
            commands[2] = lut(source, destination, 64)
            with self.assertRaisesRegex(ValueError, "outstanding producer"):
                plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), 1)

    def test_lut_crossing_reused_and_stale_rows_is_rejected(self):
        commands = [linebuffer(0, 128, 4, 2), linebuffer(0, 128, 2, 2),
                    lut(128, 0, 128), linebuffer(2, 0, 1, 2)]
        with self.assertRaisesRegex(ValueError, "contiguous producer rows"):
            plan_checkpoints(commands, 0, (1, 3, 2, 2), (1, 1, 2, 3), 1)

    def test_dma_overwrite_invalidates_producer_mapping(self):
        for kind in (2, 3, 4, 24, 25, 26):
            command = bytearray(64)
            struct.pack_into("<HHIII", command, 0, kind, 64, 0, 0, 0)
            struct.pack_into("<HHI", command, 24, 6, 0, 128)
            struct.pack_into("<I", command, 32, 1)
            if kind in (3, 4, 25, 26):
                struct.pack_into("<2I", command, 40, 32, 1)
            if kind in (4, 26):
                struct.pack_into("<2I", command, 52, 64, 1)
            commands = self.commands()
            commands.insert(1, bytes(command))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "contiguous producer rows"):
                plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), 1)

    def test_async_producer_requires_wait_and_supported_intervening_commands(self):
        commands = self.commands()
        commands[0] = linebuffer(0, 128, 4, 2, async_job=True)
        with self.assertRaisesRegex(ValueError, "outstanding producer"):
            plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), 1)
        commands = self.commands()
        commands.insert(1, struct.pack("<HHIII", 30, 32, 0, 0, 0).ljust(32, b"\0"))
        with self.assertRaisesRegex(ValueError, "no mapped outstanding"):
            plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), 1)
        commands[1] = struct.pack("<HHIII", 10, 32, 0, 0, 0).ljust(32, b"\0")
        with self.assertRaisesRegex(ValueError, "Unsupported command 10"):
            plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3), 1)

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
        compact = runtime._package(end, b"", bindings[:128], 0, 1024, 1, 1)
        memory, temporary, output, input_bytes = reset_buffers(compact)
        self.assertEqual((len(memory), len(temporary), len(output), input_bytes), (524288, 0, 96, 64))

    def test_fresh_reset_rejects_missing_duplicate_and_nonzero_bindings(self):
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        for descriptors in (((1, 0),), ((1, 0), (2, 1)), ((1, 0), (2, 0), (1, 0))):
            bindings = b"".join(runtime._binding(kind, index, dimensions=(1, 1, 1, 32))
                                for kind, index in descriptors)
            model = bytearray(runtime._package(end, b"", bindings, 0, 1024, 1, 1))
            struct.pack_into("<I", model, 64 + 3 * 32 + 20, len(descriptors))
            with self.subTest(descriptors=descriptors), self.assertRaisesRegex(ValueError, "one input and output"):
                reset_buffers(model)

    def test_fresh_reset_allows_temporary_above_one_mib(self):
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        bindings = b"".join(runtime._binding(direction, 0, dimensions=(1, 1, 1, size))
                            for direction, size in ((1, 150528), (2, 1000), (3, 2408448)))
        model = bytearray(runtime._package(end, b"", bindings, 0, 1024, 1, 1))
        struct.pack_into("<I", model, 64 + 3 * 32 + 20, 3)
        memory, temporary, output, input_bytes = reset_buffers(model)
        self.assertEqual((len(memory), len(temporary), len(output), input_bytes),
                         (524288, 2408448, 1000, 150528))

    def test_fresh_reset_rejects_binding_window_overlap(self):
        end = struct.pack("<HHIII", 0, 32, 0, 0, 0).ljust(32, b"\0")
        for direction, bad_size in ((1, 0x100001), (2, 0x100001), (3, 0xE00001)):
            sizes = {1: 64, 2: 96, 3: 128, direction: bad_size}
            bindings = b"".join(runtime._binding(kind, 0, dimensions=(1, 1, 1, size))
                                for kind, size in sizes.items())
            model = bytearray(runtime._package(end, b"", bindings, 0, 1024, 1, 1))
            struct.pack_into("<I", model, 64 + 3 * 32 + 20, 3)
            with self.subTest(direction=direction), self.assertRaisesRegex(ValueError, "non-overlapping L2"):
                reset_buffers(model)

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

    def test_rejects_async_consumer_restart_without_wait_or_multigroup_tensor(self):
        commands = self.commands()
        async_command = bytearray(commands[2])
        struct.pack_into("<H", async_command, 0, 29)
        commands[2] = bytes(async_command)
        with self.assertRaisesRegex(ValueError, "consumer wait"):
            plan_checkpoints(commands, 1, (1, 2, 2, 2), (1, 1, 2, 3))
        with self.assertRaises(ValueError):
            plan_checkpoints(self.commands(), 1, (1, 2, 2, 33), (1, 1, 2, 3))

    def async_consumer_commands(self):
        commands = self.commands()
        commands[-1] = linebuffer(2, 32, 1, 2, async_job=True)
        # A later layer can run disjoint LUT work while the consumer finishes.
        later = bytearray(lut(128, 256, 32))
        struct.pack_into("<I", later, 8, 3)
        commands.extend((bytes(later), struct.pack("<HHIII", 30, 32, 0, 2, 0).ljust(32, b"\0")))
        return commands

    def test_async_consumer_is_checked_at_wait_not_submission(self):
        checks, end = plan_checkpoints(self.async_consumer_commands(), 0, (1, 2, 2, 2), (1, 1, 2, 3))
        self.assertEqual(end, 6)
        self.assertEqual([c["command"] for c in checks], [2, 3, 6])
        self.assertEqual(checks[-1]["submit_command"], 4)
        self.assertEqual(checks[-1]["reference_first_row"], 1)

    def test_all_cascade_tiles_are_derived_from_tensor_extent(self):
        shapes = {"producer_activation": (1, 2, 2, 2), "consumer_conv": (1, 1, 2, 3)}
        checks, end = plan_checkpoints(self.async_consumer_commands(), 0, *shapes.values(), None)
        coverage = validate_cascade_coverage(checks, shapes)
        self.assertEqual((end, coverage["producer_activation"]["bytes"], coverage["consumer_conv"]["bytes"]),
                         (6, 8, 6))

    def test_incomplete_or_duplicate_cascade_rows_are_rejected(self):
        shapes = {"producer_activation": (1, 2, 2, 2), "consumer_conv": (1, 1, 2, 3)}
        checks, _ = plan_checkpoints(self.commands(), 0, *shapes.values(), None)
        for invalid in (checks[:-1], checks + [checks[0]], checks[1:]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "cascade checkpoints"):
                validate_cascade_coverage(invalid, shapes)

    def test_complete_cascade_case_revalidates_row_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case, _ = self.full_case(folder)
            shapes = {"producer_activation": [1, 2, 2, 2], "consumer_conv": [1, 1, 2, 3]}
            checks, _ = plan_checkpoints(self.commands(), 0, *shapes.values(), None)
            for check in checks:
                check["command"] += 1  # The packaged fixture starts with LOAD_QPARAMS.
                expected = folder / f"check-{check['command']}.bin"
                expected.write_bytes(bytes(check["rows"] * check["channels"]))
                check["expected"] = artifact(expected)
            case.update(full_graph=False, final_output=None, all_cascade_checkpoints=True,
                        cascade_shapes=shapes, checkpoints=checks,
                        cascade_coverage=validate_cascade_coverage(checks, shapes))
            path = folder / "case.json"
            path.write_text(json.dumps(case))
            self.assertEqual(load_case(path)[0]["cascade_coverage"]["consumer_conv"]["bytes"], 6)
            case["cascade_coverage"]["consumer_conv"]["bytes"] = 99
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError, "Stale complete-cascade coverage"):
                load_case(path)
            case["checkpoints"] = checks[:-1]
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError, "Incomplete cascade checkpoints"):
                load_case(path)

    def test_async_consumer_missing_wait_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Not enough completed"):
            plan_checkpoints(self.async_consumer_commands()[:-1], 0, (1, 2, 2, 2), (1, 1, 2, 3))

    def test_lut_cannot_touch_pending_consumer_output(self):
        for source, destination in ((32, 256), (128, 32)):
            commands = self.async_consumer_commands()
            later = bytearray(lut(source, destination, 32))
            struct.pack_into("<I", later, 8, 3)
            commands[-2] = bytes(later)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "outstanding consumer"):
                plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))

    def test_dma_cannot_read_or_write_pending_consumer_output(self):
        for kind in (2, 3, 4, 24, 25, 26):
            for source, destination in ((32, 256), (128, 32)):
                command = bytearray(64)
                struct.pack_into("<HHIII", command, 0, kind, 64, 0, 3, 0)
                struct.pack_into("<HHI", command, 16, 6, 0, source)
                struct.pack_into("<HHI", command, 24, 6, 0, destination)
                struct.pack_into("<I", command, 32, 32)
                if kind in (3, 4, 25, 26):
                    struct.pack_into("<3I", command, 36, 32, 32, 1)
                if kind in (4, 26):
                    struct.pack_into("<3I", command, 48, 32, 32, 1)
                commands = self.async_consumer_commands()
                commands[-2] = bytes(command)
                with self.subTest(kind=kind, source=source), self.assertRaisesRegex(ValueError, "outstanding consumer"):
                    plan_checkpoints(commands, 0, (1, 2, 2, 2), (1, 1, 2, 3))

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
