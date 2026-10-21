"""Offline guards for selected captures; a dump is never a byte-exact PASS."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import zlib

from compute_dumps import ComputeDumpWriter, plan_compute_dumps, select_compute_dumps, validate_compute_dump_case
from prepare_model_checkpoints import safe_layer_boundaries
from yolo_reference_replay import artifact, checked_read, runtime


class ComputeDumpTest(unittest.TestCase):
    def command(self, kind, layer=0, tile=0):
        return runtime._command_header(kind, 32, layer=layer, tile=tile) + bytes(16)

    def test_all_compute_types_include_accumulators_but_not_config_or_dma(self):
        kinds = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 31, 34]
        commands = [self.command(k) for k in kinds]
        planned = plan_compute_dumps(commands, 0, len(commands))
        self.assertEqual([p['type'] for p in planned], list(range(6, 21)) + [28, 31, 34])
        self.assertTrue(all(p['command'] == p['completion_command'] for p in planned))

    def test_async_capture_occurs_at_matching_completion(self):
        for submit in (29, 32):
            commands = [self.command(submit, 7, 2), self.command(24), self.command(30, 7, 2)]
            self.assertEqual(plan_compute_dumps(commands, 0, 3)[0]['completion_command'], 3)
            for changed in (commands[:-1], [commands[0], self.command(10), commands[-1]],
                            [commands[0], self.command(30, 7, 3)]):
                with self.assertRaises(ValueError):
                    plan_compute_dumps(changed, 0, len(changed))

    def test_bounded_selection_keeps_each_layer_and_exact_budget(self):
        commands = [self.command(10, layer, tile) for layer in range(100) for tile in range(4)]
        planned = plan_compute_dumps(commands, 0, len(commands))
        selected = select_compute_dumps(planned, 100)
        self.assertEqual(len(selected), 100)
        self.assertEqual([v['tile_id'] for v in selected], [3] * 100)
        self.assertEqual([v['layer_id'] for v in selected], list(range(100)))
        self.assertEqual(len(select_compute_dumps(planned, 125)), 125)
        self.assertEqual(len({v['command'] for v in select_compute_dumps(planned, 125)}), 125)
        self.assertEqual(select_compute_dumps(planned, 500), planned)
        for limit in (0, -1, True, 99):
            with self.assertRaises(ValueError):
                select_compute_dumps(planned, limit)

    def test_zlib_full_memory_and_flushed_manifest_are_verified_not_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            planned = plan_compute_dumps([self.command(10)], 0, 1)
            writer = ComputeDumpWriter(directory, planned, 512)
            memory = bytes(range(256)) * 2
            writer.capture(1, memory)
            entry = json.loads(writer.path.read_text())
            self.assertEqual(zlib.decompress(Path(entry['path']).read_bytes()), memory)
            self.assertEqual(entry['numerical_status'], 'NOT_COMPARED')
            self.assertEqual(entry['integrity'], 'VERIFIED')
            self.assertEqual(writer.summary(complete=True)['captured'], 1)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                writer.capture(1, memory)
            with self.assertRaises(FileExistsError):
                ComputeDumpWriter(directory, planned, 512)

    def test_missing_or_bad_size_captures_do_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ComputeDumpWriter(directory, plan_compute_dumps([self.command(8)], 0, 1), 512)
            with self.assertRaisesRegex(ValueError, 'Missing'):
                writer.summary(complete=True)
            with self.assertRaisesRegex(ValueError, 'size'):
                writer.capture(1, bytes(511))
            self.assertEqual(writer.summary()['captured'], 0)

    def test_case_scope_identities_tensor_archive_and_plan_are_pinned(self):
        self.assertEqual(validate_compute_dump_case({}, [], checked_read), [])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tensor_file = folder / 'reference.npz'
            tensor_file.write_bytes(b'pinned tensor archive')
            case = dict(full_graph=True, start_mode='reset', first_command=0, end_command=1)
            case.update({k: k for k in ('model', 'source', 'firmware', 'input', 'reference_build')})
            commands = [self.command(9)]
            manifest = {k: case[k] for k in ('model', 'source', 'firmware', 'input', 'reference_build')}
            manifest.update(reference_tensors=artifact(tensor_file), compute_checkpoints=plan_compute_dumps(commands, 0, 1))
            path = folder / 'manifest.json'
            path.write_text(json.dumps(manifest))
            case['compute_dump_reference'] = artifact(path)
            self.assertEqual(len(validate_compute_dump_case(case, commands, checked_read)), 1)
            manifest['compute_checkpoint_limit'] = 1
            path.write_text(json.dumps(manifest))
            case['compute_dump_reference'] = artifact(path)
            self.assertEqual(len(validate_compute_dump_case(case, commands, checked_read)), 1)
            for field, value in (('full_graph', False), ('start_mode', 'snapshot'), ('input', 'changed')):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_compute_dump_case({**case, field: value}, commands, checked_read)
            changed = copy.deepcopy(manifest)
            changed['compute_checkpoints'][0]['command_sha256'] = 'changed'
            path.write_text(json.dumps(changed))
            case['compute_dump_reference'] = artifact(path)
            with self.assertRaisesRegex(ValueError, 'plan'):
                validate_compute_dump_case(case, commands, checked_read)
            path.write_text(json.dumps(manifest))
            case['compute_dump_reference'] = artifact(path)
            tensor_file.write_bytes(b'corrupt archive')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                validate_compute_dump_case(case, commands, checked_read)

    def test_layer_snapshots_skip_pending_dma_and_end_is_saved_separately(self):
        load = bytearray(runtime._dma_1d(runtime._ref(1), runtime._ref(6), 32, 0, 0))
        load[0] = 24
        commands = [bytes(load), self.command(27, 1), self.command(1, 2)]
        end = self.command(0)
        model = runtime._package(b''.join(commands) + end, b'', b'', 3, 1024, 0, 0)
        self.assertEqual(safe_layer_boundaries(model, commands), [2])


if __name__ == '__main__':
    unittest.main()
