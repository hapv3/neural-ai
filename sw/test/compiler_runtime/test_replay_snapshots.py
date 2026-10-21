"""Offline tests for intermediate snapshot integrity, scope and clock control."""
import asyncio
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import yolo_reference_replay as replay
from replay_snapshots import (capture_with_paused_clock, observation_verdict, parse_snapshot_boundaries,
                              save_intermediate_snapshot, validate_snapshot_boundaries)


class IntermediateSnapshotTest(unittest.TestCase):
    def model(self, commands=None):
        rt = replay.runtime
        commands = commands or [rt._command_header(1, 32) + bytes(16)] * 3
        end = rt._command_header(0, 32) + bytes(16)
        return rt._package(b''.join(commands) + end, b'', b'', len(commands), 1024, 0, 0)

    def test_parse_and_reject_ambiguous_lists(self):
        self.assertEqual(parse_snapshot_boundaries('2733, 2804,5492'), [2733, 2804, 5492])
        for value in ('', '1,,2', '0,1', '-1,2', '2,1', '1,1', 'x', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_snapshot_boundaries(value)

    def test_boundaries_are_explicit_in_range_and_not_end(self):
        case = dict(first_command=0, end_command=3, snapshot_boundaries=[1, 2])
        self.assertEqual(validate_snapshot_boundaries(case, self.model(), replay.runtime), [1, 2])
        for values in (None, [True], [1, 1], [2, 1], [0], [3], [4], ['1']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_snapshot_boundaries({**case, 'snapshot_boundaries': values}, self.model(), replay.runtime)

    def test_pending_dma_and_systolic_snapshots_are_rejected(self):
        rt = replay.runtime
        load = bytearray(rt._dma_1d(rt._ref(1), rt._ref(6), 32, 0, 0))
        struct.pack_into('<H', load, 0, 24)
        submit = rt._command_header(29, 160) + bytes(144)
        for command in (bytes(load), submit):
            model = self.model([command, rt._command_header(1, 32) + bytes(16)])
            with self.subTest(kind=command[0]), self.assertRaisesRegex(ValueError, 'not restart-safe'):
                validate_snapshot_boundaries(dict(first_command=0, end_command=2,
                                                   snapshot_boundaries=[1]), model, rt)

    def test_observation_scope_does_not_claim_full_output_pass(self):
        planned = [dict(command=1), dict(command=2)]
        report = dict(status='FAIL', checkpoints=[dict(command=1, bytes=32, mismatches=0)],
                      diagnostic_checkpoints=[dict(command=2, bytes=64, mismatches=0)])
        verdict = observation_verdict(planned, report, 1)
        self.assertEqual((verdict['status'], verdict['comparisons'], verdict['bytes']), ('PASS_LISTED_CHECKS', 1, 32))
        report['diagnostic_checkpoints'][0]['mismatches'] = 1
        self.assertEqual(observation_verdict(planned, report, 2)['status'], 'FAIL_LISTED_CHECKS')
        self.assertEqual(observation_verdict([], {}, 1)['status'], 'UNVERIFIED')

    def test_missing_duplicate_or_invalid_observations_are_rejected(self):
        planned = [dict(command=1)]
        for seen in ([], [dict(command=2, bytes=32, mismatches=0)],
                     [dict(command=1, bytes=32, mismatches=-1)],
                     [dict(command=1, bytes=32, mismatches=0)] * 2):
            with self.subTest(seen=seen), self.assertRaises(ValueError):
                observation_verdict(planned, dict(checkpoints=seen), 2)
        with self.assertRaises(ValueError):
            observation_verdict(planned * 2, {}, 2)

    def fixture(self, folder):
        model = self.model()
        case = dict(first_command=0, end_command=3, temporary_bytes=64, output_bytes=32,
                    snapshot_boundaries=[1, 2], reference_build={'build': 'pinned'})
        for key, data in (('model', model), ('source', b'tflite'), ('firmware', b'elf'), ('input', bytes(32))):
            path = folder / key
            path.write_bytes(data)
            case[key] = replay.artifact(path)
        path = folder / 'case.json'
        path.write_text(json.dumps(case))
        state = (bytes(replay.runtime.TCDM_TOTAL_BYTES), bytes(range(64)), bytes(range(32)))
        return path, case, model, state

    def test_full_memory_and_immutable_sidecar_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path, case, model, state = self.fixture(Path(directory))
            record = save_intermediate_snapshot(path, case, model, {}, [], 1, state, replay.runtime)
            encoded = replay.checked_read(record['snapshot'])
            boundary, *decoded = replay.runtime._verify_yolo320_snapshot(model, encoded)
            self.assertEqual(boundary, 1)
            self.assertEqual(tuple(decoded), state)
            sidecar = json.loads(replay.checked_read(record['evidence']))
            self.assertEqual(sidecar['model'], case['model'])
            self.assertEqual(sidecar['integrity'], 'VERIFIED')
            self.assertEqual(sidecar['observations']['status'], 'UNVERIFIED')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                save_intermediate_snapshot(path, case, model, {}, [], 1, state, replay.runtime)

    def test_bad_memory_sizes_and_changed_identity_do_not_publish_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path, case, model, state = self.fixture(folder)
            for changed in ((state[0][:-1], *state[1:]), (state[0], b'', state[2])):
                with self.assertRaisesRegex(ValueError, 'memory sizes'):
                    save_intermediate_snapshot(path, case, model, {}, [], 1, changed, replay.runtime)
            changed = copy.deepcopy(case)
            changed['model']['sha256'] = 'other'
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                save_intermediate_snapshot(path, changed, model, {}, [], 1, state, replay.runtime)
            self.assertFalse(list(folder.glob('*.snapshot*')))

    def test_sidecar_preserves_prefix_evidence_independently_of_final_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, case, model, state = self.fixture(Path(directory))
            seen = dict(command=1, bytes=32, mismatches=0, actual_sha256='pinned actual bytes')
            report = dict(status='FAIL', checkpoints=[seen], final_output=dict(mismatches=7))
            record = save_intermediate_snapshot(path, case, model, report, [dict(command=1)],
                                                1, state, replay.runtime)
            metadata = json.loads(replay.checked_read(record['evidence']))
            self.assertEqual(metadata['observations']['status'], 'PASS_LISTED_CHECKS')
            self.assertEqual(metadata['comparison_evidence'], [seen])
            report['checkpoints'][0]['mismatches'] = 1
            self.assertEqual(json.loads(replay.checked_read(record['evidence']))['comparison_evidence'][0]['mismatches'], 0)


class PausedCaptureTest(unittest.IsolatedAsyncioTestCase):
    async def test_clock_is_paused_and_resumes_at_held_level(self):
        for level in (0, 1):
            events = []
            class Clock:
                def stop(self): events.append('stop')
                def start(self, start_high): events.append(('start', start_high))
            async def capture():
                events.append('read')
                await asyncio.sleep(0)
                events.append('save')
                return 'snapshot'
            self.assertEqual(await capture_with_paused_clock(Clock(), level, capture), 'snapshot')
            self.assertEqual(events, ['stop', 'read', 'save', ('start', bool(level))])

    async def test_capture_error_still_resumes_clock(self):
        events = []
        class Clock:
            def stop(self): events.append('stop')
            def start(self, start_high): events.append(('start', start_high))
        async def capture():
            raise ValueError('disk or backdoor error')
        with self.assertRaises(ValueError):
            await capture_with_paused_clock(Clock(), 1, capture)
        self.assertEqual(events, ['stop', ('start', True)])


if __name__ == '__main__':
    unittest.main()
