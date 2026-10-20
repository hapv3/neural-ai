"""Source mapping, finalized outputs and complete C32 Conv coverage."""

import struct
import unittest

from prepare_conv_diagnostics import plan_conv_observations
from yolo_reference_replay import runtime


class ConvObservationPlannerTest(unittest.TestCase):
    def fixture(self):
        qparams = b"".join(struct.pack("<iiIiii2I", channel, 1, 0, 0, -128, 127, 0, 0)
                           for channel in range(64))
        source = dict(tensor=7, shape=[1, 1, 2, 64], kernel=[1, 1], bias_signature=list(range(32)))
        commands = []
        for group in range(2):
            commands.append(struct.pack("<HHIII4I", 5, 32, 0, 5, 0, group * 32, 32, group, 0))
            command = bytearray(runtime._pointwise_c32(runtime._ref(1), runtime._ref(6),
                runtime._ref(6, offset=512), runtime._ref(6, offset=128 + group * 64),
                2, 2, 1, group, 64, 64, 0))
            struct.pack_into("<I", command, 8, 5)
            commands.append(bytes(command))
        return commands, qparams, [source]

    def plan(self, fixture):
        commands, qparams, sources = fixture
        return plan_conv_observations(commands, qparams, sources, 0, len(commands))

    def test_unique_bias_mapping_covers_all_output_groups(self):
        checks, mapping = self.plan(self.fixture())
        self.assertEqual(mapping[5]["tensor"], 7)
        self.assertEqual([c["command"] for c in checks], [2, 4])
        self.assertEqual([c["reference_first_channel"] for c in checks], [0, 32])
        self.assertEqual(sum(c["rows"] * c["channels"] for c in checks), 128)

    def test_missing_and_ambiguous_source_mapping_are_rejected(self):
        for mutation in ("missing", "ambiguous", "no_first_rq"):
            fixture = self.fixture()
            if mutation == "missing": fixture[2][0]["bias_signature"][0] = -1
            elif mutation == "ambiguous": fixture[2].append(dict(fixture[2][0], tensor=8))
            else: del fixture[0][0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.plan(fixture)

    def test_rq_table_truncation_and_source_identity_change_are_rejected(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(ValueError, "RQ table"):
            self.plan((fixture[0], fixture[1][:16], fixture[2]))
        commands, qparams, sources = self.fixture()
        sources.append(dict(sources[0], tensor=8, bias_signature=list(range(100, 132))))
        qparams += b"".join(struct.pack("<iiIiii2I", channel, 1, 0, 0, -128, 127, 0, 0)
                            for channel in range(100, 132))
        commands.insert(2, struct.pack("<HHIII4I", 5, 32, 0, 5, 0, 64, 32, 0, 0))
        with self.assertRaisesRegex(ValueError, "source changes"):
            self.plan((commands, qparams, sources))

    def test_missing_duplicate_output_and_wrong_kernel_fail_closed(self):
        for mutation in ("missing", "duplicate", "kernel"):
            fixture = self.fixture()
            if mutation == "missing": fixture[0].pop()
            elif mutation == "duplicate": fixture[0].append(fixture[0][-1])
            else: fixture[2][0]["kernel"] = [3, 3]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.plan(fixture)

    def test_pointwise_external_compute_addresses_are_rejected(self):
        for offset in (24, 40):
            fixture = self.fixture()
            command = bytearray(fixture[0][1])
            struct.pack_into("<H", command, offset, 5)
            fixture[0][1] = bytes(command)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "local compute"):
                self.plan(fixture)

    def linebuffer_fixture(self, asynchronous=False):
        old, qparams, sources = self.fixture()
        sources[0]["kernel"] = [3, 3]
        commands = []
        for group in range(2):
            commands.append(old[group * 2])
            for mode in (1, 2):
                command = bytearray(160)
                kind = 29 if asynchronous and mode == 2 else 9
                struct.pack_into("<HHIII", command, 0, kind, 160, 0, 5, group)
                struct.pack_into("<2H", command, 52, 3, 3)
                struct.pack_into("<5I", command, 108, 128 + group * 64, 2, mode, 64, 2)
                commands.append(bytes(command))
                if kind == 29:
                    commands.append(runtime._command_header(30, 32, layer=5, tile=group) + bytes(16))
        return commands, qparams, sources

    def test_partial_accumulators_are_not_output_observations(self):
        checks, _ = self.plan(self.linebuffer_fixture())
        self.assertEqual([c["command"] for c in checks], [3, 6])
        self.assertEqual(len(checks), 2)

    def test_async_final_output_is_observed_only_at_matching_wait(self):
        checks, _ = self.plan(self.linebuffer_fixture(True))
        self.assertEqual([(c["submit_command"], c["command"]) for c in checks], [(3, 4), (7, 8)])
        for mutation in ("missing", "wrong_wait"):
            fixture = self.linebuffer_fixture(True)
            if mutation == "missing": fixture[0].pop()
            else: fixture[0][3] = runtime._command_header(30, 32, layer=5, tile=9) + bytes(16)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.plan(fixture)

    def test_linebuffer_kernel_and_output_extent_are_verified(self):
        for mutation in ("kernel", "rows"):
            fixture = self.linebuffer_fixture()
            command = bytearray(fixture[0][2])
            struct.pack_into("<I", command, 52 if mutation == "kernel" else 112, 1 if mutation == "kernel" else 3)
            fixture[0][2] = bytes(command)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.plan(fixture)


if __name__ == "__main__":
    unittest.main()
