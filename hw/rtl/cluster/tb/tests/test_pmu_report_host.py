"""Host-only serialization, precision and verdict checks for operator PMU."""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from npu_test_utils import PMU_COUNTER_NAMES
from pmu_report import write_pmu_report


class PMUReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.counters = {name: index for index, name in enumerate(PMU_COUNTER_NAMES)}
        self.counters.update(num_counters=len(PMU_COUNTER_NAMES), overflow_status=0,
                             cycle=1234, pmu_version=2, current_context=0, current_phase=6, filter_context=0)

    def write(self, status="PASS", counters=None, **kwargs):
        return write_pmu_report(self.counters if counters is None else counters,
                                "test_copy", PMU_COUNTER_NAMES, status=status,
                                output_dir=self.folder, **kwargs)

    def load(self):
        payload = json.loads((self.folder / "test_copy_pmu.json").read_text())
        with (self.folder / "test_copy_pmu.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        return payload, rows[0]

    def test_all_163_counters_and_metadata_roundtrip(self):
        before = dict(self.counters)
        self.write()
        payload, row = self.load()
        self.assertEqual(len(PMU_COUNTER_NAMES), 163)
        self.assertEqual(payload["counters"], before)
        self.assertEqual(self.counters, before)
        self.assertTrue(payload["full_counter_set"])
        self.assertTrue(payload["performance_valid"])
        self.assertEqual(payload["captured_counter_count"], 163)
        self.assertEqual(payload["missing_counters"], [])
        for name, value in before.items():
            self.assertEqual(int(row[name]), value)

    def test_uint64_precision_and_absolute_peaks(self):
        self.counters.update(cycle=(1 << 64) - 1, axi_read_latency_max=2048,
                             axi_read_outstanding_peak=16)
        self.write()
        payload, row = self.load()
        for name in ("cycle", "axi_read_latency_max", "axi_read_outstanding_peak"):
            self.assertEqual(payload["counters"][name], self.counters[name])
            self.assertEqual(int(row[name]), self.counters[name])

    def test_missing_counter_is_not_fabricated_as_zero(self):
        del self.counters["cycle"]
        self.write()
        payload, row = self.load()
        self.assertNotIn("cycle", payload["counters"])
        self.assertEqual(row["cycle"], "")
        self.assertEqual(payload["missing_counters"], ["cycle"])
        self.assertFalse(payload["performance_valid"])

    def test_unknown_hardware_counter_count_is_incomplete(self):
        self.counters["num_counters"] = 164
        self.write()
        payload, _ = self.load()
        self.assertFalse(payload["full_counter_set"])
        self.assertFalse(payload["performance_valid"])

    def test_missing_overflow_metadata_does_not_imply_validity(self):
        del self.counters["overflow_status"]
        self.write()
        payload, _ = self.load()
        self.assertFalse(payload["performance_valid"])

    def test_missing_hardware_count_does_not_imply_validity(self):
        del self.counters["num_counters"]
        self.write()
        payload, _ = self.load()
        self.assertFalse(payload["performance_valid"])

    def test_wide_overflow_bitmap_is_preserved(self):
        self.counters["overflow_status"] = 1 << 162
        self.write()
        payload, row = self.load()
        self.assertFalse(payload["performance_valid"])
        self.assertEqual(payload["counters"]["overflow_status"], 1 << 162)
        self.assertEqual(int(row["overflow_status"]), 1 << 162)

    def test_irq_reached_is_not_numerical_pass(self):
        self.write(status="IRQ_REACHED")
        payload, _ = self.load()
        self.assertFalse(payload["performance_valid"])
        self.assertEqual(payload["status"], "IRQ_REACHED")

    def test_zero_cycle_snapshot_is_not_valid_performance(self):
        self.counters["cycle"] = 0
        self.write()
        payload, _ = self.load()
        self.assertFalse(payload["performance_valid"])

    def test_failure_and_timeout_preserve_counters(self):
        for status in ("FAIL", "TIMEOUT", "ERROR"):
            with self.subTest(status=status):
                self.write(status=status)
                payload, row = self.load()
                self.assertEqual(payload["status"], status)
                self.assertEqual(row["status"], status)
                self.assertEqual(payload["counters"], self.counters)
                self.assertFalse(payload["performance_valid"])

    def test_checker_failure_updates_preliminary_verdict(self):
        self.write(status="IRQ_REACHED")
        self.write(status="FAIL")
        payload, row = self.load()
        self.assertEqual(payload["status"], "FAIL")
        self.assertEqual(row["status"], "FAIL")

    def test_sim_build_output_directory(self):
        with patch.dict(os.environ, {"SIM_BUILD": str(self.folder)}, clear=True):
            paths = write_pmu_report(self.counters, "test_copy", PMU_COUNTER_NAMES, status="PASS")
        self.assertTrue(all(path.parent == self.folder for path in paths))

    def test_results_xml_output_directory_fallback(self):
        with patch.dict(os.environ, {"COCOTB_RESULTS_FILE": str(self.folder / "results.xml")}, clear=True):
            paths = write_pmu_report(self.counters, "test_copy", PMU_COUNTER_NAMES, status="PASS")
        self.assertTrue(all(path.parent == self.folder for path in paths))

    def test_unsafe_report_name_is_rejected_before_writing(self):
        for name in ("../escape", "/absolute", "bad/name", ".", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                write_pmu_report(self.counters, name, PMU_COUNTER_NAMES, status="PASS", output_dir=self.folder)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_invalid_values_and_status_are_rejected(self):
        for value in (-1, True, 1.5, 1 << 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.write(counters={"cycle": value})
        with self.assertRaises(ValueError):
            self.write(status="UNKNOWN")


if __name__ == "__main__":
    unittest.main()
