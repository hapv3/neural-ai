"""Exercise operator-runner reporting with mocked hardware, never Verilator."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from npu_test_utils import PMU_COUNTER_NAMES
import test_spatz_operator_library as operators


class OperatorPMUTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.counters = dict.fromkeys(PMU_COUNTER_NAMES, 0)
        self.counters.update(cycle=1234, num_counters=163, overflow_status=0)
        self.dut = SimpleNamespace(clk_i=object(), rst_ni=object(), _log=Mock())

    async def invoke(self, checker, *, status_error=None, wait_error=None):
        self.wait = AsyncMock(return_value=self.counters, side_effect=wait_error)
        self.snapshot = Mock(return_value=self.counters)
        substitutes = {
            "Clock": Mock(), "AxiLiteBus": Mock(), "AxiLiteMaster": Mock(),
            "reset_dut": AsyncMock(), "load_firmware_elf_axi": AsyncMock(),
            "firmware_path": Mock(return_value="unused.elf"),
            "release_fetch": AsyncMock(), "wait_for_host_irq": self.wait,
            "pmu_direct_snapshot": self.snapshot,
            "check_status": Mock(side_effect=status_error),
            "read_dtcm_word": Mock(return_value=0),
            "axi_read32_or_none": AsyncMock(return_value=None),
        }
        with ExitStack() as stack:
            for name, value in substitutes.items():
                stack.enter_context(patch.object(operators, name, value))
            stack.enter_context(patch.object(operators.cocotb, "start_soon", Mock()))
            stack.enter_context(patch.dict(os.environ, {"SIM_BUILD": str(self.folder)}))
            await operators.run_firmware_case(self.dut, "unused.elf", "integration", 1, checker)

    def report(self):
        return json.loads((self.folder / "integration_pmu.json").read_text())

    async def test_success_uses_frozen_snapshot_without_extra_reads(self):
        checker = Mock()
        await self.invoke(checker)
        checker.assert_called_once_with(self.dut)
        self.wait.assert_awaited_once()
        self.snapshot.assert_not_called()
        self.assertEqual(self.report()["status"], "PASS")
        self.assertEqual(self.report()["counters"], self.counters)

    async def test_async_checker_is_awaited_before_pass(self):
        checker = AsyncMock()
        await self.invoke(checker)
        checker.assert_awaited_once_with(self.dut)
        self.assertEqual(self.report()["status"], "PASS")

    async def test_byte_mismatch_is_not_converted_to_pass(self):
        checker = Mock(side_effect=AssertionError("byte mismatch"))
        with self.assertRaisesRegex(AssertionError, "byte mismatch"):
            await self.invoke(checker)
        self.assertEqual(self.report()["status"], "FAIL")
        self.assertFalse(self.report()["performance_valid"])
        self.assertEqual(self.report()["counters"], self.counters)

    async def test_firmware_status_failure_does_not_run_checker(self):
        checker = Mock()
        with self.assertRaisesRegex(AssertionError, "firmware failed"):
            await self.invoke(checker, status_error=AssertionError("firmware failed"))
        checker.assert_not_called()
        self.assertEqual(self.report()["status"], "FAIL")

    async def test_timeout_retains_live_snapshot_before_diagnostics(self):
        checker = Mock()
        with self.assertRaisesRegex(AssertionError, "timeout waiting for host irq"):
            await self.invoke(checker, wait_error=AssertionError("timeout waiting for host irq"))
        checker.assert_not_called()
        self.snapshot.assert_called_once_with(self.dut)
        self.assertEqual(self.report()["status"], "TIMEOUT")
        self.assertEqual(self.report()["counters"], self.counters)


if __name__ == "__main__":
    unittest.main()
