"""Run a pinned full model with per-command PMU and an exact hybrid-reference output check."""

import os

import cocotb

from test_yolo_reference_replay import run_reference_case


@cocotb.test()
async def test_model_reference_replay(dut):
    await run_reference_case(dut, os.environ["MODEL_REFERENCE_CASE"])
