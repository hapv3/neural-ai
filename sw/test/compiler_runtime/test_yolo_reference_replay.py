"""Shared verification for pinned full models and YOLO intermediate tensors."""

import json
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer
from cocotbext.axi import AxiLiteBus, AxiLiteMaster
import numpy as np

from yolo_reference_replay import (
    checked_read, execution_model, load_case, output_comparison, runtime, save_checkpoint_result,
)
from test_yolo_stem_byte_exact import compare_bytes, extract_c32
from yolo_c2f_checkpoints import extract_checkpoint
from diagnostic_checkpoints import load_diagnostic_checkpoints


async def run_reference_case(dut, case_path):
    case_path = Path(case_path).resolve()
    folder = case_path.parent
    if (folder / "byte_exact.json").exists():
        raise ValueError("Refusing to overwrite a replay result")
    case, full_model, memory, temporary, output = load_case(case_path)
    if Path(os.environ["NAI_REPLAY_MANIFEST"]).resolve() != Path(case["pinned_manifest"]):
        raise ValueError("Firmware replay manifest differs from the prepared case")
    if runtime._runtime_firmware_path().read_bytes() != checked_read(case["firmware"]):
        raise ValueError("Firmware override differs from prepared case")
    first, end = case["first_command"], case["end_command"]
    preamble = case["restore_preamble"]
    execution_first = first - 1 if preamble is not None else first
    model = execution_model(case, full_model)
    invocation_base = case.get("invocation_base", 0x81400000)
    binding_table_base = invocation_base + 0x1000
    assert 0x81000000 + len(model) <= invocation_base, "Model overlaps invocation/control data"
    (folder / "replay.nai").write_bytes(model)
    full_graph = case.get("full_graph", False)
    remaining_graph = case.get("remaining_graph", False)
    verify_final = full_graph or remaining_graph
    report = {"first_command": first, "end_command": end, "reference": case["reference"],
              "model": case["model"], "source": case["source"], "firmware": case["firmware"],
              "input": case["input"], "full_graph": full_graph,
              "remaining_graph": remaining_graph,
              "remaining_prefix_end_command": case.get("remaining_prefix_end_command"),
              "fail_fast": case.get("fail_fast", False),
              "cascade_coverage": case.get("cascade_coverage"),
              "post_cascade_coverage": case.get("post_cascade_coverage"),
              "c2f_coverage": case.get("c2f_coverage"),
              "c2f_output_coverage": case.get("c2f_output_coverage"),
              "downsample_coverage": case.get("downsample_coverage"),
              "resumed_pointwise_coverage": case.get("resumed_pointwise_coverage"),
              "resumed_c2f_coverage": case.get("resumed_c2f_coverage"),
              "resumed_output_coverage": case.get("resumed_output_coverage"),
              "l2_downsample_coverage": case.get("l2_downsample_coverage"),
              "initial_snapshot_checks": case.get("initial_snapshot_checks", []),
              "snapshot_evidence": case.get("snapshot_evidence"),
              "scope": ("Snapshot-to-output continuation plus listed prefix tensors; not a reset regression"
                        if remaining_graph else
                        "Full output binding plus listed intermediate tensors against the hybrid reference"
                        if full_graph else "Only listed intermediate tensors; not full-model output correctness"),
              "status": "running", "checkpoints": [], "restore_preamble": preamble,
              "diagnostic_checkpoints": [], "diagnostic_manifest": case.get("diagnostic_checkpoints"),
              "start_mode": case.get("start_mode", "snapshot")}

    def persist():
        (folder / "byte_exact.json").write_text(json.dumps(report, indent=2) + "\n")

    persist()
    cocotb.start_soon(Clock(dut.clk_i, 1, unit="ns").start())
    axi = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s_axi"), dut.clk_i, dut.rst_ni, reset_active_level=False)
    await runtime.reset_dut(dut)
    await runtime.hold_reset(dut)
    runtime.write_tcdm_bytes(dut, memory)
    await runtime.write_l2_bytes(dut, 0x80000000, checked_read(case["input"]))
    await runtime.write_l2_bytes(dut, 0x80200000, temporary)
    await runtime.write_l2_bytes(dut, 0x80100000, output)
    if runtime.read_tcdm_bytes(dut) != memory:
        raise AssertionError("Snapshot TCDM readback differs")
    for address, expected in ((0x80200000, temporary), (0x80100000, output)):
        if bytes(await runtime.read_l2_bytes(dut, address, len(expected))) != expected:
            raise AssertionError("Snapshot L2 readback differs")
    await runtime.release_reset(dut)
    if preamble is not None:
        dut._log.info("Replaying original command %d as a verified idempotent LUT restore preamble",
                      preamble["command"])
    bindings = [(1, 0, 0x80000000, case["input_bytes"]), (2, 0, 0x80100000, case["output_bytes"]),
                (3, 0, 0x80200000, case["temporary_bytes"])]
    invocation, addresses = runtime.build_invocation_with_bindings(
        model, bindings, model_base=0x81000000, binding_table_base=binding_table_base)
    for address, data in ((0x81000000, model), (binding_table_base, addresses), (invocation_base, invocation)):
        await runtime.write_l2_bytes(dut, address, data)
    records = runtime._selected_command_trace_records(full_model, execution_first, end)
    checkpoints = {check["command"]: check for check in case["checkpoints"]}
    commands, _ = runtime._logical_command_records(full_model)
    diagnostics = {check["command"]: check for check in
                   load_diagnostic_checkpoints(case, commands, len(memory), checked_read)}
    before_compute = {check.get("submit_command", check["command"]) - 1:
                      check.get("submit_command", check["command"]) for check in case["checkpoints"]
                      if check["tensor_role"] in ("consumer_conv", "pointwise_conv", "block_conv1", "block_conv2",
                                                  "block_output_conv", "downsample_conv", "resumed_pointwise_conv",
                                                  "resumed_output_conv", "l2_downsample_conv") or
                      (check["tensor_role"].startswith("resumed_c2f_") and
                       check["tensor_role"].endswith(("conv1", "conv2")))}

    async def capture(index, counters):
        record = records[index]
        if record["type"] == 0:
            return
        number = record["command"]
        if preamble is not None and number == preamble["command"]:
            runtime._write_command_pmu_csv(folder / "restore_pmu.csv", [record], [counters], append=False)
            restored = runtime.read_tcdm_bytes(dut)
            address, length = preamble["destination"], preamble["bytes"]
            actual = restored[address:address + length]
            expected = memory[address:address + length]
            (folder / "restore-actual.bin").write_bytes(actual)
            report["restore_check"] = compare_bytes(np.frombuffer(actual, np.int8),
                                                    np.frombuffer(expected, np.int8))
            persist()
            assert actual == expected, "Firmware LUT preamble changed saved output bytes"
            dut._log.info("LUT restore preamble byte-exact: %d physical bytes", length)
            return
        runtime._write_command_pmu_csv(folder / "command_pmu.csv", [record], [counters],
                                       append=number != first + 1)
        if number in before_compute:
            (folder / f"before-command-{before_compute[number]}-tcdm.bin").write_bytes(runtime.read_tcdm_bytes(dut))
        l2_check = case.get("cascade_l2_output")
        if l2_check is not None and l2_check.get("command") == number:
            # The PMU callback runs in ReadOnly. Leave that phase before driving
            # the existing backdoor address; no RTL/firmware pause is introduced.
            await Timer(1, unit="ps")
            last_row = l2_check["rows"] - 1
            span = (last_row // l2_check["tile_cols"] * l2_check["row_stride"] +
                    last_row % l2_check["tile_cols"] * 32 + l2_check["channels"])
            retained = bytes(await runtime.read_l2_bytes(dut, 0x80200000 + l2_check["address"], span))
            actual = extract_checkpoint(retained, {**l2_check, "address": 0})
            expected = np.frombuffer(checked_read(l2_check["expected"]), np.int8).reshape(actual.shape)
            (folder / "cascade-l2-actual.bin").write_bytes(actual.tobytes())
            report["cascade_l2_output"] = {**l2_check, **compare_bytes(actual, expected)}
            persist()
            dut._log.info("NEURAL-AI LIVE CASCADE L2 command %d: %d/%d mismatches", number,
                          report["cascade_l2_output"]["mismatches"], report["cascade_l2_output"]["bytes"])
            if report["fail_fast"] and report["cascade_l2_output"]["mismatches"]:
                report.update(status="FAIL", failed_command=number, error="Live cascade L2 output mismatch")
                persist()
                raise AssertionError(report["error"])
        diagnostic = number in diagnostics
        if number not in checkpoints and not diagnostic:
            return
        check = diagnostics[number] if diagnostic else checkpoints[number]
        actual = extract_checkpoint(runtime.read_tcdm_bytes(dut), check)
        stats = save_checkpoint_result(folder, report, check, actual, report["fail_fast"],
                                       "diagnostic_checkpoints" if diagnostic else "checkpoints")
        dut._log.info("NEURAL-AI BYTE-EXACT command %d tensor %d: %d/%d mismatches, first=%s max_abs=%d",
                      number, check["tensor"], stats["mismatches"], stats["bytes"],
                      stats["first_mismatch"], stats["max_abs"])
        if report["fail_fast"] and stats["mismatches"]:
            # A monitor callback can fail independently of the main coroutine;
            # save diagnostic memory and FAIL before raising from that task.
            (folder / "failure-tcdm.bin").write_bytes(runtime.read_tcdm_bytes(dut))
            raise AssertionError(report["error"])

    try:
        counters, samples = await runtime._load_and_run(
            dut, axi, invocation, timeout_cycles=case.get("timeout_cycles", 1000000),
            invocation_base=invocation_base, model=model,
            command_trace_records=records, command_trace_callback=capture)
    except Exception as error:
        if report["status"] != "FAIL":
            report.update(status="ERROR", error=str(error))
        (folder / "failure-tcdm.bin").write_bytes(runtime.read_tcdm_bytes(dut))
        persist()
        raise
    assert len(samples) == end - execution_first, "Missing per-command PMU samples"
    report["pmu"] = counters
    # Preserve complete restart state before asserting numerical correctness.
    snapshot_memory = runtime.read_tcdm_bytes(dut)
    snapshot_temporary = bytes(await runtime.read_l2_bytes(dut, 0x80200000, case["temporary_bytes"]))
    snapshot_output = bytes(await runtime.read_l2_bytes(dut, 0x80100000, case["output_bytes"]))
    snapshot_path = folder / f"command-{end}.snapshot"
    snapshot_path.write_bytes(runtime._encode_yolo320_snapshot(
        full_model, end, snapshot_memory, snapshot_temporary, snapshot_output))
    runtime._verify_yolo320_snapshot(full_model, snapshot_path.read_bytes(), expected_boundary=end,
                                    expected_tcdm=snapshot_memory, expected_temporary=snapshot_temporary,
                                    expected_output=snapshot_output)
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_STATUS) == runtime.NPU_CMD_STATUS_PASS
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_FAIL_CODE) == 0
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_FAIL_PTR) == 0
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_DONE_COUNT) == end - execution_first
    if preamble is not None:
        assert report.get("restore_check", {}).get("mismatches") == 0
    assert len(report["checkpoints"]) == len(checkpoints)
    failures = [check for check in report["checkpoints"] if check["mismatches"]]
    if case.get("cascade_l2_output") is not None and "command" not in case["cascade_l2_output"]:
        check = case["cascade_l2_output"]
        actual = extract_c32(snapshot_temporary, check["address"], check["rows"], check["channels"],
                             check["row_stride"], check["tile_cols"])
        expected = np.frombuffer(checked_read(check["expected"]), np.int8).reshape(actual.shape)
        (folder / "cascade-l2-actual.bin").write_bytes(actual.tobytes())
        report["cascade_l2_output"] = {**check, **compare_bytes(actual, expected)}
        dut._log.info("NEURAL-AI CASCADE L2 OUTPUT: %d/%d mismatches",
                      report["cascade_l2_output"]["mismatches"], report["cascade_l2_output"]["bytes"])
    if case.get("cascade_l2_output") is not None:
        assert "cascade_l2_output" in report, "Missing live L2 comparison"
        if report["cascade_l2_output"]["mismatches"]:
            failures.append(report["cascade_l2_output"])
    if case.get("resumed_output_l2") is not None:
        check = case["resumed_output_l2"]
        actual = extract_checkpoint(snapshot_temporary, check)
        expected = np.frombuffer(checked_read(check["expected"]), np.int8).reshape(actual.shape)
        (folder / "resident-l2-actual.bin").write_bytes(actual.tobytes())
        report["resumed_output_l2"] = {**check, **compare_bytes(actual, expected)}
        if report["resumed_output_l2"]["mismatches"]:
            failures.append(report["resumed_output_l2"])
        dut._log.info("NEURAL-AI RESIDENT L2 OUTPUT: %d/%d mismatches",
                      report["resumed_output_l2"]["mismatches"], report["resumed_output_l2"]["bytes"])
    if verify_final:
        (folder / "output-actual.bin").write_bytes(snapshot_output)
        report["final_output"] = {**case["final_output"],
                                  **output_comparison(case["final_output"], snapshot_output)}
        if report["final_output"]["mismatches"]:
            failures.append(report["final_output"])
        dut._log.info("NEURAL-AI FULL OUTPUT: %d/%d mismatches, first=%s max_abs=%d",
                      report["final_output"]["mismatches"], report["final_output"]["bytes"],
                      report["final_output"]["first_mismatch"], report["final_output"]["max_abs"])
    report["snapshot"] = str(snapshot_path)
    if {check["command"] for check in report["diagnostic_checkpoints"]} != set(diagnostics):
        report.update(status="ERROR", error="Missing diagnostic Conv observations")
        persist()
        raise AssertionError(report["error"])
    failures.extend(check for check in report["diagnostic_checkpoints"] if check["mismatches"])
    report["status"] = "FAIL" if failures else "PASS"
    persist()
    assert not failures, f"First mismatching Neural-AI checkpoint: {failures[0] if failures else None}"


@cocotb.test()
async def test_yolo_reference_replay(dut):
    await run_reference_case(dut, os.environ["YOLO_REFERENCE_CASE"])
