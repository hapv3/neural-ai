"""Replay a short saved YOLO cascade against precomputed Neural-AI tensors."""

import json
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotbext.axi import AxiLiteBus, AxiLiteMaster
import numpy as np
from npu_test_utils import _axi_write32

from yolo_reference_replay import checked_read, load_case, runtime
from test_yolo_stem_byte_exact import compare_bytes, extract_c32


@cocotb.test()
async def test_yolo_reference_replay(dut):
    case_path = Path(os.environ["YOLO_REFERENCE_CASE"]).resolve()
    folder = case_path.parent
    if (folder / "byte_exact.json").exists():
        raise ValueError("Refusing to overwrite a replay result")
    case, full_model, memory, temporary, output = load_case(case_path)
    if Path(os.environ["NAI_REPLAY_MANIFEST"]).resolve() != Path(case["pinned_manifest"]):
        raise ValueError("Firmware replay manifest differs from the prepared case")
    if runtime._runtime_firmware_path().read_bytes() != checked_read(case["firmware"]):
        raise ValueError("Firmware override differs from prepared case")
    first, end = case["first_command"], case["end_command"]
    model = runtime._extract_selected_yolo320_command_range(full_model, first, end)
    (folder / "replay.nai").write_bytes(model)
    report = {"first_command": first, "end_command": end, "reference": case["reference"],
              "scope": "Only listed intermediate tensors; not full-model output correctness",
              "status": "running", "checkpoints": []}

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
    restore = case["afu_lut_restore"]
    if restore is not None:
        for index, value in enumerate(bytes.fromhex(restore["data_hex"])):
            await _axi_write32(axi, 0x20003000 + index * 4, value)
        report["afu_lut_restore"] = {key: value for key, value in restore.items() if key != "data_hex"}
        persist()
        dut._log.info("Restored 256 LUT entries from command %d before unchanged reuse command %d",
                      restore["source_command"], restore["reuse_command"])
    bindings = [(1, 0, 0x80000000, case["input_bytes"]), (2, 0, 0x80100000, case["output_bytes"]),
                (3, 0, 0x80200000, case["temporary_bytes"])]
    invocation, addresses = runtime.build_invocation_with_bindings(
        model, bindings, model_base=0x81000000, binding_table_base=0x81401000)
    for address, data in ((0x81000000, model), (0x81401000, addresses), (0x81400000, invocation)):
        await runtime.write_l2_bytes(dut, address, data)
    records = runtime._selected_command_trace_records(full_model, first, end)
    checkpoints = {check["command"]: check for check in case["checkpoints"]}
    before_compute = {check["command"] - 1: check["command"] for check in case["checkpoints"]
                      if check["tensor_role"] == "consumer_conv"}

    def capture(index, counters):
        record = records[index]
        if record["type"] == 0:
            return
        runtime._write_command_pmu_csv(folder / "command_pmu.csv", [record], [counters], append=index != 0)
        number = record["command"]
        if number in before_compute:
            (folder / f"before-command-{before_compute[number]}-tcdm.bin").write_bytes(runtime.read_tcdm_bytes(dut))
        if number not in checkpoints:
            return
        check = checkpoints[number]
        actual = extract_c32(runtime.read_tcdm_bytes(dut), check["address"], check["rows"], check["channels"],
                             check["row_stride"], check["tile_cols"])
        expected = np.frombuffer(checked_read(check["expected"]), np.int8).reshape(actual.shape)
        (folder / f"command-{number}-actual.bin").write_bytes(actual.tobytes())
        stats = compare_bytes(actual, expected)
        report["checkpoints"].append({"command": number, "tensor": check["tensor"], **stats})
        persist()
        dut._log.info("NEURAL-AI BYTE-EXACT command %d tensor %d: %d/%d mismatches, first=%s max_abs=%d",
                      number, check["tensor"], stats["mismatches"], stats["bytes"],
                      stats["first_mismatch"], stats["max_abs"])

    await runtime._load_and_run(dut, axi, invocation, timeout_cycles=1000000,
                               invocation_base=0x81400000, model=model,
                               command_trace_records=records, command_trace_callback=capture)
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
    assert await runtime._axi_read32(axi, runtime.NPU_CMD_DONE_COUNT) == end - first
    assert len(report["checkpoints"]) == len(checkpoints)
    failures = [check for check in report["checkpoints"] if check["mismatches"]]
    report["snapshot"] = str(snapshot_path)
    report["status"] = "FAIL" if failures else "PASS"
    persist()
    assert not failures, f"First mismatching Neural-AI checkpoint: {failures[0] if failures else None}"
