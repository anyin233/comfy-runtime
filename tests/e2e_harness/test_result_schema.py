"""Tests for benchmark result schema dataclasses."""

import json

import pytest

from benchmarks.e2e._harness.result_schema import (
    NodeRecord,
    RunResult,
    StageRecord,
    run_result_from_dict,
    run_result_to_dict,
)


def test_node_record_fields():
    rec = NodeRecord(class_type="KSampler", call_index=0, elapsed_ns=12345)
    assert rec.class_type == "KSampler"
    assert rec.call_index == 0
    assert rec.elapsed_ns == 12345


def test_stage_record_fields():
    rec = StageRecord(name="sample", elapsed_ns=999, node_count=1)
    assert rec.name == "sample"
    assert rec.elapsed_ns == 999
    assert rec.node_count == 1


def test_run_result_round_trip():
    result = RunResult(
        workflow="sd15_text_to_image",
        side="runtime",
        run_idx=1,
        status="ok",
        error=None,
        total_ns=1_000_000_000,
        stages=[
            StageRecord(name="model_load", elapsed_ns=500_000_000, node_count=1),
            StageRecord(name="sample", elapsed_ns=400_000_000, node_count=1),
        ],
        nodes=[
            NodeRecord(class_type="CheckpointLoaderSimple", call_index=0, elapsed_ns=500_000_000),
            NodeRecord(class_type="KSampler", call_index=0, elapsed_ns=400_000_000),
        ],
        gpu_peak_allocated_bytes=1_000_000_000,
        gpu_peak_reserved_bytes=1_200_000_000,
        host_vmhwm_bytes=2_000_000_000,
        env={"gpu_name": "RTX 4090"},
    )

    as_dict = run_result_to_dict(result)
    # Should be JSON-serializable
    json_str = json.dumps(as_dict)
    restored = run_result_from_dict(json.loads(json_str))

    assert restored == result


def test_run_result_failure_status():
    result = RunResult(
        workflow="sd15_text_to_image",
        side="runtime",
        run_idx=0,
        status="failed",
        error="boom",
        total_ns=0,
        stages=[],
        nodes=[],
        gpu_peak_allocated_bytes=0,
        gpu_peak_reserved_bytes=0,
        host_vmhwm_bytes=0,
        env={},
    )
    assert result.status == "failed"
    assert result.error == "boom"
    # Round-trip still works for failures.
    restored = run_result_from_dict(run_result_to_dict(result))
    assert restored == result
