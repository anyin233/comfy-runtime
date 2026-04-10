"""Tests for timing helpers."""

import pytest

from benchmarks.e2e._harness.timing import NodeRecorder, StageRecorder


def test_node_recorder_records_single_call():
    rec = NodeRecorder()
    rec.record("KSampler", 1_000_000)
    records = rec.as_list()
    assert len(records) == 1
    assert records[0].class_type == "KSampler"
    assert records[0].call_index == 0
    assert records[0].elapsed_ns == 1_000_000


def test_node_recorder_assigns_incrementing_call_index():
    rec = NodeRecorder()
    rec.record("CLIPTextEncode", 100)
    rec.record("CLIPTextEncode", 200)
    rec.record("KSampler", 5_000)
    rec.record("CLIPTextEncode", 150)
    records = rec.as_list()
    clip_records = [r for r in records if r.class_type == "CLIPTextEncode"]
    assert [r.call_index for r in clip_records] == [0, 1, 2]
    assert [r.elapsed_ns for r in clip_records] == [100, 200, 150]
    ksampler_records = [r for r in records if r.class_type == "KSampler"]
    assert [r.call_index for r in ksampler_records] == [0]


def test_stage_recorder_maps_class_type_to_stage():
    mapping = {
        "model_load": ["CheckpointLoaderSimple"],
        "text_encode": ["CLIPTextEncode"],
        "sample": ["KSampler"],
    }
    rec = StageRecorder(mapping)
    rec.record("CheckpointLoaderSimple", 500)
    rec.record("CLIPTextEncode", 100)
    rec.record("CLIPTextEncode", 200)
    rec.record("KSampler", 1000)

    stages = {s.name: s for s in rec.as_list()}
    assert stages["model_load"].elapsed_ns == 500
    assert stages["model_load"].node_count == 1
    assert stages["text_encode"].elapsed_ns == 300
    assert stages["text_encode"].node_count == 2
    assert stages["sample"].elapsed_ns == 1000
    assert stages["sample"].node_count == 1


def test_stage_recorder_unknown_class_type_goes_to_other():
    mapping = {"sample": ["KSampler"]}
    rec = StageRecorder(mapping)
    rec.record("KSampler", 100)
    rec.record("SomeUnmappedNode", 50)
    stages = {s.name: s for s in rec.as_list()}
    assert stages["sample"].elapsed_ns == 100
    assert stages["other"].elapsed_ns == 50
    assert stages["other"].node_count == 1


def test_stage_recorder_preserves_mapping_order():
    mapping = {
        "model_load": ["CheckpointLoaderSimple"],
        "sample": ["KSampler"],
        "decode": ["VAEDecode"],
    }
    rec = StageRecorder(mapping)
    # Record in reverse order to confirm output follows mapping order, not
    # insertion order.
    rec.record("VAEDecode", 30)
    rec.record("KSampler", 20)
    rec.record("CheckpointLoaderSimple", 10)
    stage_names = [s.name for s in rec.as_list()]
    assert stage_names[:3] == ["model_load", "sample", "decode"]
