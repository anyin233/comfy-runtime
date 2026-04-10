"""Tests for the statistical aggregator."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks.e2e._harness.result_schema import (
    NodeRecord,
    RunResult,
    StageRecord,
)
from benchmarks.e2e.aggregate import (
    compute_stats,
    group_by_workflow_side,
    load_results,
)


def _make_result(workflow, side, run_idx, total_ns, status="ok", nodes=None, stages=None):
    return RunResult(
        workflow=workflow,
        side=side,
        run_idx=run_idx,
        status=status,
        error=None,
        total_ns=total_ns,
        stages=stages or [StageRecord(name="total", elapsed_ns=total_ns, node_count=1)],
        nodes=nodes or [NodeRecord(class_type="FakeNode", call_index=0, elapsed_ns=total_ns)],
        gpu_peak_allocated_bytes=1_000_000 * (run_idx + 1),
        gpu_peak_reserved_bytes=2_000_000 * (run_idx + 1),
        host_vmhwm_bytes=3_000_000 * (run_idx + 1),
        env={},
    )


def _write_fixtures(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    # 4 runs per (workflow, side): warmup=0 + three timed
    for workflow in ["sd15", "flux"]:
        for side, base_ns in (("runtime", 1_000_000_000), ("comfyui", 1_500_000_000)):
            for run_idx, bias in [(0, 9_000_000_000), (1, 0), (2, 50_000_000), (3, -50_000_000)]:
                # run_idx=0 is warmup, intentionally much slower so we can
                # confirm it is discarded.
                total = base_ns + bias
                result = _make_result(workflow, side, run_idx, total)
                (runs_dir / f"{workflow}_{side}_{run_idx}.json").write_text(
                    json.dumps(asdict(result))
                )
    return runs_dir


def test_load_results_reads_all_json(tmp_path):
    runs_dir = _write_fixtures(tmp_path)
    loaded = load_results(runs_dir)
    assert len(loaded) == 2 * 2 * 4


def test_group_by_workflow_side(tmp_path):
    runs_dir = _write_fixtures(tmp_path)
    groups = group_by_workflow_side(load_results(runs_dir))
    assert set(groups.keys()) == {("sd15", "runtime"), ("sd15", "comfyui"),
                                  ("flux", "runtime"), ("flux", "comfyui")}
    for group in groups.values():
        assert len(group) == 4


def test_compute_stats_discards_warmup(tmp_path):
    runs_dir = _write_fixtures(tmp_path)
    groups = group_by_workflow_side(load_results(runs_dir))
    stats = compute_stats(groups[("sd15", "runtime")])
    # Mean of (1.0e9, 1.05e9, 0.95e9) ns = 1.0e9 ns
    assert stats["total_ns"]["mean"] == pytest.approx(1_000_000_000, rel=1e-6)
    assert stats["total_ns"]["min"] == 950_000_000
    assert stats["total_ns"]["max"] == 1_050_000_000
    # Warmup (10e9 ns) must not be in the dataset.
    assert stats["total_ns"]["max"] < 9_000_000_000


def test_compute_stats_includes_memory(tmp_path):
    runs_dir = _write_fixtures(tmp_path)
    groups = group_by_workflow_side(load_results(runs_dir))
    stats = compute_stats(groups[("sd15", "runtime")])
    assert "gpu_peak_allocated_bytes" in stats
    assert "host_vmhwm_bytes" in stats
    assert stats["gpu_peak_allocated_bytes"]["mean"] > 0


def test_compute_stats_includes_stages_and_nodes(tmp_path):
    runs_dir = _write_fixtures(tmp_path)
    groups = group_by_workflow_side(load_results(runs_dir))
    stats = compute_stats(groups[("sd15", "runtime")])
    assert "stages" in stats
    assert "nodes" in stats
    assert "total" in stats["stages"]


# --- Rendering tests ---

from benchmarks.e2e.aggregate import render_report


def test_render_report_writes_readme_and_workflow_pages(tmp_path):
    runs_dir = _write_fixtures(tmp_path)
    docs_root = tmp_path / "docs" / "benchmarks"

    render_report(
        runs_dir=runs_dir,
        docs_root=docs_root,
        copy_raw_data=False,
        render_figures=False,
    )

    readme = docs_root / "README.md"
    assert readme.exists()
    content = readme.read_text()
    assert "sd15" in content
    assert "flux" in content
    assert "runtime" in content.lower() and "comfyui" in content.lower()

    wf_page = docs_root / "workflows" / "sd15.md"
    assert wf_page.exists()
    wf_content = wf_page.read_text()
    assert "stage" in wf_content.lower()


def test_render_report_copies_raw_data(tmp_path):
    runs_dir = _write_fixtures(tmp_path)
    docs_root = tmp_path / "docs" / "benchmarks"
    render_report(
        runs_dir=runs_dir,
        docs_root=docs_root,
        copy_raw_data=True,
        render_figures=False,
    )
    data_dir = docs_root / "data"
    assert data_dir.exists()
    assert len(list(data_dir.glob("*.json"))) == 2 * 2 * 4
