"""Tests for run_all.py orchestration logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.e2e import run_all


def test_plan_runs_produces_expected_count():
    entries = list(run_all.plan_runs(
        workflows=["a", "b"],
        runs_per_side=4,
    ))
    assert len(entries) == 2 * 2 * 4
    sides = {e.side for e in entries}
    assert sides == {"runtime", "comfyui"}
    run_indices = {e.run_idx for e in entries}
    assert run_indices == {0, 1, 2, 3}


def test_plan_runs_order_is_workflow_then_side_then_run():
    entries = list(run_all.plan_runs(
        workflows=["a", "b"],
        runs_per_side=2,
    ))
    # a/runtime/0, a/runtime/1, a/comfyui/0, a/comfyui/1, b/runtime/0, ...
    assert entries[0].workflow == "a"
    assert entries[0].side == "runtime"
    assert entries[0].run_idx == 0
    assert entries[1].workflow == "a"
    assert entries[1].side == "runtime"
    assert entries[1].run_idx == 1
    assert entries[2].workflow == "a"
    assert entries[2].side == "comfyui"
    assert entries[4].workflow == "b"


def _scaffold_bench_root(tmp_path: Path) -> Path:
    bench_root = tmp_path / "benchmarks" / "e2e"
    (bench_root / "runners").mkdir(parents=True)
    (bench_root / "runtime-env" / ".venv" / "bin").mkdir(parents=True)
    (bench_root / "comfyui-env" / ".venv" / "bin").mkdir(parents=True)
    (bench_root / "runtime-env" / ".venv" / "bin" / "python").touch()
    (bench_root / "comfyui-env" / ".venv" / "bin" / "python").touch()
    (bench_root / "runners" / "runtime_runner.py").touch()
    (bench_root / "runners" / "comfyui_runner.py").touch()
    return bench_root


def test_run_all_calls_subprocess_for_each_entry(tmp_path):
    bench_root = _scaffold_bench_root(tmp_path)

    with patch("benchmarks.e2e.run_all.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_all.run_all(
            bench_root=bench_root,
            workflows=["a"],
            runs_per_side=2,
            out_root=tmp_path / "results",
        )

    # 2 sides x 2 runs = 4 subprocess calls
    assert mock_run.call_count == 4


def test_run_all_continues_after_failure(tmp_path):
    bench_root = _scaffold_bench_root(tmp_path)

    with patch("benchmarks.e2e.run_all.subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0),
            MagicMock(returncode=0),
            MagicMock(returncode=0),
        ]
        run_all.run_all(
            bench_root=bench_root,
            workflows=["a"],
            runs_per_side=2,
            out_root=tmp_path / "results",
        )
    assert mock_run.call_count == 4   # driver did not abort
