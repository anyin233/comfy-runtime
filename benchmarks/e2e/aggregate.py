"""Read benchmark result JSONs and compute summary statistics.

This module exposes the *core* of the aggregation pipeline:

* :func:`load_results` — read every ``*.json`` under a directory as
  :class:`~benchmarks.e2e._harness.result_schema.RunResult` instances.
* :func:`group_by_workflow_side` — bucket the results by (workflow, side).
* :func:`compute_stats` — for one (workflow, side) bucket, compute
  min/max/mean/median/stddev/p95 over total_ns, stage-level totals,
  per-node totals, and peak memory. Discards the warmup run
  (``run_idx == 0``) and any failed runs.

Markdown rendering (Jinja2 templates) and matplotlib figure generation
live in subsequent tasks (H2, H3); this file is the pure-data layer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.e2e._harness.result_schema import (
    RunResult,
    run_result_from_dict,
)

# ---------------------------------------------------------------------------
# Core: load + group + stats
# ---------------------------------------------------------------------------


def load_results(runs_dir: Path) -> list[RunResult]:
    """Load every ``*.json`` under runs_dir as a :class:`RunResult`.

    Args:
        runs_dir: Directory containing ``*.json`` result files.

    Returns:
        Sorted list of :class:`RunResult` instances (sorted by filename).
    """
    results: list[RunResult] = []
    for path in sorted(runs_dir.glob("*.json")):
        data = json.loads(path.read_text())
        results.append(run_result_from_dict(data))
    return results


def group_by_workflow_side(
    results: list[RunResult],
) -> dict[tuple[str, str], list[RunResult]]:
    """Bucket results by (workflow, side). Each bucket is sorted by run_idx.

    Args:
        results: Flat list of :class:`RunResult` instances.

    Returns:
        Dict keyed by ``(workflow, side)`` tuples; values are lists sorted by
        ``run_idx`` ascending.
    """
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.workflow, r.side), []).append(r)
    for v in groups.values():
        v.sort(key=lambda x: x.run_idx)
    return groups


def _stats_dict(values: list[int | float]) -> dict[str, float]:
    """Compute summary statistics over a list of numeric values.

    Args:
        values: Non-empty or possibly empty list of ints or floats.

    Returns:
        Dict with keys: min, max, mean, median, stddev, p95, count. All zeros
        when ``values`` is empty.
    """
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "stddev": 0,
            "p95": 0,
            "count": 0,
        }
    sorted_vals = sorted(values)
    # Use nearest-rank for p95; clamp index to valid range.
    p95_idx = max(0, min(len(sorted_vals) - 1, int(round(0.95 * (len(sorted_vals) - 1)))))
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        # Population stddev: we have the full sample for this group, not an
        # estimate of a larger population, so pstdev is more appropriate here.
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p95": sorted_vals[p95_idx],
        "count": len(values),
    }


def compute_stats(group: list[RunResult]) -> dict[str, Any]:
    """Compute summary stats for one (workflow, side) group.

    Discards warmup runs (``run_idx == 0``) and failed runs. Returns a dict
    keyed by metric with sub-dicts of {min, max, mean, median, stddev, p95,
    count}. Stage and node breakdowns are nested under ``"stages"`` /
    ``"nodes"``, indexed by stage name / ``f"{class_type}#{call_index}"``.

    Args:
        group: All :class:`RunResult` instances for a single (workflow, side)
            pair, including the warmup run.

    Returns:
        Nested stats dict. Top-level keys: total_ns, gpu_peak_allocated_bytes,
        gpu_peak_reserved_bytes, host_vmhwm_bytes, stages, nodes, failures,
        timed_runs, env.
    """
    # Discard warmup (run_idx==0) and any runs that did not succeed.
    timed_ok = [r for r in group if r.run_idx != 0 and r.status == "ok"]
    failures = [r for r in group if r.run_idx != 0 and r.status != "ok"]

    stats: dict[str, Any] = {
        "total_ns": _stats_dict([r.total_ns for r in timed_ok]),
        "gpu_peak_allocated_bytes": _stats_dict(
            [r.gpu_peak_allocated_bytes for r in timed_ok]
        ),
        "gpu_peak_reserved_bytes": _stats_dict(
            [r.gpu_peak_reserved_bytes for r in timed_ok]
        ),
        "host_vmhwm_bytes": _stats_dict([r.host_vmhwm_bytes for r in timed_ok]),
        "failures": len(failures),
        "timed_runs": len(timed_ok),
    }

    # Stage stats: preserve first-seen order across runs.
    stage_names: list[str] = []
    seen_stages: set[str] = set()
    for r in timed_ok:
        for s in r.stages:
            if s.name not in seen_stages:
                seen_stages.add(s.name)
                stage_names.append(s.name)
    stats["stages"] = {
        name: _stats_dict(
            [
                next((s.elapsed_ns for s in r.stages if s.name == name), 0)
                for r in timed_ok
            ]
        )
        for name in stage_names
    }

    # Per-node stats: keyed by (class_type, call_index) to handle repeated
    # nodes of the same class within a single workflow execution.
    node_keys: list[tuple[str, int]] = []
    seen_nodes: set[tuple[str, int]] = set()
    for r in timed_ok:
        for n in r.nodes:
            key = (n.class_type, n.call_index)
            if key not in seen_nodes:
                seen_nodes.add(key)
                node_keys.append(key)
    stats["nodes"] = {
        f"{ct}#{idx}": _stats_dict(
            [
                next(
                    (
                        n.elapsed_ns
                        for n in r.nodes
                        if n.class_type == ct and n.call_index == idx
                    ),
                    0,
                )
                for r in timed_ok
            ]
        )
        for (ct, idx) in node_keys
    }

    # Env snapshot from the first timed run; identical across runs in practice.
    stats["env"] = timed_ok[0].env if timed_ok else {}
    return stats


# ---------------------------------------------------------------------------
# CLI stub (rendering added in Task H2)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace with ``runs_dir`` and ``docs_root``.
    """
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir", type=Path, help="benchmarks/e2e/results/<timestamp>/")
    p.add_argument(
        "--docs-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "benchmarks",
    )
    return p.parse_args()


def main() -> None:
    """Entry point: load results, group, compute stats, print summary."""
    args = _parse_args()
    results = load_results(args.runs_dir)
    groups = group_by_workflow_side(results)
    stats = {key: compute_stats(group) for key, group in groups.items()}
    print(f"[aggregate] loaded {len(results)} results, {len(groups)} groups")
    for (wf, side), s in stats.items():
        print(
            f"  {wf}/{side}: mean_total_ns={s['total_ns']['mean']:.0f} "
            f"failures={s['failures']}"
        )


if __name__ == "__main__":
    main()
