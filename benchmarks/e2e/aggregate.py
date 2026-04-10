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
# Rendering helpers (Jinja2)
# ---------------------------------------------------------------------------

import datetime as _dt
import shutil


def _ns_to_ms(v: float) -> float:
    return v / 1_000_000.0


def _bytes_to_mb(v: float) -> float:
    return v / (1024 * 1024)


def _fmt_ms(v: float) -> str:
    return f"{_ns_to_ms(v):.1f}"


def _fmt_mb(v: float) -> str:
    return f"{_bytes_to_mb(v):.1f}"


def _fmt_pct(a: float, b: float) -> str:
    """Return (a - b) / b * 100% formatted (negative = a is smaller than b)."""
    if b == 0:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%"


def _build_summary_rows(stats_by_key: dict, workflows: list[str]) -> list[dict]:
    rows = []
    for wf in workflows:
        r = stats_by_key.get((wf, "runtime"), {})
        c = stats_by_key.get((wf, "comfyui"), {})
        r_total = r.get("total_ns", {}).get("mean", 0)
        c_total = c.get("total_ns", {}).get("mean", 0)
        speedup = f"{c_total / r_total:.2f}x" if r_total else "n/a"
        rows.append(
            {
                "workflow": wf,
                "runtime_total_ms": _fmt_ms(r_total),
                "comfyui_total_ms": _fmt_ms(c_total),
                "speedup": speedup,
                "runtime_gpu_mb": _fmt_mb(
                    r.get("gpu_peak_allocated_bytes", {}).get("mean", 0)
                ),
                "comfyui_gpu_mb": _fmt_mb(
                    c.get("gpu_peak_allocated_bytes", {}).get("mean", 0)
                ),
                "runtime_vmhwm_mb": _fmt_mb(
                    r.get("host_vmhwm_bytes", {}).get("mean", 0)
                ),
                "comfyui_vmhwm_mb": _fmt_mb(
                    c.get("host_vmhwm_bytes", {}).get("mean", 0)
                ),
            }
        )
    return rows


def _build_workflow_context(
    workflow: str,
    runtime_stats: dict,
    comfyui_stats: dict,
    raw_files: list[str],
) -> dict:
    """Build the Jinja2 context dict for one per-workflow page."""
    stage_names = list(runtime_stats.get("stages", {}).keys())
    for n in comfyui_stats.get("stages", {}):
        if n not in stage_names:
            stage_names.append(n)

    stage_rows = []
    for name in stage_names:
        r = runtime_stats.get("stages", {}).get(name, {})
        c = comfyui_stats.get("stages", {}).get(name, {})
        stage_rows.append(
            {
                "name": name,
                "runtime_min_ms": _fmt_ms(r.get("min", 0)),
                "runtime_mean_ms": _fmt_ms(r.get("mean", 0)),
                "runtime_median_ms": _fmt_ms(r.get("median", 0)),
                "runtime_stddev_ms": _fmt_ms(r.get("stddev", 0)),
                "comfyui_min_ms": _fmt_ms(c.get("min", 0)),
                "comfyui_mean_ms": _fmt_ms(c.get("mean", 0)),
                "comfyui_median_ms": _fmt_ms(c.get("median", 0)),
                "comfyui_stddev_ms": _fmt_ms(c.get("stddev", 0)),
                "delta_mean_pct": _fmt_pct(r.get("mean", 0), c.get("mean", 0)),
            }
        )

    r_total = runtime_stats.get("total_ns", {})
    c_total = comfyui_stats.get("total_ns", {})
    total = {
        "runtime_min_ms": _fmt_ms(r_total.get("min", 0)),
        "runtime_mean_ms": _fmt_ms(r_total.get("mean", 0)),
        "runtime_median_ms": _fmt_ms(r_total.get("median", 0)),
        "runtime_stddev_ms": _fmt_ms(r_total.get("stddev", 0)),
        "comfyui_min_ms": _fmt_ms(c_total.get("min", 0)),
        "comfyui_mean_ms": _fmt_ms(c_total.get("mean", 0)),
        "comfyui_median_ms": _fmt_ms(c_total.get("median", 0)),
        "comfyui_stddev_ms": _fmt_ms(c_total.get("stddev", 0)),
        "delta_mean_pct": _fmt_pct(r_total.get("mean", 0), c_total.get("mean", 0)),
    }

    mem = {
        "runtime_alloc_mb": _fmt_mb(
            runtime_stats.get("gpu_peak_allocated_bytes", {}).get("mean", 0)
        ),
        "comfyui_alloc_mb": _fmt_mb(
            comfyui_stats.get("gpu_peak_allocated_bytes", {}).get("mean", 0)
        ),
        "runtime_reserved_mb": _fmt_mb(
            runtime_stats.get("gpu_peak_reserved_bytes", {}).get("mean", 0)
        ),
        "comfyui_reserved_mb": _fmt_mb(
            comfyui_stats.get("gpu_peak_reserved_bytes", {}).get("mean", 0)
        ),
        "runtime_vmhwm_mb": _fmt_mb(
            runtime_stats.get("host_vmhwm_bytes", {}).get("mean", 0)
        ),
        "comfyui_vmhwm_mb": _fmt_mb(
            comfyui_stats.get("host_vmhwm_bytes", {}).get("mean", 0)
        ),
        "delta_alloc_pct": _fmt_pct(
            runtime_stats.get("gpu_peak_allocated_bytes", {}).get("mean", 0),
            comfyui_stats.get("gpu_peak_allocated_bytes", {}).get("mean", 0),
        ),
        "delta_reserved_pct": _fmt_pct(
            runtime_stats.get("gpu_peak_reserved_bytes", {}).get("mean", 0),
            comfyui_stats.get("gpu_peak_reserved_bytes", {}).get("mean", 0),
        ),
        "delta_vmhwm_pct": _fmt_pct(
            runtime_stats.get("host_vmhwm_bytes", {}).get("mean", 0),
            comfyui_stats.get("host_vmhwm_bytes", {}).get("mean", 0),
        ),
    }

    node_keys = list(runtime_stats.get("nodes", {}).keys())
    for k in comfyui_stats.get("nodes", {}):
        if k not in node_keys:
            node_keys.append(k)

    node_rows = []
    for key in node_keys:
        class_type, call_index = key.rsplit("#", 1)
        r = runtime_stats.get("nodes", {}).get(key, {})
        c = comfyui_stats.get("nodes", {}).get(key, {})
        node_rows.append(
            {
                "class_type": class_type,
                "call_index": call_index,
                "runtime_mean_ms": _fmt_ms(r.get("mean", 0)),
                "comfyui_mean_ms": _fmt_ms(c.get("mean", 0)),
                "delta_pct": _fmt_pct(r.get("mean", 0), c.get("mean", 0)),
            }
        )

    return {
        "workflow": workflow,
        "stage_rows": stage_rows,
        "total": total,
        "mem": mem,
        "node_rows": node_rows,
        "raw_data_files": raw_files,
    }


# ---------------------------------------------------------------------------
# Figures (matplotlib)
# ---------------------------------------------------------------------------


def _render_figures(
    stats_by_key: dict,
    workflows: list[str],
    figures_dir: Path,
) -> None:
    """Render the e2e, stage-breakdown, and memory figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_means = [
        stats_by_key.get((wf, "runtime"), {}).get("total_ns", {}).get("mean", 0) / 1e6
        for wf in workflows
    ]
    runtime_stds = [
        stats_by_key.get((wf, "runtime"), {}).get("total_ns", {}).get("stddev", 0) / 1e6
        for wf in workflows
    ]
    comfyui_means = [
        stats_by_key.get((wf, "comfyui"), {}).get("total_ns", {}).get("mean", 0) / 1e6
        for wf in workflows
    ]
    comfyui_stds = [
        stats_by_key.get((wf, "comfyui"), {}).get("total_ns", {}).get("stddev", 0) / 1e6
        for wf in workflows
    ]

    x = list(range(len(workflows)))
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        [xi - bar_w / 2 for xi in x],
        runtime_means,
        bar_w,
        yerr=runtime_stds,
        label="comfy_runtime",
        capsize=4,
    )
    ax.bar(
        [xi + bar_w / 2 for xi in x],
        comfyui_means,
        bar_w,
        yerr=comfyui_stds,
        label="ComfyUI",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(workflows, rotation=30, ha="right")
    ax.set_ylabel("End-to-end time (ms)")
    ax.set_title("End-to-End Wall Time — comfy_runtime vs ComfyUI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "e2e_comparison.png", dpi=150)
    plt.close(fig)

    for wf in workflows:
        r = stats_by_key.get((wf, "runtime"), {})
        c = stats_by_key.get((wf, "comfyui"), {})
        stage_names: list[str] = []
        for s in r.get("stages", {}):
            if s not in stage_names:
                stage_names.append(s)
        for s in c.get("stages", {}):
            if s not in stage_names:
                stage_names.append(s)

        runtime_stage_ms = [
            r.get("stages", {}).get(s, {}).get("mean", 0) / 1e6 for s in stage_names
        ]
        comfyui_stage_ms = [
            c.get("stages", {}).get(s, {}).get("mean", 0) / 1e6 for s in stage_names
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        bottoms_r = 0.0
        bottoms_c = 0.0
        cmap = plt.get_cmap("tab10")
        for i, stage_name in enumerate(stage_names):
            ax.bar(
                0,
                runtime_stage_ms[i],
                bottom=bottoms_r,
                color=cmap(i),
                label=stage_name,
            )
            ax.bar(1, comfyui_stage_ms[i], bottom=bottoms_c, color=cmap(i))
            bottoms_r += runtime_stage_ms[i]
            bottoms_c += comfyui_stage_ms[i]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["comfy_runtime", "ComfyUI"])
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{wf} — stage breakdown")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"stage_breakdown_{wf}.png", dpi=150)
        plt.close(fig)

    metrics = [
        ("gpu_peak_allocated_bytes", "GPU allocated"),
        ("gpu_peak_reserved_bytes", "GPU reserved"),
        ("host_vmhwm_bytes", "Host VmHWM"),
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    n_workflows = len(workflows)
    group_w = 0.9
    bar_w = group_w / (len(metrics) * 2)

    for mi, (key, label) in enumerate(metrics):
        runtime_mb = [
            stats_by_key.get((wf, "runtime"), {}).get(key, {}).get("mean", 0)
            / (1024 * 1024)
            for wf in workflows
        ]
        comfyui_mb = [
            stats_by_key.get((wf, "comfyui"), {}).get(key, {}).get("mean", 0)
            / (1024 * 1024)
            for wf in workflows
        ]
        r_offset = (mi * 2) * bar_w - group_w / 2 + bar_w / 2
        c_offset = (mi * 2 + 1) * bar_w - group_w / 2 + bar_w / 2
        ax.bar(
            [i + r_offset for i in range(n_workflows)],
            runtime_mb,
            bar_w,
            label=f"{label} (runtime)",
        )
        ax.bar(
            [i + c_offset for i in range(n_workflows)],
            comfyui_mb,
            bar_w,
            label=f"{label} (ComfyUI)",
        )

    ax.set_xticks(list(range(n_workflows)))
    ax.set_xticklabels(workflows, rotation=30, ha="right")
    ax.set_ylabel("Memory (MB)")
    ax.set_yscale("log")
    ax.set_title("Memory Usage — comfy_runtime vs ComfyUI")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "memory_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level entry: render_report()
# ---------------------------------------------------------------------------


def render_report(
    runs_dir: Path,
    docs_root: Path,
    copy_raw_data: bool = True,
    render_figures: bool = True,
) -> None:
    """End-to-end rendering: load -> stats -> templates -> write files."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    templates_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
    )

    results = load_results(runs_dir)
    groups = group_by_workflow_side(results)
    stats_by_key = {key: compute_stats(group) for key, group in groups.items()}

    workflows = sorted({key[0] for key in stats_by_key.keys()})

    any_env = next(iter(stats_by_key.values()), {}).get("env", {})
    timed_runs_counts = [s.get("timed_runs", 0) for s in stats_by_key.values()]
    summary_rows = _build_summary_rows(stats_by_key, workflows)

    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "workflows").mkdir(exist_ok=True)
    if copy_raw_data:
        (docs_root / "data").mkdir(exist_ok=True)
    (docs_root / "figures").mkdir(exist_ok=True)

    readme_tpl = env.get_template("readme.md.j2")
    readme_out = readme_tpl.render(
        generated_at=_dt.datetime.utcnow().isoformat() + "Z",
        env=any_env,
        workflows=workflows,
        summary_rows=summary_rows,
        timed_runs=min(timed_runs_counts) if timed_runs_counts else 0,
    )
    (docs_root / "README.md").write_text(readme_out)

    workflow_tpl = env.get_template("workflow.md.j2")
    for wf in workflows:
        runtime_stats = stats_by_key.get((wf, "runtime"), {})
        comfyui_stats = stats_by_key.get((wf, "comfyui"), {})
        raw_files = sorted(p.name for p in runs_dir.glob(f"{wf}_*.json"))
        ctx = _build_workflow_context(wf, runtime_stats, comfyui_stats, raw_files)
        (docs_root / "workflows" / f"{wf}.md").write_text(
            workflow_tpl.render(**ctx)
        )

    if render_figures:
        _render_figures(stats_by_key, workflows, docs_root / "figures")

    if copy_raw_data:
        for json_file in runs_dir.glob("*.json"):
            shutil.copy2(json_file, docs_root / "data" / json_file.name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir", type=Path, help="benchmarks/e2e/results/<timestamp>/")
    p.add_argument(
        "--docs-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "benchmarks",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    render_report(runs_dir=args.runs_dir, docs_root=args.docs_root)
    print(f"[aggregate] wrote {args.docs_root}")


if __name__ == "__main__":
    main()
