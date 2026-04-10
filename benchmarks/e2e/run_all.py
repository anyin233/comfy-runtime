"""Top-level benchmark driver.

Dispatches one subprocess per (workflow, side, run_idx) combination, strictly
serially. Subprocess failures do not abort the batch — they land as a failed
JSON for aggregation to surface.

Usage:
    python run_all.py
    python run_all.py --workflow sd15_text_to_image
    python run_all.py --runs 5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# NOTE: flux2_klein_text_to_image is excluded by default because of a
# pre-existing matmul shape error in the vendored Flux2 sampler that
# reproduces even from the workflow's own .venv in the parent repo. Pass
# --workflow flux2_klein_text_to_image explicitly to attempt it.
ALL_WORKFLOWS = [
    "sd15_text_to_image",
    "img2img",
    "inpainting",
    "hires_fix",
    "area_composition",
    "esrgan_upscale",
]

DEFAULT_RUNS_PER_SIDE = 4   # 1 warmup + 3 timed


@dataclass(frozen=True)
class RunEntry:
    """One (workflow, side, run_idx) scheduling entry."""

    workflow: str
    side: str
    run_idx: int


def plan_runs(workflows: list[str], runs_per_side: int) -> Iterator[RunEntry]:
    """Yield scheduling entries in (workflow, side, run_idx) order.

    Args:
        workflows: List of workflow names to schedule.
        runs_per_side: Number of runs per side (runtime and comfyui).

    Yields:
        RunEntry for each (workflow, side, run_idx) combination.
    """
    for wf in workflows:
        for side in ("runtime", "comfyui"):
            for r in range(runs_per_side):
                yield RunEntry(workflow=wf, side=side, run_idx=r)


def _clean_env() -> dict[str, str]:
    """Build a curated environment for each subprocess.

    Fixes CUDA_VISIBLE_DEVICES / PYTHONHASHSEED / CUBLAS_WORKSPACE_CONFIG
    so re-runs produce comparable numbers. Forwards HF caches so model
    downloads are not repeated.

    Returns:
        Dictionary of environment variables for subprocess invocation.
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for k in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME", "LD_LIBRARY_PATH"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def _runner_for_side(bench_root: Path, side: str) -> tuple[Path, Path]:
    """Return the (python_bin, runner_script) pair for a given side.

    Args:
        bench_root: Root directory of the benchmark suite.
        side: Either "runtime" or "comfyui".

    Returns:
        Tuple of (python_binary_path, runner_script_path).
    """
    if side == "runtime":
        python_bin = bench_root / "runtime-env" / ".venv" / "bin" / "python"
        runner = bench_root / "runners" / "runtime_runner.py"
    else:
        python_bin = bench_root / "comfyui-env" / ".venv" / "bin" / "python"
        runner = bench_root / "runners" / "comfyui_runner.py"
    return python_bin, runner


def run_all(
    bench_root: Path,
    workflows: list[str],
    runs_per_side: int,
    out_root: Path,
) -> None:
    """Run every (workflow, side, run_idx) once, collecting JSONs under out_root.

    Subprocess failures are logged but do not abort the batch — they land as
    failed JSON entries for aggregation to surface later.

    Args:
        bench_root: Root directory of the benchmark suite.
        workflows: List of workflow names to run.
        runs_per_side: Number of runs per (workflow, side) pair.
        out_root: Directory where result JSONs are written.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    env = _clean_env()

    entries = list(plan_runs(workflows, runs_per_side))
    total = len(entries)
    for i, entry in enumerate(entries, start=1):
        out_json = out_root / f"{entry.workflow}_{entry.side}_{entry.run_idx}.json"
        python_bin, runner = _runner_for_side(bench_root, entry.side)

        print(
            f"[run_all] [{i}/{total}] {entry.workflow} / {entry.side} / run {entry.run_idx}",
            flush=True,
        )
        result = subprocess.run(
            [str(python_bin), str(runner),
             "--workflow", entry.workflow,
             "--run-idx", str(entry.run_idx),
             "--out", str(out_json)],
            env=env,
        )
        if result.returncode != 0:
            print(f"[run_all]   -> returncode={result.returncode} (continuing)", flush=True)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark driver.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--workflow", action="append", dest="workflows",
                   help="Run only this workflow (repeatable).")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_SIDE,
                   help=f"Runs per side (default: {DEFAULT_RUNS_PER_SIDE} = 1 warmup + 3 timed)")
    p.add_argument("--out-root", type=Path, default=None,
                   help="Override output directory (default: results/{timestamp}/)")
    return p.parse_args()


def main() -> None:
    """CLI entry point: parse args and dispatch the full benchmark run."""
    args = _parse_args()
    bench_root = Path(__file__).resolve().parent
    workflows = args.workflows or ALL_WORKFLOWS
    if args.out_root is None:
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_root = bench_root / "results" / ts
    else:
        out_root = args.out_root

    print(f"[run_all] writing results to {out_root}", flush=True)
    run_all(
        bench_root=bench_root,
        workflows=workflows,
        runs_per_side=args.runs,
        out_root=out_root,
    )

    # Maintain a stable "latest" symlink for aggregate.py.
    latest = bench_root / "results" / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_root.name)
        print(f"[run_all] updated {latest} -> {out_root.name}", flush=True)
    except OSError as exc:
        print(f"[run_all] warn: could not update latest symlink: {exc}", flush=True)


if __name__ == "__main__":
    main()
