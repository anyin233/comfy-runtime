"""Timing and comparison primitives for benchmark scripts.

Kept deliberately minimal: no third-party dependencies, no decorators,
no session state. Every bench_*.py file composes these functions to
produce a dict of results that run_all.py aggregates to JSON.
"""

import statistics
import time
from typing import Callable, Dict


def run_block(
    name: str,
    fn: Callable[[], object],
    *,
    warmup: int = 3,
    iters: int = 100,
) -> Dict[str, object]:
    """Time *fn* over *iters* runs after *warmup* warmup runs.

    Args:
        name: Identifier recorded in the result dict.
        fn: Zero-argument callable to time.
        warmup: Number of untimed warmup invocations.
        iters: Number of timed invocations.

    Returns:
        dict with keys: name, iters, min_ns, median_ns, mean_ns, p95_ns, stddev_ns.
    """
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)

    samples_sorted = sorted(samples)
    p95_index = max(0, min(len(samples_sorted) - 1, int(round(0.95 * (len(samples_sorted) - 1)))))
    stddev = statistics.pstdev(samples) if len(samples) > 1 else 0

    return {
        "name": name,
        "iters": iters,
        "min_ns": samples_sorted[0],
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
        "p95_ns": samples_sorted[p95_index],
        "stddev_ns": stddev,
    }


def compare(before: Dict[str, dict], after: Dict[str, dict]) -> str:
    """Build a markdown table comparing two sets of benchmark results.

    Benches present only in one side are included with a placeholder
    in the missing column.
    """
    all_names = sorted(set(before.keys()) | set(after.keys()))
    lines = [
        "| Benchmark | Before (median ns) | After (median ns) | Ratio | Delta |",
        "|-----------|--------------------|-------------------|-------|-------|",
    ]
    for name in all_names:
        b = before.get(name)
        a = after.get(name)
        if b is None:
            lines.append(
                f"| {name} | — | {a['median_ns']:.0f} | new | new |"
            )
            continue
        if a is None:
            lines.append(
                f"| {name} | {b['median_ns']:.0f} | — | removed | removed |"
            )
            continue
        ratio = a["median_ns"] / b["median_ns"] if b["median_ns"] else float("inf")
        pct = (ratio - 1.0) * 100.0
        lines.append(
            f"| {name} | {b['median_ns']:.0f} | {a['median_ns']:.0f} | {ratio:.2f}x | {pct:+.1f}% |"
        )
    return "\n".join(lines) + "\n"
