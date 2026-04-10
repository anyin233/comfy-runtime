"""Run every bench_*.py module under benchmarks/ and dump results to JSON.

Usage:
    python -m benchmarks.run_all --output benchmarks/results/baseline.json
    python -m benchmarks.run_all --compare benchmarks/results/baseline.json \\
        --output benchmarks/results/after.json
"""

import argparse
import importlib
import json
import os
import pkgutil
import sys
from typing import Dict

import benchmarks
from benchmarks._harness import compare


def _discover_bench_modules():
    """Return sorted list of bench_* module names under the benchmarks package."""
    names = []
    for mod_info in pkgutil.iter_modules(benchmarks.__path__):
        if mod_info.name.startswith("bench_"):
            names.append(mod_info.name)
    return sorted(names)


def _run_bench_module(name: str) -> Dict[str, dict]:
    """Import benchmarks.<name> and call its run() function."""
    mod = importlib.import_module(f"benchmarks.{name}")
    if not hasattr(mod, "run"):
        print(f"[skip] {name}: no run() function defined", file=sys.stderr)
        return {}
    results = mod.run()
    if not isinstance(results, dict):
        raise TypeError(
            f"{name}.run() must return dict[str, dict], got {type(results)}"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Write results to this JSON path.",
    )
    parser.add_argument(
        "--compare",
        help="Path to a prior baseline JSON to diff against.",
    )
    args = parser.parse_args()

    all_results: Dict[str, dict] = {}
    modules = _discover_bench_modules()
    if not modules:
        print("[info] no benches registered", file=sys.stderr)
    for name in modules:
        print(f"[run ] {name}", file=sys.stderr)
        all_results.update(_run_bench_module(name))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[wrote] {args.output}", file=sys.stderr)

    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        md = compare(baseline, all_results)
        print(md)


if __name__ == "__main__":
    main()
