"""Benchmark for configure() idempotency (Step 4)."""

import tempfile

import comfy_runtime
from comfy_runtime import config as cfg_mod
from benchmarks._harness import run_block


def run():
    results = {}

    tmp = tempfile.mkdtemp(prefix="bench_configure_")

    # Cold configure: reset snapshot each time so the full path runs.
    def cold():
        cfg_mod._LAST_CONFIG = None
        comfy_runtime.configure(models_dir=tmp)

    results["configure.cold"] = run_block(
        "configure.cold", cold, warmup=3, iters=50
    )

    # Steady-state: identical kwargs; should hit the snapshot short-circuit.
    # Ensure _LAST_CONFIG is set once, then hammer configure().
    comfy_runtime.configure(models_dir=tmp)

    def warm():
        comfy_runtime.configure(models_dir=tmp)

    results["configure.steady"] = run_block(
        "configure.steady", warm, warmup=20, iters=5000
    )

    return results
