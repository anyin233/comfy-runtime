"""Benchmark for load_nodes_from_path caching (Step 5)."""

import os
import tempfile
import textwrap

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime import registry
from comfy_runtime.compat import nodes as compat_nodes
from benchmarks._harness import run_block


NODE_SRC = textwrap.dedent(
    """
    class _BenchProbeNode:
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
        RETURN_TYPES = ()
        def run(self):
            return ()

    NODE_CLASS_MAPPINGS = {"_BenchProbeNode": _BenchProbeNode}
    """
)


def run():
    tmp = tempfile.mkdtemp(prefix="bench_loadnodes_")
    path = os.path.join(tmp, "probe_bench.py")
    with open(path, "w") as f:
        f.write(NODE_SRC)

    # Cold: clear cache each iteration to force full exec_module.
    def cold():
        registry._LOAD_CACHE.clear()
        compat_nodes.NODE_CLASS_MAPPINGS.pop("_BenchProbeNode", None)
        registry.load_nodes_from_path(path)

    results = {
        "load_nodes.cold": run_block(
            "load_nodes.cold", cold, warmup=3, iters=100
        )
    }

    # Warm: single prime call, then repeated lookups should hit cache.
    registry.load_nodes_from_path(path)

    def warm():
        registry.load_nodes_from_path(path)

    results["load_nodes.warm"] = run_block(
        "load_nodes.warm", warm, warmup=20, iters=5000
    )
    return results
