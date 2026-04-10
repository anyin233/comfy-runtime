"""Benchmarks for cheap execute_node calls.

Measures per-call dispatch overhead isolated from real model work.
Targets Steps 1 (introspection memoize) and 3 (V1 instance pool).
"""

import torch

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime import executor
from benchmarks._harness import run_block


def _empty_latent_image():
    executor.execute_node("EmptyLatentImage", width=512, height=512, batch_size=1)


def _list_nodes():
    executor.list_nodes()


_COND_FIXTURE = None


def _conditioning_zero_out():
    global _COND_FIXTURE
    if _COND_FIXTURE is None:
        _COND_FIXTURE = [[torch.randn(1, 77, 768), {}]]
    executor.execute_node("ConditioningZeroOut", conditioning=_COND_FIXTURE)


def run():
    results = {}
    results["execute_node.EmptyLatentImage"] = run_block(
        "execute_node.EmptyLatentImage", _empty_latent_image, warmup=20, iters=2000
    )
    results["list_nodes"] = run_block(
        "list_nodes", _list_nodes, warmup=20, iters=5000
    )
    results["execute_node.ConditioningZeroOut"] = run_block(
        "execute_node.ConditioningZeroOut",
        _conditioning_zero_out,
        warmup=20,
        iters=1000,
    )
    return results
