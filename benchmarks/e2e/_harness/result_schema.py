"""Dataclasses defining the schema of one benchmark run's JSON output.

A single benchmark run writes a single JSON file that conforms to
:class:`RunResult`. Both sides (runtime and comfyui) emit the same schema so
the aggregator can process them uniformly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NodeRecord:
    """One node-level timing record."""

    class_type: str
    call_index: int       # 0 for the first call of this class_type, 1 for the second, ...
    elapsed_ns: int


@dataclass
class StageRecord:
    """One stage-level timing record (sum across all nodes in the stage)."""

    name: str
    elapsed_ns: int
    node_count: int


@dataclass
class RunResult:
    """One subprocess run of one workflow on one side.

    ``run_idx == 0`` is the warmup run and is discarded during aggregation.
    ``status == "failed"`` carries a human-readable ``error`` and zeroed
    measurements; aggregation surfaces failures as explicit cells in the
    report rather than dropping them.
    """

    workflow: str
    side: str             # "runtime" | "comfyui"
    run_idx: int
    status: str           # "ok" | "failed"
    error: str | None
    total_ns: int
    stages: list[StageRecord]
    nodes: list[NodeRecord]
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int
    host_vmhwm_bytes: int
    env: dict[str, Any] = field(default_factory=dict)


def run_result_to_dict(result: RunResult) -> dict[str, Any]:
    """Serialize a :class:`RunResult` to a plain dict (JSON-ready)."""
    return asdict(result)


def run_result_from_dict(data: dict[str, Any]) -> RunResult:
    """Deserialize a dict produced by :func:`run_result_to_dict`.

    Reconstructs the nested :class:`StageRecord` / :class:`NodeRecord`
    instances rather than leaving them as dicts, so equality comparisons
    round-trip cleanly.
    """
    stages = [StageRecord(**s) for s in data.get("stages", [])]
    nodes = [NodeRecord(**n) for n in data.get("nodes", [])]
    return RunResult(
        workflow=data["workflow"],
        side=data["side"],
        run_idx=data["run_idx"],
        status=data["status"],
        error=data.get("error"),
        total_ns=data["total_ns"],
        stages=stages,
        nodes=nodes,
        gpu_peak_allocated_bytes=data["gpu_peak_allocated_bytes"],
        gpu_peak_reserved_bytes=data["gpu_peak_reserved_bytes"],
        host_vmhwm_bytes=data["host_vmhwm_bytes"],
        env=data.get("env", {}),
    )
