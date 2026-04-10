"""Subprocess entrypoint — runs one workflow on the comfy_runtime side.

CLI: ``python runtime_runner.py --workflow NAME --run-idx N --out PATH``

Can also be imported and driven via :func:`run` for testing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Bootstrap sys.path so `benchmarks.e2e._harness.*` imports resolve when this
# module is launched as a subprocess from a venv that does not have the
# worktree root on PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from benchmarks.e2e._harness.env import gather_env
from benchmarks.e2e._harness.memory import read_gpu_peak, read_vmhwm, reset_gpu_peak
from benchmarks.e2e._harness.result_schema import (
    RunResult,
    run_result_to_dict,
)
from benchmarks.e2e._harness.timing import NodeRecorder, StageRecorder


REPO_ROOT = _REPO_ROOT


def _load_stages(stages_yaml: Path) -> dict[str, list[str]]:
    """Load the stage-to-node mapping from a stages.yaml file.

    Args:
        stages_yaml: Path to the YAML file containing a ``stages`` key whose
            value is a mapping from stage name to list of class_type strings.

    Returns:
        Dict mapping stage name to list of class_type strings.
    """
    data = yaml.safe_load(stages_yaml.read_text())
    return data["stages"]


def _try_cuda_sync() -> None:
    """Call ``torch.cuda.synchronize()`` if CUDA is available.

    Silently skips if torch is not installed — this keeps the runner usable
    in CPU-only test environments.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _install_instrumentation(
    stage_mapping: dict[str, list[str]],
) -> tuple[NodeRecorder, StageRecorder]:
    """Monkey-patch ``comfy_runtime.execute_node`` to record timings.

    Wraps the real (or fake, in tests) ``execute_node`` with a timer that
    calls :meth:`NodeRecorder.record` and :meth:`StageRecorder.record` after
    every invocation.  ``torch.cuda.synchronize()`` is called inside the
    ``finally`` block so elapsed time captures real GPU kernel completion.

    Args:
        stage_mapping: Mapping from stage name to list of class_type strings,
            forwarded verbatim to :class:`StageRecorder`.

    Returns:
        A ``(NodeRecorder, StageRecorder)`` pair that accumulate measurements
        as the workflow runs.
    """
    import comfy_runtime

    node_rec = NodeRecorder()
    stage_rec = StageRecorder(stage_mapping)
    original = comfy_runtime.execute_node

    def timed(class_type: str, **kwargs: Any):
        t0 = time.perf_counter_ns()
        try:
            result = original(class_type, **kwargs)
        finally:
            _try_cuda_sync()
            elapsed = time.perf_counter_ns() - t0
            node_rec.record(class_type, elapsed)
            stage_rec.record(class_type, elapsed)
        return result

    comfy_runtime.execute_node = timed  # type: ignore[assignment]
    return node_rec, stage_rec


def _load_workflow_module(workflow_main_py: Path):
    """Import a workflow's ``main.py`` by absolute path.

    Inserts the workflow directory at the front of ``sys.path`` so sibling
    packages (e.g. ``workflow_utils``) can be imported from within the
    workflow script.

    Args:
        workflow_main_py: Absolute path to the workflow's ``main.py``.

    Returns:
        The loaded module object.

    Raises:
        RuntimeError: If ``importlib.util.spec_from_file_location`` returns
            ``None`` (path does not exist or is not a Python file).
    """
    spec = importlib.util.spec_from_file_location(
        f"workflow_main_{workflow_main_py.parent.name}",
        str(workflow_main_py),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workflow module from {workflow_main_py}")
    module = importlib.util.module_from_spec(spec)
    # Ensure the workflow can import its sibling packages (workflow_utils, etc.)
    sys.path.insert(0, str(workflow_main_py.parent))
    spec.loader.exec_module(module)
    return module


def run(
    workflow_name: str,
    workflow_main_py: Path,
    stages_yaml: Path,
    run_idx: int,
    out_path: Path,
) -> None:
    """Run one workflow once and write a single RunResult JSON.

    Seeds ``random`` and ``torch`` to 42 *before* loading the workflow module
    so that module-level ``SEED = random.randint(...)`` calls pick up the
    deterministic seed.  After loading, overwrites the module's ``SEED``
    attribute to 42 explicitly to handle any other assignment patterns.

    Args:
        workflow_name: Human-readable workflow identifier stored in the result.
        workflow_main_py: Absolute path to the workflow's ``main.py``.
        stages_yaml: Absolute path to the workflow's ``stages.yaml``.
        run_idx: Run index (0 = warmup, ≥1 = measurement run).
        out_path: Path where the :class:`RunResult` JSON will be written.
            Parent directories are created if they do not exist.
    """
    # Step 1: Deterministic seeds BEFORE loading the workflow module, since
    # several workflows compute SEED at import time via random.randint().
    random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass

    # Step 2: Prepare stage mapping & instrumentation.
    stage_mapping = _load_stages(stages_yaml)
    node_rec, stage_rec = _install_instrumentation(stage_mapping)

    # Step 3: Snapshot memory baselines.
    reset_gpu_peak()

    status = "ok"
    error: str | None = None
    t_start = time.perf_counter_ns()
    try:
        module = _load_workflow_module(workflow_main_py)
        # Overwrite SEED so the workflow uses the deterministic seed that the
        # ComfyUI side's prompt JSON also uses.
        if hasattr(module, "SEED"):
            module.SEED = 42
        if not hasattr(module, "main"):
            raise RuntimeError(f"workflow {workflow_name} has no main()")
        module.main()
        _try_cuda_sync()
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    total_ns = time.perf_counter_ns() - t_start

    # Step 4: Read memory peaks.
    gpu_peak = read_gpu_peak()
    vmhwm = read_vmhwm()

    # Step 5: Assemble and write JSON.
    result = RunResult(
        workflow=workflow_name,
        side="runtime",
        run_idx=run_idx,
        status=status,
        error=error,
        total_ns=total_ns,
        stages=stage_rec.as_list(),
        nodes=node_rec.as_list(),
        gpu_peak_allocated_bytes=gpu_peak["allocated_bytes"],
        gpu_peak_reserved_bytes=gpu_peak["reserved_bytes"],
        host_vmhwm_bytes=vmhwm,
        env=gather_env(),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(run_result_to_dict(result), indent=2))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the subprocess entrypoint.

    Returns:
        Parsed namespace with ``workflow``, ``run_idx``, and ``out`` attributes.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--workflow", required=True)
    p.add_argument("--run-idx", type=int, required=True)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    """CLI entrypoint — resolve workflow paths from REPO_ROOT and call :func:`run`.

    Expects the workflow to live at ``<REPO_ROOT>/workflows/<name>/main.py``
    and its stages config at
    ``<runner_dir>/../workflows/<name>/stages.yaml``.
    """
    args = _parse_args()
    workflow_main = REPO_ROOT / "workflows" / args.workflow / "main.py"
    stages_yaml = (
        Path(__file__).resolve().parent.parent
        / "workflows"
        / args.workflow
        / "stages.yaml"
    )
    run(
        workflow_name=args.workflow,
        workflow_main_py=workflow_main,
        stages_yaml=stages_yaml,
        run_idx=args.run_idx,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
