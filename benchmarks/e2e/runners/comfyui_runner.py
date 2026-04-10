"""Subprocess entrypoint — runs one workflow on the upstream ComfyUI side.

CLI: ``python comfyui_runner.py --workflow NAME --run-idx N --out PATH``

Expects to be launched from ``benchmarks/e2e/comfyui-env/.venv``. On import,
it prepends ``/home/yanweiye/Project/ComfyUI`` to ``sys.path`` so the upstream
modules can be imported.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

COMFYUI_PATH = "/home/yanweiye/Project/ComfyUI"
if COMFYUI_PATH not in sys.path:
    sys.path.insert(0, COMFYUI_PATH)

# Bootstrap sys.path so `benchmarks.e2e._harness.*` imports resolve when this
# module is launched as a subprocess from a venv that does not have the
# worktree root on PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.e2e._harness.env import gather_env
from benchmarks.e2e._harness.memory import read_gpu_peak, read_vmhwm, reset_gpu_peak
from benchmarks.e2e._harness.result_schema import RunResult, run_result_to_dict
from benchmarks.e2e._harness.timing import NodeRecorder, StageRecorder


REPO_ROOT = _REPO_ROOT


class MockServer:
    """Minimal no-op server stub satisfying PromptExecutor's interface."""

    client_id = None
    last_node_id = None
    last_prompt_id = None
    prompt_queue = None

    def send_sync(self, event, data, sid=None):
        """Accept and discard server events.

        Args:
            event: Event name string.
            data: Event payload.
            sid: Optional session ID.
        """
        pass


def _try_cuda_sync() -> None:
    """Synchronize the CUDA device if available; silently no-op otherwise."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _load_stages(stages_yaml: Path) -> dict[str, list[str]]:
    """Parse the stages.yaml mapping for a workflow.

    Args:
        stages_yaml: Path to the YAML file containing a ``stages`` key.

    Returns:
        Dict mapping stage names to lists of class_type strings.
    """
    return yaml.safe_load(stages_yaml.read_text())["stages"]


def _configure_folder_paths(
    workflow_models_dir: Path,
    workflow_input_dir: Path | None = None,
    workflow_output_dir: Path | None = None,
) -> None:
    """Point ComfyUI's folder_paths at the workflow's models/input/output subtree.

    Also calls ``nodes.init_extra_nodes()`` to register comfy_extras nodes
    (Flux2 sampler, custom samplers, upscalers, etc.). Without this call,
    only the core ``nodes.NODE_CLASS_MAPPINGS`` are registered and nodes
    like ``SamplerCustomAdvanced``, ``Flux2Scheduler``, ``UpscaleModelLoader``
    are missing. Best-effort — some ComfyUI versions init on import; newer
    versions expose it as an async coroutine that must be driven via asyncio.
    """
    import asyncio
    import inspect

    import folder_paths
    import nodes

    if hasattr(nodes, "init_extra_nodes"):
        try:
            fn = nodes.init_extra_nodes
            if inspect.iscoroutinefunction(fn):
                # Newer ComfyUI (post-2024) made init_extra_nodes async.
                # Skip custom-node discovery to avoid unrelated plugin errors.
                asyncio.run(fn(init_custom_nodes=False, init_api_nodes=False))
            else:
                fn()
        except Exception:
            pass

    if workflow_input_dir is not None and workflow_input_dir.exists():
        folder_paths.set_input_directory(str(workflow_input_dir))
    if workflow_output_dir is not None:
        workflow_output_dir.mkdir(parents=True, exist_ok=True)
        folder_paths.set_output_directory(str(workflow_output_dir))

    if not workflow_models_dir.exists():
        return

    for category_dir in workflow_models_dir.iterdir():
        if category_dir.is_dir():
            folder_paths.add_model_folder_path(category_dir.name, str(category_dir))


def _load_custom_nodes(workflow_nodes_dir: Path | None) -> None:
    """Load custom node Python files from the workflow's nodes/ directory.

    Handles both the sync (older ComfyUI) and async (newer ComfyUI 0.18+)
    signatures of ``nodes.load_custom_node``.

    Args:
        workflow_nodes_dir: Directory containing custom ``*.py`` node files, or
            ``None`` to skip loading. Silently skips if the directory does not
            exist.
    """
    if workflow_nodes_dir is None or not workflow_nodes_dir.exists():
        return
    import asyncio
    import inspect

    import nodes

    loader = nodes.load_custom_node
    is_async = inspect.iscoroutinefunction(loader)
    for py_file in sorted(workflow_nodes_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        try:
            if is_async:
                asyncio.run(loader(str(py_file)))
            else:
                loader(str(py_file))
        except Exception as exc:
            print(f"[comfyui_runner] warn: failed to load {py_file.name}: {exc}", flush=True)


def _install_instrumentation(
    stage_mapping: dict[str, list[str]],
) -> tuple[NodeRecorder, StageRecorder]:
    """Monkey-patch ``execution.execute`` to record per-node timings.

    Works with both the stub PromptExecutor used in tests (which calls
    ``execution.execute(node_id, class_type, cls, inputs)``) and the real
    upstream one (which calls ``execute(server, dynprompt, caches,
    current_item, extra_data, executed, prompt_id, execution_list, ...)``).
    We use ``*args, **kwargs`` passthrough and recover ``class_type`` from
    whichever calling convention we received.

    Args:
        stage_mapping: Mapping from stage name to list of class_type strings.

    Returns:
        A ``(NodeRecorder, StageRecorder)`` tuple populated during execution.
    """
    import execution

    node_rec = NodeRecorder()
    stage_rec = StageRecorder(stage_mapping)
    original = execution.execute

    def _recover_class_type(args: tuple, kwargs: dict) -> str:
        """Extract the node's class_type from either calling convention.

        Args:
            args: Positional arguments passed to the patched execute function.
            kwargs: Keyword arguments passed to the patched execute function.

        Returns:
            The ``class_type`` string, or ``"unknown"`` if it cannot be found.
        """
        # Stub path: args = (node_id: str, class_type: str, cls, inputs)
        # args[1] is a plain string — return it directly.
        if len(args) >= 2 and isinstance(args[1], str):
            return args[1]
        # Real path: args = (server, dynprompt, caches, current_item, ...)
        # args[1] is the dynprompt object; args[3] is the node_id string.
        if len(args) >= 4:
            dynprompt = args[1]
            current_item = args[3]
            try:
                node = dynprompt.get_node(current_item)
                return node["class_type"]
            except Exception:
                return "unknown"
        # kwargs path used by some future refactors of the upstream caller.
        if "current_item" in kwargs and "dynprompt" in kwargs:
            try:
                node = kwargs["dynprompt"].get_node(kwargs["current_item"])
                return node["class_type"]
            except Exception:
                return "unknown"
        return "unknown"

    import inspect

    original_is_async = inspect.iscoroutinefunction(original)

    if original_is_async:
        # ComfyUI 0.18+ made ``execution.execute`` an async coroutine that is
        # awaited from ``execute_async``. The wrapper therefore must also be
        # async — a sync wrapper would return the coroutine object to the
        # caller without awaiting it, and our timing would only measure
        # coroutine creation (microseconds), not the actual node execution.
        async def timed(*args: Any, **kwargs: Any):
            """Async wrapper around execution.execute.

            Awaits the original coroutine, synchronizes CUDA, and records
            elapsed wall-clock time under the recovered class_type.
            """
            class_type = _recover_class_type(args, kwargs)
            t0 = time.perf_counter_ns()
            try:
                result = await original(*args, **kwargs)
            finally:
                _try_cuda_sync()
                elapsed = time.perf_counter_ns() - t0
                node_rec.record(class_type, elapsed)
                stage_rec.record(class_type, elapsed)
            return result
    else:
        def timed(*args: Any, **kwargs: Any):
            """Sync wrapper around execution.execute (used by the stub test)."""
            class_type = _recover_class_type(args, kwargs)
            t0 = time.perf_counter_ns()
            try:
                result = original(*args, **kwargs)
            finally:
                _try_cuda_sync()
                elapsed = time.perf_counter_ns() - t0
                node_rec.record(class_type, elapsed)
                stage_rec.record(class_type, elapsed)
            return result

    execution.execute = timed  # type: ignore[assignment]
    return node_rec, stage_rec


def _execute_outputs_for_prompt(prompt: dict[str, dict]) -> list[str]:
    """Return node IDs that PromptExecutor should treat as output targets.

    Prefers SaveImage-like nodes. Falls back to the node with the highest
    numeric ID so PromptExecutor always has something to run.

    Args:
        prompt: ComfyUI API-format prompt dict.

    Returns:
        List of node ID strings to pass as ``execute_outputs``.
    """
    outputs = []
    for node_id, node in prompt.items():
        if node["class_type"] in {"SaveImage", "PreviewImage", "SaveAudio"}:
            outputs.append(node_id)
    if not outputs:
        # Fall back: run everything by picking the max-id node.
        outputs.append(max(prompt, key=int))
    return outputs


def run(
    workflow_name: str,
    prompt_json: Path,
    stages_yaml: Path,
    workflow_models_dir: Path,
    workflow_nodes_dir: Path | None,
    run_idx: int,
    out_path: Path,
    workflow_input_dir: Path | None = None,
    workflow_output_dir: Path | None = None,
) -> None:
    """Run one workflow once on the ComfyUI side and write a RunResult JSON.

    Sets a fixed random seed for reproducibility, configures folder_paths,
    optionally loads custom nodes, installs timing instrumentation, constructs
    a fresh PromptExecutor with caching disabled, executes the prompt, and
    serializes the result to ``out_path``.

    Args:
        workflow_name: Logical name of the workflow (embedded in the result).
        prompt_json: Path to the ComfyUI API-format prompt JSON file.
        stages_yaml: Path to the stages.yaml mapping file for this workflow.
        workflow_models_dir: Root of the workflow's models directory tree.
        workflow_nodes_dir: Optional directory of custom ``.py`` node files.
        run_idx: Index of this run (0 = warmup, discarded during aggregation).
        out_path: Destination path for the JSON result file.
        workflow_input_dir: Optional directory holding input images for
            ``LoadImage`` nodes. If provided, passed to
            ``folder_paths.set_input_directory``.
        workflow_output_dir: Optional directory where ``SaveImage`` nodes
            should write. If provided, passed to
            ``folder_paths.set_output_directory``.
    """
    random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass

    prompt = json.loads(prompt_json.read_text())
    stage_mapping = _load_stages(stages_yaml)

    _configure_folder_paths(
        workflow_models_dir,
        workflow_input_dir=workflow_input_dir,
        workflow_output_dir=workflow_output_dir,
    )
    _load_custom_nodes(workflow_nodes_dir)

    node_rec, stage_rec = _install_instrumentation(stage_mapping)

    import execution
    # cache_type=False disables the HierarchicalCache, but PromptExecutor
    # still reads cache_args["ram"] in execute_async to compute a RAM
    # headroom. Pass a minimal dict to keep it happy.
    executor = execution.PromptExecutor(
        MockServer(),
        cache_type=False,
        cache_args={"ram": 0.0, "lru": 0},
    )

    reset_gpu_peak()

    status = "ok"
    error: str | None = None
    t_start = time.perf_counter_ns()
    try:
        execute_outputs = _execute_outputs_for_prompt(prompt)
        executor.execute(
            prompt,
            prompt_id="bench",
            extra_data={},
            execute_outputs=execute_outputs,
        )
        _try_cuda_sync()
        # PromptExecutor catches node exceptions internally and sets
        # self.success = False rather than raising. Surface that as a failure.
        if getattr(executor, "success", True) is False:
            status = "failed"
            # Pull the last execution_error event if present.
            messages = getattr(executor, "status_messages", []) or []
            err_events = [m for m in messages if isinstance(m, tuple) and m[0] == "execution_error"]
            if err_events:
                error = f"execution_error: {err_events[-1][1]}"
            else:
                error = "PromptExecutor reported success=False with no execution_error event"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    total_ns = time.perf_counter_ns() - t_start

    gpu_peak = read_gpu_peak()
    vmhwm = read_vmhwm()

    result = RunResult(
        workflow=workflow_name,
        side="comfyui",
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
        Namespace with ``workflow``, ``run_idx``, and ``out`` attributes.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--workflow", required=True)
    p.add_argument("--run-idx", type=int, required=True)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    """CLI entrypoint: resolve paths from the workflow name and delegate to run()."""
    args = _parse_args()
    bench_root = Path(__file__).resolve().parent.parent
    workflow_dir = bench_root / "workflows" / args.workflow
    prompt_json = workflow_dir / "comfyui_prompt.json"
    stages_yaml = workflow_dir / "stages.yaml"
    upstream_workflow = REPO_ROOT / "workflows" / args.workflow
    models_dir = upstream_workflow / "models"
    nodes_dir = upstream_workflow / "nodes"
    input_dir = upstream_workflow / "input"
    output_dir = upstream_workflow / "output"
    run(
        workflow_name=args.workflow,
        prompt_json=prompt_json,
        stages_yaml=stages_yaml,
        workflow_models_dir=models_dir,
        workflow_nodes_dir=nodes_dir if nodes_dir.exists() else None,
        run_idx=args.run_idx,
        out_path=args.out,
        workflow_input_dir=input_dir if input_dir.exists() else None,
        workflow_output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
