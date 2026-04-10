"""Environment metadata collector for benchmark runs.

Gathers everything the final report needs to disclose about the machine and
software versions used during a run: GPU, driver, torch, python, ComfyUI
commit, etc. Values that cannot be determined default to ``"unknown"`` so
callers can embed the dict in JSON unconditionally.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

# Upstream ComfyUI clone used for vendored commit detection. The default
# matches the original dev machine; override via the ``COMFYUI_PATH`` env
# var on any other host. Matches the pattern used by
# ``benchmarks/e2e/runners/comfyui_runner.py`` and the profiling scripts.
COMFYUI_PATH = Path(os.environ.get("COMFYUI_PATH", "/home/yanweiye/Project/ComfyUI"))


def _safe_run(cmd: list[str]) -> str:
    """Run a subprocess command and return stripped stdout, or 'unknown' on failure.

    Args:
        cmd: The command and arguments to execute.

    Returns:
        Stripped stdout string, or ``"unknown"`` if the command fails or times out.
    """
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        return out.decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def _torch_info() -> dict[str, Any]:
    """Collect torch version, CUDA version, GPU name, and GPU memory.

    Returns:
        Dict with keys: torch_version, cuda_version, gpu_name, gpu_memory_total_mb.
        All values default to ``"unknown"`` if torch is not importable or CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        return {"torch_version": "unknown", "cuda_version": "unknown",
                "gpu_name": "unknown", "gpu_memory_total_mb": "unknown"}
    info: dict[str, Any] = {"torch_version": torch.__version__}
    info["cuda_version"] = torch.version.cuda or "unknown"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["gpu_memory_total_mb"] = int(props.total_memory // (1024 * 1024))
    else:
        info["gpu_name"] = "unknown"
        info["gpu_memory_total_mb"] = "unknown"
    return info


def _comfyui_commit() -> str:
    """Read the short git SHA of the ComfyUI installation.

    Returns:
        Short SHA string, or ``"unknown"`` if COMFYUI_PATH does not exist or git fails.
    """
    if not COMFYUI_PATH.exists():
        return "unknown"
    return _safe_run(["git", "-C", str(COMFYUI_PATH), "rev-parse", "--short", "HEAD"])


def _comfy_runtime_version() -> str:
    """Read the installed comfy_runtime package version.

    Returns:
        Version string from ``comfy_runtime.__version__``, or ``"unknown"`` if not installed.
    """
    try:
        import comfy_runtime
        return getattr(comfy_runtime, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _driver_version() -> str:
    """Query the NVIDIA driver version via nvidia-smi.

    Returns:
        Driver version string, or ``"unknown"`` if nvidia-smi is unavailable or fails.
    """
    out = _safe_run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if out == "unknown" or not out:
        return "unknown"
    return out.splitlines()[0].strip()


def gather_env() -> dict[str, Any]:
    """Snapshot environment metadata for embedding in a result JSON.

    Collects hostname, timestamp, Python version, platform, NVIDIA driver version,
    ComfyUI commit SHA, comfy_runtime version, torch/CUDA/GPU info, and process
    context. Values that cannot be determined default to ``"unknown"``.

    Returns:
        Dict with string keys and JSON-serializable values.
    """
    env: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "driver_version": _driver_version(),
        "comfyui_commit": _comfyui_commit(),
        "comfy_runtime_version": _comfy_runtime_version(),
        "argv": sys.argv[:],
        "cwd": os.getcwd(),
    }
    env.update(_torch_info())
    return env
