"""Memory measurement helpers.

Host memory uses VmHWM from /proc/self/status — a kernel-maintained peak
that costs zero to read. GPU memory uses torch.cuda peak stats; when torch
or CUDA is unavailable the reader returns zeros so callers can always rely
on the same dict shape.
"""

from __future__ import annotations


def read_vmhwm(status_path: str = "/proc/self/status") -> int:
    """Return VmHWM (peak resident set size) in bytes.

    Args:
        status_path: Path to a /proc/<pid>/status-formatted file. Defaults
            to the current process. Exposed as a parameter so tests can
            point at a synthetic file.

    Raises:
        RuntimeError: If the VmHWM line is absent from the file.
    """
    with open(status_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                # Format: "VmHWM:\t  98765 kB"
                parts = line.split()
                kb = int(parts[1])
                return kb * 1024
    raise RuntimeError(f"VmHWM line not found in {status_path}")


def read_gpu_peak() -> dict[str, int]:
    """Return torch.cuda peak allocation stats for the current process.

    Returns a dict with keys ``allocated_bytes`` and ``reserved_bytes``.
    When torch or CUDA is unavailable, returns zeros — this lets callers
    always use the same key set without branching.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_bytes": 0, "reserved_bytes": 0}
        return {
            "allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except ImportError:
        return {"allocated_bytes": 0, "reserved_bytes": 0}


def reset_gpu_peak() -> None:
    """Reset torch.cuda peak-stats counters. No-op without torch/CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
