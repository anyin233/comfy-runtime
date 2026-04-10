"""Correctness verification gate for the benchmark.

Runs each workflow once per side, compares the output image's statistical
summary (shape, dtype, mean, stddev), and aborts the benchmark if any pair
diverges beyond ``rel_tol``. Intended to be run once before a full benchmark
batch to catch authoring errors in ``comfyui_prompt.json``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageStats:
    shape: tuple
    dtype: str
    mean: float
    std: float


def compute_stats(array: Any) -> ImageStats:
    """Compute a statistical summary of an image ndarray.

    Args:
        array: Any array-like object (numpy ndarray, PIL image data, etc.).

    Returns:
        ImageStats with shape, dtype, mean, and std of the array.
    """
    import numpy as np

    arr = np.asarray(array)
    return ImageStats(
        shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        mean=float(arr.mean()),
        std=float(arr.std()),
    )


def compare_stats(a: ImageStats, b: ImageStats, rel_tol: float = 0.01) -> tuple[bool, str]:
    """Compare two image stats within relative tolerance.

    Args:
        a: Stats from the first image (e.g. runtime side).
        b: Stats from the second image (e.g. comfyui side).
        rel_tol: Maximum allowed relative deviation for mean and std.

    Returns:
        Tuple of (ok, reason) where ok is True if images are within tolerance.
    """
    if a.shape != b.shape:
        return False, f"shape mismatch: {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        return False, f"dtype mismatch: {a.dtype} vs {b.dtype}"
    if a.mean == 0 and b.mean == 0:
        pass  # zero-image is fine.
    else:
        denom = max(abs(a.mean), 1e-6)
        if abs(a.mean - b.mean) / denom > rel_tol:
            return False, f"mean drift: {a.mean} vs {b.mean} (rel > {rel_tol})"
    if abs(a.std - b.std) > rel_tol * max(a.std, 1.0):
        return False, f"std drift: {a.std} vs {b.std}"
    return True, "ok"


def _latest_png_in(directory: Path) -> Path:
    """Return the most recently modified .png in directory.

    Args:
        directory: Directory path to search for PNG files.

    Returns:
        Path to the most recently modified PNG file.

    Raises:
        RuntimeError: If no PNG files are found in the directory.
    """
    # Exclude hidden files (starting with ".") so the runtime-side PNG
    # stashed as ``.verify_runtime_*.png`` does not leak into the ComfyUI-side
    # lookup on the second half of each verification round.
    pngs = sorted(
        (p for p in directory.glob("*.png") if not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
    )
    if not pngs:
        raise RuntimeError(f"no .png in {directory}")
    return pngs[-1]


def load_image(path: Path) -> Any:
    """Load an image file and return it as a numpy ndarray in RGB format.

    Args:
        path: Path to the image file to load.

    Returns:
        Numpy ndarray with shape (H, W, 3) in uint8 RGB format.
    """
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the verification gate.

    Returns:
        Parsed argument namespace with workflow and rel_tol fields.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--workflow", required=False, help="Run a single workflow only")
    p.add_argument("--rel-tol", type=float, default=0.01)
    return p.parse_args()


def main() -> int:
    """Entry point: run each workflow once per side, compare output stats.

    Returns:
        0 on success (all workflows within tolerance), 1 on any failure.
    """
    args = _parse_args()
    bench_root = Path(__file__).resolve().parent
    repo_root = bench_root.parent.parent

    all_workflows = [
        "sd15_text_to_image",
        "img2img",
        "inpainting",
        "hires_fix",
        "area_composition",
        "esrgan_upscale",
        "flux2_klein_text_to_image",
    ]
    workflows = [args.workflow] if args.workflow else all_workflows

    import json as _json
    import subprocess
    failures = []
    for wf in workflows:
        output_dir = repo_root / "workflows" / wf / "output"

        print(f"[verify] === {wf} ===", flush=True)

        # Clear previous output so _latest_png_in picks up only the verification run.
        for old in output_dir.glob("*.png"):
            old.unlink()

        # Runtime side
        runtime_json_path = bench_root / "results" / "verify" / f"{wf}_runtime.json"
        runtime_json_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(bench_root / "runtime-env" / ".venv" / "bin" / "python"),
             str(bench_root / "runners" / "runtime_runner.py"),
             "--workflow", wf, "--run-idx", "0",
             "--out", str(runtime_json_path)],
            check=False,
        )
        if not runtime_json_path.exists():
            print(f"[verify] {wf}: FAIL — runtime subprocess produced no JSON", flush=True)
            failures.append(wf)
            continue
        runtime_result = _json.loads(runtime_json_path.read_text())
        if runtime_result.get("status") != "ok":
            print(f"[verify] {wf}: FAIL — runtime subprocess failed:", flush=True)
            print(runtime_result.get("error", "")[:2000])
            failures.append(wf)
            continue
        try:
            runtime_png = _latest_png_in(output_dir)
        except RuntimeError as e:
            print(f"[verify] {wf}: FAIL — runtime produced no PNG: {e}", flush=True)
            failures.append(wf)
            continue
        runtime_img = load_image(runtime_png)
        runtime_stats = compute_stats(runtime_img)

        # Move or rename so the ComfyUI side's output does not clobber it.
        runtime_preserved = output_dir / f".verify_runtime_{runtime_png.name}"
        runtime_png.rename(runtime_preserved)

        # ComfyUI side
        comfyui_json_path = bench_root / "results" / "verify" / f"{wf}_comfyui.json"
        subprocess.run(
            [str(bench_root / "comfyui-env" / ".venv" / "bin" / "python"),
             str(bench_root / "runners" / "comfyui_runner.py"),
             "--workflow", wf, "--run-idx", "0",
             "--out", str(comfyui_json_path)],
            check=False,
        )
        if not comfyui_json_path.exists():
            print(f"[verify] {wf}: FAIL — comfyui subprocess produced no JSON", flush=True)
            failures.append(wf)
            continue
        comfyui_result = _json.loads(comfyui_json_path.read_text())
        if comfyui_result.get("status") != "ok":
            print(f"[verify] {wf}: FAIL — comfyui subprocess failed:", flush=True)
            print(comfyui_result.get("error", "")[:2000])
            failures.append(wf)
            continue
        try:
            comfyui_png = _latest_png_in(output_dir)
        except RuntimeError as e:
            print(f"[verify] {wf}: FAIL — comfyui produced no PNG: {e}", flush=True)
            failures.append(wf)
            continue
        comfyui_img = load_image(comfyui_png)
        comfyui_stats = compute_stats(comfyui_img)

        ok, reason = compare_stats(runtime_stats, comfyui_stats, rel_tol=args.rel_tol)
        if ok:
            print(f"[verify] {wf}: OK ({reason})", flush=True)
        else:
            print(f"[verify] {wf}: FAIL ({reason})", flush=True)
            print(f"  runtime: mean={runtime_stats.mean:.3f} std={runtime_stats.std:.3f}")
            print(f"  comfyui: mean={comfyui_stats.mean:.3f} std={comfyui_stats.std:.3f}")
            failures.append(wf)

    if failures:
        print(f"[verify] {len(failures)} workflow(s) failed: {failures}")
        return 1
    print("[verify] all workflows passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
