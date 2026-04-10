"""Test a fast-unload path that avoids moving weights CPU-ward.

Compares the baseline ``mm.free_memory(1e30, cuda)`` path (which calls
``ModelPatcher.detach(unpatch_all=True)`` and spends ~3.3 s copying 7-8 GB
GPU→CPU) against a fast-unload helper that uses ``detach(unpatch_all=False)``
to skip the ``self.model.to(offload_device)`` call and lets Python GC +
``torch.cuda.empty_cache()`` reclaim the freed blocks.

Runs in two phases:
  Phase A — load UNET + CLIP, call the SLOW path, measure.
  Phase B — reload UNET + CLIP, call the FAST path, measure.

Both phases print before/after allocated memory so you can confirm the
fast path actually frees memory (not just returns quickly without doing
the work).
"""
import gc
import os
import sys
import time
from pathlib import Path

# Repo root: this file lives at $REPO/benchmarks/e2e/profiling/, so three
# parents up is the repo root.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Path to upstream ComfyUI clone (needed for the comfyui-side comparison
# probes). Override via COMFYUI_PATH env var if your clone lives elsewhere.
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/home/yanweiye/Project/ComfyUI")

# Where each workflow's model files live. Defaults to the repo-local
# workflows/<name>/models/ directory. Override via WORKFLOW_MODELS env var
# if you share a single models tree across workflows.
WORKFLOW_MODELS = Path(os.environ.get("WORKFLOW_MODELS", REPO / "workflows"))

# Flux2-specific model locations derived from the above.
FLUX2_MODELS_DIR = str(WORKFLOW_MODELS / "flux2_klein_text_to_image" / "models")
FLUX2_NODES_DIR = str(REPO / "workflows" / "flux2_klein_text_to_image" / "nodes")

import comfy_runtime  # noqa: E402  (sys.path mutated above)
import torch  # noqa: E402

comfy_runtime.configure(
    models_dir=FLUX2_MODELS_DIR,
)
comfy_runtime.load_nodes_from_path(f"{FLUX2_NODES_DIR}/nodes_custom_sampler.py")

import comfy.model_management as mm  # noqa: E402  (must come after configure)


def _mb(nbytes: int) -> float:
    return nbytes / 1024 / 1024


def _snapshot(label: str) -> None:
    loaded = [type(lm.model.model).__name__ for lm in mm.current_loaded_models]
    print(
        f"{label}: alloc={_mb(torch.cuda.memory_allocated()):.0f}MB "
        f"reserved={_mb(torch.cuda.memory_reserved()):.0f}MB "
        f"loaded={loaded}"
    )


def _load_clip():
    """Load CLIP and run a tiny encode so its weights land on GPU.

    Intentionally skips UNETLoader because we're measuring the *first* of
    the two real-workflow unloads: CLIP eviction after text encoding,
    before the sampler loads the UNET. In that state only the CLIP
    ModelPatcher sits in ``current_loaded_models``. Including a dangling
    UNETLoader result here adds a 500 ms CPU-RAM free to the measurement
    that does not exist in the real workflow (where the UNET local stays
    alive through sampling).
    """
    clip = comfy_runtime.execute_node(
        "CLIPLoader",
        clip_name="qwen_3_4b.safetensors",
        type="flux2",
    )[0]
    _ = comfy_runtime.execute_node("CLIPTextEncode", text="a hedgehog", clip=clip)[0]
    return clip


# ---------------------------------------------------------------------------
# Phase A: slow path (baseline) — mm.free_memory(1e30, cuda)
# ---------------------------------------------------------------------------
print("=" * 72)
print("Phase A: SLOW path (mm.free_memory → detach(unpatch_all=True) → .to(cpu))")
print("=" * 72)

clip_a = _load_clip()
_snapshot("before slow")

cuda0 = torch.device("cuda:0")
torch.cuda.synchronize()
t0 = time.perf_counter()
mm.free_memory(1e30, cuda0)
del clip_a
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
slow_ms = (time.perf_counter() - t0) * 1000
print(f"SLOW unload took: {slow_ms:.1f} ms")
_snapshot("after slow")
print()

# ---------------------------------------------------------------------------
# Phase B: fast path — detach(unpatch_all=False), no GPU→CPU copy
# ---------------------------------------------------------------------------
print("=" * 72)
print("Phase B: FAST path (detach(unpatch_all=False), drop refs, empty_cache)")
print("=" * 72)


def fast_unload(device):
    """Pop loaded models for ``device`` WITHOUT moving weights CPU-ward.

    ``detach(unpatch_all=False)`` skips the expensive
    ``self.model.to(offload_device)`` branch in ``unpatch_model`` (the 3.3 s
    PCIe copy for a 7-8 GB model). The weights are left on GPU until Python
    GC reclaims them, which happens once the caller drops all strong refs
    to the ModelPatcher (both the LoadedModel entry we pop here and any
    user-level locals like ``clip``/``model``).
    """
    to_pop = [i for i, lm in enumerate(mm.current_loaded_models) if lm.device == device]
    for i in sorted(to_pop, reverse=True):
        lm = mm.current_loaded_models.pop(i)
        lm.model.detach(unpatch_all=False)
        if lm.model_finalizer is not None:
            lm.model_finalizer.detach()
        lm.model_finalizer = None
        lm.real_model = None


def _tick(label: str, state: dict) -> None:
    """Sync the device then log ms since the previous tick."""
    torch.cuda.synchronize()
    now = time.perf_counter()
    print(f"  [{label:<18}] {(now - state['prev']) * 1000:7.1f} ms")
    state["prev"] = now


clip_b = _load_clip()
_snapshot("before fast (with gc)")

torch.cuda.synchronize()
t_start = time.perf_counter()
state = {"prev": t_start}

fast_unload(cuda0)
_tick("fast_unload()", state)
del clip_b
_tick("del clip_b", state)
gc.collect()
_tick("gc.collect()", state)
torch.cuda.empty_cache()
_tick("empty_cache()", state)

fast_ms = (time.perf_counter() - t_start) * 1000
print(f"FAST unload with gc (total): {fast_ms:.1f} ms")
_snapshot("after fast (with gc)")
print()


# ---------------------------------------------------------------------------
# Phase C: fast path WITHOUT gc.collect() — rely on refcount only
# ---------------------------------------------------------------------------
print("=" * 72)
print("Phase C: FAST path WITHOUT gc.collect()")
print("=" * 72)

clip_c = _load_clip()
_snapshot("before fast (no gc)")

torch.cuda.synchronize()
t_start = time.perf_counter()
state = {"prev": t_start}

fast_unload(cuda0)
_tick("fast_unload()", state)
del clip_c
_tick("del clip_c", state)
torch.cuda.empty_cache()
_tick("empty_cache()", state)

fast_nogc_ms = (time.perf_counter() - t_start) * 1000
print(f"FAST unload no-gc (total): {fast_nogc_ms:.1f} ms")
_snapshot("after fast (no gc)")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 72)
print("Summary")
print("=" * 72)
print(f"  SLOW (mm.free_memory):              {slow_ms:8.1f} ms")
print(f"  FAST with gc.collect():             {fast_ms:8.1f} ms")
print(f"  FAST without gc.collect():          {fast_nogc_ms:8.1f} ms")
if fast_nogc_ms > 0:
    print(f"  Speedup (no-gc vs slow):            {slow_ms / fast_nogc_ms:8.1f}x")
print(f"  Savings per call (no-gc vs slow):   {slow_ms - fast_nogc_ms:8.1f} ms")
