"""Memory management helpers exposed as public comfy_runtime API.

These wrappers evict loaded models from GPU between workflow stages, which
matters for large models (multi-gigabyte diffusion models + text encoders)
running the direct-call API of comfy_runtime.

Why this exists
---------------

``comfy.model_management`` tracks every loaded ModelPatcher in a global
``current_loaded_models`` list. That list holds strong references, so
``del model`` at the user level does NOT free GPU memory on its own — the
LoadedModel entry keeps the weights alive.

ComfyUI's PromptExecutor sidesteps this by walking the prompt graph with
``execution_list.complete_node_execution`` popping each completed node's
entries from the execution cache, which indirectly triggers
``load_models_gpu`` to evict the unused models when the next node asks
for GPU memory. comfy_runtime's direct-call API has no such walker, so
workflow authors should call :func:`unload_all_models` explicitly at
stage boundaries where a large model is no longer needed.

Why the fast path
-----------------

The naive path (``mm.free_memory(1e30, cuda)``) calls
``LoadedModel.model_unload(...)`` which in turn calls
``ModelPatcher.detach(unpatch_all=True)`` — that branch runs
``unpatch_model(offload_device, unpatch_weights=True)`` which contains
``self.model.to(offload_device)`` on line 938 of
``comfy/model_patcher.py``. For a 7-8 GB text encoder or diffusion model
that's a **4.5-5.0 s PCIe copy GPU → CPU**, performed synchronously.

For a workflow stage boundary where we are done with the model for this
run and will not need it again, the CPU-ward copy is pure waste. The
upstream API already supports skipping it: ``model_unload`` accepts
``unpatch_weights=False`` which propagates to ``detach(unpatch_all=False)``
which skips the ``unpatch_model`` branch entirely. Upstream uses this
exact pattern internally in ``load_models_gpu`` (lines 764-766) when
evicting a clone of a model that's about to be reloaded.

Profiling on an RTX A6000 (48 GB) with Flux2 Klein's Qwen 3 4B text
encoder (~7.5 GB GPU):

    slow path (unpatch_weights=True):  ~4500 ms
    fast path (unpatch_weights=False):  ~170 ms
    speedup:                           ~26x

The remaining ~170 ms is dominated by Python refcount cleanup of the
ModelPatcher's internal state (~30 ms) and ``torch.cuda.empty_cache()``
returning freed blocks to the driver (~140 ms) — both unavoidable.

Typical pattern for a text2image workflow with multiple large models
(e.g. Flux2 Klein with a 4B Qwen text encoder and a 4B Flux diffusion
model)::

    clip = comfy_runtime.execute_node("CLIPLoader", ...)
    pos = comfy_runtime.execute_node("CLIPTextEncode", clip=clip, text=...)
    neg = comfy_runtime.execute_node("CLIPTextEncode", clip=clip, text="")
    del clip  # drop user-level ref so GC can reclaim once unloaded
    comfy_runtime.unload_all_models()

    model = comfy_runtime.execute_node("UNETLoader", ...)
    sampled = comfy_runtime.execute_node("KSampler", model=model, ...)
    del model
    comfy_runtime.unload_all_models()

    vae = comfy_runtime.execute_node("VAELoader", ...)
    images = comfy_runtime.execute_node("VAEDecode", samples=sampled, vae=vae)

Without the two :func:`unload_all_models` calls, running Flux2 Klein at
1024x1024 on a 24 GB GPU pushes total peak to ~23.4 GB, crossing the
threshold where ``VAE.decode`` falls back to tiled decoding and the
decode stage takes ~1 second instead of ~250 ms. With the fast path,
peak drops to ~12 GB and decode runs single-pass — and the per-unload
overhead is ~170 ms instead of ~3.3 s each.

For SD1.5-scale workflows (<5 GB peak), calling these helpers is
harmless but unnecessary.
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional


def free_memory(
    memory_required: float = 1e30,
    device=None,
    keep_loaded: Optional[Iterable] = None,
) -> list:
    """Fast-evict loaded models until ``memory_required`` bytes are free.

    Mirrors ``comfy.model_management.free_memory`` — same filtering,
    same sort order, same budget semantics — but routes each eviction
    through ``LoadedModel.model_unload(..., unpatch_weights=False)``
    to skip the 3-5 s GPU→CPU weight copy that the stock call makes.

    Args:
        memory_required: Bytes of free GPU memory to ensure on ``device``.
            Defaults to ``1e30`` (effectively infinity), which evicts
            every eligible model on ``device`` unconditionally.
        device: Torch device to free memory on. Defaults to the current
            CUDA device if CUDA is available; if not, the call is a
            no-op. Pass ``torch.device(...)`` explicitly for
            multi-device setups.
        keep_loaded: Optional iterable of ``LoadedModel`` entries to
            pin — they are skipped even if they are on ``device``. Mirrors
            the stock ``keep_loaded`` parameter. ``None`` means no
            pinning.

    Returns:
        List of LoadedModel entries that were removed from
        ``mm.current_loaded_models``. May be empty.

    Notes:
        The caller is expected to drop user-level strong references
        (``del clip``, ``del model``, ...) BEFORE calling this helper,
        otherwise the ModelPatcher's weights remain reachable through
        the caller's locals and ``torch.cuda.empty_cache()`` cannot
        reclaim them. See the module docstring for the recommended
        pattern.
    """
    try:
        import comfy.model_management as mm
        import torch
    except ImportError:
        return []

    if keep_loaded is None:
        keep_loaded = ()

    if device is None:
        if not torch.cuda.is_available():
            return []
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Drop dead weakref entries before we walk the list, matching stock.
    mm.cleanup_models_gc()

    # Phase 1: build the candidate list with the same filter + sort order
    # as stock mm.free_memory (model_management.py:670-677). The sort key
    # puts models with more *offloaded* memory first (negated so sort
    # ascending still evicts them earliest), then breaks ties by refcount
    # and total memory. This preserves the heuristic upstream uses to
    # decide *which* model to drop when only some need to go.
    can_unload = []
    for i in range(len(mm.current_loaded_models) - 1, -1, -1):
        lm = mm.current_loaded_models[i]
        if lm.device != device:
            continue
        if lm in keep_loaded:
            continue
        if lm.is_dead():
            continue
        can_unload.append((
            -lm.model_offloaded_memory(),
            sys.getrefcount(lm.model),
            lm.model_memory(),
            i,
        ))
        lm.currently_used = False

    can_unload.sort()

    # Phase 2: evict in priority order, stopping once the budget is met.
    # Pass ``unpatch_weights=False`` so ``model_unload`` routes through
    # ``detach(unpatch_all=False)`` and skips the slow GPU→CPU copy.
    unloaded_indices = []
    for _, _, _, i in can_unload:
        memory_to_free = memory_required - mm.get_free_memory(device)
        if memory_to_free <= 0:
            break
        if mm.current_loaded_models[i].model_unload(
            memory_to_free, unpatch_weights=False
        ):
            unloaded_indices.append(i)

    # Phase 3: pop the unloaded entries. Reverse order so each pop leaves
    # earlier indices valid.
    unloaded_models = []
    for i in sorted(unloaded_indices, reverse=True):
        unloaded_models.append(mm.current_loaded_models.pop(i))

    # Phase 4: return freed blocks to the driver so downstream allocations
    # (e.g. VAE decode) see the full free pool, not just torch's cached
    # pool. Matches stock free_memory's soft_empty_cache tail.
    if unloaded_models:
        try:
            mm.soft_empty_cache()
        except Exception:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return unloaded_models


def unload_all_models() -> list:
    """Fast-evict every loaded model from the current CUDA device.

    Convenience wrapper around :func:`free_memory` with an infinite
    budget. Use between workflow stages when the previously-loaded
    models are no longer needed, particularly before a stage that will
    load a new large model.

    Returns:
        List of LoadedModel entries that were removed.
    """
    return free_memory(memory_required=1e30)
