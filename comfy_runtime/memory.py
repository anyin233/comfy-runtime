"""Memory management helpers exposed as public comfy_runtime API.

These are thin wrappers around ``comfy.model_management``'s eviction
primitives. They let workflow authors explicitly release models from GPU
between workflow stages, which is particularly important for large models
(multi-gigabyte diffusion models + text encoders) running the direct-call
API of comfy_runtime.

Why this exists
---------------

comfy.model_management tracks every loaded ModelPatcher in a global
``current_loaded_models`` list. That list holds strong references, so
``del model`` at the user level does NOT free GPU memory on its own — the
LoadedModel entry keeps the weights alive.

ComfyUI's PromptExecutor sidesteps this by walking the graph with
``execution_list.complete_node_execution`` popping each completed node's
entries from the execution cache, which indirectly triggers
``load_models_gpu`` to evict the unused models when the next node asks
for GPU memory. comfy_runtime's direct-call API has no such walker, so
workflow authors should call :func:`unload_all_models` explicitly at
stage boundaries where a large model is no longer needed.

Typical pattern for a text2image workflow with multiple large models
(e.g. Flux2 Klein with a 4B Qwen text encoder and a 4B Flux diffusion
model)::

    clip = comfy_runtime.execute_node("CLIPLoader", ...)
    pos = comfy_runtime.execute_node("CLIPTextEncode", clip=clip, text=...)
    neg = comfy_runtime.execute_node("CLIPTextEncode", clip=clip, text="")
    comfy_runtime.unload_all_models()  # evict the text encoder

    model = comfy_runtime.execute_node("UNETLoader", ...)
    sampled = comfy_runtime.execute_node("KSampler", model=model, ...)
    comfy_runtime.unload_all_models()  # evict the diffusion model

    vae = comfy_runtime.execute_node("VAELoader", ...)
    images = comfy_runtime.execute_node("VAEDecode", samples=sampled, vae=vae)

Without the two :func:`unload_all_models` calls, running Flux2 Klein at
1024x1024 on a 24 GB GPU pushes total peak to ~23.4 GB, crossing the
threshold where ``VAE.decode`` falls back to tiled decoding and the
decode stage takes ~1 second instead of ~250 ms. With them, peak drops
to ~12 GB and decode runs single-pass.

For SD1.5-scale workflows (<5 GB peak), calling these helpers is
harmless but unnecessary.
"""

from __future__ import annotations


def free_memory(memory_required: float = 1e30, device=None) -> None:
    """Evict loaded models until ``memory_required`` bytes are free on ``device``.

    Thin wrapper around ``comfy.model_management.free_memory``. Passing
    the default ``memory_required=1e30`` evicts every loaded model from
    ``device`` unconditionally.

    Args:
        memory_required: Bytes of free GPU memory to ensure. Defaults to
            ``1e30`` (effectively infinity), which evicts everything.
        device: Torch device to free memory on. Defaults to the current
            CUDA device if CUDA is available, otherwise does nothing.
    """
    try:
        import comfy.model_management as mm
        import torch
    except ImportError:
        return

    if device is None:
        if not torch.cuda.is_available():
            return
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    mm.free_memory(memory_required, device)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def unload_all_models() -> None:
    """Evict every loaded model from every device and clear the cache.

    Convenience wrapper around
    ``comfy.model_management.unload_all_models`` plus
    ``torch.cuda.empty_cache``. Use this between workflow stages when the
    previously-loaded models are no longer needed, particularly before a
    stage that will load a new large model.
    """
    try:
        import comfy.model_management as mm
        import torch
    except ImportError:
        return

    mm.unload_all_models()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
