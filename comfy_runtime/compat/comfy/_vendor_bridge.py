"""Bridge to vendored ComfyUI inference code.

This module provides access to the vendored ComfyUI implementation
for operations that require actual model inference (checkpoint loading,
sampling, VAE encode/decode, text encoding).

The compat layer handles namespace registration and non-inference APIs.
This bridge is a temporary measure until all inference is reimplemented
with diffusers (Phase 2-3).

TODO(Phase2): Replace each function with a diffusers-based implementation.
TODO(Phase3): Remove this file entirely once all inference is MIT.
"""

import importlib
import os
import sys
import logging

logger = logging.getLogger(__name__)

_bridge_initialized = False


def _ensure_vendor_imports():
    """Switch from compat modules to vendored ComfyUI for inference.

    The compat layer provides the ``comfy.*`` namespace at import time
    so that custom nodes can be loaded. When actual inference is needed
    (model loading, sampling, VAE), we swap the compat modules with the
    full vendored ComfyUI code.

    This is a wholesale replacement — after this call, ``import comfy.sd``
    resolves to the vendored version with full model loading support.
    """
    global _bridge_initialized
    if _bridge_initialized:
        return
    _bridge_initialized = True

    try:
        # Step 1: Import the vendored comfy top-level package
        vendor_comfy = importlib.import_module("comfy_runtime._vendor.comfy")

        # Step 2: Replace comfy in sys.modules with the vendored version.
        # Preserve attributes that the compat bootstrap set (like options).
        compat_comfy = sys.modules.get("comfy")
        if compat_comfy is not None:
            # Copy bootstrap-set attrs (options, args_parsing) to vendor
            for attr in ("options",):
                if hasattr(compat_comfy, attr) and not hasattr(vendor_comfy, attr):
                    setattr(vendor_comfy, attr, getattr(compat_comfy, attr))

        sys.modules["comfy"] = vendor_comfy

        # Step 3: Remove ALL compat comfy.* submodules from sys.modules
        # so they get re-imported from vendor on next access.
        # This is critical — leftover compat stubs for comfy.ldm.*,
        # comfy.sd, comfy.ops etc. will block vendored imports.
        to_remove = []
        for key in list(sys.modules.keys()):
            if not key.startswith("comfy."):
                continue
            if key == "comfy.options":
                continue  # Preserve bootstrap-set options
            mod = sys.modules.get(key)
            if mod is None:
                continue
            mod_file = getattr(mod, "__file__", None) or ""
            if "compat" in mod_file:
                to_remove.append(key)
        for key in to_remove:
            del sys.modules[key]

        # Also remove compat comfy_api.* and comfy_execution.*
        for prefix in ("comfy_api.", "comfy_execution."):
            for key in list(sys.modules.keys()):
                if not key.startswith(prefix):
                    continue
                mod = sys.modules.get(key)
                if mod is None:
                    continue
                mod_file = getattr(mod, "__file__", None) or ""
                if "compat" in mod_file:
                    del sys.modules[key]

        # Step 3b: Wire the full ldm attribute chain for vendored comfy
        # Many nodes access comfy.ldm.modules.diffusionmodules.mmdit.MMDiT
        # which requires the full chain to be importable and wired as attrs.
        _ldm_chain = [
            "comfy.ldm",
            "comfy.ldm.modules",
            "comfy.ldm.modules.diffusionmodules",
            "comfy.ldm.modules.diffusionmodules.mmdit",
        ]
        for mod_short in _ldm_chain:
            vendor_name = f"comfy_runtime._vendor.{mod_short}"
            try:
                vendor_mod = importlib.import_module(vendor_name)
                sys.modules[mod_short] = vendor_mod
            except Exception:
                pass

        # Wire attribute chain: parent.child = child_module
        try:
            comfy_ldm = sys.modules.get("comfy.ldm")
            comfy_ldm_modules = sys.modules.get("comfy.ldm.modules")
            comfy_ldm_diffmod = sys.modules.get("comfy.ldm.modules.diffusionmodules")
            comfy_ldm_mmdit = sys.modules.get("comfy.ldm.modules.diffusionmodules.mmdit")
            if comfy_ldm and comfy_ldm_modules:
                comfy_ldm.modules = comfy_ldm_modules
            if comfy_ldm_modules and comfy_ldm_diffmod:
                comfy_ldm_modules.diffusionmodules = comfy_ldm_diffmod
            if comfy_ldm_diffmod and comfy_ldm_mmdit:
                comfy_ldm_diffmod.mmdit = comfy_ldm_mmdit
            # Also wire ldm onto comfy
            if comfy_ldm:
                vendor_comfy.ldm = comfy_ldm
        except Exception:
            pass

        # Step 4: Replace standalone modules (each independently)
        for mod_name in ("comfy_execution", "comfy_api"):
            vendor_name = f"comfy_runtime._vendor.{mod_name}"
            try:
                vendor_mod = importlib.import_module(vendor_name)
                sys.modules[mod_name] = vendor_mod
            except Exception:
                pass

        for mod_name in ("folder_paths", "execution", "protocol", "node_helpers",
                         "comfyui_version"):
            vendor_name = f"comfy_runtime._vendor.{mod_name}"
            try:
                vendor_mod = importlib.import_module(vendor_name)
                # Carry over configured directories from compat folder_paths
                if mod_name == "folder_paths":
                    compat_fp = sys.modules.get(mod_name)
                    if compat_fp is not None:
                        for attr in ("output_directory", "input_directory", "temp_directory",
                                     "folder_names_and_paths"):
                            val = getattr(compat_fp, attr, None)
                            if val is not None:
                                setattr(vendor_mod, attr, val)
                sys.modules[mod_name] = vendor_mod
            except Exception:
                pass

        # Step 4b: Register comfy_extras from the ComfyUI installation
        # Many custom nodes import from comfy_extras.* (e.g. nodes_mask, chainner_models)
        comfyui_root = os.environ.get("COMFYUI_ROOT")
        if comfyui_root is None:
            # Try common locations
            for candidate in [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))), "..", "ComfyUI"),
                "/home/yanweiye/Project/ComfyUI",
            ]:
                if os.path.isdir(os.path.join(candidate, "comfy_extras")):
                    comfyui_root = os.path.abspath(candidate)
                    break

        if comfyui_root and os.path.isdir(os.path.join(comfyui_root, "comfy_extras")):
            if comfyui_root not in sys.path:
                sys.path.insert(0, comfyui_root)
            # Ensure comfy_extras is importable
            try:
                importlib.import_module("comfy_extras")
            except Exception:
                pass

        # nodes.py has heavy import chains (controlnet->mmdit).
        # Don't import it eagerly — let it be imported on demand.

    except Exception as e:
        logger.warning(f"Failed to initialize vendor bridge: {e}")
        import traceback
        traceback.print_exc()


def activate_vendor_bridge():
    """Public API to activate the vendor bridge early.

    Call this before loading custom nodes that need full ComfyUI
    inference support (e.g., SamplerCustomAdvanced, CFGGuider).
    """
    _ensure_vendor_imports()


def _vendor_import(dotted_name):
    """Import a module directly from the _vendor package.

    Bypasses the shim completely so that we always get the vendored
    version, not the compat stub.
    """
    _ensure_vendor_imports()
    return importlib.import_module(f"comfy_runtime._vendor.{dotted_name}")


def load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                  embedding_directory=None, output_clipvision=False):
    """Load a checkpoint file and return (model, clip, vae, ...)."""
    vendor_sd = _vendor_import("comfy.sd")
    return vendor_sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=output_vae,
        output_clip=output_clip,
        embedding_directory=embedding_directory,
        output_clipvision=output_clipvision,
    )


def load_unet(unet_path, dtype=None):
    """Load a standalone UNET/transformer model."""
    vendor_sd = _vendor_import("comfy.sd")
    model_options = {}
    if dtype is not None:
        model_options["dtype"] = dtype
    return vendor_sd.load_diffusion_model(unet_path, model_options=model_options)


def load_clip(clip_paths, clip_type=None, model_options=None):
    """Load a CLIP text encoder."""
    vendor_sd = _vendor_import("comfy.sd")
    if model_options is None:
        model_options = {}
    return vendor_sd.load_clip(
        ckpt_paths=clip_paths,
        embedding_directory=None,
        clip_type=clip_type,
        model_options=model_options,
    )


def load_vae(vae_path):
    """Load a VAE model."""
    vendor_sd = _vendor_import("comfy.sd")
    vendor_utils = _vendor_import("comfy.utils")
    sd = vendor_utils.load_torch_file(vae_path)
    return vendor_sd.VAE(sd=sd)


def encode_clip_text(clip, text):
    """Encode text using a CLIP model.

    Returns:
        Conditioning as returned by ``clip.encode_from_tokens_scheduled()``.
    """
    _ensure_vendor_imports()
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive,
             negative, latent_image, denoise=1.0, disable_noise=False,
             start_step=None, last_step=None, force_full_denoise=False):
    """Run the KSampler diffusion sampling loop.

    Uses vendored comfy.sample.sample() directly.
    """
    _ensure_vendor_imports()
    import comfy.sample
    import comfy.utils

    latent = latent_image
    latent_samples = latent["samples"]
    latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples,
                                                             latent.get("downscale_ratio_spacial", None))

    if disable_noise:
        noise = torch.zeros(latent_samples.size(), dtype=latent_samples.dtype,
                            layout=latent_samples.layout, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_samples,
        denoise=denoise, disable_noise=disable_noise,
        start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask, callback=None,
        disable_pbar=disable_pbar, seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out,)


def vae_decode(vae, samples):
    """Decode latent samples to images using VAE."""
    _ensure_vendor_imports()
    return vae.decode(samples["samples"])


def vae_encode(vae, pixels):
    """Encode pixel images to latent using VAE."""
    _ensure_vendor_imports()
    t = vae.encode(pixels[:, :, :, :3])
    return {"samples": t}
