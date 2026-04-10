"""Sampler helper utilities for comfy_runtime.

MIT reimplementation of comfy.sampler_helpers — provides helper functions
used by samplers during the denoising process.

Functions are stubs for Phase 3 implementation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def get_area_and_mult(conds: List, x_in: torch.Tensor, timestep: float) -> List[Tuple]:
    """Extract area and multiplier information from conditioning.

    Args:
        conds: List of conditioning entries.
        x_in: Current latent tensor (for shape reference).
        timestep: Current timestep value.

    Returns:
        List of (area, mult, conditioning, ...) tuples.
    """
    # TODO(Phase3): Implement area/mult extraction from conditioning.
    return [(None, 1.0, c) for c in conds]


def can_concat_cond(c1, c2) -> bool:
    """Check if two conditioning entries can be batched together.

    Args:
        c1: First conditioning entry.
        c2: Second conditioning entry.

    Returns:
        True if the entries can be concatenated.
    """
    # TODO(Phase3): Implement compatibility checking.
    return True


def cond_cat(c_list: List) -> Any:
    """Concatenate a list of compatible conditioning entries.

    Args:
        c_list: List of conditioning entries.

    Returns:
        Concatenated conditioning.
    """
    # TODO(Phase3): Implement conditioning concatenation.
    if not c_list:
        return None
    if len(c_list) == 1:
        return c_list[0]
    if isinstance(c_list[0], torch.Tensor):
        return torch.cat(c_list, dim=0)
    return c_list[0]


def calc_cond_uncond_batch(
    model, cond, uncond, x_in, timestep, model_options
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate conditioned and unconditioned predictions in a batch.

    Args:
        model: The denoising model.
        cond: Positive conditioning.
        uncond: Negative conditioning.
        x_in: Current noisy latent.
        timestep: Current timestep.
        model_options: Model options dict.

    Returns:
        Tuple of (cond_pred, uncond_pred) tensors.
    """
    # TODO(Phase3): Implement batched cond/uncond calculation.
    raise NotImplementedError(
        "calc_cond_uncond_batch is a stub. Will be implemented in Phase 3."
    )


def prepare_sampling_callback(steps: int, callback=None):
    """Prepare a callback wrapper for sampling progress.

    Args:
        steps: Total number of sampling steps.
        callback: User-provided callback function.

    Returns:
        Wrapped callback suitable for sampler use.
    """
    # TODO(Phase3): Implement callback wrapping.
    return callback


def cleanup_additional_models(models: set):
    """Clean up additional models after sampling.

    Args:
        models: Set of model objects to clean up.
    """
    # TODO(Phase3): Implement model cleanup.
    pass


def prepare_mask(noise_mask: torch.Tensor, shape, device) -> torch.Tensor:
    """Prepare a noise mask for the given latent shape.

    Import-compat stub: custom nodes (e.g. ComfyUI-KJNodes) reference
    ``comfy.sampler_helpers.prepare_mask`` at import time.  The compat
    implementation interpolates the mask to match the target spatial
    dims and moves it to the target device.

    Args:
        noise_mask: ``(B, 1, H, W)`` mask tensor in ``[0, 1]``.
        shape:      Target latent shape tuple.
        device:     Target device.
    """
    if noise_mask is None:
        return None
    mask = noise_mask
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    # Resize to match the target spatial dims
    target_h, target_w = shape[-2], shape[-1]
    if mask.shape[-2] != target_h or mask.shape[-1] != target_w:
        mask = torch.nn.functional.interpolate(
            mask, size=(target_h, target_w), mode="bilinear"
        )
    return mask.to(device)


def prepare_noise(latent_image: torch.Tensor, seed: int, noise_inds=None) -> torch.Tensor:
    """Deterministic per-seed noise generator for sampling.

    Import-compat re-export (ComfyUI's original lives in
    ``comfy.sample.prepare_noise``; many custom nodes import it from
    ``comfy.sampler_helpers`` instead).  Uses a torch.Generator seeded
    with the provided seed so two calls with the same seed produce
    identical noise.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(
        latent_image.shape,
        generator=generator,
        dtype=latent_image.dtype,
        device="cpu",
    )
