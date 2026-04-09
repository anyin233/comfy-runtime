"""Sampling entry points for comfy_runtime.

MIT reimplementation of comfy.sample — provides the top-level sample
functions and noise preparation utilities that ComfyUI nodes call.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise preparation
# ---------------------------------------------------------------------------


def prepare_noise(
    latent_image: torch.Tensor, seed: int, noise_inds: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate reproducible noise matching the shape of latent_image.

    Creates a CPU-based random noise tensor using the given seed for
    reproducibility. If noise_inds is provided, generates noise for
    each index independently and stacks the results.

    Args:
        latent_image: Reference latent tensor whose shape/dtype to match.
        seed: Integer seed for the random generator.
        noise_inds: Optional tensor of batch indices for per-frame noise.

    Returns:
        Noise tensor on CPU with the same shape and dtype as latent_image.
    """
    generator = torch.manual_seed(seed)
    if noise_inds is not None:
        # Generate unique noise per index by offsetting the seed
        unique_inds = noise_inds.unique()
        noises = []
        for ind in unique_inds:
            # Use a distinct generator state per index
            gen = torch.manual_seed(seed + int(ind.item()))
            noise = torch.randn(
                latent_image[0:1].size(),
                dtype=latent_image.dtype,
                layout=latent_image.layout,
                generator=gen,
                device="cpu",
            )
            noises.append(noise)
        # Map indices to generated noises
        noise_map = {int(ind.item()): noises[i] for i, ind in enumerate(unique_inds)}
        result = torch.cat([noise_map[int(idx.item())] for idx in noise_inds], dim=0)
        return result

    return torch.randn(
        latent_image.size(),
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        generator=generator,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Latent channel fixing
# ---------------------------------------------------------------------------


def fix_empty_latent_channels(
    model, latent_image: torch.Tensor, downscale_ratio_spacial=None
) -> torch.Tensor:
    """Ensure latent_image has the correct number of channels for the model.

    If the model expects more channels than the latent has, pads with zeros.
    If the model expects fewer, truncates.

    Args:
        model: Model object, checked for `.model.latent_format.latent_channels`
            or `.latent_channels` attribute.
        latent_image: Latent tensor of shape (B, C, ...).

    Returns:
        Latent tensor with the correct channel count.
    """
    # Determine expected channels
    expected_channels = None
    if hasattr(model, "get_model_object"):
        try:
            latent_format = model.get_model_object("latent_format")
            if hasattr(latent_format, "latent_channels"):
                expected_channels = latent_format.latent_channels
        except Exception:
            pass

    if expected_channels is None and hasattr(model, "model"):
        m = model.model
        if hasattr(m, "latent_format") and hasattr(m.latent_format, "latent_channels"):
            expected_channels = m.latent_format.latent_channels

    if expected_channels is None:
        # Cannot determine expected channels — return as-is
        return latent_image

    current_channels = latent_image.shape[1]
    if current_channels == expected_channels:
        return latent_image

    if current_channels < expected_channels:
        # Pad with zeros
        pad_shape = list(latent_image.shape)
        pad_shape[1] = expected_channels - current_channels
        padding = torch.zeros(
            pad_shape, dtype=latent_image.dtype, device=latent_image.device
        )
        return torch.cat([latent_image, padding], dim=1)

    # Truncate extra channels
    return latent_image[:, :expected_channels]


# ---------------------------------------------------------------------------
# Top-level sampling stubs
# ---------------------------------------------------------------------------


def sample(
    model,
    noise,
    positive,
    negative,
    cfg,
    device,
    sampler,
    sigmas,
    model_options=None,
    latent_image=None,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    """Run the standard sampling pipeline.

    This is the main entry point called by KSampler and related nodes.
    Currently a stub that will be implemented in Phase 3.

    Args:
        model: The model patcher.
        noise: Initial noise tensor.
        positive: Positive conditioning.
        negative: Negative conditioning.
        cfg: Classifier-free guidance scale.
        device: Target device.
        sampler: Sampler object or KSAMPLER.
        sigmas: Sigma schedule tensor.
        model_options: Optional model options dict.
        latent_image: Optional starting latent for img2img.
        denoise_mask: Optional mask for inpainting.
        callback: Progress callback.
        disable_pbar: Disable progress bar.
        seed: Random seed.

    Returns:
        Denoised latent tensor.

    Raises:
        NotImplementedError: Always (Phase 3 work).
    """
    # TODO(Phase3): Implement full sampling pipeline with model loading,
    # conditioning encoding, CFG guidance, and sampler execution.
    raise NotImplementedError(
        "comfy.sample.sample is a stub. "
        "Full sampling pipeline will be implemented in Phase 3."
    )


def sample_custom(
    model,
    noise,
    cfg,
    sampler_object,
    sigmas,
    positive=None,
    negative=None,
    latent_image=None,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    model_options=None,
):
    """Run sampling with a custom sampler object.

    Provides a lower-level interface for custom sampling workflows.
    Currently a stub that will be implemented in Phase 3.

    Args:
        model: The model patcher.
        noise: Initial noise tensor.
        cfg: Classifier-free guidance scale.
        sampler_object: Custom sampler callable.
        sigmas: Sigma schedule tensor.
        positive: Optional positive conditioning.
        negative: Optional negative conditioning.
        latent_image: Optional starting latent.
        denoise_mask: Optional inpainting mask.
        callback: Progress callback.
        disable_pbar: Disable progress bar.
        seed: Random seed.
        model_options: Optional model options dict.

    Returns:
        Denoised latent tensor.

    Raises:
        NotImplementedError: Always (Phase 3 work).
    """
    # TODO(Phase3): Implement custom sampling pipeline.
    raise NotImplementedError(
        "comfy.sample.sample_custom is a stub. "
        "Custom sampling will be implemented in Phase 3."
    )
