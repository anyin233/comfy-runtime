"""Regression tests for :class:`KSAMPLER.sample` device handling.

The img2img and inpainting workflows hit a device-mixing crash after the
GPU-residency fix landed: ``VAEEncode`` returns a latent that lives on the
compute device (because the VAE is now on CUDA), but ``_common_ksampler``
creates the initial noise on CPU (for cross-platform reproducibility of
``torch.Generator(device="cpu")``).  The sampler then ran
``latent_image + noise`` *before* moving either tensor to the target
device, which raised::

    RuntimeError: Expected all tensors to be on the same device, but
    found at least two devices, cuda:0 and cpu!

These tests pin down the sampler's contract: it must accept
``latent_image`` and ``noise`` on *different* devices (at least one CPU
and one CUDA) and move them to the UNet's device before combining.
"""
import pytest
import torch

from comfy_runtime.compat.comfy import samplers
from tests.fixtures.tiny_sd15 import make_tiny_sd15


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="device-mixing bug only reproduces on hosts with CUDA",
)


def _make_conditioning(hidden_size: int) -> list:
    """Synthesize a ComfyUI-shape conditioning for the tiny fixture."""
    cond = torch.randn(1, 77, hidden_size)
    return [[cond, {"pooled_output": cond[:, 0, :]}]]


@needs_cuda
def test_ksampler_accepts_cuda_latent_and_cpu_noise():
    """Mirror the img2img flow: latent comes from a CUDA VAE, noise is CPU."""
    comp = make_tiny_sd15()
    unet = comp["unet"].to(device="cuda", dtype=torch.float16)
    text_encoder = comp["text_encoder"]
    hidden = text_encoder.config.hidden_size

    sampler = samplers.sampler_object("euler")
    sigmas = samplers.calculate_sigmas(None, "normal", 2)

    latent_image = torch.zeros(1, 4, 8, 8, device="cuda", dtype=torch.float32)
    noise = torch.randn(1, 4, 8, 8, device="cpu", dtype=torch.float32)

    # Before the fix this raised:
    # ``RuntimeError: Expected all tensors to be on the same device ...``
    out = sampler.sample(
        model=unet,
        noise=noise,
        positive=_make_conditioning(hidden),
        negative=_make_conditioning(hidden),
        cfg=7.0,
        latent_image=latent_image,
        sigmas=sigmas,
        disable_pbar=True,
        seed=42,
    )
    assert out.shape == latent_image.shape
    assert torch.isfinite(out).all(), "sampler produced NaN/Inf"
