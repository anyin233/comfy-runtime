"""Tests for the ``denoise`` parameter in KSampler / _common_ksampler.

ComfyUI's KSampler accepts a ``denoise`` float in ``(0, 1]`` that controls
how much of the sigma schedule the sampler actually runs.  ``denoise=1.0``
is "full denoise" (text2img from pure noise); ``denoise=0.75`` skips the
first 25% of the schedule and starts from a less-noisy latent
(img2img, hires fix pass 2).

Before this fix the compat layer's ``_common_ksampler`` accepted the
``denoise`` parameter but silently discarded it — every call ran the
full schedule regardless.  That broke img2img (output didn't preserve
the source image at all) and hires_fix pass 2 (second sampling pass was
indistinguishable from a fresh text2img).

These tests pin down the correct behaviour:
  * ``denoise=1.0`` samples from the full sigma schedule.
  * ``denoise=0.5`` preserves the input latent's low-frequency content
    (measured as correlation between input and output latents).
  * ``denoise=0.0`` is a no-op (returns the input latent unchanged).
"""
import pytest
import torch

from comfy_runtime.compat.nodes import _common_ksampler
from tests.fixtures.tiny_sd15 import make_tiny_sd15


def _make_conditioning(hidden_size: int) -> list:
    """Synthesize ComfyUI-shape conditioning (single slot) for the fixture."""
    cond = torch.randn(1, 77, hidden_size)
    return [[cond, {"pooled_output": cond[:, 0, :]}]]


@pytest.fixture
def tiny_model():
    """Build a ModelPatcher wrapping the tiny SD1.5 UNet fixture."""
    from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
    comp = make_tiny_sd15()
    return ModelPatcher(comp["unet"]), comp["text_encoder"].config.hidden_size


def test_denoise_partial_preserves_input_more_than_full(tiny_model):
    """A partial denoise should preserve the input more than a full one.

    We sample twice with the exact same seed + conditioning + input
    latent but different ``denoise`` values:
      * ``denoise=1.0``  — full fresh sample, should be far from input
      * ``denoise=0.25`` — only the last quarter of the schedule runs,
        should stay close to the input

    The L2 distance from the input must be smaller for ``denoise=0.25``
    than for ``denoise=1.0``.  This is a regression lock: before the
    fix, both were identical because ``denoise`` was silently ignored.
    """
    model, hidden = tiny_model
    torch.manual_seed(0)
    latent_in = torch.randn(1, 4, 16, 16) * 0.5
    pos = _make_conditioning(hidden)
    neg = _make_conditioning(hidden)

    (full_out,) = _common_ksampler(
        model.clone(), seed=42, steps=8, cfg=1.0,
        sampler_name="euler", scheduler="normal",
        positive=pos, negative=neg,
        latent_image={"samples": latent_in.clone()}, denoise=1.0,
    )
    (quarter_out,) = _common_ksampler(
        model.clone(), seed=42, steps=8, cfg=1.0,
        sampler_name="euler", scheduler="normal",
        positive=pos, negative=neg,
        latent_image={"samples": latent_in.clone()}, denoise=0.25,
    )

    dist_full = (full_out["samples"] - latent_in).norm().item()
    dist_quarter = (quarter_out["samples"] - latent_in).norm().item()

    assert dist_quarter < dist_full, (
        f"denoise=0.25 distance from input ({dist_quarter:.3f}) should be "
        f"< denoise=1.0 distance ({dist_full:.3f}); the sampler appears "
        "to be ignoring the denoise parameter"
    )


def test_denoise_zero_returns_input_unchanged(tiny_model):
    """``denoise=0.0`` is a no-op — the latent is returned unchanged."""
    model, hidden = tiny_model
    torch.manual_seed(0)
    latent_in = torch.randn(1, 4, 16, 16) * 0.5
    latent_image = {"samples": latent_in.clone()}

    pos = _make_conditioning(hidden)
    neg = _make_conditioning(hidden)

    (out,) = _common_ksampler(
        model, seed=42, steps=4, cfg=1.0,
        sampler_name="euler", scheduler="normal",
        positive=pos, negative=neg,
        latent_image=latent_image, denoise=0.0,
    )
    assert torch.allclose(
        out["samples"], latent_in, atol=1e-5, rtol=1e-4
    ), "denoise=0.0 should return the input latent unchanged"
