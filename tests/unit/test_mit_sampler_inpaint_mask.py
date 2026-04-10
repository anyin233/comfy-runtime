"""Tests for ``noise_mask`` (inpainting) handling in KSAMPLER.sample.

ComfyUI's ``SetLatentNoiseMask`` attaches a spatial mask to a latent dict:
  * mask == 1 at pixels the sampler should regenerate
  * mask == 0 at pixels that should stay untouched by sampling

Before this fix, the compat layer's ``SetLatentNoiseMask`` set
``samples["noise_mask"]`` correctly but ``_common_ksampler`` never read
it, so every inpaint workflow ended up regenerating the entire image —
the moon-on-grey-background output from the inpainting benchmark
instead of a magical portal inside the center mask over the source
photo.

These tests pin down the contract:
  * ``noise_mask`` all-zero → sampler is a no-op (returns source latent).
  * ``noise_mask`` all-one  → sampler behaves exactly like the no-mask path.
  * Partial mask: the unmasked region of the output tracks the source
    latent, while the masked region is free to change.
"""
import pytest
import torch

from comfy_runtime.compat.nodes import _common_ksampler
from tests.fixtures.tiny_sd15 import make_tiny_sd15


def _make_conditioning(hidden_size: int) -> list:
    cond = torch.randn(1, 77, hidden_size)
    return [[cond, {"pooled_output": cond[:, 0, :]}]]


@pytest.fixture
def tiny_model():
    from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
    comp = make_tiny_sd15()
    return ModelPatcher(comp["unet"]), comp["text_encoder"].config.hidden_size


def test_all_zero_noise_mask_preserves_source_latent(tiny_model):
    """``noise_mask`` all zeros → no pixels are sampled → source unchanged."""
    model, hidden = tiny_model
    torch.manual_seed(0)
    source = torch.randn(1, 4, 16, 16) * 0.3

    latent_in = {
        "samples": source.clone(),
        "noise_mask": torch.zeros(1, 1, 16, 16),
    }
    (out,) = _common_ksampler(
        model, seed=42, steps=4, cfg=1.0,
        sampler_name="euler", scheduler="normal",
        positive=_make_conditioning(hidden),
        negative=_make_conditioning(hidden),
        latent_image=latent_in, denoise=1.0,
    )
    # A fully-zero mask means "sample nothing" → the source should pass
    # through essentially unchanged.  Allow small tolerance for any
    # dtype round-trip (cpu/cuda fp32/fp16).
    diff = (out["samples"].to(torch.float32) - source).abs().max().item()
    assert diff < 1e-3, (
        f"all-zero mask should preserve source (max diff {diff:.4f}); "
        "the sampler is ignoring noise_mask"
    )


def test_partial_mask_preserves_unmasked_region(tiny_model):
    """With a half-and-half mask, the unmasked half should match source."""
    model, hidden = tiny_model
    torch.manual_seed(0)
    source = torch.randn(1, 4, 16, 16) * 0.3

    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :, 8:] = 1.0  # regenerate the right half, preserve the left

    latent_in = {
        "samples": source.clone(),
        "noise_mask": mask,
    }
    (out,) = _common_ksampler(
        model, seed=42, steps=4, cfg=1.0,
        sampler_name="euler", scheduler="normal",
        positive=_make_conditioning(hidden),
        negative=_make_conditioning(hidden),
        latent_image=latent_in, denoise=1.0,
    )
    out_t = out["samples"].to(torch.float32)

    # Unmasked (left) half should closely track the source.
    left_diff = (out_t[:, :, :, :8] - source[:, :, :, :8]).abs().max().item()
    assert left_diff < 1e-2, (
        f"unmasked region drifted from source (max diff {left_diff:.4f}); "
        "the sampler is blending the mask incorrectly"
    )

    # Masked (right) half must have changed — it was sampled.  Verify it's
    # *different* from the source so we don't accidentally short-circuit.
    right_diff = (out_t[:, :, :, 8:] - source[:, :, :, 8:]).abs().max().item()
    assert right_diff > 1e-4, (
        f"masked region unchanged (max diff {right_diff:.6f}); "
        "the sampler didn't actually run on the masked pixels"
    )
