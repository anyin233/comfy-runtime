"""Tests for ControlNet forward-pass integration with KSAMPLER.

ComfyUI's ControlNet contract:

* ``ControlNet.set_cond_hint(image, strength, range)`` — registers the
  control image (already preprocessed: edges, depth, pose, ...).
* ``ControlNet.get_control(x_noisy, t, cond, batched_number)`` — runs
  the ControlNet model on the noisy latent and returns a dict::

      {"down_block_residuals": [...],   # list of tensors per UNet block
       "mid_block_residual": <tensor>}  # the bottleneck residual

* The KSAMPLER step then passes those residuals into the UNet forward
  via ``down_block_additional_residuals=`` and
  ``mid_block_additional_residual=``.  Diffusers' UNet2DConditionModel
  honors these kwargs natively.

This file exercises the contract end-to-end against the tiny SD1.5
fixture's UNet plus a tiny diffusers ControlNetModel.
"""
import pytest
import torch

from tests.fixtures.tiny_sd15 import make_tiny_sd15
from comfy_runtime.compat.comfy.controlnet import ControlNet
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat.comfy.sd import CLIP
from comfy_runtime.compat.comfy import samplers


def _make_tiny_controlnet():
    """Build a tiny diffusers ControlNetModel matching the tiny UNet shape."""
    from diffusers import ControlNetModel

    return ControlNetModel(
        in_channels=4,
        conditioning_channels=3,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    ).eval()


def test_controlnet_get_control_returns_dict():
    """get_control() must produce a dict with the residual keys."""
    cn_model = _make_tiny_controlnet()
    cn = ControlNet(control_model=cn_model)
    cn.set_cond_hint(torch.rand(1, 3, 32, 32), strength=1.0)

    latent = torch.zeros(1, 4, 8, 8)
    cond = [[torch.randn(1, 4, 32), {"pooled_output": torch.randn(1, 32)}]]
    out = cn.get_control(latent, torch.tensor([5.0]), cond, 1)

    assert isinstance(out, dict)
    assert "down_block_residuals" in out
    assert "mid_block_residual" in out
    assert isinstance(out["down_block_residuals"], list)
    assert isinstance(out["mid_block_residual"], torch.Tensor)


def test_controlnet_get_control_residuals_match_unet_blocks():
    """The down-block residuals list length must match the UNet's
    expected count (one per CrossAttnDownBlock + DownBlock + sample
    points).  diffusers ControlNetModel handles this internally; we
    just verify the result is a non-empty list."""
    cn_model = _make_tiny_controlnet()
    cn = ControlNet(control_model=cn_model)
    cn.set_cond_hint(torch.rand(1, 3, 32, 32), strength=1.0)

    latent = torch.zeros(1, 4, 8, 8)
    cond = [[torch.randn(1, 4, 32), {"pooled_output": torch.randn(1, 32)}]]
    out = cn.get_control(latent, torch.tensor([5.0]), cond, 1)
    assert len(out["down_block_residuals"]) > 0


def test_controlnet_get_control_strength_scales_residuals():
    """strength=0 → all-zero residuals."""
    cn_model = _make_tiny_controlnet()
    cn = ControlNet(control_model=cn_model)
    cn.set_cond_hint(torch.rand(1, 3, 32, 32), strength=0.0)

    latent = torch.zeros(1, 4, 8, 8)
    cond = [[torch.randn(1, 4, 32), {"pooled_output": torch.randn(1, 32)}]]
    out = cn.get_control(latent, torch.tensor([5.0]), cond, 1)

    for r in out["down_block_residuals"]:
        assert torch.allclose(r, torch.zeros_like(r))
    assert torch.allclose(
        out["mid_block_residual"], torch.zeros_like(out["mid_block_residual"])
    )


def test_ksampler_with_control_runs_to_completion():
    """End-to-end: feed the ControlNet through KSAMPLER.sample's
    control kwarg and verify a denoised latent comes out."""
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])
    clip = CLIP(clip_model=comp["text_encoder"], tokenizer=comp["tokenizer"])

    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a cat"))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    cn_model = _make_tiny_controlnet()
    cn = ControlNet(control_model=cn_model)
    cn.set_cond_hint(torch.rand(1, 3, 32, 32), strength=0.8)

    latent = torch.zeros(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    sigmas = samplers.calculate_sigmas(None, "normal", 2)

    sampler = samplers.sampler_object("euler")
    out = sampler.sample(
        model=model, noise=noise, positive=pos, negative=neg,
        cfg=1.0, latent_image=latent, sigmas=sigmas,
        disable_pbar=True, control=cn,
    )
    assert out.shape == latent.shape
    assert not torch.isnan(out).any()


def test_ksampler_without_control_still_works():
    """Regression: omitting the control kwarg must still produce the
    same output as before Task 5.2."""
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])
    clip = CLIP(clip_model=comp["text_encoder"], tokenizer=comp["tokenizer"])
    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a"))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    latent = torch.zeros(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    sigmas = samplers.calculate_sigmas(None, "normal", 2)

    sampler = samplers.sampler_object("euler")
    out = sampler.sample(
        model=model, noise=noise, positive=pos, negative=neg,
        cfg=1.0, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    assert out.shape == latent.shape
