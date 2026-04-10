"""Tests for the MIT KSAMPLER implementation (Phase 1: euler family)."""
import pytest
import torch

from tests.fixtures.tiny_sd15 import make_tiny_sd15
from comfy_runtime.compat.comfy.sd import CLIP
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat.comfy import samplers


def _build_tiny_stack():
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])
    clip = CLIP(clip_model=comp["text_encoder"], tokenizer=comp["tokenizer"])
    return model, clip


def test_sampler_names_include_phase1_euler_family():
    """Phase-1 supported samplers must stay in the public list."""
    for name in ("euler", "euler_ancestral", "dpmpp_2m", "ddim"):
        assert name in samplers.SAMPLER_NAMES


def test_sampler_object_returns_ksampler_instance():
    sampler = samplers.sampler_object("euler")
    assert hasattr(sampler, "sample")


def test_ksampler_euler_runs_without_cfg():
    """CFG=1.0 → no classifier-free guidance, single forward pass per step."""
    model, clip = _build_tiny_stack()
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    latent = torch.zeros(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    sigmas = samplers.calculate_sigmas(None, "normal", 4)

    sampler = samplers.sampler_object("euler")
    out = sampler.sample(
        model=model,
        noise=noise,
        positive=positive,
        negative=negative,
        cfg=1.0,
        latent_image=latent,
        sigmas=sigmas,
        disable_pbar=True,
    )
    assert out.shape == latent.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_ksampler_euler_runs_with_cfg():
    """CFG>1.0 → batched uncond/cond noise prediction."""
    model, clip = _build_tiny_stack()
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    latent = torch.zeros(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    sigmas = samplers.calculate_sigmas(None, "normal", 2)

    sampler = samplers.sampler_object("euler")
    out = sampler.sample(
        model=model,
        noise=noise,
        positive=positive,
        negative=negative,
        cfg=7.5,
        latent_image=latent,
        sigmas=sigmas,
        disable_pbar=True,
    )
    assert out.shape == latent.shape
    assert not torch.isnan(out).any()


def test_ksampler_is_deterministic_across_runs():
    """Same inputs → identical output."""
    model, clip = _build_tiny_stack()
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    latent = torch.zeros(1, 4, 8, 8)
    torch.manual_seed(42)
    noise1 = torch.randn(1, 4, 8, 8)
    torch.manual_seed(42)
    noise2 = torch.randn(1, 4, 8, 8)

    sigmas = samplers.calculate_sigmas(None, "normal", 2)

    sampler = samplers.sampler_object("euler")
    out1 = sampler.sample(
        model=model, noise=noise1, positive=positive, negative=negative,
        cfg=1.0, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    out2 = sampler.sample(
        model=model, noise=noise2, positive=positive, negative=negative,
        cfg=1.0, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    assert torch.allclose(out1, out2)


def test_ksampler_runs_with_dpmpp_2m():
    """dpmpp_2m is the most common SDXL sampler; must work."""
    model, clip = _build_tiny_stack()
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    latent = torch.zeros(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    sigmas = samplers.calculate_sigmas(None, "normal", 2)

    sampler = samplers.sampler_object("dpmpp_2m")
    out = sampler.sample(
        model=model, noise=noise, positive=positive, negative=negative,
        cfg=1.0, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    assert out.shape == latent.shape
    assert not torch.isnan(out).any()


def test_ksampler_runs_with_uni_pc():
    """uni_pc is ComfyUI's fastest standard sampler; must work."""
    model, clip = _build_tiny_stack()
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    latent = torch.zeros(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    sigmas = samplers.calculate_sigmas(None, "normal", 2)

    sampler = samplers.sampler_object("uni_pc")
    out = sampler.sample(
        model=model, noise=noise, positive=positive, negative=negative,
        cfg=1.0, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    assert out.shape == latent.shape
    assert not torch.isnan(out).any()


def test_scheduler_map_covers_common_samplers():
    """Every sampler a typical workflow uses must be mapped."""
    from comfy_runtime.compat.comfy._scheduler_map import supported_sampler_names

    supported = set(supported_sampler_names())
    must_have = {
        "euler", "euler_ancestral", "heun", "lms",
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde",
        "dpmpp_2s_ancestral", "ddim", "ddpm", "lcm",
        "uni_pc", "uni_pc_bh2",
    }
    missing = must_have - supported
    assert not missing, f"missing sampler mappings: {missing}"


def test_unsupported_sampler_raises_helpful_error():
    """Phase-2 unsupported samplers should raise NotImplementedError with a hint."""
    from comfy_runtime.compat.comfy._scheduler_map import make_diffusers_scheduler

    with pytest.raises(NotImplementedError, match="Use 'dpmpp_2m'"):
        make_diffusers_scheduler("dpm_fast", "normal")


def test_unknown_sampler_raises_keyerror():
    from comfy_runtime.compat.comfy._scheduler_map import make_diffusers_scheduler

    with pytest.raises(KeyError):
        make_diffusers_scheduler("totally_made_up", "normal")
