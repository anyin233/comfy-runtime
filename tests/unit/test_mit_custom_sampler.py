"""Tests for the ported comfy_extras.nodes_custom_sampler.

ComfyUI's custom-sampler nodes split the sampling pipeline into
discrete pieces so workflows can compose them:

  Noise        — RandomNoise / DisableNoise nodes produce a noise
                 generator object
  Sigmas       — BasicScheduler / KarrasScheduler / ExponentialScheduler
                 produce a sigma schedule tensor
  Sampler      — KSamplerSelect produces a SAMPLER object
  Guider       — BasicGuider / CFGGuider wrap a model with guidance
  Drive        — SamplerCustomAdvanced ties them together and runs
                 the loop

The tests below verify each piece returns the expected ComfyUI
contract type.
"""
import pytest
import torch

from tests.fixtures.tiny_sd15 import make_tiny_sd15
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat.comfy.sd import CLIP
from comfy_runtime.compat.comfy_extras import nodes_custom_sampler as ncs


# --- Schedulers ------------------------------------------------------


def test_basic_scheduler_returns_sigma_tensor():
    out = ncs.BasicScheduler().get_sigmas(
        model=None, scheduler="normal", steps=10, denoise=1.0
    )
    sigmas = out[0]
    assert isinstance(sigmas, torch.Tensor)
    assert sigmas.shape == (11,)  # steps + 1


def test_karras_scheduler_returns_sigma_tensor():
    sigmas = ncs.KarrasScheduler().get_sigmas(
        steps=10, sigma_max=14.6, sigma_min=0.0292, rho=7.0
    )[0]
    assert sigmas.shape == (11,)
    assert sigmas[0] >= sigmas[-1]  # decreasing


def test_exponential_scheduler_returns_sigma_tensor():
    sigmas = ncs.ExponentialScheduler().get_sigmas(
        steps=10, sigma_max=14.6, sigma_min=0.0292
    )[0]
    assert sigmas.shape == (11,)


# --- KSamplerSelect --------------------------------------------------


def test_ksampler_select_returns_named_sampler():
    (sampler,) = ncs.KSamplerSelect().get_sampler(sampler_name="euler")
    assert hasattr(sampler, "sample")


# --- RandomNoise -----------------------------------------------------


def test_random_noise_produces_deterministic_noise():
    (noise,) = ncs.RandomNoise().get_noise(noise_seed=42)
    latent = {"samples": torch.zeros(1, 4, 8, 8)}

    n1 = noise.generate_noise(latent)
    n2 = noise.generate_noise(latent)
    assert torch.allclose(n1, n2)


def test_random_noise_different_seeds_diverge():
    (n1,) = ncs.RandomNoise().get_noise(noise_seed=1)
    (n2,) = ncs.RandomNoise().get_noise(noise_seed=2)
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    assert not torch.allclose(n1.generate_noise(latent), n2.generate_noise(latent))


# --- BasicGuider -----------------------------------------------------


def test_basic_guider_wraps_model_and_conditioning():
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])
    clip = CLIP(clip_model=comp["text_encoder"], tokenizer=comp["tokenizer"])
    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a"))

    (guider,) = ncs.BasicGuider().get_guider(model=model, conditioning=pos)
    assert hasattr(guider, "model_patcher")
    assert hasattr(guider, "predict_noise")


# --- SamplerCustomAdvanced -------------------------------------------


def test_sampler_custom_advanced_runs_full_pipeline():
    """End-to-end: noise + sigmas + sampler + guider → denoised latent."""
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])
    clip = CLIP(clip_model=comp["text_encoder"], tokenizer=comp["tokenizer"])
    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a"))

    (noise,) = ncs.RandomNoise().get_noise(noise_seed=7)
    sigmas = ncs.BasicScheduler().get_sigmas(
        model=model, scheduler="normal", steps=2, denoise=1.0
    )[0]
    (sampler,) = ncs.KSamplerSelect().get_sampler(sampler_name="euler")
    (guider,) = ncs.BasicGuider().get_guider(model=model, conditioning=pos)

    latent_dict = {"samples": torch.zeros(1, 4, 8, 8)}

    out, _ = ncs.SamplerCustomAdvanced().sample(
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=latent_dict,
    )
    assert "samples" in out
    assert out["samples"].shape == latent_dict["samples"].shape
    assert not torch.isnan(out["samples"]).any()
