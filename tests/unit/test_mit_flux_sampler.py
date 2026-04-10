"""Tests for the Flux sampling path.

Flux uses :class:`diffusers.FluxTransformer2DModel` which has a
forward signature very different from SD1/SDXL's UNet2DConditionModel:

    forward(
        hidden_states,           # (B, num_patches, in_channels)
        encoder_hidden_states,   # (B, txt_seq, joint_attention_dim)
        pooled_projections,      # (B, pooled_projection_dim)
        timestep,                # (B,)
        img_ids,                 # (num_patches, 3)
        txt_ids,                 # (txt_seq, 3)
        guidance,                # (B,)  — Flux is CFG-distilled
    )

The companion scheduler is :class:`FlowMatchEulerDiscreteScheduler`,
not the regular EulerDiscreteScheduler.

Phase 5 introduces :class:`FluxKSAMPLER` that handles this signature
and a tiny ``make_tiny_flux()`` fixture so we can exercise the
end-to-end sampling loop without needing a real 24 GB Flux checkpoint.
"""
import pytest
import torch

from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat.comfy import samplers
from comfy_runtime.compat.comfy.sd import CLIP


# --- Tiny Flux fixture ----------------------------------------------


def _make_tiny_flux():
    """Build a 83 KB FluxTransformer2DModel + tiny CLIP-L + T5 stand-in."""
    from diffusers import FluxTransformer2DModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=64,
        pooled_projection_dim=32,
        axes_dims_rope=(4, 6, 6),
        guidance_embeds=True,
    ).eval()

    l_cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=77,
    )
    text_encoder_l = CLIPTextModel(l_cfg).eval()

    t5_cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=77,
    )
    text_encoder_t5 = CLIPTextModel(t5_cfg).eval()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    return {
        "transformer": transformer,
        "text_encoder_l": text_encoder_l,
        "text_encoder_t5": text_encoder_t5,
        "tokenizer": tokenizer,
    }


def _build_flux_stack():
    comp = _make_tiny_flux()
    model = ModelPatcher(comp["transformer"])
    clip = CLIP(
        clip_model=comp["text_encoder_l"],
        tokenizer=comp["tokenizer"],
        clip_model2=comp["text_encoder_t5"],
        family="flux",
    )
    return model, clip


# --- Sampler dispatch ------------------------------------------------


def test_flux_ksampler_class_exists():
    """KSAMPLER must accept a model_type='flux' kwarg or expose a
    FluxKSAMPLER class."""
    assert hasattr(samplers, "FluxKSAMPLER")


def test_flux_sampler_object_factory():
    """sampler_object should still return a usable sampler when the
    underlying model is a FluxTransformer2DModel; the dispatch happens
    inside .sample() based on the model type."""
    sampler = samplers.sampler_object("euler")
    assert hasattr(sampler, "sample")


def test_flux_ksampler_runs_through_tiny_transformer():
    """End-to-end: encode prompt → run flow-match scheduler loop →
    return a denoised latent of the same shape."""
    model, clip = _build_flux_stack()

    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a cat"))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))

    # Flux latents are packed: (B, num_patches, in_channels)
    # For our tiny config: in_channels=64, so patch the latent to that.
    latent = torch.zeros(1, 16, 64)
    noise = torch.randn(1, 16, 64)

    sigmas = torch.linspace(1.0, 0.0, 5)  # 4-step schedule

    sampler = samplers.FluxKSAMPLER(sampler_name="euler")
    out = sampler.sample(
        model=model,
        noise=noise,
        positive=pos,
        negative=neg,
        cfg=3.5,
        latent_image=latent,
        sigmas=sigmas,
        disable_pbar=True,
    )
    assert out.shape == latent.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_flux_ksampler_produces_deterministic_output():
    """Same noise + same prompt → same output."""
    model, clip = _build_flux_stack()
    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a cat"))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    latent = torch.zeros(1, 16, 64)

    torch.manual_seed(7)
    n1 = torch.randn(1, 16, 64)
    torch.manual_seed(7)
    n2 = torch.randn(1, 16, 64)

    sigmas = torch.linspace(1.0, 0.0, 3)
    sampler = samplers.FluxKSAMPLER(sampler_name="euler")

    out1 = sampler.sample(
        model=model, noise=n1, positive=pos, negative=neg,
        cfg=3.5, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    out2 = sampler.sample(
        model=model, noise=n2, positive=pos, negative=neg,
        cfg=3.5, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    assert torch.allclose(out1, out2)


def test_flux_dispatch_via_modelpatcher_class():
    """When ModelPatcher wraps a FluxTransformer2DModel, the standard
    KSAMPLER should auto-dispatch to FluxKSAMPLER on .sample() so
    callers don't need to know which sampler class to use."""
    model, clip = _build_flux_stack()
    pos = clip.encode_from_tokens_scheduled(clip.tokenize("a cat"))
    neg = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    latent = torch.zeros(1, 16, 64)
    noise = torch.randn(1, 16, 64)
    sigmas = torch.linspace(1.0, 0.0, 3)

    # Use the regular KSAMPLER — it should detect the Flux transformer
    # and route through FluxKSAMPLER internally.
    sampler = samplers.sampler_object("euler")
    out = sampler.sample(
        model=model, noise=noise, positive=pos, negative=neg,
        cfg=3.5, latent_image=latent, sigmas=sigmas, disable_pbar=True,
    )
    assert out.shape == latent.shape
