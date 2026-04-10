"""Tests for standalone UNet / CLIP / VAE file loaders (not full checkpoints).

These power the ``UNETLoader``, ``CLIPLoader``, and ``VAELoader`` nodes
for workflows that ship the three components as separate safetensors
files (common for Flux, SD3, and large custom models).

Phase 1 supports: a standalone VAE file whose state dict matches the
diffusers ``AutoencoderKL`` format.  Standalone UNet and CLIP are
stubbed with the tiny fixture fallback for the unit tests — real
workflows need actual large weights and will be covered by integration
tests once a checkpoint download fixture exists (Phase 5).
"""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from comfy_runtime.compat.comfy.sd import (
    CLIP,
    CLIPType,
    VAE,
    load_clip,
    load_unet,
    load_vae,
)
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


def test_load_vae_returns_vae_wrapper(tmp_path):
    """Loading a plausible VAE state dict via load_vae() returns a VAE."""
    # Write a tiny placeholder safetensors that our loader can't read with
    # diffusers (no full config) — it will fall back to the tiny fixture.
    sd = {"encoder.conv_in.weight": torch.randn(16, 3, 3, 3)}
    path = tmp_path / "tiny_vae.safetensors"
    save_file(sd, str(path))

    vae = load_vae(str(path))
    assert isinstance(vae, VAE)
    assert vae.vae_model is not None


def test_load_vae_encode_decode_roundtrips(tmp_path):
    sd = {"encoder.conv_in.weight": torch.randn(16, 3, 3, 3)}
    path = tmp_path / "tiny_vae.safetensors"
    save_file(sd, str(path))

    vae = load_vae(str(path))
    img = torch.rand(1, 32, 32, 3)
    latent = vae.encode(img)
    out = vae.decode(latent)
    assert out.shape == img.shape


def test_load_clip_returns_clip_wrapper(tmp_path):
    """Standalone CLIP loading returns a CLIP wrapper with tokenizer."""
    sd = {"text_model.embeddings.token_embedding.weight": torch.randn(1000, 32)}
    path = tmp_path / "tiny_clip.safetensors"
    save_file(sd, str(path))

    clip = load_clip(str(path), clip_type=CLIPType.SD1)
    assert isinstance(clip, CLIP)
    assert clip.clip_model is not None
    assert clip.tokenizer is not None


def test_load_clip_can_encode_after_load(tmp_path):
    sd = {"text_model.embeddings.token_embedding.weight": torch.randn(1000, 32)}
    path = tmp_path / "tiny_clip.safetensors"
    save_file(sd, str(path))

    clip = load_clip(str(path), clip_type=CLIPType.SD1)
    tokens = clip.tokenize("a dog")
    cond = clip.encode_from_tokens(tokens)
    assert cond.dim() == 3  # (B, seq, hidden)


def test_load_unet_returns_model_patcher(tmp_path):
    sd = {"time_embed.0.weight": torch.randn(128, 32)}
    path = tmp_path / "tiny_unet.safetensors"
    save_file(sd, str(path))

    patcher = load_unet(str(path))
    assert isinstance(patcher, ModelPatcher)
    assert patcher.model is not None


def test_load_unet_accepts_dtype_arg(tmp_path):
    """dtype arg is accepted (fp8 casting is Phase 3)."""
    sd = {"time_embed.0.weight": torch.randn(128, 32)}
    path = tmp_path / "tiny_unet.safetensors"
    save_file(sd, str(path))

    patcher = load_unet(str(path), dtype=torch.float16)
    assert patcher.model is not None
