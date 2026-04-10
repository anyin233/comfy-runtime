"""Synthetic tiny SD1.5-shaped pipeline for fast unit tests.

Produces a random-initialized diffusers UNet2DConditionModel,
AutoencoderKL, CLIPTextModel, CLIPTokenizer, and DDIMScheduler at
<20 MB total, no real weights.

All tests in the MIT-rewrite plan use this fixture as a fast stand-in
for a real SD1.5 checkpoint.  No network access is required *if* the
host's HuggingFace cache already contains ``openai/clip-vit-base-patch32``
(which is nearly universal since it's the CLIPTokenizer default).
"""
from typing import Dict

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


def _make_tiny_unet() -> UNet2DConditionModel:
    """A 2-block cross-attention UNet at <1 M params."""
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    )


def _make_tiny_vae() -> AutoencoderKL:
    """A 2-block VAE with 2x spatial downsample (mimics SD1.5 at mini scale).

    Two blocks produce exactly one downsample, so a 32x32 image becomes
    a 16x16 latent — the same 2x factor as real SD1.5 (except real uses 8x
    from 4 blocks).
    """
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
    )


def _make_tiny_text_encoder() -> CLIPTextModel:
    """A 2-layer CLIPTextModel with 32-d hidden size matching the tiny UNet."""
    config = CLIPTextConfig(
        vocab_size=49408,  # match the real CLIP vocab size so tokens are valid
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=77,
    )
    return CLIPTextModel(config)


def _make_tiny_tokenizer() -> CLIPTokenizer:
    """Real CLIP tokenizer from HuggingFace cache.

    Uses the canonical `openai/clip-vit-base-patch32` tokenizer because
    every SD1.5 tool depends on it, so it's nearly always cached locally.
    If not cached and the host is offline, this raises — fine for CI
    where the cache is warmed on first run.
    """
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def make_tiny_sd15() -> Dict:
    """Construct a tiny SD1.5-shaped pipeline dict.

    Returns:
        Dict with keys: unet, vae, text_encoder, tokenizer, scheduler
    """
    return {
        "unet": _make_tiny_unet().eval(),
        "vae": _make_tiny_vae().eval(),
        "text_encoder": _make_tiny_text_encoder().eval(),
        "tokenizer": _make_tiny_tokenizer(),
        "scheduler": DDIMScheduler(num_train_timesteps=1000),
    }
