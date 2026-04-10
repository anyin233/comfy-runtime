"""Tests for SDXL dual-encoder checkpoint loading.

SDXL differs from SD1.5 in two important ways:

1. **Dual text encoders** — SDXL concatenates the CLIP-L (hidden=768)
   and OpenCLIP-G (hidden=1280) outputs to produce a 2048-dim
   conditioning, plus a separate pooled output from the G encoder.
   ComfyUI's ``CLIP`` wrapper handles both via the ``"l"`` and ``"g"``
   slots in the tokens dict.

2. **Different state-dict namespace** — SDXL checkpoints have keys
   under ``conditioner.embedders.0`` (CLIP-L) and
   ``conditioner.embedders.1`` (OpenCLIP-G), whereas SD1.5 uses the
   single ``cond_stage_model.transformer`` path.

This test file verifies that:

* ``detect_model_family`` recognizes SDXL from the ``conditioner.embedders.1``
  key pattern.
* ``load_checkpoint_guess_config`` returns a ``CLIP`` wrapper with both
  encoders populated.
* ``CLIP.tokenize`` produces both ``"l"`` and ``"g"`` slots.
* ``CLIP.encode_from_tokens`` concatenates the 768 + 1280 = 2048 outputs
  and returns a separate pooled tensor.

For unit tests we use a tiny SDXL fixture — two synthesized
CLIPTextModel instances with hidden_size 32 and 48 — so we can verify
the concatenation dim is 80 without shipping a real 6 GB checkpoint.
"""
import pytest
import torch
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from comfy_runtime.compat.comfy import _diffusers_loader
from comfy_runtime.compat.comfy._diffusers_loader import detect_model_family
from comfy_runtime.compat.comfy.sd import CLIP, load_checkpoint_guess_config


def _make_tiny_sdxl_clip():
    """Build the two-text-encoder stack used by SDXL (both are CLIPTextModel)."""
    l_cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77,
    )
    g_cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=48, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=3, max_position_embeddings=77,
    )
    l = CLIPTextModel(l_cfg).eval()
    g = CLIPTextModel(g_cfg).eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return l, g, tok


def _write_tiny_sdxl_state(tmp_path):
    """Synthetic SDXL-shaped safetensors for family detection only."""
    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(32, 4, 3, 3),
        "first_stage_model.encoder.conv_in.weight": torch.randn(16, 3, 3, 3),
        # The SDXL-distinguishing key:
        "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(1000, 32),
        "conditioner.embedders.1.model.token_embedding.weight": torch.randn(1000, 48),
    }
    path = tmp_path / "tiny_sdxl.safetensors"
    save_file(sd, str(path))
    return str(path)


# --- detect_model_family ---------------------------------------------


def test_detect_sdxl_from_conditioner_embedders_key():
    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(4),
        "first_stage_model.encoder.conv_in.weight": torch.zeros(4),
        "conditioner.embedders.1.model.token_embedding.weight": torch.zeros(4),
    }
    assert detect_model_family(sd) == "sdxl"


def test_detect_sd15_still_works_after_sdxl_branch():
    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(4),
        "first_stage_model.encoder.conv_in.weight": torch.zeros(4),
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.zeros(4),
    }
    assert detect_model_family(sd) == "sd15"


# --- CLIP dual-slot tokenize / encode ---------------------------------


def test_clip_dual_slot_tokenize():
    """When both encoders are present, tokenize() produces 'l' and 'g'."""
    l, g, tok = _make_tiny_sdxl_clip()
    clip = CLIP(clip_model=l, tokenizer=tok)
    # Tell the wrapper about the second encoder
    clip.clip_model2 = g

    tokens = clip.tokenize("a cat")
    assert "l" in tokens
    assert "g" in tokens
    assert len(tokens["l"][0]) == 77
    assert len(tokens["g"][0]) == 77


def test_clip_dual_slot_encode_concatenates_hidden_dims():
    """encode_from_tokens on a dual-slot tokens dict concatenates along the hidden dim."""
    l, g, tok = _make_tiny_sdxl_clip()
    clip = CLIP(clip_model=l, tokenizer=tok)
    clip.clip_model2 = g

    tokens = clip.tokenize("a cat")
    cond = clip.encode_from_tokens(tokens)
    # (B, 77, 32 + 48) = (1, 77, 80)
    assert cond.shape == (1, 77, 80)


def test_clip_dual_slot_encode_with_pooled_uses_g():
    """SDXL's pooled output comes from OpenCLIP-G only, not CLIP-L."""
    l, g, tok = _make_tiny_sdxl_clip()
    clip = CLIP(clip_model=l, tokenizer=tok)
    clip.clip_model2 = g

    tokens = clip.tokenize("a cat")
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    assert cond.shape == (1, 77, 80)
    # Pooled from G only → shape (1, 48)
    assert pooled.shape == (1, 48)


def test_clip_single_slot_still_works_after_dual_support():
    """Existing SD1.5 single-slot path must not regress."""
    from tests.fixtures.tiny_sd15 import make_tiny_sd15

    comp = make_tiny_sd15()
    clip = CLIP(clip_model=comp["text_encoder"], tokenizer=comp["tokenizer"])

    tokens = clip.tokenize("a cat")
    cond = clip.encode_from_tokens(tokens)
    assert cond.shape == (1, 77, 32)


# --- load_checkpoint_guess_config dispatch ---------------------------


def test_load_checkpoint_sdxl_returns_dual_encoder_clip(tmp_path, monkeypatch):
    """A state-dict with conditioner.embedders.1 keys triggers SDXL loading."""
    ckpt = _write_tiny_sdxl_state(tmp_path)

    # Monkeypatch the _diffusers_loader to fall back to the tiny SDXL fixture.
    # The real StableDiffusionXLPipeline.from_single_file would try to load
    # full weights and choke on the synthetic state dict.
    def _fake_load_sdxl_single_file(path):
        l, g, tok = _make_tiny_sdxl_clip()
        from tests.fixtures.tiny_sd15 import make_tiny_sd15

        fallback = make_tiny_sd15()
        return fallback["unet"], fallback["vae"], l, g, tok

    monkeypatch.setattr(
        _diffusers_loader,
        "load_sdxl_single_file",
        _fake_load_sdxl_single_file,
    )

    model, clip, vae, _ = load_checkpoint_guess_config(ckpt)
    assert clip is not None
    assert clip.clip_model is not None
    assert hasattr(clip, "clip_model2") and clip.clip_model2 is not None
