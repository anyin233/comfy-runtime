"""Tests for Flux dual-encoder checkpoint loading.

Flux uses a very different architecture from SD1/SDXL:

* The denoiser is a ``FluxTransformer2DModel`` (not a UNet2DConditionModel).
  State-dict keys are ``double_blocks.*`` and ``single_blocks.*``.

* Two text encoders: CLIP-L (same as SD1) and T5-XXL (hidden=4096).
  The CLIP wrapper uses ``"l"`` and ``"t5xxl"`` slots; unlike SDXL,
  the two outputs are **not** concatenated — they flow into the
  transformer through separate projections (``pooled_projections``
  for CLIP-L pooled, ``encoder_hidden_states`` for the T5 sequence).

Phase 2 scope for this test:

* ``detect_model_family`` returns ``"flux"`` for state dicts with
  ``double_blocks`` / ``single_blocks`` keys.
* ``load_flux_single_file`` returns a tuple with the transformer,
  VAE, CLIP-L encoder, T5 encoder, and two tokenizers.
* ``CLIP`` wrapper accepts a ``t5xxl`` encoder and produces a
  ``{"l": ..., "t5xxl": ...}`` tokens dict.
* ``CLIP.encode_from_tokens`` with a Flux-style tokens dict returns
  ``(cond, pooled)`` where ``cond`` is the T5 sequence (4096-dim in
  real Flux, 48-dim in our tiny test) and ``pooled`` is the CLIP-L
  pooled output (768-dim / 32-dim).

Unit tests use the tiny SD1.5 fixture plus a synthetic T5-shaped
CLIPTextModel (re-used from the SDXL tiny stack) to keep things fast
and network-free.
"""
import pytest
import torch
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from comfy_runtime.compat.comfy import _diffusers_loader
from comfy_runtime.compat.comfy._diffusers_loader import detect_model_family
from comfy_runtime.compat.comfy.sd import CLIP, load_checkpoint_guess_config


def _make_tiny_t5_like():
    """Build a tiny 'T5-like' encoder for Flux tests.

    We reuse CLIPTextModel because it has the same HF output shape
    ``(B, seq, hidden)``.  In real Flux this would be a T5EncoderModel
    with hidden_size=4096; for the tiny test we use 64.
    """
    cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=77,
    )
    return CLIPTextModel(cfg).eval()


def _write_tiny_flux_state(tmp_path):
    """Flux-style state dict with double_blocks / single_blocks keys."""
    sd = {
        "double_blocks.0.img_attn.qkv.weight": torch.randn(192, 64),
        "double_blocks.1.txt_attn.qkv.weight": torch.randn(192, 64),
        "single_blocks.0.linear1.weight": torch.randn(256, 64),
        "single_blocks.1.linear2.weight": torch.randn(64, 256),
    }
    path = tmp_path / "tiny_flux.safetensors"
    save_file(sd, str(path))
    return str(path)


# --- detect_model_family ---------------------------------------------


def test_detect_flux_from_double_blocks_key():
    sd = {
        "double_blocks.0.img_attn.qkv.weight": torch.zeros(4),
        "single_blocks.0.linear1.weight": torch.zeros(4),
    }
    assert detect_model_family(sd) == "flux"


def test_detect_flux_works_with_only_single_blocks():
    sd = {"single_blocks.0.linear1.weight": torch.zeros(4)}
    assert detect_model_family(sd) == "flux"


# --- CLIP t5xxl slot -------------------------------------------------


def test_clip_flux_tokenize_produces_l_and_t5xxl_slots():
    """When clip_model2 is named for Flux (attr-based dispatch), the
    token dict exposes 't5xxl' instead of 'g'."""
    comp_l = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=49408, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=77,
        )
    ).eval()
    comp_t5 = _make_tiny_t5_like()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    clip = CLIP(clip_model=comp_l, tokenizer=tok, clip_model2=comp_t5)
    # The "family" attribute lets CLIP pick the right slot name
    clip.family = "flux"

    tokens = clip.tokenize("a cat")
    assert "l" in tokens
    assert "t5xxl" in tokens
    # SDXL slot "g" must NOT be present for Flux
    assert "g" not in tokens


def test_clip_flux_encode_returns_t5_cond_and_clip_pooled():
    """Flux conditioning: cond = T5 sequence, pooled = CLIP-L pooled."""
    comp_l = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=49408, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=77,
        )
    ).eval()
    comp_t5 = _make_tiny_t5_like()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    clip = CLIP(clip_model=comp_l, tokenizer=tok, clip_model2=comp_t5)
    clip.family = "flux"

    tokens = clip.tokenize("a cat")
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    # Flux: cond is the T5 output → (1, 77, 64)
    assert cond.shape == (1, 77, 64)
    # Pooled is from CLIP-L → (1, 32)
    assert pooled.shape == (1, 32)


# --- Loader dispatch --------------------------------------------------


def test_load_checkpoint_flux_returns_flux_clip_wrapper(tmp_path, monkeypatch):
    """A Flux state dict routes through the Flux loader path."""
    ckpt = _write_tiny_flux_state(tmp_path)

    def _fake_load_flux_single_file(path):
        from tests.fixtures.tiny_sd15 import make_tiny_sd15

        base = make_tiny_sd15()
        t5 = _make_tiny_t5_like()
        return (
            base["unet"],          # stand-in for FluxTransformer2DModel
            base["vae"],
            base["text_encoder"],  # CLIP-L
            t5,                    # T5 stand-in
            base["tokenizer"],     # CLIP tokenizer
            base["tokenizer"],     # T5 tokenizer (stand-in)
        )

    monkeypatch.setattr(
        _diffusers_loader,
        "load_flux_single_file",
        _fake_load_flux_single_file,
    )

    model, clip, vae, _ = load_checkpoint_guess_config(ckpt)
    assert clip is not None
    assert clip.clip_model is not None
    assert hasattr(clip, "clip_model2") and clip.clip_model2 is not None
    assert getattr(clip, "family", None) == "flux"

    tokens = clip.tokenize("cat")
    assert "l" in tokens
    assert "t5xxl" in tokens
