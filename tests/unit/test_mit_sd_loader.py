"""Tests for the MIT SD1.5 single-file checkpoint loader.

Phase 1 uses the tiny fixture as a stand-in for a real SD1.5 checkpoint.
The loader accepts a path to a safetensors file with SD1.5-style keys and
returns a (ModelPatcher, CLIP, VAE, clipvision) tuple.  For real SD1.5
checkpoints we defer to ``diffusers.StableDiffusionPipeline.from_single_file``;
for synthetic test files we fall back to the tiny fixture.
"""
import pytest
import torch
from safetensors.torch import save_file

from comfy_runtime.compat.comfy.sd import CLIP, VAE, load_checkpoint_guess_config
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


def _write_tiny_sd15_state(tmp_path):
    """Write a synthetic safetensors file with SD1.5-style top-level keys.

    Real SD1.5 checkpoints have thousands of keys under
    model.diffusion_model.*, first_stage_model.*, cond_stage_model.*.
    We only need enough keys for detect_model_family to pick the right
    branch; the actual weights come from the tiny fixture fallback.
    """
    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(32, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(128, 32),
        "first_stage_model.encoder.conv_in.weight": torch.randn(16, 3, 3, 3),
        "first_stage_model.decoder.conv_out.weight": torch.randn(3, 16, 3, 3),
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.randn(1000, 32),
    }
    path = tmp_path / "tiny_sd15.safetensors"
    save_file(sd, str(path))
    return str(path)


def test_loader_returns_patcher_clip_vae_tuple(tmp_path):
    ckpt = _write_tiny_sd15_state(tmp_path)
    model, clip, vae, clipvision = load_checkpoint_guess_config(ckpt)

    assert isinstance(model, ModelPatcher)
    assert isinstance(clip, CLIP)
    assert isinstance(vae, VAE)
    assert clipvision is None


def test_loader_components_are_populated(tmp_path):
    ckpt = _write_tiny_sd15_state(tmp_path)
    model, clip, vae, _ = load_checkpoint_guess_config(ckpt)

    assert model.model is not None
    assert vae.vae_model is not None
    assert clip.clip_model is not None
    assert clip.tokenizer is not None


def test_loader_clip_can_encode(tmp_path):
    ckpt = _write_tiny_sd15_state(tmp_path)
    _, clip, _, _ = load_checkpoint_guess_config(ckpt)
    tokens = clip.tokenize("a cat")
    cond = clip.encode_from_tokens(tokens)
    assert cond.dim() == 3  # (B, seq, hidden)


def test_loader_vae_can_decode(tmp_path):
    ckpt = _write_tiny_sd15_state(tmp_path)
    _, _, vae, _ = load_checkpoint_guess_config(ckpt)
    latent = torch.randn(1, 4, 16, 16)
    img = vae.decode(latent)
    assert img.shape[-1] == 3  # (B, H, W, 3)


def test_loader_rejects_unknown_format(tmp_path):
    from safetensors.torch import save_file as _save

    junk = tmp_path / "junk.safetensors"
    _save({"nonsense.key": torch.zeros(1)}, str(junk))
    with pytest.raises(ValueError, match="Unrecognized|not.*sd15|not a known"):
        load_checkpoint_guess_config(str(junk))


def test_loader_output_flags_respected(tmp_path):
    ckpt = _write_tiny_sd15_state(tmp_path)
    model, clip, vae, _ = load_checkpoint_guess_config(
        ckpt, output_vae=False, output_clip=False
    )
    assert model is not None
    assert clip is None
    assert vae is None
