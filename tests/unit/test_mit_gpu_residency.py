"""Tests that loaded models actually live on the compute device.

The compat layer historically hard-coded ``load_device=torch.device("cpu")``
in every loader and then never called ``load_models_gpu`` — so inference
ran entirely on CPU.  The e2e benchmark caught this: slim-vendor sampled
at 62 s per workflow vs. ComfyUI's 1.5 s (40× slower), with
``gpu_peak_allocated_bytes == 0``.  This file pins down the fix: after
loading, the UNet / CLIP / VAE must all be resident on the compute
device that :func:`get_torch_device` reports.

Tests are CUDA-only — on CPU-only hosts there is no regression to
assert, so they skip.
"""
import pytest
import torch
from safetensors.torch import save_file

from comfy_runtime.compat.comfy.model_management import get_torch_device
from comfy_runtime.compat.comfy.sd import load_checkpoint_guess_config


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU residency regression only applies when CUDA is available",
)


def _write_tiny_sd15_state(tmp_path):
    """Synthetic SD1.5-shaped safetensors — triggers the SD1.5 branch of
    ``detect_model_family`` without needing a real 4 GB checkpoint.

    The loader falls back to the tiny fixture for the actual weights
    because diffusers' ``from_single_file`` can't build a real pipeline
    from five junk tensors — that's fine for this test since the
    residency assertion doesn't care whether the weights are real.
    """
    sd = {
        "model.diffusion_model.input_blocks.0.0.weight": torch.randn(32, 4, 3, 3),
        "model.diffusion_model.time_embed.0.weight": torch.randn(128, 32),
        "first_stage_model.encoder.conv_in.weight": torch.randn(16, 3, 3, 3),
        "first_stage_model.decoder.conv_out.weight": torch.randn(3, 16, 3, 3),
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight":
            torch.randn(1000, 32),
    }
    path = tmp_path / "tiny_sd15.safetensors"
    save_file(sd, str(path))
    return str(path)


@needs_cuda
def test_checkpoint_loader_places_unet_on_compute_device(tmp_path):
    """Every UNet parameter must live on ``get_torch_device()``."""
    ckpt = _write_tiny_sd15_state(tmp_path)
    model, _, _, _ = load_checkpoint_guess_config(ckpt)

    target = get_torch_device()
    assert target.type == "cuda", "test precondition: compute device is cuda"

    params = list(model.model.parameters())
    assert params, "tiny fixture UNet has no parameters — test fixture broken"

    for p in params:
        assert p.device.type == "cuda", (
            f"UNet parameter on {p.device}, expected cuda — "
            "the loader didn't move the model to the compute device"
        )


@needs_cuda
def test_checkpoint_loader_places_clip_on_compute_device(tmp_path):
    """Every CLIP text-encoder parameter must live on the compute device."""
    ckpt = _write_tiny_sd15_state(tmp_path)
    _, clip, _, _ = load_checkpoint_guess_config(ckpt)

    params = list(clip.clip_model.parameters())
    assert params, "tiny fixture CLIP has no parameters — test fixture broken"

    for p in params:
        assert p.device.type == "cuda", (
            f"CLIP parameter on {p.device}, expected cuda"
        )


@needs_cuda
def test_checkpoint_loader_places_vae_on_compute_device(tmp_path):
    """Every VAE parameter must live on the compute device."""
    ckpt = _write_tiny_sd15_state(tmp_path)
    _, _, vae, _ = load_checkpoint_guess_config(ckpt)

    params = list(vae.vae_model.parameters())
    assert params, "tiny fixture VAE has no parameters — test fixture broken"

    for p in params:
        assert p.device.type == "cuda", (
            f"VAE parameter on {p.device}, expected cuda"
        )
