"""Tests for the MIT VAE.encode / VAE.decode implementation."""
import torch

from tests.fixtures.tiny_sd15 import make_tiny_sd15
from comfy_runtime.compat.comfy.sd import VAE


def _tiny_vae() -> VAE:
    comp = make_tiny_sd15()
    return VAE(vae_model=comp["vae"])


def test_encode_image_returns_latent_shape():
    """ComfyUI input is (B, H, W, 3) in [0, 1]; latent halves H/W once (tiny VAE)."""
    vae = _tiny_vae()
    image = torch.rand(1, 32, 32, 3)
    latent = vae.encode(image)
    assert latent.shape[0] == 1
    assert latent.shape[1] == 4  # latent_channels
    assert latent.shape[2] == 16
    assert latent.shape[3] == 16


def test_decode_latent_returns_image_shape():
    """Output is (B, H, W, 3) in [0, 1]."""
    vae = _tiny_vae()
    latent = torch.randn(1, 4, 16, 16)
    image = vae.decode(latent)
    assert image.shape == (1, 32, 32, 3)
    assert image.dtype == torch.float32


def test_decode_output_is_clipped_to_unit_range():
    vae = _tiny_vae()
    latent = torch.randn(1, 4, 16, 16) * 5.0  # large magnitude
    image = vae.decode(latent)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_encode_decode_roundtrip_shape_is_stable():
    """Random-init VAE won't be visually accurate, but shape must roundtrip."""
    vae = _tiny_vae()
    image = torch.rand(1, 32, 32, 3)
    latent = vae.encode(image)
    recon = vae.decode(latent)
    assert recon.shape == image.shape


def test_encode_accepts_batch_of_two():
    vae = _tiny_vae()
    image = torch.rand(2, 32, 32, 3)
    latent = vae.encode(image)
    assert latent.shape[0] == 2
