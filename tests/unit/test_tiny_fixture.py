"""Ensure the synthetic tiny SD1.5 fixture produces a usable mini pipeline.

The fixture is used by every Phase-1 TDD test as a fast, network-free,
~5-20 MB stand-in for a real SD1.5 checkpoint.
"""
import torch

from tests.fixtures.tiny_sd15 import make_tiny_sd15


def test_tiny_sd15_has_expected_components():
    components = make_tiny_sd15()
    for key in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        assert key in components, f"missing {key!r} in fixture"


def test_tiny_sd15_unet_forward_pass_preserves_shape():
    """The tiny UNet should accept latent + timestep + ctx and return same-shape noise."""
    components = make_tiny_sd15()
    unet = components["unet"]

    latent = torch.randn(1, unet.config.in_channels, 8, 8)
    t = torch.tensor([0])
    cross_dim = unet.config.cross_attention_dim
    ctx = torch.randn(1, 4, cross_dim)

    out = unet(latent, t, encoder_hidden_states=ctx).sample
    assert out.shape == latent.shape


def test_tiny_sd15_vae_encode_decode_roundtrips_shape():
    """The tiny VAE should encode an image and decode back to the same shape."""
    components = make_tiny_sd15()
    vae = components["vae"]

    img = torch.randn(1, 3, 32, 32)
    enc = vae.encode(img).latent_dist.sample()
    dec = vae.decode(enc).sample
    assert dec.shape == img.shape


def test_tiny_sd15_text_encoder_shapes():
    """The tiny CLIPTextModel should produce an embedding matrix."""
    components = make_tiny_sd15()
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    ids = tokenizer(["a cat"], padding="max_length", max_length=77,
                    return_tensors="pt").input_ids
    out = text_encoder(ids)
    assert out.last_hidden_state.shape == (1, 77, text_encoder.config.hidden_size)


def test_tiny_sd15_is_small():
    """Fixture must stay under 20 MB so unit tests remain fast."""
    components = make_tiny_sd15()
    total_bytes = 0
    for key, m in components.items():
        if hasattr(m, "parameters"):
            total_bytes += sum(p.numel() * p.element_size() for p in m.parameters())
    assert total_bytes < 20 * 1024 * 1024, (
        f"Fixture too large: {total_bytes / 1e6:.1f} MB"
    )
