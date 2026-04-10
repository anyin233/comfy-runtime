"""End-to-end node pipeline test — SD1.5 happy path with the tiny fixture.

This is the integration test that proves Phase 1 closes the loop from
CheckpointLoaderSimple → CLIPTextEncode → KSampler → VAEDecode through
the MIT compat layer without any _vendor_bridge involvement.
"""
import pytest
import torch

from tests.fixtures.tiny_sd15 import make_tiny_sd15
from comfy_runtime.compat.comfy.sd import CLIP, VAE
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat import nodes


@pytest.fixture
def tiny_stack():
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])
    clip = CLIP(clip_model=comp["text_encoder"], tokenizer=comp["tokenizer"])
    vae = VAE(vae_model=comp["vae"])
    return model, clip, vae


def test_cliptextencode_produces_conditioning(tiny_stack):
    _, clip, _ = tiny_stack
    (result,) = nodes.CLIPTextEncode().encode(clip, "a cat")
    assert isinstance(result, list)
    assert len(result[0]) == 2
    assert isinstance(result[0][0], torch.Tensor)
    assert "pooled_output" in result[0][1]


def test_empty_latent_image_node_produces_latent_dict():
    (latent,) = nodes.EmptyLatentImage().generate(width=32, height=32, batch_size=1)
    assert "samples" in latent
    assert latent["samples"].shape == (1, 4, 4, 4)  # 32/8 = 4


def test_ksampler_produces_latent_dict(tiny_stack):
    model, clip, _ = tiny_stack
    (pos,) = nodes.CLIPTextEncode().encode(clip, "a cat")
    (neg,) = nodes.CLIPTextEncode().encode(clip, "")
    # Tiny VAE downsamples 2x not 8x, so we supply the latent directly.
    latent = {"samples": torch.zeros(1, 4, 8, 8)}

    (sampled,) = nodes.KSampler().sample(
        model=model,
        seed=42,
        steps=2,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        positive=pos,
        negative=neg,
        latent_image=latent,
        denoise=1.0,
    )
    assert "samples" in sampled
    assert sampled["samples"].shape == latent["samples"].shape
    assert not torch.isnan(sampled["samples"]).any()


def test_vaedecode_produces_image_tensor(tiny_stack):
    _, _, vae = tiny_stack
    latent = {"samples": torch.randn(1, 4, 16, 16)}
    (image,) = nodes.VAEDecode().decode(vae=vae, samples=latent)
    assert image.shape == (1, 32, 32, 3)
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0


def test_vaeencode_produces_latent_dict(tiny_stack):
    _, _, vae = tiny_stack
    image = torch.rand(1, 32, 32, 3)
    (latent,) = nodes.VAEEncode().encode(vae=vae, pixels=image)
    assert "samples" in latent
    assert latent["samples"].shape == (1, 4, 16, 16)


def test_full_txt2img_pipeline(tiny_stack):
    """CLIP encode → KSampler → VAEDecode, single flow."""
    model, clip, vae = tiny_stack
    (pos,) = nodes.CLIPTextEncode().encode(clip, "a cat")
    (neg,) = nodes.CLIPTextEncode().encode(clip, "")
    latent = {"samples": torch.zeros(1, 4, 8, 8)}

    (sampled,) = nodes.KSampler().sample(
        model=model, seed=123, steps=2, cfg=1.0,
        sampler_name="euler", scheduler="normal",
        positive=pos, negative=neg, latent_image=latent, denoise=1.0,
    )
    (image,) = nodes.VAEDecode().decode(vae=vae, samples=sampled)
    assert image.shape[-1] == 3
    assert not torch.isnan(image).any()
