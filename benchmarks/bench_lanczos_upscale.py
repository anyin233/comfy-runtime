"""Benchmark for the Lanczos upscale fast path (Step 8)."""

import torch

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime.compat.comfy import utils
from benchmarks._harness import run_block


def run():
    # Latent-sized tensor (typical LatentUpscale input).
    latent = torch.rand(1, 4, 64, 64)

    def latent_upscale():
        utils.lanczos(latent, 128, 128)

    # Image-sized tensor (typical ImageScale input).
    image = torch.rand(1, 3, 256, 256)

    def image_upscale():
        utils.lanczos(image, 512, 512)

    return {
        "lanczos.latent_64_to_128": run_block(
            "lanczos.latent_64_to_128", latent_upscale, warmup=3, iters=200
        ),
        "lanczos.image_256_to_512": run_block(
            "lanczos.image_256_to_512", image_upscale, warmup=3, iters=50
        ),
    }
