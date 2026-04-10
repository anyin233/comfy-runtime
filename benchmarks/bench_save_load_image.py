"""Benchmark for SaveImage / LoadImage hot path (Steps 6 and 7)."""

import os
import tempfile

import numpy as np
import torch
from PIL import Image as PILImage

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime import executor
from comfy_runtime import config as cfg_mod
from benchmarks._harness import run_block


def run():
    tmp = tempfile.mkdtemp(prefix="bench_imageio_")
    cfg_mod._LAST_CONFIG = None
    comfy_runtime.configure(output_dir=tmp, input_dir=tmp)

    # Pre-materialize a reusable batch on CPU (simulate generated images).
    imgs = torch.rand(4, 128, 128, 3)

    def save_batch():
        executor.execute_node(
            "SaveImage", images=imgs, filename_prefix="bench_save"
        )

    # Pre-write a natural-looking RGB PNG for LoadImage warm path.
    rgb_arr = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    rgb_path = os.path.join(tmp, "bench_load_rgb.png")
    PILImage.fromarray(rgb_arr, "RGB").save(rgb_path)

    rgba_arr = np.zeros((256, 256, 4), dtype="uint8")
    rgba_arr[..., :3] = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    rgba_arr[..., 3] = 255
    rgba_path = os.path.join(tmp, "bench_load_rgba.png")
    PILImage.fromarray(rgba_arr, "RGBA").save(rgba_path)

    def load_rgb():
        executor.execute_node("LoadImage", image="bench_load_rgb.png")

    def load_rgba():
        executor.execute_node("LoadImage", image="bench_load_rgba.png")

    return {
        "save_image.batch4_128x128": run_block(
            "save_image.batch4_128x128", save_batch, warmup=3, iters=200
        ),
        "load_image.rgb_256x256": run_block(
            "load_image.rgb_256x256", load_rgb, warmup=3, iters=200
        ),
        "load_image.rgba_256x256": run_block(
            "load_image.rgba_256x256", load_rgba, warmup=3, iters=200
        ),
    }
