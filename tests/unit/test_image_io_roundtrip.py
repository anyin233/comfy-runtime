"""Tests for SaveImage / LoadImage optimizations (Steps 6 and 7)."""

import os
import tempfile

import numpy as np
import torch
from PIL import Image as PILImage

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime import executor


def _setup(tmp: str):
    """Configure both output_dir and input_dir to the same tmp for roundtrips."""
    # Reset snapshot so repeated tmp dirs don't short-circuit into stale config.
    from comfy_runtime import config as cfg_mod

    cfg_mod._LAST_CONFIG = None
    comfy_runtime.configure(output_dir=tmp, input_dir=tmp)


# ---------------------------------------------------------------------------
# Step 6 — SaveImage batch
# ---------------------------------------------------------------------------


def test_save_image_roundtrip_preserves_pixels():
    """Round-trip a known tensor through SaveImage + LoadImage."""
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        torch.manual_seed(0)
        # uint8-quantized so roundtrip is deterministic.
        imgs = (torch.rand(4, 32, 32, 3) * 255).round() / 255.0

        result = executor.execute_node(
            "SaveImage", images=imgs, filename_prefix="probe"
        )
        saved = result["ui"]["images"]
        assert len(saved) == 4

        loaded, _mask = executor.execute_node(
            "LoadImage", image=saved[0]["filename"]
        )
        diff = (loaded[0] - imgs[0]).abs().max().item()
        assert diff <= 1.5 / 255.0, f"max diff {diff}"


def test_save_image_does_not_mutate_input_tensor():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        imgs = torch.rand(2, 16, 16, 3)
        snapshot = imgs.clone()
        executor.execute_node("SaveImage", images=imgs, filename_prefix="nomut")
        assert torch.equal(imgs, snapshot), "SaveImage mutated caller tensor"


def test_save_image_batch_uses_single_cpu_transfer():
    """Batch=4 must fold the per-image .cpu() calls into a single transfer."""
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        imgs = torch.rand(4, 16, 16, 3)

        cpu_call_count = {"n": 0}
        orig_cpu = torch.Tensor.cpu

        def counting_cpu(self, *a, **kw):
            cpu_call_count["n"] += 1
            return orig_cpu(self, *a, **kw)

        try:
            torch.Tensor.cpu = counting_cpu
            executor.execute_node(
                "SaveImage", images=imgs, filename_prefix="batchcpu"
            )
        finally:
            torch.Tensor.cpu = orig_cpu

        # The batched path may still bounce to CPU indirectly (e.g., dtype
        # conversions). We assert it's substantially fewer than "once per
        # batch item" — 4 images should not produce 4 separate .cpu() calls.
        assert cpu_call_count["n"] <= 2, (
            f"expected batched transfer (<=2 .cpu() calls), got {cpu_call_count['n']}"
        )


# ---------------------------------------------------------------------------
# Step 7 — LoadImage skip redundant RGBA conversion
# ---------------------------------------------------------------------------


def test_load_image_rgb_shape_and_mask():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        arr = (np.random.rand(24, 24, 3) * 255).astype("uint8")
        PILImage.fromarray(arr, "RGB").save(os.path.join(tmp, "rgb.png"))

        tensor, mask = executor.execute_node("LoadImage", image="rgb.png")

        assert tensor.shape == (1, 24, 24, 3)
        assert mask.shape == (1, 24, 24)
        assert mask.abs().sum().item() == 0.0


def test_load_image_rgba_mask_from_alpha():
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        arr = np.zeros((24, 24, 4), dtype="uint8")
        arr[..., :3] = 128
        arr[..., 3] = 255  # fully opaque → mask should be all zeros
        PILImage.fromarray(arr, "RGBA").save(os.path.join(tmp, "rgba.png"))

        tensor, mask = executor.execute_node("LoadImage", image="rgba.png")

        assert tensor.shape == (1, 24, 24, 3)
        assert mask.shape == (1, 24, 24)
        assert mask.abs().max().item() < 1e-6


def test_load_image_rgb_does_not_convert_to_rgba():
    """Pure RGB input must NOT trigger an RGBA conversion."""
    convert_modes = []
    orig_convert = PILImage.Image.convert

    def spy_convert(self, mode, *a, **kw):
        convert_modes.append(mode)
        return orig_convert(self, mode, *a, **kw)

    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp)
        arr = (np.random.rand(24, 24, 3) * 255).astype("uint8")
        PILImage.fromarray(arr, "RGB").save(os.path.join(tmp, "rgb2.png"))

        try:
            PILImage.Image.convert = spy_convert
            executor.execute_node("LoadImage", image="rgb2.png")
        finally:
            PILImage.Image.convert = orig_convert

        assert "RGBA" not in convert_modes, (
            f"RGB input should skip RGBA conversion; modes={convert_modes}"
        )
