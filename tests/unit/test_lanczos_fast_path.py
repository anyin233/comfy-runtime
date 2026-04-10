"""Tests for the lanczos() torch fast path (Step 8)."""

import torch
import torch.nn.functional as F

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime.compat.comfy import utils


def test_lanczos_upscale_shape():
    x = torch.rand(1, 3, 16, 16)
    out = utils.lanczos(x, 32, 32)
    assert out.shape == (1, 3, 32, 32)
    assert out.device == x.device


def test_lanczos_numerically_close_on_smooth_gradient():
    """Smooth gradient → upscale mean should be close to input mean."""
    g = torch.linspace(0, 1, 32).view(1, 1, 32, 1).expand(1, 3, 32, 32).contiguous()
    out = utils.lanczos(g, 64, 64)
    assert abs(out.mean().item() - g.mean().item()) < 0.05


def test_lanczos_uses_torch_interpolate_fast_path(monkeypatch):
    """The fast path should call torch.nn.functional.interpolate exactly once."""
    called = {"n": 0}
    orig = F.interpolate

    def spy(*args, **kwargs):
        called["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(F, "interpolate", spy)

    x = torch.rand(1, 3, 8, 8)
    utils.lanczos(x, 16, 16)

    assert called["n"] == 1, (
        f"fast path should call F.interpolate once, got {called['n']}"
    )


def test_lanczos_downscale_shape():
    """Downscaling must also produce the requested output shape."""
    x = torch.rand(1, 3, 64, 64)
    out = utils.lanczos(x, 32, 32)
    assert out.shape == (1, 3, 32, 32)
