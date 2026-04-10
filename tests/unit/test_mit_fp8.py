"""Tests for fp8 weight cast support in load_unet / ModelPatcher.

ComfyUI's ``UNETLoader`` accepts ``weight_dtype`` strings including
``fp8_e4m3fn`` and ``fp8_e5m2``.  When set, the loader casts every
parameter to that dtype at load time, reducing VRAM footprint by ~50%
compared to fp16 at the cost of accuracy (fp8 has only 4 or 5 mantissa
bits).  Forward-pass computation still happens in fp16/bf16 via a
just-in-time upcast; only the stored weights are fp8.

Phase 3 scope for this test:
  * load_unet(path, dtype=torch.float8_e4m3fn) returns a ModelPatcher
    whose parameters are all fp8_e4m3fn dtype.
  * ModelPatcher.partially_load still accounts for the correct byte
    count (1 byte per fp8 element rather than 4 per fp32).
  * Casting a tiny nn.Module via the helper survives a .to(device)
    roundtrip.
"""
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from comfy_runtime.compat.comfy.sd import load_unet
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2


def _write_tiny_unet_sd(tmp_path):
    sd = {"time_embed.0.weight": torch.randn(128, 32)}
    path = tmp_path / "tiny_unet.safetensors"
    save_file(sd, str(path))
    return str(path)


def test_load_unet_casts_to_fp8_e4m3fn(tmp_path):
    ckpt = _write_tiny_unet_sd(tmp_path)
    patcher = load_unet(ckpt, dtype=FP8_E4M3)
    for p in patcher.model.parameters():
        assert p.dtype == FP8_E4M3, f"param not cast: {p.dtype}"


def test_load_unet_casts_to_fp8_e5m2(tmp_path):
    ckpt = _write_tiny_unet_sd(tmp_path)
    patcher = load_unet(ckpt, dtype=FP8_E5M2)
    for p in patcher.model.parameters():
        assert p.dtype == FP8_E5M2


def test_load_unet_fp16_passthrough(tmp_path):
    """dtype=torch.float16 should also work (non-fp8 path)."""
    ckpt = _write_tiny_unet_sd(tmp_path)
    patcher = load_unet(ckpt, dtype=torch.float16)
    for p in patcher.model.parameters():
        assert p.dtype == torch.float16


def test_load_unet_default_dtype_preserves_original(tmp_path):
    """dtype=None should leave the model in its original dtype (fp32)."""
    ckpt = _write_tiny_unet_sd(tmp_path)
    patcher = load_unet(ckpt)
    dtypes = {p.dtype for p in patcher.model.parameters()}
    assert torch.float32 in dtypes


def test_fp8_model_size_is_quarter_of_fp32():
    """fp8 parameters should account for 1 byte/elem vs 4 for fp32."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1000))  # 4000 B in fp32

    model_fp32 = Tiny()
    size_fp32 = sum(p.nelement() * p.element_size() for p in model_fp32.parameters())
    assert size_fp32 == 4000

    model_fp8 = Tiny()
    for name, param in model_fp8.named_parameters():
        param.data = param.data.to(FP8_E4M3)
    size_fp8 = sum(p.nelement() * p.element_size() for p in model_fp8.parameters())
    assert size_fp8 == 1000  # 1 byte per element


def test_modelpatcher_partial_load_tracks_fp8_bytes():
    """ModelPatcher.current_loaded_size must report fp8 sizes correctly."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.big = nn.Parameter(torch.zeros(1000))
            self.small = nn.Parameter(torch.zeros(250))

    model = Tiny()
    for p in model.parameters():
        p.data = p.data.to(FP8_E4M3)
    # total fp8 = 1000 + 250 = 1250 B
    model.to("cpu")
    load_dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    patcher = ModelPatcher(model, load_device=load_dev, offload_device=torch.device("cpu"))

    # Load only the big param (1000 B budget fits big, not big+small=1250)
    patcher.partially_load(device=load_dev, extra_memory=1000)
    assert patcher.current_loaded_size() == 1000
