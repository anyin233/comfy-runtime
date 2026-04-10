"""Tests for the MIT LoRA application path.

ComfyUI LoRA state dicts use the naming convention::

    <target_key>.lora_up.weight     — shape (out_features, rank)
    <target_key>.lora_down.weight   — shape (rank, in_features)
    <target_key>.alpha              — scalar, usually rank

The full LoRA delta applied to the target weight is::

    delta = (lora_up @ lora_down) * (alpha / rank) * strength

The Phase 1 ModelPatcher only understands raw delta patches, so the
LoRA loader's job is to translate the up/down/alpha triples into raw
delta tensors and register them via ``ModelPatcher.add_patches``.
"""
import pytest
import torch
import torch.nn as nn

from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat.comfy._lora_peft import (
    extract_lora_deltas,
    apply_lora_to_patcher,
)


class _TargetModel(nn.Module):
    """Mimics a diffusers sub-module named ``diffusion_model.proj``."""

    def __init__(self):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.proj = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.diffusion_model.proj.weight.copy_(torch.zeros(8, 8))


def _make_tiny_lora_sd(target_key: str = "diffusion_model.proj.weight",
                      rank: int = 2, alpha: float = 2.0,
                      scale: float = 1.0):
    """Synthesize a ComfyUI-format LoRA state dict for a single linear layer."""
    lora_up = torch.ones(8, rank) * scale
    lora_down = torch.ones(rank, 8) * scale
    # Strip the trailing ".weight" for the LoRA key prefix — this is how
    # ComfyUI writes them.
    prefix = target_key.rsplit(".", 1)[0]
    return {
        f"{prefix}.lora_up.weight": lora_up,
        f"{prefix}.lora_down.weight": lora_down,
        f"{prefix}.alpha": torch.tensor(float(alpha)),
    }


def test_extract_lora_deltas_produces_one_delta_per_target():
    lora_sd = _make_tiny_lora_sd()
    deltas = extract_lora_deltas(lora_sd)
    assert "diffusion_model.proj.weight" in deltas
    delta = deltas["diffusion_model.proj.weight"]
    assert delta.shape == (8, 8)


def test_extract_lora_deltas_applies_alpha_over_rank_scaling():
    """delta = (up @ down) * (alpha / rank)."""
    lora_sd = _make_tiny_lora_sd(rank=2, alpha=4.0, scale=1.0)
    deltas = extract_lora_deltas(lora_sd)
    delta = deltas["diffusion_model.proj.weight"]
    # (1 @ 1) = 2 per cell, * (alpha/rank = 4/2 = 2) = 4
    expected = torch.full((8, 8), 4.0)
    assert torch.allclose(delta, expected)


def test_extract_lora_deltas_ignores_unrelated_keys():
    lora_sd = _make_tiny_lora_sd()
    lora_sd["some.unrelated.key"] = torch.randn(3, 3)
    deltas = extract_lora_deltas(lora_sd)
    # Still produces exactly one delta for the real target
    assert len(deltas) == 1


def test_apply_lora_modifies_target_weight_after_patch():
    target = _TargetModel()
    patcher = ModelPatcher(target)
    # Target starts at zeros
    assert torch.allclose(target.diffusion_model.proj.weight, torch.zeros(8, 8))

    lora_sd = _make_tiny_lora_sd(rank=2, alpha=2.0, scale=1.0)
    # (1@1)*2=2 per cell, *(alpha/rank = 2/2 = 1) = 2 per cell
    apply_lora_to_patcher(patcher, lora_sd, strength=1.0)
    patcher.patch_model()

    expected = torch.full((8, 8), 2.0)
    assert torch.allclose(target.diffusion_model.proj.weight, expected)


def test_apply_lora_strength_scales_delta():
    target = _TargetModel()
    patcher = ModelPatcher(target)
    lora_sd = _make_tiny_lora_sd(rank=2, alpha=2.0, scale=1.0)
    # strength=0.5 → delta/2
    apply_lora_to_patcher(patcher, lora_sd, strength=0.5)
    patcher.patch_model()
    expected = torch.full((8, 8), 1.0)
    assert torch.allclose(target.diffusion_model.proj.weight, expected)


def test_apply_lora_roundtrips_via_unpatch():
    target = _TargetModel()
    original = target.diffusion_model.proj.weight.clone()
    patcher = ModelPatcher(target)
    lora_sd = _make_tiny_lora_sd()
    apply_lora_to_patcher(patcher, lora_sd, strength=1.0)
    patcher.patch_model()
    patcher.unpatch_model()
    assert torch.allclose(target.diffusion_model.proj.weight, original)


def test_apply_lora_ignores_keys_not_in_model():
    target = _TargetModel()
    patcher = ModelPatcher(target)
    # Use a key the model doesn't have
    lora_sd = _make_tiny_lora_sd(target_key="nonexistent.layer.weight")
    apply_lora_to_patcher(patcher, lora_sd, strength=1.0)
    patcher.patch_model()
    assert torch.allclose(
        target.diffusion_model.proj.weight, torch.zeros(8, 8)
    )


def test_extract_lora_deltas_handles_missing_alpha():
    """When alpha is absent, scaling defaults to 1.0 (matches ComfyUI)."""
    lora_sd = {
        "diffusion_model.proj.lora_up.weight": torch.ones(8, 4),
        "diffusion_model.proj.lora_down.weight": torch.ones(4, 8),
        # no alpha key
    }
    deltas = extract_lora_deltas(lora_sd)
    delta = deltas["diffusion_model.proj.weight"]
    # (1@1) = 4 per cell, scale = 1.0 → 4
    expected = torch.full((8, 8), 4.0)
    assert torch.allclose(delta, expected)


def test_load_lora_for_models_returns_clones():
    """load_lora_for_models must not mutate the input ModelPatcher."""
    from comfy_runtime.compat.comfy.sd import load_lora_for_models

    target = _TargetModel()
    patcher = ModelPatcher(target)
    lora_sd = _make_tiny_lora_sd()

    new_model, new_clip = load_lora_for_models(
        patcher, None, lora_sd,
        strength_model=1.0, strength_clip=1.0,
    )

    # The clone must have the patch registered; the original must not.
    assert len(new_model.patches) == 1
    assert len(patcher.patches) == 0
    # Clip is None → clone is None
    assert new_clip is None


def test_load_lora_for_models_patch_then_unpatch_roundtrips():
    from comfy_runtime.compat.comfy.sd import load_lora_for_models

    target = _TargetModel()
    patcher = ModelPatcher(target)
    original = target.diffusion_model.proj.weight.clone()

    lora_sd = _make_tiny_lora_sd()
    new_model, _ = load_lora_for_models(
        patcher, None, lora_sd,
        strength_model=1.0, strength_clip=1.0,
    )

    new_model.patch_model()
    assert not torch.allclose(target.diffusion_model.proj.weight, original)

    new_model.unpatch_model()
    assert torch.allclose(target.diffusion_model.proj.weight, original)
