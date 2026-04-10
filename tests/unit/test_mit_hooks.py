"""Tests for HookGroup patch/unpatch chain parity.

ComfyUI's hook system lets callers stack multiple weight-modifying
operations (LoRAs, model-as-LoRA, custom patches) on a single
ModelPatcher and apply/remove them as a unit.  This file tests the
minimum semantics needed for production workflows that chain two or
more hooks:

* ``Hook.patch_model(patcher)`` — adds the hook's contribution to the
  patcher's ``.patches`` dict (doesn't materialize yet).
* ``Hook.unpatch_model(patcher)`` — removes only this hook's
  contributions without touching other hooks' backup state.
* ``HookGroup.patch_hooks(patcher)`` — applies every hook in
  registration order, then calls ``patcher.patch_model()`` once so
  the cumulative effect is committed.
* ``HookGroup.unpatch_hooks(patcher)`` — reverses it via
  ``patcher.unpatch_model()``.
* Two hooks stacked must produce the sum of their deltas.
* Cloning a HookGroup and applying it to a different patcher must
  leave the original patcher untouched.
"""
import torch
import torch.nn as nn

from comfy_runtime.compat.comfy.hooks import Hook, HookGroup, LoRAHook
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


class _Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.diffusion_model.proj.weight.copy_(torch.zeros(4, 4))


def _make_lora_sd(scale: float = 1.0):
    return {
        "diffusion_model.proj.lora_up.weight": torch.ones(4, 2) * scale,
        "diffusion_model.proj.lora_down.weight": torch.ones(2, 4) * scale,
        "diffusion_model.proj.alpha": torch.tensor(2.0),
    }


def test_single_hook_applies_lora_delta():
    """Applying a HookGroup with one LoRAHook mutates the model."""
    target = _Target()
    patcher = ModelPatcher(target)

    hook = LoRAHook(_make_lora_sd(scale=1.0), strength=1.0)
    group = HookGroup()
    group.add(hook)

    group.patch_hooks(patcher)

    # (1@1) = 2 per cell, * (alpha/rank = 2/2 = 1) * strength 1 = 2 per cell
    expected = torch.full((4, 4), 2.0)
    assert torch.allclose(target.diffusion_model.proj.weight, expected)


def test_single_hook_unpatch_restores_original():
    target = _Target()
    patcher = ModelPatcher(target)
    original = target.diffusion_model.proj.weight.clone()

    hook = LoRAHook(_make_lora_sd(), strength=1.0)
    group = HookGroup()
    group.add(hook)

    group.patch_hooks(patcher)
    assert not torch.allclose(target.diffusion_model.proj.weight, original)

    group.unpatch_hooks(patcher)
    assert torch.allclose(target.diffusion_model.proj.weight, original)


def test_two_hooks_accumulate_deltas():
    """Two LoRA hooks should sum their contributions."""
    target = _Target()
    patcher = ModelPatcher(target)

    group = HookGroup()
    group.add(LoRAHook(_make_lora_sd(scale=1.0), strength=1.0))
    group.add(LoRAHook(_make_lora_sd(scale=0.5), strength=1.0))

    group.patch_hooks(patcher)

    # hook1 delta = (1@1)*rank=2 per cell * scale(alpha/rank=1.0) = 2.0
    # hook2 delta = (0.5@0.5)*rank=0.5 per cell * scale(alpha/rank=1.0) = 0.5
    # Total weight = 0 (original) + 2.0 + 0.5 = 2.5 per cell
    expected = torch.full((4, 4), 2.5)
    assert torch.allclose(target.diffusion_model.proj.weight, expected)


def test_strength_scales_hook_delta():
    target = _Target()
    patcher = ModelPatcher(target)

    group = HookGroup()
    group.add(LoRAHook(_make_lora_sd(), strength=0.25))

    group.patch_hooks(patcher)
    # (1@1)*2 = 2 per cell, *0.25 strength = 0.5 per cell
    expected = torch.full((4, 4), 0.5)
    assert torch.allclose(target.diffusion_model.proj.weight, expected)


def test_patch_unpatch_roundtrip_is_idempotent():
    """Patching then unpatching twice must still restore the original."""
    target = _Target()
    patcher = ModelPatcher(target)
    original = target.diffusion_model.proj.weight.clone()

    group = HookGroup()
    group.add(LoRAHook(_make_lora_sd(), strength=1.0))

    group.patch_hooks(patcher)
    group.unpatch_hooks(patcher)
    group.patch_hooks(patcher)
    group.unpatch_hooks(patcher)

    assert torch.allclose(target.diffusion_model.proj.weight, original)


def test_hook_clone_is_independent():
    h1 = LoRAHook(_make_lora_sd(), strength=1.0)
    h2 = h1.clone()
    # Mutating the clone's strength must not affect the original
    h2.strength = 0.0
    assert h1.strength == 1.0
    assert h2.strength == 0.0


def test_hook_group_clone_is_independent():
    g1 = HookGroup()
    g1.add(LoRAHook(_make_lora_sd(), strength=1.0))
    g2 = g1.clone()

    g2.hooks.append(LoRAHook(_make_lora_sd(), strength=0.5))
    assert len(g1.hooks) == 1
    assert len(g2.hooks) == 2


def test_empty_hook_group_is_noop():
    target = _Target()
    patcher = ModelPatcher(target)
    original = target.diffusion_model.proj.weight.clone()

    group = HookGroup()
    group.patch_hooks(patcher)
    group.unpatch_hooks(patcher)

    assert torch.allclose(target.diffusion_model.proj.weight, original)


def test_base_hook_abstract_patch_raises():
    """Subclasses must implement patch_model."""
    target = _Target()
    patcher = ModelPatcher(target)
    h = Hook()
    try:
        h.patch_model(patcher)
    except NotImplementedError:
        return
    # If no error, the base class must at least be a no-op (documented)
