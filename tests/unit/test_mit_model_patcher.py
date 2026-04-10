"""Tests for the MIT ModelPatcher weight-patching implementation."""
import torch
import torch.nn as nn

from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


class _Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4))


def _new_dummy_patcher():
    m = _Dummy()
    return m, ModelPatcher(m)


def test_patch_adds_raw_delta_to_weight():
    """Simple delta patch: new_weight = original + strength_patch * delta."""
    m, patcher = _new_dummy_patcher()
    delta = torch.ones(4, 4) * 0.1

    added = patcher.add_patches({"linear.weight": (delta,)})
    assert "linear.weight" in added

    patcher.patch_model()
    expected = torch.eye(4) + delta
    assert torch.allclose(m.linear.weight, expected)


def test_unpatch_restores_original_weight():
    m, patcher = _new_dummy_patcher()
    original = m.linear.weight.clone()

    delta = torch.ones(4, 4) * 0.1
    patcher.add_patches({"linear.weight": (delta,)})
    patcher.patch_model()
    patcher.unpatch_model()

    assert torch.allclose(m.linear.weight, original)
    assert not patcher.is_patched


def test_strength_patch_scales_delta():
    m, patcher = _new_dummy_patcher()
    delta = torch.ones(4, 4)

    patcher.add_patches(
        {"linear.weight": (delta,)},
        strength_patch=0.5,
        strength_model=1.0,
    )
    patcher.patch_model()
    expected = torch.eye(4) + 0.5 * delta
    assert torch.allclose(m.linear.weight, expected)


def test_strength_model_scales_original():
    m, patcher = _new_dummy_patcher()
    delta = torch.zeros(4, 4)

    patcher.add_patches(
        {"linear.weight": (delta,)},
        strength_patch=1.0,
        strength_model=0.0,  # wipe out the original
    )
    patcher.patch_model()
    assert torch.allclose(m.linear.weight, torch.zeros(4, 4))


def test_multiple_patches_accumulate():
    m, patcher = _new_dummy_patcher()
    delta_a = torch.ones(4, 4) * 0.1
    delta_b = torch.ones(4, 4) * 0.2

    patcher.add_patches({"linear.weight": (delta_a,)})
    patcher.add_patches({"linear.weight": (delta_b,)})
    patcher.patch_model()

    expected = torch.eye(4) + delta_a + delta_b
    assert torch.allclose(m.linear.weight, expected)


def test_patch_is_noop_for_unknown_key():
    m, patcher = _new_dummy_patcher()
    delta = torch.ones(4, 4)
    added = patcher.add_patches({"nonexistent.key": (delta,)})
    assert "nonexistent.key" not in added

    patcher.patch_model()
    assert torch.allclose(m.linear.weight, torch.eye(4))


def test_patch_moves_model_to_device_on_request():
    m, patcher = _new_dummy_patcher()
    patcher.patch_model(device_to=torch.device("cpu"))
    assert patcher.current_device == torch.device("cpu")
    assert patcher.is_patched
