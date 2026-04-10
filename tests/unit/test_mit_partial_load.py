"""Tests for ModelPatcher.partially_load / partially_unload — the
lowvram / novram offload primitive.

ComfyUI semantics:

* ``partially_load(device, extra_memory=0)`` —
  Move as much of the model to ``device`` as fits within ``extra_memory``
  bytes, starting from the biggest parameters first.  Leaves the
  remainder on ``offload_device``.  When ``extra_memory == 0``, nothing
  new is moved.  When ``extra_memory`` is larger than the whole model,
  everything moves.

* ``partially_unload(device, extra_memory=0)`` —
  Free at least ``extra_memory`` bytes from ``self.current_device`` by
  moving parameters back to ``device``, starting from the smallest
  parameters (cheapest-eviction-first so the most "useful" big layers
  stay resident).

* ``current_loaded_size()`` — bytes currently resident on the
  ``load_device``.  Used by :func:`load_models_gpu` to budget lowvram
  state transitions.

These methods power :func:`compat.comfy.model_management.load_models_gpu`
when VRAMState is LOW_VRAM or NO_VRAM (Task 3.2).

Tests use real CPU↔CUDA movement when CUDA is available, and fall back
to tracking ``current_loaded_size()`` on the CPU-only path (which still
exercises the per-parameter budget accounting).
"""
import pytest
import torch
import torch.nn as nn

from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


HAS_CUDA = torch.cuda.is_available()
LOAD_DEV = torch.device("cuda:0") if HAS_CUDA else torch.device("cpu")
OFFLOAD_DEV = torch.device("cpu")


class _LargeIsh(nn.Module):
    """Deterministic model with three param tensors of distinct sizes.

    * ``a`` → 400 bytes (100 fp32 elements)
    * ``b`` → 200 bytes (50 fp32 elements)
    * ``c`` → 100 bytes (25 fp32 elements)

    Total = 700 bytes.  Big-first eviction order: a, b, c.
    Small-first eviction order: c, b, a.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(100))
        self.b = nn.Parameter(torch.zeros(50))
        self.c = nn.Parameter(torch.zeros(25))


def _param_bytes(p: torch.Tensor) -> int:
    return p.nelement() * p.element_size()


def _bytes_on(model: nn.Module, device: torch.device) -> int:
    total = 0
    for p in model.parameters():
        if p.device.type == device.type:
            total += _param_bytes(p)
    return total


def _fresh_patcher():
    model = _LargeIsh()
    # Start on the offload device
    model.to(OFFLOAD_DEV)
    patcher = ModelPatcher(
        model,
        load_device=LOAD_DEV,
        offload_device=OFFLOAD_DEV,
    )
    return model, patcher


def test_current_loaded_size_reports_zero_initially():
    model, patcher = _fresh_patcher()
    # Before any load, nothing resident on load_device
    assert patcher.current_loaded_size() == 0


def test_partially_load_zero_budget_is_noop():
    _, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=0)
    assert patcher.current_loaded_size() == 0


def test_partially_load_full_budget_moves_everything():
    _, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=10_000)
    assert patcher.current_loaded_size() == 700


def test_partially_load_partial_budget_moves_big_first():
    """Budget=600 fits a(400) + b(200) but not c(100)."""
    model, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=600)
    assert patcher.current_loaded_size() == 600
    assert model.a.device.type == LOAD_DEV.type  # biggest moved first
    assert model.b.device.type == LOAD_DEV.type
    # c stays on offload
    assert model.c.device.type == OFFLOAD_DEV.type


def test_partially_load_tight_budget_only_fits_biggest():
    """Budget=400 fits only a (400); b and c stay."""
    model, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=400)
    assert patcher.current_loaded_size() == 400
    assert model.a.device.type == LOAD_DEV.type
    assert model.b.device.type == OFFLOAD_DEV.type
    assert model.c.device.type == OFFLOAD_DEV.type


def test_partially_load_too_small_budget_moves_nothing():
    """Budget=50 doesn't fit even the smallest parameter (100 bytes)."""
    model, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=50)
    assert patcher.current_loaded_size() == 0


def test_partially_unload_evicts_smallest_first():
    """After full load, unload 100 bytes → c (100) evicted first."""
    model, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=10_000)
    assert patcher.current_loaded_size() == 700

    patcher.partially_unload(device=OFFLOAD_DEV, extra_memory=100)
    assert patcher.current_loaded_size() == 600
    assert model.c.device.type == OFFLOAD_DEV.type
    assert model.a.device.type == LOAD_DEV.type
    assert model.b.device.type == LOAD_DEV.type


def test_partially_unload_evicts_enough_to_hit_budget():
    """Requesting 250 bytes freed evicts c(100)+b(200)=300 bytes."""
    model, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=10_000)

    patcher.partially_unload(device=OFFLOAD_DEV, extra_memory=250)
    assert patcher.current_loaded_size() == 400
    assert model.a.device.type == LOAD_DEV.type


def test_partially_load_unload_roundtrip():
    model, patcher = _fresh_patcher()
    patcher.partially_load(device=LOAD_DEV, extra_memory=10_000)
    assert patcher.current_loaded_size() == 700

    patcher.partially_unload(device=OFFLOAD_DEV, extra_memory=10_000)
    assert patcher.current_loaded_size() == 0
    for p in model.parameters():
        assert p.device.type == OFFLOAD_DEV.type


def test_partially_load_handles_empty_model():
    class Empty(nn.Module):
        pass

    model = Empty()
    patcher = ModelPatcher(model, load_device=LOAD_DEV, offload_device=OFFLOAD_DEV)
    patcher.partially_load(device=LOAD_DEV, extra_memory=1000)
    assert patcher.current_loaded_size() == 0


def test_partially_load_ignores_buffers_that_cannot_move():
    """Non-Parameter buffers should not crash the loader.  They're simply
    not counted toward the budget."""

    class WithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(10))
            self.register_buffer("running_mean", torch.zeros(10))

    model = WithBuffer()
    model.to(OFFLOAD_DEV)
    patcher = ModelPatcher(model, load_device=LOAD_DEV, offload_device=OFFLOAD_DEV)
    patcher.partially_load(device=LOAD_DEV, extra_memory=10_000)
    # Parameter moved; buffer tracking is implementation-defined
    assert model.w.device.type == LOAD_DEV.type
