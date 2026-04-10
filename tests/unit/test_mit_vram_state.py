"""Tests for VRAMState wiring in ``load_models_gpu`` and ``LoadedModel.model_load``.

ComfyUI's ``model_management.vram_state`` is an enum: ``HIGH_VRAM`` /
``NORMAL_VRAM`` / ``LOW_VRAM`` / ``NO_VRAM`` / ``DISABLED``.  The state
governs what ``load_models_gpu`` actually does:

* ``HIGH_VRAM``   — keep everything on GPU.  ``free_memory`` never evicts.
* ``NORMAL_VRAM`` — default: full-model .to(device) via partially_load
                    with an infinite budget.
* ``LOW_VRAM``    — partial residency via ``partially_load(budget)`` where
                    ``budget == lowvram_model_memory`` passed in by the
                    caller.  Layers that don't fit stay on CPU.
* ``NO_VRAM``     — keep everything on CPU; budget=0.
* ``DISABLED``    — the model manager isn't supposed to run.  Callers
                    should bypass it.

This test file exercises the budget → ``partially_load`` path without
actually needing a real large model.  We use a tiny 700-byte fixture
(same as the Task 3.1 tests) and assert that after each vram_state
transition the resident-on-device byte count matches the expected
budget.
"""
import pytest
import torch
import torch.nn as nn

from comfy_runtime.compat.comfy import model_management as mm
from comfy_runtime.compat.comfy.model_management import (
    LoadedModel,
    VRAMState,
    load_models_gpu,
    unload_all_models,
)
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


HAS_CUDA = torch.cuda.is_available()
LOAD_DEV = torch.device("cuda:0") if HAS_CUDA else torch.device("cpu")
OFFLOAD_DEV = torch.device("cpu")


class _Tiny(nn.Module):
    """700-byte fixture: a(400) + b(200) + c(100)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(100))  # 400 B
        self.b = nn.Parameter(torch.zeros(50))   # 200 B
        self.c = nn.Parameter(torch.zeros(25))   # 100 B


def _fresh_patcher():
    model = _Tiny()
    model.to(OFFLOAD_DEV)
    return ModelPatcher(
        model, load_device=LOAD_DEV, offload_device=OFFLOAD_DEV
    )


@pytest.fixture(autouse=True)
def _reset_vram_state():
    """Each test starts with a clean vram_state + empty loaded list."""
    orig_state = mm.vram_state
    mm.current_loaded_models = []
    yield
    mm.vram_state = orig_state
    mm.current_loaded_models = []


def test_normal_vram_full_load_moves_everything():
    patcher = _fresh_patcher()
    mm.vram_state = VRAMState.NORMAL_VRAM
    load_models_gpu([patcher])
    assert patcher.current_loaded_size() == 700


def test_low_vram_honors_budget():
    """LOW_VRAM with a 600-byte budget loads a(400)+b(200), leaves c(100)."""
    patcher = _fresh_patcher()
    mm.vram_state = VRAMState.LOW_VRAM
    load_models_gpu([patcher], minimum_memory_required=600)
    assert patcher.current_loaded_size() == 600


def test_no_vram_keeps_everything_on_cpu():
    patcher = _fresh_patcher()
    mm.vram_state = VRAMState.NO_VRAM
    load_models_gpu([patcher])
    # Nothing should land on the compute device.
    assert patcher.current_loaded_size() == 0


def test_unload_all_models_frees_everything():
    patcher = _fresh_patcher()
    mm.vram_state = VRAMState.NORMAL_VRAM
    load_models_gpu([patcher])
    assert patcher.current_loaded_size() == 700

    unload_all_models()
    assert patcher.current_loaded_size() == 0
    assert mm.current_loaded_models == []


def test_reload_same_model_is_idempotent():
    """Loading the same ModelPatcher twice is a no-op on the second call."""
    patcher = _fresh_patcher()
    mm.vram_state = VRAMState.NORMAL_VRAM
    load_models_gpu([patcher])
    first_size = patcher.current_loaded_size()

    load_models_gpu([patcher])
    second_size = patcher.current_loaded_size()

    assert first_size == second_size == 700
    # Not double-inserted
    assert len(mm.current_loaded_models) == 1


def test_loaded_model_model_load_respects_lowvram_budget():
    """LoadedModel.model_load with lowvram_model_memory uses partially_load."""
    patcher = _fresh_patcher()
    loaded = LoadedModel(patcher)
    loaded.model_load(lowvram_model_memory=400)
    assert patcher.current_loaded_size() == 400


def test_loaded_model_model_load_full_when_no_budget():
    patcher = _fresh_patcher()
    loaded = LoadedModel(patcher)
    loaded.model_load()  # default: load everything
    assert patcher.current_loaded_size() == 700


def test_loaded_model_model_unload_returns_to_offload():
    patcher = _fresh_patcher()
    loaded = LoadedModel(patcher)
    loaded.model_load()
    assert patcher.current_loaded_size() == 700

    loaded.model_unload()
    assert patcher.current_loaded_size() == 0
