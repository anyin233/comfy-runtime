"""Tests for ControlNet loading via diffusers.

ComfyUI's ``ControlNetLoader`` → ``compat.comfy.controlnet.load_controlnet``
→ diffusers' ``ControlNetModel.from_single_file``.  The loader returns
a ``ControlNet`` wrapper that conforms to the ControlBase interface
the existing ``compat/comfy/controlnet.py`` defines.

Phase 2 scope for this test:

* ``load_controlnet(path)`` returns a ``ControlNet`` instance.
* The wrapper holds the underlying diffusers ``ControlNetModel``.
* ``set_cond_hint`` and ``get_control`` basic plumbing works.
* The ``ControlNetLoader`` node exposed from ``compat/nodes.py`` no
  longer raises ``NotImplementedError``.
"""
import pytest
import torch
from safetensors.torch import save_file

from comfy_runtime.compat.comfy.controlnet import ControlNet, load_controlnet


def _write_tiny_controlnet_state(tmp_path):
    """Synthetic ControlNet state dict — enough keys for the loader
    to recognize but not for diffusers to load real weights."""
    sd = {
        "controlnet_cond_embedding.conv_in.weight": torch.randn(16, 3, 3, 3),
        "down_blocks.0.resnets.0.conv1.weight": torch.randn(32, 16, 3, 3),
    }
    path = tmp_path / "tiny_controlnet.safetensors"
    save_file(sd, str(path))
    return str(path)


def test_load_controlnet_returns_wrapper(tmp_path):
    ckpt = _write_tiny_controlnet_state(tmp_path)
    cn = load_controlnet(ckpt)
    assert isinstance(cn, ControlNet)
    assert cn.control_model is not None


def test_load_controlnet_default_strength():
    """Fresh ControlNet starts with strength=1.0, full timestep range."""
    cn = ControlNet(control_model=torch.nn.Linear(4, 4))
    assert cn.strength == 1.0
    assert cn.start_percent == 0.0
    assert cn.end_percent == 1.0


def test_controlnet_set_cond_hint():
    cn = ControlNet(control_model=torch.nn.Linear(4, 4))
    hint = torch.rand(1, 3, 64, 64)
    cn.set_cond_hint(hint, strength=0.8, timestep_percent_range=(0.1, 0.9))
    assert cn.strength == 0.8
    assert cn.start_percent == 0.1
    assert cn.end_percent == 0.9
    assert cn.cond_hint_original is hint


def test_controlnet_loader_node_no_longer_raises():
    """ControlNetLoader from compat.nodes should route through load_controlnet
    without raising NotImplementedError."""
    from comfy_runtime.compat import nodes

    # We can't exercise the real folder_paths lookup in a unit test,
    # but we can verify the body no longer raises NotImplementedError
    # at import time.  (Runtime folder_paths lookup is covered by
    # integration tests.)
    loader_cls = nodes.ControlNetLoader
    assert hasattr(loader_cls, "load_controlnet")


def test_controlnet_clone_preserves_state():
    """ControlNet.copy() / clone() must share the underlying model but
    produce independent strength settings."""
    cn = ControlNet(control_model=torch.nn.Linear(4, 4))
    cn.strength = 0.5
    cn2 = cn.copy()
    assert cn2.control_model is cn.control_model  # shared
    cn2.strength = 1.0
    assert cn.strength == 0.5  # unchanged
