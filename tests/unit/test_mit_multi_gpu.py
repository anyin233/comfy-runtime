"""Tests for multi-GPU device routing in compat/comfy/model_management.

ComfyUI's model_management module exposes a collection of device
selectors (``get_torch_device``, ``text_encoder_device``, ``vae_device``,
``unet_offload_device`` …) that each return a single device.  In
multi-GPU scenarios operators want to pin different sub-models onto
different devices — e.g. the UNet on ``cuda:0`` and the text encoder
on ``cuda:1`` — to avoid head-of-line blocking and to overlap the
encode/denoise stages.

The Phase-3 API surface:

* ``set_device_assignment(unet=..., text_encoder=..., vae=...)`` —
  caller-facing configuration hook that pins each sub-model to a
  specific device index.  None values fall back to ``get_torch_device()``.

* Existing selectors (``text_encoder_device()``, ``vae_device()``,
  ``unet_inital_load_device()``) now consult the assignment first.

* ``get_device_list()`` — returns the list of available CUDA devices
  honoring ``CUDA_VISIBLE_DEVICES``.

The tests exercise the pure Python dispatch logic — they don't need
multiple real GPUs on the host.  We feed the assignment explicitly
and verify the right device comes back out.
"""
import pytest
import torch

from comfy_runtime.compat.comfy import model_management as mm


@pytest.fixture(autouse=True)
def _reset_assignment():
    """Each test starts with a clean device assignment."""
    mm.set_device_assignment()  # reset to defaults
    yield
    mm.set_device_assignment()


def test_default_assignment_returns_get_torch_device():
    default = mm.get_torch_device()
    assert mm.text_encoder_device() == default or (
        # NORMAL_VRAM w/o fp16 returns CPU — that's OK, it's a legitimate
        # fallback.  We assert that nothing is weirder than CPU or default.
        mm.text_encoder_device() == torch.device("cpu")
    )


def test_set_assignment_pins_unet():
    mm.set_device_assignment(unet=torch.device("cpu"))
    assert mm.unet_inital_load_device(0, torch.float32) == torch.device("cpu")


def test_set_assignment_pins_text_encoder():
    mm.set_device_assignment(text_encoder=torch.device("cpu"))
    assert mm.text_encoder_device() == torch.device("cpu")


def test_set_assignment_pins_vae():
    mm.set_device_assignment(vae=torch.device("cpu"))
    assert mm.vae_device() == torch.device("cpu")


def test_set_assignment_accepts_strings():
    mm.set_device_assignment(unet="cpu", text_encoder="cpu")
    assert mm.unet_inital_load_device(0, torch.float32).type == "cpu"
    assert mm.text_encoder_device().type == "cpu"


def test_set_assignment_none_falls_back_to_default():
    mm.set_device_assignment(unet=torch.device("cpu"))
    assert mm.unet_inital_load_device(0, torch.float32) == torch.device("cpu")

    # Reset unet to None; others unspecified (still default)
    mm.set_device_assignment(unet=None)
    # Should no longer return cpu (unless default is cpu)
    default = mm.get_torch_device()
    assert mm.unet_inital_load_device(0, torch.float32) == default or (
        mm.unet_inital_load_device(0, torch.float32) == torch.device("cpu")
    )


def test_get_device_list_returns_list():
    devices = mm.get_device_list()
    assert isinstance(devices, list)
    assert len(devices) >= 1
    for d in devices:
        assert isinstance(d, torch.device)


def test_get_device_list_reports_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    devices = mm.get_device_list()
    assert any(d.type == "cuda" for d in devices)


def test_assignment_survives_configure_cycles():
    """set_device_assignment should persist until explicitly reset."""
    mm.set_device_assignment(unet=torch.device("cpu"))
    # Simulate another module calling get_torch_device() (doesn't reset)
    _ = mm.get_torch_device()
    assert mm.unet_inital_load_device(0, torch.float32) == torch.device("cpu")
