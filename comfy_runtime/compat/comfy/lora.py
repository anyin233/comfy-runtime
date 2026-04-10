"""Import-compat shim for ``comfy.lora``.

ComfyUI's ``comfy.lora`` exposes lower-level LoRA conversion helpers
(model_lora_keys_unet, model_lora_keys_clip, load_lora) that some
custom nodes import directly.  Our MIT compat layer keeps the
high-level path in :mod:`comfy_runtime.compat.comfy._lora_peft`;
this module re-exports the public symbols by name.
"""
from typing import Any, Dict


def model_lora_keys_unet(model, key_map=None) -> Dict[str, str]:
    """Return ``{lora_key: target_state_dict_key}`` for a UNet.

    Stub: returns an empty mapping.  Real callers should use
    :func:`comfy_runtime.compat.comfy.sd.load_lora_for_models` which
    handles the kohya/peft conventions directly without needing this
    explicit key-map step.
    """
    return key_map or {}


def model_lora_keys_clip(model, key_map=None) -> Dict[str, str]:
    """Return ``{lora_key: target_state_dict_key}`` for a CLIP encoder."""
    return key_map or {}


def load_lora(lora_state: Dict[str, Any], to_load: Dict[str, str]) -> Dict[str, Any]:
    """Re-key a LoRA state dict according to ``to_load``.

    Stub: returns the input unchanged.  The real
    :func:`comfy_runtime.compat.comfy._lora_peft.extract_lora_deltas`
    handles all kohya/diffusers naming conventions natively.
    """
    return dict(lora_state)
