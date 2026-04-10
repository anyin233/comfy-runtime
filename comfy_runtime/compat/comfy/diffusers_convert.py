"""Import-compat shim for ``comfy.diffusers_convert``.

Custom nodes (e.g. was-node-suite-comfyui) reference
``comfy.diffusers_convert`` at import time for its state-dict
conversion helpers that translate between ComfyUI's old
single-file checkpoint format and diffusers' directory-based format.

We don't ship a real converter — the Phase-1/2 MIT rewrite loads
checkpoints directly via diffusers' ``from_single_file`` path, which
has a battle-tested ComfyUI → diffusers converter built in.  So any
custom node that actually *calls* these functions will hit a
NotImplementedError with a pointer to the proper loading path.

This module exists purely so ``import comfy.diffusers_convert``
succeeds.
"""
from typing import Any, Dict


def convert_unet_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ComfyUI-style UNet state dict to diffusers layout.

    Raises:
        NotImplementedError: Always — callers should use
            ``StableDiffusionPipeline.from_single_file`` which handles
            the conversion automatically.
    """
    raise NotImplementedError(
        "comfy.diffusers_convert.convert_unet_state_dict is not "
        "implemented in the MIT compat layer.  Use "
        "diffusers.StableDiffusionPipeline.from_single_file(path) "
        "or comfy_runtime.compat.comfy.sd.load_checkpoint_guess_config "
        "which both handle the ComfyUI → diffusers conversion."
    )


def convert_vae_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ComfyUI-style VAE state dict to diffusers layout.

    Raises:
        NotImplementedError: Always.  Use
            ``AutoencoderKL.from_single_file`` instead.
    """
    raise NotImplementedError(
        "comfy.diffusers_convert.convert_vae_state_dict is not "
        "implemented.  Use diffusers.AutoencoderKL.from_single_file."
    )


def convert_text_enc_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ComfyUI-style text encoder state dict to HF layout.

    Raises:
        NotImplementedError: Always.  Use
            ``transformers.CLIPTextModel.from_pretrained`` instead.
    """
    raise NotImplementedError(
        "comfy.diffusers_convert.convert_text_enc_state_dict is not "
        "implemented.  Use transformers.CLIPTextModel.from_pretrained."
    )
