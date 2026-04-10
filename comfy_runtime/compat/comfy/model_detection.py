"""Import-compat shim for ``comfy.model_detection``.

Custom nodes (e.g. ComfyUI-Advanced-ControlNet) reference this
module at import time for state-dict-based model family detection.
Our MIT compat layer performs the equivalent detection inside
``_diffusers_loader.detect_model_family`` — this module re-exports
that function under the ComfyUI name so ``from comfy.model_detection
import detect_unet_config`` (and similar) succeeds.

Functions that have no direct compat equivalent raise
NotImplementedError with a pointer to the compat API.
"""
from typing import Any, Dict

from comfy_runtime.compat.comfy._diffusers_loader import detect_model_family


def detect_unet_config(state_dict: Dict[str, Any], unet_key_prefix: str = ""):
    """Return a best-effort UNet config dict based on state-dict keys.

    This is a thin compat shim: ComfyUI's original ``detect_unet_config``
    reads dozens of keys to reconstruct SD1/SDXL/Flux architectures.
    Our MIT layer relies on ``diffusers.*.from_single_file`` to do the
    heavy lifting, so this function only returns the family name and
    the detected block count — enough for nodes that branch on it.
    """
    family = detect_model_family(state_dict)
    return {
        "family": family,
        "model_type": family,  # aliasing for ComfyUI-Advanced-ControlNet
    }


def model_config_from_unet(state_dict: Dict[str, Any], unet_key_prefix: str = ""):
    """Return a ComfyUI ``BASE`` config object for the detected family.

    Raises:
        NotImplementedError: The compat layer doesn't expose ComfyUI's
            ``supported_models_base.BASE`` class hierarchy.  Callers
            that need the full object should use
            :func:`comfy_runtime.compat.comfy.sd.load_checkpoint_guess_config`
            directly.
    """
    raise NotImplementedError(
        "comfy.model_detection.model_config_from_unet is not implemented "
        "in the MIT compat layer.  Use "
        "comfy_runtime.compat.comfy.sd.load_checkpoint_guess_config(path) "
        "which dispatches by family internally."
    )


def convert_config(unet_config: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through stub for legacy unet_config → new_config transforms."""
    return dict(unet_config)
