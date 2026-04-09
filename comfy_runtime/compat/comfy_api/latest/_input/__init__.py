"""Stub for comfy_api.latest._input.

Re-exports input types for the ComfyUI V3 API layer.
"""

from comfy_runtime.compat.comfy_api.input import (  # noqa: F401
    ImageInput,
    AudioInput,
    MaskInput,
    LatentInput,
    VideoInput,
)

__all__ = [
    "ImageInput",
    "AudioInput",
    "MaskInput",
    "LatentInput",
    "VideoInput",
]
