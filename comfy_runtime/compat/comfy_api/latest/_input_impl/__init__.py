"""Stub for comfy_api.latest._input_impl.

Re-exports input implementation types for the ComfyUI V3 API layer.
"""

from comfy_runtime.compat.comfy_api.input_impl import (  # noqa: F401
    VideoFromFile,
    VideoFromComponents,
)

__all__ = [
    "VideoFromFile",
    "VideoFromComponents",
]
