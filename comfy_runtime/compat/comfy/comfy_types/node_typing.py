"""Node typing definitions for comfy_runtime.

Provides type hints used by ComfyUI node definitions.
"""

from typing import Any


# Type aliases for INPUT_TYPES dict structure
InputTypeDict = dict[str, Any]
InputTypeOptions = dict[str, Any]


class IO(str):
    """String-based type enum for ComfyUI node I/O types.

    Each class attribute holds the canonical string identifier for a
    particular data type flowing between nodes.  Instances behave as
    plain strings so they can be compared with ``==`` or used as dict
    keys.
    """

    IMAGE = "IMAGE"
    MASK = "MASK"
    LATENT = "LATENT"
    MODEL = "MODEL"
    CLIP = "CLIP"
    VAE = "VAE"
    CONDITIONING = "CONDITIONING"
    CONTROL_NET = "CONTROL_NET"
    STYLE_MODEL = "STYLE_MODEL"
    CLIP_VISION = "CLIP_VISION"
    CLIP_VISION_OUTPUT = "CLIP_VISION_OUTPUT"
    GLIGEN = "GLIGEN"
    UPSCALE_MODEL = "UPSCALE_MODEL"
    SAMPLER = "SAMPLER"
    SIGMAS = "SIGMAS"
    NOISE = "NOISE"
    GUIDER = "GUIDER"
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    COMBO = "COMBO"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    POINT = "POINT"
    FACE_ANALYSIS = "FACE_ANALYSIS"
    BBOX = "BBOX"
    SEGS = "SEGS"
    MESH = "MESH"
    VOXEL = "VOXEL"
    WEBCAM = "WEBCAM"
    ANY = "*"

    # Allow attribute-style access for forwards-compatibility: if a node
    # references ``IO.SOME_NEW_TYPE`` that we haven't listed yet, return
    # the attribute name as-is rather than raising AttributeError.
    def __class_getitem__(cls, name: str) -> str:  # type: ignore[override]
        return name
