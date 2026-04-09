"""Stub for comfy_api.latest._util.

Provides utility types for the ComfyUI V3 API.
"""

import enum
from typing import Any


class VideoCodec(str, enum.Enum):
    """Enum of supported video codecs."""

    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


class VideoContainer(str, enum.Enum):
    """Enum of supported video container formats."""

    MP4 = "mp4"
    WEBM = "webm"
    MKV = "mkv"
    MOV = "mov"


class VideoComponents:
    """Stub for video component data (frames, audio, metadata).

    Holds the decomposed parts of a video for processing.
    """

    def __init__(self, **kwargs):
        """Initialize VideoComponents.

        Args:
            **kwargs: Video component data.
        """
        self.frames = kwargs.get("frames", None)
        self.audio = kwargs.get("audio", None)
        self.fps = kwargs.get("fps", 24.0)


class MESH:
    """Stub for 3D mesh data type.

    Represents a 3D mesh object in the ComfyUI API.
    """

    def __init__(self, **kwargs):
        """Initialize MESH.

        Args:
            **kwargs: Mesh data and configuration.
        """
        pass


class VOXEL:
    """Stub for 3D voxel data type.

    Represents a voxel grid in the ComfyUI API.
    """

    def __init__(self, **kwargs):
        """Initialize VOXEL.

        Args:
            **kwargs: Voxel data and configuration.
        """
        pass


class File3D:
    """Stub for generic 3D file data type.

    Wraps a 3D file (GLTF, OBJ, etc.) for the ComfyUI API.
    """

    def __init__(self, **kwargs):
        """Initialize File3D.

        Args:
            **kwargs: File data and configuration.
        """
        pass


class SVG:
    """Stub for SVG data type.

    Represents an SVG image in the ComfyUI API.
    """

    def __init__(self, **kwargs):
        """Initialize SVG.

        Args:
            **kwargs: SVG data and configuration.
        """
        pass
