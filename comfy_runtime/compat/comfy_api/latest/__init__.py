"""comfy_api.latest — current V3 node API.

MIT reimplementation of the ComfyUI V3 node protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from . import _io as io

# Alias io to both lowercase and uppercase for compatibility
IO = io


class ComfyExtension(ABC):
    """Abstract base class for ComfyUI extensions.

    Extensions declare their node list via ``get_node_list()``.
    """

    async def on_load(self) -> None:
        """Called when the extension is loaded."""
        pass

    @abstractmethod
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """Return the list of node classes this extension provides."""
        ...


# UI namespace stub — nodes that import UI/ui get a minimal namespace
class _UIStub:
    """Minimal UI stub — Phase 5 will add full UI support."""
    pass


ui = _UIStub()
UI = ui


class _InputStub:
    """Stub for Input namespace used by some nodes."""
    pass


Input = _InputStub()


class _InputImplStub:
    """Stub for InputImpl namespace."""
    pass


InputImpl = _InputImplStub()


try:
    from ._util import VideoCodec, VideoContainer, VideoComponents, MESH, VOXEL, File3D, SVG as _SVG

    class Types:
        """Types namespace with video/3D type stubs."""
        VideoCodec = VideoCodec
        VideoContainer = VideoContainer
        VideoComponents = VideoComponents
        MESH = MESH
        VOXEL = VOXEL
        File3D = File3D
except ImportError:
    class Types:
        """Fallback Types namespace."""
        pass


class _ComfyAPIStub:
    """Stub for ComfyAPI class."""
    pass


ComfyAPI = _ComfyAPIStub


# Re-export _io for `from comfy_api.latest._io import ...`
_io = io

__all__ = [
    "ComfyAPI",
    "ComfyExtension",
    "io",
    "IO",
    "ui",
    "UI",
    "Input",
    "InputImpl",
    "Types",
]
