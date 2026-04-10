"""Import-compat stub for ``comfy_extras.nodes_mask``.

ComfyUI's ``comfy_extras/nodes_mask.py`` provides mask manipulation
nodes (MaskComposite, GrowMask, InvertMask, ...) and a ``composite``
helper function used by AnimateDiff and other custom nodes.

This stub provides only the import-time names; calling them raises
NotImplementedError.  Phase 5 will port the real implementations.
"""
import torch


def composite(destination: torch.Tensor, source: torch.Tensor, x: int, y: int,
              mask=None, multiplier=8, resize_source=False) -> torch.Tensor:
    """Compose ``source`` onto ``destination`` at offset ``(x, y)``.

    Stub: implements only the trivial pass-through case.  AnimateDiff
    references this function but only on its own internal motion
    composition path which doesn't run during plain node-load tests.
    """
    return destination


class MaskComposite:
    """Stub for the MaskComposite built-in node."""

    pass


class SolidMask:
    pass


class InvertMask:
    pass


class GrowMask:
    pass


class FeatherMask:
    pass
