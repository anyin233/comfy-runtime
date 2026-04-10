"""Stub for comfy.clip_vision.

Provides clip_preprocess, Output, and ClipVisionModel for compatibility.
"""

from typing import Any


def clip_preprocess(**kwargs) -> Any:
    """Preprocess images for CLIP vision model.

    Args:
        **kwargs: Preprocessing arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("clip_preprocess is a stub")


class Output:
    """Generic output container for CLIP vision results.

    Attributes:
        penultimate_hidden_states: Hidden states from the penultimate layer.
        image_embeds: Image embeddings from CLIP.
    """

    def __init__(self, **kwargs):
        """Initialize Output.

        Args:
            **kwargs: Arbitrary output attributes.
        """
        self.penultimate_hidden_states = None
        self.image_embeds = None
        for k, v in kwargs.items():
            setattr(self, k, v)


class ClipVisionModel:
    """Stub for CLIP vision model wrapper.

    Wraps a CLIP vision model for use in ComfyUI pipelines.
    """

    def __init__(self, **kwargs):
        """Initialize ClipVisionModel.

        Args:
            **kwargs: Model configuration arguments.
        """
        pass

    def encode_image(self, image, **kwargs) -> Output:
        """Encode an image using the CLIP vision model.

        Args:
            image: Input image tensor.
            **kwargs: Additional arguments.

        Returns:
            Output object with encoded representations.
        """
        raise NotImplementedError("ClipVisionModel.encode_image is a stub")


def load(ckpt_path: str):
    """Load a CLIP vision model from a single-file checkpoint.

    Import-compat stub: custom nodes (e.g. IPAdapter_plus) reference
    ``comfy.clip_vision.load`` at import time, so we need this symbol
    to exist.  Routes through transformers ``CLIPVisionModel`` when
    available, else raises a clear error pointing at the ControlNet
    loader pattern as a template for Phase 5 work.
    """
    try:
        from transformers import CLIPVisionModel

        return CLIPVisionModel.from_pretrained(ckpt_path)
    except Exception as e:
        raise NotImplementedError(
            f"comfy.clip_vision.load for {ckpt_path!r} not implemented: {e}"
        )
