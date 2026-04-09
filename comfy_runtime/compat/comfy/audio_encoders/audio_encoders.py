"""Stub for comfy.audio_encoders.audio_encoders.

Provides load_audio_encoder_from_sd for compatibility.
"""

from typing import Any


def load_audio_encoder_from_sd(sd, **kwargs) -> Any:
    """Load an audio encoder model from a state dict.

    Args:
        sd: State dict containing audio encoder weights.
        **kwargs: Additional loading arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("load_audio_encoder_from_sd is a stub")
