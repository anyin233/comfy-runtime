"""Stub for comfy.ldm.wan.model_multitalk.

Provides MultiTalk model stubs for Wan video generation compatibility.
"""

from typing import Any


class InfiniteTalkOuterSampleWrapper:
    """Stub wrapper for infinite-length multi-talk sampling.

    Wraps the outer sampling loop for multi-speaker video generation.
    """

    def __init__(self, **kwargs):
        """Initialize InfiniteTalkOuterSampleWrapper.

        Args:
            **kwargs: Wrapper configuration.
        """
        pass


class MultiTalkCrossAttnPatch:
    """Stub patch for multi-talk cross-attention layers.

    Modifies cross-attention to incorporate multi-speaker conditioning.
    """

    def __init__(self, **kwargs):
        """Initialize MultiTalkCrossAttnPatch.

        Args:
            **kwargs: Patch configuration.
        """
        pass


class MultiTalkGetAttnMapPatch:
    """Stub patch for extracting attention maps from multi-talk models.

    Captures attention maps during multi-speaker video generation.
    """

    def __init__(self, **kwargs):
        """Initialize MultiTalkGetAttnMapPatch.

        Args:
            **kwargs: Patch configuration.
        """
        pass


class WanMultiTalkAttentionBlock:
    """Stub attention block for Wan multi-talk models.

    Custom attention block supporting multi-speaker conditioning.
    """

    def __init__(self, **kwargs):
        """Initialize WanMultiTalkAttentionBlock.

        Args:
            **kwargs: Block configuration.
        """
        pass


class MultiTalkAudioProjModel:
    """Stub for audio projection model used in multi-talk generation.

    Projects audio features into the conditioning space.
    """

    def __init__(self, **kwargs):
        """Initialize MultiTalkAudioProjModel.

        Args:
            **kwargs: Model configuration.
        """
        pass


def project_audio_features(**kwargs) -> Any:
    """Project audio features into the multi-talk conditioning space.

    Args:
        **kwargs: Audio features and projection arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("project_audio_features is a stub")
