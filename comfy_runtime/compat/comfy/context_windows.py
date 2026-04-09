"""Stub for comfy.context_windows.

Provides context-window scheduling and fusion for temporal/video generation.
"""

import enum
from typing import Any, Optional


class ContextSchedules(enum.Enum):
    """Enum of available context-window scheduling strategies."""

    STATIC_STANDARD = "static_standard"
    UNIFORM_STANDARD = "uniform_standard"
    UNIFORM_LOOPED = "uniform_looped"
    BATCHED = "batched"

    @classmethod
    @property
    def LIST_STATIC(cls):
        return [e.value for e in cls]


class ContextFuseMethods(enum.Enum):
    """Enum of available context-fusion methods."""

    PYRAMID = "pyramid"

    @classmethod
    @property
    def LIST_STATIC(cls):
        return [e.value for e in cls]


class IndexListContextHandler:
    """Stub handler for index-list based context windows.

    Manages which frames belong to each context window during generation.
    """

    def __init__(self, **kwargs):
        """Initialize IndexListContextHandler.

        Args:
            **kwargs: Handler configuration.
        """
        pass


def create_prepare_sampling_wrapper(**kwargs) -> Any:
    """Create a wrapper that prepares sampling for context-windowed generation.

    Args:
        **kwargs: Configuration arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("create_prepare_sampling_wrapper is a stub")


def create_sampler_sample_wrapper(**kwargs) -> Any:
    """Create a wrapper around the sampler's sample method for context windows.

    Args:
        **kwargs: Configuration arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("create_sampler_sample_wrapper is a stub")


def get_matching_context_schedule(schedule_name: Optional[str] = None, **kwargs) -> Any:
    """Look up a context schedule by name.

    Args:
        schedule_name: Name of the context schedule to find.
        **kwargs: Additional lookup arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("get_matching_context_schedule is a stub")


def get_matching_fuse_method(fuse_name: Optional[str] = None, **kwargs) -> Any:
    """Look up a context-fusion method by name.

    Args:
        fuse_name: Name of the fusion method to find.
        **kwargs: Additional lookup arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("get_matching_fuse_method is a stub")
