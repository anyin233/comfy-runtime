"""Stub for comfy_api.input_impl.

Provides input implementation stubs for the ComfyUI API layer.
"""


class VideoFromFile:
    """Stub for creating video input from a file path.

    Loads video data from disk for use in ComfyUI pipelines.
    """

    def __init__(self, **kwargs):
        """Initialize VideoFromFile.

        Args:
            **kwargs: File path and loading configuration.
        """
        pass


class VideoFromComponents:
    """Stub for creating video input from component tensors.

    Assembles video data from separate frame/audio tensors.
    """

    def __init__(self, **kwargs):
        """Initialize VideoFromComponents.

        Args:
            **kwargs: Component tensors and configuration.
        """
        pass
