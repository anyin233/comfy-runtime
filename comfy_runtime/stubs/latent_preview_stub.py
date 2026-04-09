"""Stub replacement for ComfyUI's ``latent_preview`` module.

The real module imports torch, PIL, and several heavy comfy sub-modules.
This stub provides the public constants and functions that node code
references, all as silent no-ops.
"""

PREVIEW_NONE = 0


def set_preview_method(method):
    """No-op — ignores preview method changes."""
    pass


def get_previewer(device, latent_format):
    """No-op — always returns ``None`` (no previewer available).

    Args:
        device: Ignored.
        latent_format: Ignored.

    Returns:
        ``None``.
    """
    return None


def preview_to_image(latent_image, do_scale=True):
    """No-op — always returns ``None`` (no preview image).

    Args:
        latent_image: Ignored.
        do_scale: Ignored.

    Returns:
        ``None``.
    """
    return None


def prepare_callback(model, steps, x0_output_dict=None):
    """No-op — returns a callback that does nothing.

    Args:
        model: Ignored.
        steps: Ignored.
        x0_output_dict: Ignored.

    Returns:
        A no-op callback function.
    """

    def callback(step, x0, x, total_steps):
        pass

    return callback
