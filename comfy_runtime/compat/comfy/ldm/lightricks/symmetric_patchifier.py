"""Stub for comfy.ldm.lightricks.symmetric_patchifier.

Provides SymmetricPatchifier for compatibility.
"""


class SymmetricPatchifier:
    """Stub for symmetric patchification used in Lightricks models."""

    def __init__(self, patch_size=1, *args, **kwargs):
        self.patch_size = patch_size


def latent_to_pixel_coords(latent_coords, vae_scale_factor=8):
    """Convert latent coordinates to pixel coordinates.

    Args:
        latent_coords: Coordinates in latent space.
        vae_scale_factor: VAE spatial downscale factor.

    Returns:
        Pixel-space coordinates.
    """
    return latent_coords * vae_scale_factor
