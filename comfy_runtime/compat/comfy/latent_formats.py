"""Latent format definitions for comfy_runtime.

MIT reimplementation of comfy.latent_formats — pure data classes that
define scale factors, channel counts, and latent-space transformations
for each supported model architecture.
"""

import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LatentFormat:
    """Base latent format with configurable scale and channel count.

    Subclasses override class attributes to define architecture-specific
    latent space parameters.

    Attributes:
        scale_factor: Multiplier applied when feeding latents to the model.
        latent_channels: Number of channels in the latent space.
        latent_rgb_factors: Optional RGB decode factors for preview.
        taesd_decoder_name: Optional name for TAESD tiny decoder.
    """

    scale_factor = 1.0
    latent_channels = 4
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        """Scale latent for model input.

        Args:
            latent: Raw latent tensor.

        Returns:
            Scaled latent tensor.
        """
        return latent * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        """Unscale model output latent.

        Args:
            latent: Model output latent tensor.

        Returns:
            Unscaled latent tensor.
        """
        return latent / self.scale_factor


# ---------------------------------------------------------------------------
# Architecture-specific formats
# ---------------------------------------------------------------------------

class SD15(LatentFormat):
    """Stable Diffusion 1.5 latent format."""

    scale_factor = 0.18215
    latent_channels = 4


class SDXL(LatentFormat):
    """Stable Diffusion XL latent format."""

    scale_factor = 0.13025
    latent_channels = 4


class SDXL_Playground_2_5(LatentFormat):
    """SDXL Playground v2.5 latent format."""

    scale_factor = 0.5
    latent_channels = 4


class SD3(LatentFormat):
    """Stable Diffusion 3 latent format."""

    scale_factor = 1.5305
    shift_factor = 0.0609
    latent_channels = 16

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        """Scale and shift latent for SD3 model input.

        Args:
            latent: Raw latent tensor.

        Returns:
            Scaled and shifted latent tensor.
        """
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        """Unscale and unshift SD3 model output.

        Args:
            latent: Model output latent tensor.

        Returns:
            Unscaled latent tensor.
        """
        return latent / self.scale_factor + self.shift_factor


class Flux(LatentFormat):
    """Flux latent format."""

    scale_factor = 0.3611
    shift_factor = 0.1159
    latent_channels = 16

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        """Scale and shift latent for Flux model input.

        Args:
            latent: Raw latent tensor.

        Returns:
            Scaled and shifted latent tensor.
        """
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        """Unscale and unshift Flux model output.

        Args:
            latent: Model output latent tensor.

        Returns:
            Unscaled latent tensor.
        """
        return latent / self.scale_factor + self.shift_factor


class Flux2(LatentFormat):
    """Flux 2 latent format (high-channel variant)."""

    scale_factor = 1.0
    latent_channels = 128


class Mochi(LatentFormat):
    """Mochi video model latent format."""

    scale_factor = 1.0
    latent_channels = 12


class Wan21(LatentFormat):
    """Wan 2.1 model latent format."""

    scale_factor = 1.0
    latent_channels = 16


class Wan22(LatentFormat):
    """Wan 2.2 model latent format (high-channel variant)."""

    scale_factor = 1.0
    latent_channels = 48


class HunyuanVideo(LatentFormat):
    """HunyuanVideo latent format."""

    scale_factor = 0.476986
    latent_channels = 16
