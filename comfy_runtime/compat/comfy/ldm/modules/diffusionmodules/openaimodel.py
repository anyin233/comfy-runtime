"""Stub for ``comfy.ldm.modules.diffusionmodules.openaimodel``.

The original ComfyUI module wraps the OpenAI guided-diffusion UNet
architecture used by SD1.x.  Custom nodes (ComfyUI-Advanced-ControlNet)
import a few class names from here at module-load time.

Our compat layer doesn't ship the full OpenAI UNet — instead, real
inference goes through diffusers' ``UNet2DConditionModel``.  These
stub classes exist purely so import succeeds.
"""
import torch.nn as nn


class UNetModel(nn.Module):
    """Import-compat stub for the OpenAI UNet."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class TimestepEmbedSequential(nn.Sequential):
    """Sequential block whose forward accepts a timestep argument."""

    pass


class ResBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class AttentionBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class Downsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class Upsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class SpatialTransformer(nn.Module):
    """Import-compat stub for the OpenAI SpatialTransformer block."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def forward_timestep_embed(*args, **kwargs):
    """Stub for the timestep-embedding forward helper."""
    return None


class VideoResBlock(nn.Module):
    """Import-compat stub for the video residual block (SVD)."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class TimestepBlock(nn.Module):
    """Marker base class — modules that consume the timestep embedding."""

    pass
