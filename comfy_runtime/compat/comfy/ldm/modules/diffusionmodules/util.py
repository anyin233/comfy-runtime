"""Stub for ``comfy.ldm.modules.diffusionmodules.util``.

ComfyUI's util module exposes a handful of helper functions used by
diffusion model architectures (timestep_embedding, checkpoint helpers,
zero_module, etc.).  Custom nodes (ComfyUI-Advanced-ControlNet) import
some of them at module load time.

This stub provides the names; calling them produces a clear error
pointing at the diffusers / torch native equivalents.
"""
from typing import Any

import torch


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """Sinusoidal timestep embedding.

    Lightweight torch implementation matching ComfyUI's behavior so
    custom nodes that actually call this at runtime get a working
    result rather than a NotImplementedError.
    """
    import math

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module: Any) -> Any:
    """Zero out every parameter of ``module`` in place."""
    for p in module.parameters():
        with torch.no_grad():
            p.zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """Gradient-checkpointed forward.

    Stub: simply calls ``func(*inputs)``.  Custom nodes that need real
    activation checkpointing should switch to torch's native
    ``torch.utils.checkpoint.checkpoint``.
    """
    return func(*inputs)


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    """Beta schedule generator (stub)."""
    raise NotImplementedError(
        "comfy.ldm.modules.diffusionmodules.util.make_beta_schedule is a stub. "
        "Use diffusers.SchedulerMixin subclasses for beta schedules."
    )


def normalization(channels: int):
    """GroupNorm helper used by ResNet blocks."""
    return torch.nn.GroupNorm(32, channels)


def conv_nd(dims: int, *args, **kwargs):
    """N-dimensional convolution factory."""
    if dims == 1:
        return torch.nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return torch.nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return torch.nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return torch.nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs):
    if dims == 1:
        return torch.nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return torch.nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return torch.nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
