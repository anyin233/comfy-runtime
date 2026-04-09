"""Custom operator wrappers for comfy_runtime.

MIT reimplementation of comfy.ops — provides manual_cast context,
disable_weight_init module namespace, and bias/weight casting utilities
that ComfyUI model code uses.
"""

import logging
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# cast_bias_weight
# ---------------------------------------------------------------------------


def cast_bias_weight(
    s: nn.Module,
    input: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    bias_dtype: Optional[torch.dtype] = None,
):
    """Cast a module's weight and bias to match the input tensor.

    Moves the module's weight (and optionally bias) to the same dtype
    and device as the input, or to explicitly specified targets.

    Args:
        s: The nn.Module whose weight/bias to cast.
        input: Reference tensor for dtype/device inference.
        dtype: Explicit target dtype (overrides input inference).
        device: Explicit target device (overrides input inference).
        bias_dtype: Separate dtype for bias (defaults to dtype).

    Returns:
        Tuple of (weight, bias) cast to the target dtype/device.
    """
    if dtype is None and input is not None:
        dtype = input.dtype
    if device is None and input is not None:
        device = input.device
    if bias_dtype is None:
        bias_dtype = dtype

    weight = s.weight
    bias = s.bias if hasattr(s, "bias") else None

    if weight is not None:
        if dtype is not None:
            weight = weight.to(dtype=dtype)
        if device is not None:
            weight = weight.to(device=device)

    if bias is not None:
        if bias_dtype is not None:
            bias = bias.to(dtype=bias_dtype)
        if device is not None:
            bias = bias.to(device=device)

    return weight, bias


# ---------------------------------------------------------------------------
# disable_weight_init — module namespace with init-skipping layers
# ---------------------------------------------------------------------------


class _Linear(nn.Linear):
    """Linear layer that skips weight initialization.

    Identical to nn.Linear but overrides reset_parameters to be a no-op,
    avoiding the cost of random initialization when weights will be
    loaded from a checkpoint.
    """

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _Conv1d(nn.Conv1d):
    """Conv1d that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _Conv2d(nn.Conv2d):
    """Conv2d that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _Conv3d(nn.Conv3d):
    """Conv3d that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _ConvTranspose1d(nn.ConvTranspose1d):
    """ConvTranspose1d that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _ConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _ConvTranspose3d(nn.ConvTranspose3d):
    """ConvTranspose3d that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _Embedding(nn.Embedding):
    """Embedding that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _GroupNorm(nn.GroupNorm):
    """GroupNorm that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _LayerNorm(nn.LayerNorm):
    """LayerNorm that skips weight initialization."""

    def reset_parameters(self):
        """No-op: skip random weight initialization."""
        pass


class _DisableWeightInit:
    """Namespace providing nn.Module subclasses that skip weight init.

    Usage::

        from comfy.ops import disable_weight_init
        layer = disable_weight_init.Linear(768, 768)

    All classes are drop-in replacements for their torch.nn counterparts,
    differing only in that reset_parameters() is a no-op.
    """

    Linear = _Linear
    Conv1d = _Conv1d
    Conv2d = _Conv2d
    Conv3d = _Conv3d
    ConvTranspose1d = _ConvTranspose1d
    ConvTranspose2d = _ConvTranspose2d
    ConvTranspose3d = _ConvTranspose3d
    Embedding = _Embedding
    GroupNorm = _GroupNorm
    LayerNorm = _LayerNorm


disable_weight_init = _DisableWeightInit()


# ---------------------------------------------------------------------------
# manual_cast — layers that cast inputs to match weight dtype
# ---------------------------------------------------------------------------


class _CastLinear(_Linear):
    """Linear that casts input to weight dtype before forward pass."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic input casting.

        Args:
            input: Input tensor (may be any dtype).

        Returns:
            Output tensor in the input's original dtype.
        """
        weight, bias = cast_bias_weight(self, input)
        return torch.nn.functional.linear(input, weight, bias)


class _CastConv1d(_Conv1d):
    """Conv1d that casts input to weight dtype before forward pass."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic input casting.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        weight, bias = cast_bias_weight(self, input)
        return self._conv_forward(input, weight, bias)


class _CastConv2d(_Conv2d):
    """Conv2d that casts input to weight dtype before forward pass."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic input casting.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        weight, bias = cast_bias_weight(self, input)
        return self._conv_forward(input, weight, bias)


class _CastConv3d(_Conv3d):
    """Conv3d that casts input to weight dtype before forward pass."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic input casting.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.
        """
        weight, bias = cast_bias_weight(self, input)
        return torch.nn.functional.conv3d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _ManualCast:
    """Namespace providing nn.Module subclasses that auto-cast input dtype.

    Same init-skipping behavior as disable_weight_init, plus automatic
    input dtype casting to match weight dtype during forward.

    Usage::

        from comfy.ops import manual_cast
        layer = manual_cast.Linear(768, 768)
    """

    Linear = _CastLinear
    Conv1d = _CastConv1d
    Conv2d = _CastConv2d
    Conv3d = _CastConv3d
    ConvTranspose1d = _ConvTranspose1d
    ConvTranspose2d = _ConvTranspose2d
    ConvTranspose3d = _ConvTranspose3d
    Embedding = _Embedding
    GroupNorm = _GroupNorm
    LayerNorm = _LayerNorm


manual_cast = _ManualCast()
