"""Stub for comfy.ldm.modules.attention.

Provides optimized_attention plus a ``wrap_attn`` decorator and the
common attention block class names that custom nodes (KJNodes,
AnimateDiff-Evolved) reference at import time.
"""
from typing import Any, Callable


def wrap_attn(attn_fn: Callable) -> Callable:
    """Decorator stub — returns the function unchanged.

    ComfyUI's ``wrap_attn`` adds conditional dispatch logging.  The
    compat layer relies on torch's native SDPA so this decorator is
    a no-op.
    """
    return attn_fn


class CrossAttention:
    """Import-compat stub for the CrossAttention block."""

    pass


class BasicTransformerBlock:
    """Import-compat stub for BasicTransformerBlock."""

    pass


class SpatialTransformer:
    pass


def optimized_attention(q, k, v, heads, **kwargs) -> Any:
    """Compute attention with platform-specific optimizations.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        heads: Number of attention heads.
        **kwargs: Additional attention arguments.

    Returns:
        None (stub).
    """
    raise NotImplementedError("optimized_attention is a stub")


def optimized_attention_masked(q, k, v, heads, mask=None, **kwargs):
    """Masked variant of :func:`optimized_attention`."""
    raise NotImplementedError("optimized_attention_masked is a stub")


def attention_basic(q, k, v, heads, mask=None, **kwargs):
    """Basic (unoptimized) attention path stub."""
    raise NotImplementedError("attention_basic is a stub")


def attention_pytorch(q, k, v, heads, mask=None, **kwargs):
    """torch.nn.functional.scaled_dot_product_attention path.

    Provides the named export so KJNodes' attention dispatch table
    can register it.  Real implementation goes through SDPA.
    """
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def attention_xformers(q, k, v, heads, mask=None, **kwargs):
    """xformers attention path stub.  We don't bundle xformers."""
    raise NotImplementedError(
        "attention_xformers is a stub — install xformers and import it directly."
    )


def attention_split(q, k, v, heads, mask=None, **kwargs):
    """Memory-split attention path stub."""
    raise NotImplementedError("attention_split is a stub")


def attention_sub_quad(q, k, v, heads, mask=None, **kwargs):
    """Sub-quadratic attention path stub.

    ComfyUI's sub-quad attention is a memory-efficient implementation
    that processes attention in chunks.  Custom nodes
    (AnimateDiff-Evolved) reference this name in their attention
    dispatch table.
    """
    raise NotImplementedError("attention_sub_quad is a stub")


def default(val, d):
    """ComfyUI helper: return ``val`` if not None, else ``d`` (or ``d()``).

    Custom nodes (AnimateDiff-Evolved) re-import this from
    ``comfy.ldm.modules.attention`` for use in their own attention
    plumbing.
    """
    if val is not None:
        return val
    return d() if callable(d) else d


def exists(val):
    """ComfyUI helper: True if ``val is not None``."""
    return val is not None


class FeedForward:
    """Import-compat stub for the FeedForward block class.

    ComfyUI's FeedForward is a 2-layer GeGLU MLP used inside
    BasicTransformerBlock.  Custom nodes (AnimateDiff-Evolved)
    reference the name at module-load time so we expose it here as
    a stub class.
    """

    pass


class GEGLU:
    """Import-compat stub for the GEGLU activation block."""

    pass


class SpatialVideoTransformer:
    """Import-compat stub for the SpatialVideoTransformer block.

    Used by SVD and ControlNet-Advanced for video conditioning.
    """

    pass


class TemporalTransformer:
    """Import-compat stub for the TemporalTransformer block."""

    pass
