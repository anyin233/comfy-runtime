"""Flux attention math utilities.

Standard attention operations reimplemented from the Flux paper.
"""

import torch


def attention(q, k, v, heads, mask=None):
    """Scaled dot-product attention.

    Args:
        q: Query tensor of shape ``(B, seq_len, dim)``.
        k: Key tensor of shape ``(B, seq_len, dim)``.
        v: Value tensor of shape ``(B, seq_len, dim)``.
        heads: Number of attention heads.
        mask: Optional attention mask.

    Returns:
        Output tensor of shape ``(B, seq_len, dim)``.
    """
    b, seq_len, dim = q.shape
    head_dim = dim // heads

    q = q.view(b, seq_len, heads, head_dim).transpose(1, 2)
    k = k.view(b, -1, heads, head_dim).transpose(1, 2)
    v = v.view(b, -1, heads, head_dim).transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    out = out.transpose(1, 2).contiguous().view(b, seq_len, dim)
    return out


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary position embedding to a query/key pair.

    Implements Flux's rotary embedding from the paper: each pair of
    consecutive features in q and k is rotated by the corresponding
    angle from ``freqs_cis``.  Returns the rotated ``(xq, xk)`` tuple.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rope(pos: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
    """Build the rotary frequency tensor for the Flux attention layers."""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out), torch.cos(out)], dim=-1)
    out = out.view(*out.shape[:-1], 2, 2)
    return out.float()
