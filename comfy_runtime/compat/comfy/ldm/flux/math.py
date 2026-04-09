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

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask
    )

    out = out.transpose(1, 2).contiguous().view(b, seq_len, dim)
    return out
