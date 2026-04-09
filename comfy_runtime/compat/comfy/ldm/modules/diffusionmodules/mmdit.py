"""MMDiT stub — import compatibility for custom nodes.

Custom nodes that reference ``comfy.ldm.modules.diffusionmodules.mmdit``
will get this stub.  Actual model implementations use diffusers.
"""

import math
import torch


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """Generate 1D sincos positional embedding from a grid.

    Args:
        embed_dim: Embedding dimension (must be even).
        pos: Position tensor of shape ``(M,)``.

    Returns:
        Embedding tensor of shape ``(M, embed_dim)``.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos.double(), omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1).float()
