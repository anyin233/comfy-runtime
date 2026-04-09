"""Sigma-space sampling algorithms and schedule functions.

MIT reimplementation of k-diffusion sampling utilities.
Sigma schedule functions are derived from published papers:
- Karras et al. 2022 (EDM)
- Song et al. 2020 (Score-based)
"""

import math

import torch


# ---------------------------------------------------------------------------
# Sigma schedule functions
# ---------------------------------------------------------------------------

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Karras et al. 2022 sigma schedule.

    From "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_exponential(n, sigma_min, sigma_max, device="cpu"):
    """Exponential sigma schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.0, device="cpu"):
    """Polyexponential sigma schedule."""
    ramp = torch.linspace(1, 0, n, device=device)
    sigmas = sigma_min ** (1 - ramp) * sigma_max ** ramp
    if rho != 1.0:
        sigmas = sigmas ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0.0, beta=0.5, device="cpu"):
    """Laplace-distribution sigma schedule."""
    eps = 1e-5
    t = torch.linspace(0, 1, n, device=device)
    # Laplace CDF inverse
    sign = torch.sign(t - 0.5)
    log_term = -torch.log(1 - 2 * (t - 0.5).abs().clamp(min=eps))
    u = mu + beta * sign * log_term
    # Map u to sigma range
    u_min = u.min()
    u_max = u.max()
    if u_max - u_min > eps:
        u_normalized = (u - u_min) / (u_max - u_min)
    else:
        u_normalized = torch.linspace(0, 1, n, device=device)
    sigmas = sigma_max ** (1 - u_normalized) * sigma_min ** u_normalized
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Variance-preserving sigma schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


# ---------------------------------------------------------------------------
# Helper functions referenced by nodes
# ---------------------------------------------------------------------------

def sigma_to_half_log_snr(sigma):
    """Convert sigma to half log SNR: 0.5 * log(1/sigma^2)."""
    return -sigma.log()


def to_d(x, sigma, denoised):
    """Convert a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def append_zero(x):
    """Append a zero to the end of a 1D tensor."""
    return torch.cat([x, x.new_zeros([1])])
