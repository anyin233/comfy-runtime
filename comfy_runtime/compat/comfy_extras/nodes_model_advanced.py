"""Import-compat stub for ``comfy_extras.nodes_model_advanced``.

Provides ModelSamplingDiscrete, ModelSamplingContinuousEDM, LCM,
RescaleCFG, and friends as named placeholders so custom nodes that
import from this module load successfully.  Real implementations
land in Phase 5 (benchmark suite drives the priority order).
"""
import torch


class ModelSamplingDiscrete:
    """Stub for the ModelSamplingDiscrete node — switches a model
    between EPS / V_PREDICTION sampling parameterizations."""

    pass


class ModelSamplingContinuousEDM:
    pass


class ModelSamplingDiscreteDistilled:
    pass


class ModelSamplingStableCascade:
    pass


class LCM:
    """Latent Consistency Model wrapper stub."""

    pass


class RescaleCFG:
    pass


class ModelSamplingFlux:
    pass


def rescale_zero_terminal_snr_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    """Rescale a sigma schedule for zero-terminal-SNR training.

    Stub implementation: returns the input unchanged.  AnimateDiff and
    related projects reference this function for their own modified
    schedules; they substitute their own copy when actually running.
    """
    return sigmas
