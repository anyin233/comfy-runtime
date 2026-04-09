"""Model sampling type definitions for comfy_runtime.

MIT reimplementation of comfy.model_sampling — provides the sampling type
classes and prediction type mixins that ComfyUI nodes use to configure
model noise schedules and prediction targets.

Actual sampling math is deferred to Phase 3; this module provides the
class hierarchy that nodes expect to instantiate and manipulate.
"""

import logging
import math
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction type mixins — set model_type on the class
# ---------------------------------------------------------------------------

class EPS:
    """Epsilon (noise) prediction mixin.

    Models predict the noise added to the sample.
    """
    model_type = "eps"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Calculate denoised sample from epsilon prediction.

        Args:
            sigma: Current noise level.
            model_output: Model's epsilon prediction.
            model_input: Noisy input sample.

        Returns:
            Estimated clean sample.
        """
        # x0 = (x - sigma * eps) / (1 - sigma^2)^0.5  (simplified)
        # For simple sigma parameterization: x0 = x - sigma * eps
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_input.ndim - 1))
        return model_input - model_output * sigma


class V_PREDICTION:
    """Velocity prediction mixin.

    Models predict velocity v = alpha * eps - sigma * x0.
    """
    model_type = "v_prediction"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Calculate denoised sample from velocity prediction.

        Args:
            sigma: Current noise level.
            model_output: Model's velocity prediction.
            model_input: Noisy input sample.

        Returns:
            Estimated clean sample.
        """
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_input.ndim - 1))
        alpha = (1.0 - sigma ** 2) ** 0.5
        return model_input * alpha - model_output * sigma


class X0:
    """Direct x0 prediction mixin.

    Models directly predict the clean sample.
    """
    model_type = "x0"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Return model output directly as the denoised prediction.

        Args:
            sigma: Current noise level (unused).
            model_output: Model's x0 prediction.
            model_input: Noisy input sample (unused).

        Returns:
            The model output as-is.
        """
        return model_output


class EDM:
    """EDM (Elucidating Diffusion Models) prediction mixin."""
    model_type = "edm"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Calculate denoised sample using EDM parameterization.

        Args:
            sigma: Current noise level.
            model_output: Model prediction.
            model_input: Noisy input sample.

        Returns:
            Estimated clean sample.
        """
        return model_output


class CONST:
    """Constant noise schedule mixin."""
    model_type = "const"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Calculate denoised for constant noise model.

        Args:
            sigma: Current noise level.
            model_output: Model prediction.
            model_input: Noisy input sample.

        Returns:
            Estimated clean sample.
        """
        return model_output


class IMG_TO_IMG:
    """Image-to-image prediction mixin."""
    model_type = "img_to_img"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Calculate denoised for img2img model.

        Args:
            sigma: Current noise level.
            model_output: Model prediction.
            model_input: Noisy input sample.

        Returns:
            Estimated clean sample.
        """
        return model_output


class IMG_TO_IMG_FLOW:
    """Image-to-image flow matching prediction mixin."""
    model_type = "img_to_img_flow"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Calculate denoised for img2img flow model.

        Args:
            sigma: Current noise level.
            model_output: Model prediction.
            model_input: Noisy input sample.

        Returns:
            Estimated clean sample.
        """
        return model_output


class COSMOS_RFLOW:
    """Cosmos RFlow prediction mixin."""
    model_type = "cosmos_rflow"

    def calculate_denoised(self, sigma, model_output, model_input):
        """Calculate denoised for Cosmos RFlow model.

        Args:
            sigma: Current noise level.
            model_output: Model prediction.
            model_input: Noisy input sample.

        Returns:
            Estimated clean sample.
        """
        return model_output


# ---------------------------------------------------------------------------
# Discrete sampling base
# ---------------------------------------------------------------------------

class ModelSamplingDiscrete:
    """Discrete noise schedule sampling.

    Provides a table of sigmas indexed by timestep, along with
    conversion methods between sigma space and timestep space.

    Attributes:
        sigmas: 1-D tensor of sigma values indexed by timestep.
        log_sigmas: Log of sigmas for interpolation.
        sigma_min: Minimum sigma in the schedule.
        sigma_max: Maximum sigma in the schedule.
        num_timesteps: Number of discrete timesteps.
    """

    def __init__(self, model_config=None, num_timesteps: int = 1000):
        """Initialize discrete sampling schedule.

        Args:
            model_config: Optional model configuration dict/object.
            num_timesteps: Number of discrete timesteps (default 1000).
        """
        self.num_timesteps = num_timesteps
        self.model_config = model_config
        self._set_default_sigmas(num_timesteps)

    def _set_default_sigmas(self, num_timesteps: int):
        """Set up a default linear beta schedule.

        Args:
            num_timesteps: Number of timesteps.
        """
        # Standard DDPM linear beta schedule
        beta_start = 0.00085
        beta_end = 0.012
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_min = self.sigmas[0]
        self.sigma_max = self.sigmas[-1]

    def set_sigmas(self, sigmas: torch.Tensor):
        """Override the sigma schedule.

        Args:
            sigmas: New sigma schedule tensor.
        """
        self.sigmas = sigmas
        self.log_sigmas = sigmas.log()
        if len(sigmas) > 0:
            positive = sigmas[sigmas > 0]
            self.sigma_min = positive.min() if len(positive) > 0 else sigmas[0]
            self.sigma_max = sigmas.max()

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep by finding nearest in schedule.

        Args:
            sigma: Sigma value(s) to convert.

        Returns:
            Corresponding timestep(s).
        """
        log_sigma = sigma.log()
        dists = log_sigma.view(-1, 1) - self.log_sigmas.view(1, -1)
        return dists.abs().argmin(dim=1).float()

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert timestep to sigma via interpolation.

        Args:
            timestep: Timestep value(s) to convert.

        Returns:
            Corresponding sigma(s).
        """
        t = timestep.float()
        low = t.floor().long().clamp(0, len(self.sigmas) - 1)
        high = t.ceil().long().clamp(0, len(self.sigmas) - 1)
        frac = t - t.floor()
        log_sigma = (1.0 - frac) * self.log_sigmas[low] + frac * self.log_sigmas[high]
        return log_sigma.exp()

    def percent_to_sigma(self, percent: float) -> float:
        """Convert a denoise percentage to a sigma value.

        Args:
            percent: Denoise fraction (0.0 = no noise, 1.0 = max noise).

        Returns:
            Corresponding sigma value.
        """
        if percent <= 0.0:
            return float(self.sigma_max)
        if percent >= 1.0:
            return 0.0
        # Map percent to timestep index
        idx = int(round(percent * (self.num_timesteps - 1)))
        idx = max(0, min(idx, len(self.sigmas) - 1))
        return float(self.sigmas[-(idx + 1)])


# ---------------------------------------------------------------------------
# Continuous EDM sampling
# ---------------------------------------------------------------------------

class ModelSamplingContinuousEDM:
    """Continuous EDM noise schedule.

    Uses a continuous sigma parameterization typical of EDM models.

    Attributes:
        sigma_min: Minimum sigma.
        sigma_max: Maximum sigma.
        sigma_data: Data standard deviation for EDM scaling.
    """

    def __init__(self, model_config=None):
        """Initialize continuous EDM sampling.

        Args:
            model_config: Optional model configuration.
        """
        self.model_config = model_config
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.sigma_data = 0.5
        self.sigmas = None

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to EDM timestep (log-sigma scaling).

        Args:
            sigma: Sigma value(s).

        Returns:
            Timestep(s) in log-sigma space.
        """
        return 0.25 * sigma.log()

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert EDM timestep back to sigma.

        Args:
            timestep: Timestep in log-sigma space.

        Returns:
            Sigma value(s).
        """
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent: float) -> float:
        """Convert denoise percentage to sigma.

        Args:
            percent: Denoise fraction (0.0 = no noise, 1.0 = max noise).

        Returns:
            Corresponding sigma value.
        """
        if percent <= 0.0:
            return self.sigma_max
        if percent >= 1.0:
            return 0.0
        # Log-linear interpolation
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        return math.exp(log_max + percent * (log_min - log_max))


# ---------------------------------------------------------------------------
# Continuous V-prediction sampling
# ---------------------------------------------------------------------------

class ModelSamplingContinuousV:
    """Continuous V-prediction noise schedule.

    Attributes:
        sigma_min: Minimum sigma.
        sigma_max: Maximum sigma.
    """

    def __init__(self, model_config=None):
        """Initialize continuous V-prediction sampling.

        Args:
            model_config: Optional model configuration.
        """
        self.model_config = model_config
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.sigmas = None

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep.

        Args:
            sigma: Sigma value(s).

        Returns:
            Timestep(s).
        """
        return 0.25 * sigma.log()

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert timestep to sigma.

        Args:
            timestep: Timestep value(s).

        Returns:
            Sigma value(s).
        """
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent: float) -> float:
        """Convert denoise percentage to sigma.

        Args:
            percent: Denoise fraction.

        Returns:
            Corresponding sigma.
        """
        if percent <= 0.0:
            return self.sigma_max
        if percent >= 1.0:
            return 0.0
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        return math.exp(log_max + percent * (log_min - log_max))


# ---------------------------------------------------------------------------
# Flux-specific sampling
# ---------------------------------------------------------------------------

class ModelSamplingFlux:
    """Flux-specific sampling with shift parameter.

    Uses a flow-matching schedule with configurable shift for controlling
    the noise-to-signal transition.

    Attributes:
        shift: Shift parameter for the Flux schedule.
        sigma_min: Minimum sigma.
        sigma_max: Maximum sigma.
    """

    def __init__(self, model_config=None):
        """Initialize Flux sampling.

        Args:
            model_config: Optional model configuration.
        """
        self.model_config = model_config
        self.shift = 1.0
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        self.sigmas = None
        self.multiplier = 1000.0

    def set_parameters(self, shift: float = 1.0, timesteps: int = 1000):
        """Configure the Flux sampling schedule.

        Args:
            shift: Shift parameter (higher = more noise steps at high noise).
            timesteps: Number of discrete timesteps.
        """
        self.shift = shift
        ts = torch.arange(1, timesteps + 1, dtype=torch.float32) / timesteps
        sigmas = self._shift(ts, shift)
        self.sigmas = sigmas
        if len(sigmas) > 0:
            self.sigma_min = float(sigmas.min())
            self.sigma_max = float(sigmas.max())

    def _shift(self, timesteps: torch.Tensor, shift: float) -> torch.Tensor:
        """Apply shift transformation to timesteps.

        Args:
            timesteps: Raw timestep fractions in [0, 1].
            shift: Shift parameter.

        Returns:
            Shifted sigma values.
        """
        return shift * timesteps / (1.0 + (shift - 1.0) * timesteps)

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to Flux timestep.

        Args:
            sigma: Sigma value(s).

        Returns:
            Timestep(s) scaled by multiplier.
        """
        return sigma * self.multiplier

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert Flux timestep to sigma.

        Args:
            timestep: Timestep value(s).

        Returns:
            Sigma value(s).
        """
        return timestep / self.multiplier

    def percent_to_sigma(self, percent: float) -> float:
        """Convert denoise percentage to sigma.

        Args:
            percent: Denoise fraction (0.0 = max noise, 1.0 = clean).

        Returns:
            Corresponding sigma.
        """
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent


# ---------------------------------------------------------------------------
# Discrete flow matching
# ---------------------------------------------------------------------------

class ModelSamplingDiscreteFlow:
    """Discrete flow matching sampling.

    Uses a linear flow from noise to data with discrete timesteps.

    Attributes:
        sigma_min: Minimum sigma (typically 0).
        sigma_max: Maximum sigma (typically 1).
        shift: Optional shift parameter.
    """

    def __init__(self, model_config=None):
        """Initialize discrete flow matching sampling.

        Args:
            model_config: Optional model configuration.
        """
        self.model_config = model_config
        self.shift = 1.0
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        self.sigmas = None
        self.multiplier = 1000.0

    def set_parameters(self, shift: float = 1.0, timesteps: int = 1000):
        """Configure the flow matching schedule.

        Args:
            shift: Shift parameter.
            timesteps: Number of discrete timesteps.
        """
        self.shift = shift
        ts = torch.arange(1, timesteps + 1, dtype=torch.float32) / timesteps
        if shift != 1.0:
            ts = shift * ts / (1.0 + (shift - 1.0) * ts)
        self.sigmas = ts

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to flow timestep.

        Args:
            sigma: Sigma value(s).

        Returns:
            Timestep(s).
        """
        return sigma * self.multiplier

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert flow timestep to sigma.

        Args:
            timestep: Timestep value(s).

        Returns:
            Sigma value(s).
        """
        return timestep / self.multiplier

    def percent_to_sigma(self, percent: float) -> float:
        """Convert denoise percentage to sigma.

        Args:
            percent: Denoise fraction.

        Returns:
            Corresponding sigma.
        """
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent


# ---------------------------------------------------------------------------
# Cosmos RFlow sampling
# ---------------------------------------------------------------------------

class ModelSamplingCosmosRFlow:
    """Cosmos RFlow sampling schedule.

    Attributes:
        sigma_min: Minimum sigma.
        sigma_max: Maximum sigma.
    """

    def __init__(self, model_config=None):
        """Initialize Cosmos RFlow sampling.

        Args:
            model_config: Optional model configuration.
        """
        self.model_config = model_config
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        self.sigmas = None
        self.multiplier = 1000.0

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep.

        Args:
            sigma: Sigma value(s).

        Returns:
            Timestep(s).
        """
        return sigma * self.multiplier

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert timestep to sigma.

        Args:
            timestep: Timestep value(s).

        Returns:
            Sigma value(s).
        """
        return timestep / self.multiplier

    def percent_to_sigma(self, percent: float) -> float:
        """Convert denoise percentage to sigma.

        Args:
            percent: Denoise fraction.

        Returns:
            Corresponding sigma.
        """
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent


# ---------------------------------------------------------------------------
# Stable Cascade sampling
# ---------------------------------------------------------------------------

class StableCascadeSampling:
    """Stable Cascade noise schedule.

    Uses a cosine-based schedule specific to Stable Cascade models.

    Attributes:
        sigma_min: Minimum sigma.
        sigma_max: Maximum sigma.
    """

    def __init__(self, model_config=None):
        """Initialize Stable Cascade sampling.

        Args:
            model_config: Optional model configuration.
        """
        self.model_config = model_config
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        self.sigmas = None

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to Cascade timestep.

        Args:
            sigma: Sigma value(s).

        Returns:
            Timestep(s).
        """
        return sigma

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert Cascade timestep to sigma.

        Args:
            timestep: Timestep value(s).

        Returns:
            Sigma value(s).
        """
        return timestep

    def percent_to_sigma(self, percent: float) -> float:
        """Convert denoise percentage to sigma.

        Args:
            percent: Denoise fraction.

        Returns:
            Corresponding sigma.
        """
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent
