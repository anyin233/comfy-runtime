"""Sampler and scheduler definitions for comfy_runtime.

MIT reimplementation of comfy.samplers — provides sampler/scheduler name
lists, sigma schedule computation, and stub sampler/guider classes that
nodes can instantiate and configure.

Actual sampling algorithms are deferred to Phase 3; this module provides
the structural API surface that comfy_extras nodes depend on.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Set

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name lists — authoritative sets of sampler/scheduler identifiers
# ---------------------------------------------------------------------------

SAMPLER_NAMES: List[str] = [
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_sde",
    "gradient_estimation",
    "er_sde",
]

SCHEDULER_NAMES: List[str] = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "kl_optimal",
    "simple_trailing_zeros",
    "simple_trailing_zeros_sqrt",
    "ays",
]


# ---------------------------------------------------------------------------
# Sigma schedule computation
# ---------------------------------------------------------------------------


def _linear_sigmas(steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Compute linearly spaced sigmas from sigma_max to sigma_min.

    Args:
        steps: Number of sampling steps.
        sigma_min: Minimum sigma value.
        sigma_max: Maximum sigma value.

    Returns:
        Tensor of shape (steps + 1,) ending with 0.0.
    """
    sigmas = torch.linspace(sigma_max, sigma_min, steps)
    # Append terminal zero
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def _karras_sigmas(
    steps: int, sigma_min: float, sigma_max: float, rho: float = 7.0
) -> torch.Tensor:
    """Compute Karras et al. sigma schedule.

    Uses the formula:
        sigma_i = (sigma_min^(1/rho) + i/(n-1) * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho

    Args:
        steps: Number of sampling steps.
        sigma_min: Minimum sigma value.
        sigma_max: Maximum sigma value.
        rho: Rho parameter controlling the schedule curvature.

    Returns:
        Tensor of shape (steps + 1,) ending with 0.0.
    """
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = torch.zeros(steps)
    for i in range(steps):
        sigmas[i] = (max_inv_rho + i / (steps - 1) * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def _exponential_sigmas(steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Compute exponentially spaced sigmas (log-linear).

    Args:
        steps: Number of sampling steps.
        sigma_min: Minimum sigma value.
        sigma_max: Maximum sigma value.

    Returns:
        Tensor of shape (steps + 1,) ending with 0.0.
    """
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def calculate_sigmas(model_sampling, scheduler_name: str, steps: int) -> torch.Tensor:
    """Compute a sigma schedule for the given model and scheduler.

    Args:
        model_sampling: A model sampling object that provides `sigma_min`
            and `sigma_max` attributes (or a `.sigmas` tensor).
        scheduler_name: Name from SCHEDULER_NAMES.
        steps: Number of sampling steps.

    Returns:
        Tensor of sigmas with shape (steps + 1,), ending with 0.0.
    """
    if steps == 0:
        return torch.FloatTensor([])

    # Extract sigma bounds from model_sampling
    if hasattr(model_sampling, "sigma_max"):
        sigma_max = float(model_sampling.sigma_max)
        sigma_min = float(model_sampling.sigma_min)
    elif hasattr(model_sampling, "sigmas") and model_sampling.sigmas is not None:
        sigma_max = float(model_sampling.sigmas.max())
        sigma_min = float(model_sampling.sigmas[model_sampling.sigmas > 0].min())
    else:
        # Sensible defaults for DDPM-like models
        sigma_max = 14.6146
        sigma_min = 0.0292

    if scheduler_name == "karras":
        return _karras_sigmas(steps, sigma_min, sigma_max)
    elif scheduler_name == "exponential":
        return _exponential_sigmas(steps, sigma_min, sigma_max)
    elif scheduler_name == "sgm_uniform":
        return _linear_sigmas(steps, sigma_min, sigma_max)
    elif scheduler_name == "simple":
        # Simple schedule: linear in timestep, convert via model if possible
        if hasattr(model_sampling, "sigmas") and model_sampling.sigmas is not None:
            ss = model_sampling.sigmas.flip(0)
            total = len(ss)
            indices = torch.linspace(0, total - 1, steps + 1).long()
            return ss[indices]
        return _linear_sigmas(steps, sigma_min, sigma_max)
    else:
        # "normal" and all other schedulers fall back to linear spacing
        return _linear_sigmas(steps, sigma_min, sigma_max)


# ---------------------------------------------------------------------------
# Sampler object factories
# ---------------------------------------------------------------------------


def sampler_object(name: str) -> "KSAMPLER":
    """Return a ``KSAMPLER`` instance configured for the given sampler name.

    The returned object has a ``.sample()`` method that runs the full
    diffusers scheduler loop.  Unimplemented samplers (Phase 2+) raise
    ``KeyError`` when ``.sample()`` is called, not at factory time — this
    matches the ComfyUI behavior where ``samplers.sampler_object("lms")``
    succeeds but running it may not.

    Args:
        name: A sampler name from :data:`SAMPLER_NAMES`.

    Raises:
        ValueError: If the sampler name is not in the public list.
    """
    if name not in SAMPLER_NAMES:
        raise ValueError(f"Unknown sampler: {name!r}. Valid: {SAMPLER_NAMES}")
    return KSAMPLER(sampler_name=name)


class KSAMPLER:
    """Concrete sampler that runs a diffusers scheduler loop.

    Phase 1 handles the euler family (``euler``, ``euler_ancestral``,
    ``dpmpp_2m``, ``ddim``).  Other samplers raise ``KeyError`` on
    ``.sample()`` — the full name table is in
    :mod:`comfy_runtime.compat.comfy._scheduler_map`.

    CFG (classifier-free guidance) is implemented inside this class
    by batching the conditional and unconditional predictions through
    a single UNet forward pass per step.  Phase 2 will factor this out
    into a CFGGuider so that custom guidance strategies can hook in.
    """

    def __init__(
        self,
        sampler_function=None,
        extra_options: Optional[Dict] = None,
        inpaint_options: Optional[Dict] = None,
        sampler_name: Optional[str] = None,
    ):
        """Initialize the KSAMPLER.

        The constructor accepts either ``sampler_function`` (legacy
        ComfyUI callable API) or ``sampler_name`` (preferred MIT API).
        When both are passed, ``sampler_name`` wins.

        Args:
            sampler_function: Legacy callable for API compatibility.
            extra_options:    Extra kwargs forwarded to the scheduler.
            inpaint_options:  Inpaint-specific options (Phase 2).
            sampler_name:     ComfyUI sampler identifier (e.g. ``"euler"``).
        """
        self.sampler_function = sampler_function
        self.sampler_name = sampler_name or "euler"
        self.extra_options = extra_options or {}
        self.inpaint_options = inpaint_options or {}

    def sample(
        self,
        model,
        noise: torch.Tensor,
        positive=None,
        negative=None,
        cfg: float = 1.0,
        latent_image: Optional[torch.Tensor] = None,
        sigmas: Optional[torch.Tensor] = None,
        denoise_mask=None,
        callback=None,
        disable_pbar: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run a diffusers scheduler loop through the ModelPatcher's UNet.

        Args:
            model: A :class:`ModelPatcher` wrapping a diffusers UNet2DCondition
                (or any module with a ``(sample, timestep, encoder_hidden_states)``
                signature returning ``.sample``).
            noise: Initial Gaussian noise at max sigma, shape
                ``(B, latent_channels, H, W)``.
            positive: ComfyUI conditioning list for the positive prompt,
                ``[[cond, {"pooled_output": ...}]]``.
            negative: Same for the negative prompt.  May be ``None`` when
                ``cfg <= 1.0``.
            cfg: Classifier-free guidance scale.  ``cfg <= 1.0`` runs without
                batching the uncond path.
            latent_image: Starting latent (zeros for txt2img).
            sigmas: Sigma schedule from :func:`calculate_sigmas`.  Only the
                length is used (diffusers derives the timesteps from its own
                beta schedule); the sigma values are retained for callback
                compatibility.
            disable_pbar: Unused in Phase 1 (no tqdm integration yet).
            seed: Unused in Phase 1; retained for signature parity.
            **kwargs: Unused in Phase 1; retained for forward compatibility.

        Returns:
            Denoised latent tensor on the same device/dtype as
            ``latent_image``.
        """
        from comfy_runtime.compat.comfy._scheduler_map import (
            make_diffusers_scheduler,
        )

        if latent_image is None:
            raise ValueError("KSAMPLER.sample requires a latent_image tensor")

        unet = model.model if hasattr(model, "model") else model

        scheduler = make_diffusers_scheduler(self.sampler_name, "normal")
        num_steps = max(1, (sigmas.shape[0] - 1) if sigmas is not None else 1)
        scheduler.set_timesteps(num_steps)

        # Extract plain tensors from ComfyUI conditioning format
        pos_cond = positive[0][0] if positive else None
        neg_cond = negative[0][0] if negative else None

        unet_param = next(unet.parameters())
        device = unet_param.device
        dtype = unet_param.dtype

        latent = (latent_image + noise).to(device=device, dtype=dtype)
        if pos_cond is not None:
            pos_cond = pos_cond.to(device=device, dtype=dtype)
        if neg_cond is not None:
            neg_cond = neg_cond.to(device=device, dtype=dtype)

        do_cfg = cfg > 1.0 and neg_cond is not None

        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                if do_cfg:
                    latent_in = torch.cat([latent, latent], dim=0)
                    cond_in = torch.cat([neg_cond, pos_cond], dim=0)
                    noise_pred = unet(
                        latent_in, t, encoder_hidden_states=cond_in
                    ).sample
                    uncond, cond = noise_pred.chunk(2)
                    noise_pred = uncond + cfg * (cond - uncond)
                else:
                    noise_pred = unet(
                        latent, t, encoder_hidden_states=pos_cond
                    ).sample

            step_out = scheduler.step(noise_pred, t, latent)
            latent = step_out.prev_sample

            if callback is not None:
                try:
                    callback({"i": i, "denoised": latent, "x": latent,
                              "sigma": sigmas[i] if sigmas is not None else None})
                except Exception:
                    logger.debug("callback raised; continuing", exc_info=True)

        return latent.to(
            device=latent_image.device, dtype=latent_image.dtype
        )


def ksampler(
    sampler_name: str, extra_options: Dict = None, inpaint_options: Dict = None
) -> KSAMPLER:
    """Create a KSAMPLER from a sampler name.

    Args:
        sampler_name: Name from SAMPLER_NAMES.
        extra_options: Extra options dict.
        inpaint_options: Inpaint options dict.

    Returns:
        A KSAMPLER instance wrapping the requested sampler.
    """
    if extra_options is None:
        extra_options = {}
    if inpaint_options is None:
        inpaint_options = {}
    return KSAMPLER(sampler_object(sampler_name), extra_options, inpaint_options)


# ---------------------------------------------------------------------------
# Sampler base class
# ---------------------------------------------------------------------------


class Sampler:
    """Base sampler class that nodes can subclass.

    Provides the interface contract for custom sampler implementations.
    """

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        """Run the sampling loop.

        Args:
            model_wrap: The wrapped denoising model.
            sigmas: Sigma schedule tensor.
            extra_args: Extra arguments forwarded to the model.
            callback: Progress callback ``(step, x0, x, total_steps)``.
            noise: Initial noise tensor.
            latent_image: Optional starting latent for img2img.
            denoise_mask: Optional mask for inpainting.
            disable_pbar: Whether to disable the progress bar.

        Returns:
            Denoised latent tensor.
        """
        # TODO(Phase3): Implement base sampling loop.
        raise NotImplementedError("Sampler.sample must be overridden by subclass.")

    def max_denoise(self, model_wrap, sigmas):
        """Return whether this is a full denoise (maximum noise level).

        Args:
            model_wrap: The wrapped denoising model.
            sigmas: Sigma schedule tensor.

        Returns:
            True if the first sigma is at maximum noise level.
        """
        max_sigma = float(sigmas.max())
        if hasattr(model_wrap, "inner_model") and hasattr(
            model_wrap.inner_model, "model_sampling"
        ):
            model_sigma_max = float(model_wrap.inner_model.model_sampling.sigma_max)
            return math.isclose(max_sigma, model_sigma_max, rel_tol=0.05)
        return True


# ---------------------------------------------------------------------------
# KSampler class — high-level sampling interface
# ---------------------------------------------------------------------------


class KSampler:
    """High-level sampler wrapping model, steps, device, sampler, scheduler, denoise.

    This is the class instantiated by the KSampler node.

    Attributes:
        model: The model (typically a ModelPatcher).
        steps: Number of sampling steps.
        device: Torch device string or object.
        sampler: The KSAMPLER instance.
        scheduler: Scheduler name string.
        denoise: Denoise strength (0.0 to 1.0).
        model_options: Options dict from the model.
    """

    SAMPLERS = SAMPLER_NAMES
    SCHEDULERS = SCHEDULER_NAMES

    def __init__(
        self,
        model,
        steps: int,
        device,
        sampler: str,
        scheduler: str,
        denoise: float = 1.0,
        model_options: Dict = None,
    ):
        """Initialize KSampler.

        Args:
            model: The model patcher or model object.
            steps: Number of sampling steps.
            device: Target device for sampling.
            sampler: Sampler name string.
            scheduler: Scheduler name string.
            denoise: Denoise strength (1.0 = full denoise).
            model_options: Model options dict.
        """
        self.model = model
        self.steps = steps
        self.device = device
        self.sampler_name = sampler
        self.scheduler = scheduler
        self.denoise = denoise
        self.model_options = model_options or {}

        self.sampler = ksampler(sampler)

    def calculate_sigmas(self, steps: int) -> torch.Tensor:
        """Calculate the sigma schedule for the given number of steps.

        Args:
            steps: Number of sampling steps.

        Returns:
            Tensor of sigmas.
        """
        model_sampling = None
        if hasattr(self.model, "get_model_object"):
            try:
                model_sampling = self.model.get_model_object("model_sampling")
            except Exception:
                pass
        if (
            model_sampling is None
            and hasattr(self.model, "model")
            and hasattr(self.model.model, "model_sampling")
        ):
            model_sampling = self.model.model.model_sampling

        # Create a simple fallback if no model_sampling available
        if model_sampling is None:
            model_sampling = type(
                "FallbackSampling",
                (),
                {
                    "sigma_max": 14.6146,
                    "sigma_min": 0.0292,
                    "sigmas": None,
                },
            )()

        return calculate_sigmas(model_sampling, self.scheduler, steps)

    def sample(
        self,
        noise,
        positive,
        negative,
        cfg,
        latent_image=None,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        denoise_mask=None,
        sigmas=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        """Run the full sampling pipeline.

        Args:
            noise: Initial noise tensor.
            positive: Positive conditioning.
            negative: Negative conditioning.
            cfg: Classifier-free guidance scale.
            latent_image: Optional starting latent.
            start_step: Optional start step for partial sampling.
            last_step: Optional last step for partial sampling.
            force_full_denoise: Force full denoise on last step.
            denoise_mask: Optional inpainting mask.
            sigmas: Optional pre-computed sigma schedule.
            callback: Progress callback.
            disable_pbar: Disable progress bar.
            seed: Random seed.

        Returns:
            Denoised latent tensor.
        """
        # TODO(Phase3): Implement full sampling pipeline.
        raise NotImplementedError(
            "KSampler.sample is a stub. Full sampling will be implemented in Phase 3."
        )


# ---------------------------------------------------------------------------
# CFGGuider — classifier-free guidance wrapper
# ---------------------------------------------------------------------------


class CFGGuider:
    """Wraps a model with positive/negative conditioning and CFG scale.

    Nodes use this to set up guided sampling with configurable
    conditioning and guidance strength.

    Attributes:
        model_patcher: The underlying model patcher.
        conds: Dict of conditioning data (positive, negative).
        cfg: Guidance scale.
        model_options: Model options dict.
    """

    def __init__(self, model_patcher=None):
        """Initialize CFGGuider.

        Args:
            model_patcher: The model patcher to wrap for guided sampling.
        """
        self.model_patcher = model_patcher
        self.conds = {}
        self.cfg = 1.0
        self.model_options = (
            model_patcher.model_options.copy()
            if model_patcher and hasattr(model_patcher, "model_options")
            else {}
        )
        self.inner_model = None

    def set_conds(self, positive=None, negative=None, **kwargs):
        """Set conditioning data.

        Args:
            positive: Positive conditioning.
            negative: Negative conditioning.
            **kwargs: Additional named conditioning.
        """
        if positive is not None:
            self.conds["positive"] = positive
        if negative is not None:
            self.conds["negative"] = negative
        self.conds.update(kwargs)

    def set_cfg(self, cfg: float):
        """Set the classifier-free guidance scale.

        Args:
            cfg: Guidance scale value.
        """
        self.cfg = cfg

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        """Predict noise using CFG.

        Args:
            x: Current noisy latent.
            timestep: Current timestep/sigma.
            model_options: Optional model options override.
            seed: Random seed.

        Returns:
            Predicted noise tensor.
        """
        # TODO(Phase3): Implement CFG prediction.
        raise NotImplementedError(
            "CFGGuider.predict_noise is a stub. Will be implemented in Phase 3."
        )

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        """Run guided sampling.

        Args:
            noise: Initial noise tensor.
            latent_image: Starting latent (for img2img).
            sampler: Sampler instance.
            sigmas: Sigma schedule tensor.
            denoise_mask: Optional mask for inpainting.
            callback: Progress callback.
            disable_pbar: Disable progress bar.
            seed: Random seed.

        Returns:
            Denoised latent tensor.
        """
        # TODO(Phase3): Implement guided sampling.
        raise NotImplementedError(
            "CFGGuider.sample is a stub. Will be implemented in Phase 3."
        )

    def outer_sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        """Outer sampling entry point (wraps sample with model load/unload).

        Args:
            noise: Initial noise tensor.
            latent_image: Starting latent.
            sampler: Sampler instance.
            sigmas: Sigma schedule tensor.
            denoise_mask: Optional inpainting mask.
            callback: Progress callback.
            disable_pbar: Disable progress bar.
            seed: Random seed.

        Returns:
            Denoised latent tensor.
        """
        # TODO(Phase3): Implement with model loading/unloading.
        return self.sample(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask=denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def cfg_function(args: Dict[str, Any]) -> torch.Tensor:
    """Apply classifier-free guidance to model predictions.

    Args:
        args: Dict with keys:
            - "cond": Conditional prediction tensor.
            - "uncond": Unconditional prediction tensor.
            - "cond_scale": CFG scale factor.
            - (optional) "model_options": Model options dict.

    Returns:
        Guided prediction tensor.
    """
    cond = args.get("cond")
    uncond = args.get("uncond")
    cond_scale = args.get("cond_scale", 1.0)

    if cond is None or uncond is None:
        # TODO(Phase3): Handle missing cond/uncond gracefully.
        raise NotImplementedError(
            "cfg_function requires both cond and uncond predictions."
        )

    return uncond + cond_scale * (cond - uncond)


def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    """Encode model conditions from raw conditioning data.

    Args:
        model_function: The model's conditioning encoder.
        conds: List of conditioning dicts.
        noise: Noise tensor (used for shape reference).
        device: Target device.
        prompt_type: Type of prompt ("positive" or "negative").
        **kwargs: Additional arguments.

    Returns:
        Encoded conditioning list.
    """
    # TODO(Phase3): Implement condition encoding pipeline.
    return conds


def calc_cond_batch(model, conds, x, timestep, model_options):
    """Calculate a batch of conditional/unconditional predictions.

    Args:
        model: The denoising model.
        conds: List of conditioning data.
        x: Current noisy latent batch.
        timestep: Current timestep/sigma.
        model_options: Model options dict.

    Returns:
        List of prediction tensors.
    """
    # TODO(Phase3): Implement batched condition calculation.
    raise NotImplementedError(
        "calc_cond_batch is a stub. Will be implemented in Phase 3."
    )


def cast_to_load_options(model_options, device=None, dtype=None):
    """Cast model options for loading on a specific device/dtype.

    Args:
        model_options: Model options dict.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Modified model options dict.
    """
    # TODO(Phase3): Implement proper options casting.
    options = dict(model_options) if model_options else {}
    if device is not None:
        options["device"] = device
    if dtype is not None:
        options["dtype"] = dtype
    return options
