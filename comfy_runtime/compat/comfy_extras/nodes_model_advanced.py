"""MIT port of ``comfy_extras.nodes_model_advanced``.

Provides real implementations for the two commonly-used nodes in this
module:

* :class:`ModelSamplingDiscrete` — switches a model between ``eps``
  and ``v_prediction`` sampling parameterizations (plus an ``lcm`` /
  ``x0`` passthrough).  The choice is recorded in
  ``model_options["transformer_options"]`` so downstream samplers can
  dispatch on it without reaching for the underlying model.

* :class:`RescaleCFG` — applies the terminal-SNR-safe CFG rescaling of
  *Common Diffusion Noise Schedules and Sample Steps Are Flawed*
  (Lin et al., 2024) as a post-cfg hook on the cloned
  :class:`ModelPatcher`.

The remaining node classes (``ModelSamplingContinuousEDM``, ``LCM``,
``ModelSamplingFlux``, …) stay as import-compat stubs; real workflows
rarely reach for them and their use cases are either covered by the
two classes above or shipped on top of the scheduler/sampler layer
already.
"""
from typing import List

import torch

from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


def _rescale_cfg(
    cfg_pred: torch.Tensor, cond_pred: torch.Tensor, multiplier: float
) -> torch.Tensor:
    """Rescale a CFG prediction to preserve the cond's per-sample std.

    Blend between the raw CFG prediction and a rescaled version that
    matches the standard deviation of the conditional prediction.  A
    ``multiplier`` of ``0`` is a no-op (returns ``cfg_pred``), ``1``
    fully rescales.

    Args:
        cfg_pred: The output of the standard CFG combination
            ``uncond + scale * (cond - uncond)``.
        cond_pred: The raw conditional prediction whose std should be
            preserved.
        multiplier: Blend factor in ``[0, 1]``.

    Returns:
        The rescaled prediction, same shape as ``cfg_pred``.

    Reference:
        Lin et al., *Common Diffusion Noise Schedules and Sample Steps
        Are Flawed*, WACV 2024.
    """
    # Per-sample std across everything but the batch dim.
    spatial = list(range(1, cfg_pred.dim()))
    std_cond = cond_pred.std(dim=spatial, keepdim=True)
    std_cfg = cfg_pred.std(dim=spatial, keepdim=True).clamp(min=1e-7)
    rescaled = cfg_pred * (std_cond / std_cfg)
    return multiplier * rescaled + (1.0 - multiplier) * cfg_pred


class ModelSamplingDiscrete:
    """Switch a diffusion model between ``eps``/``v_prediction``/``lcm``/
    ``x0`` sampling parameterizations.

    The switch itself doesn't retrain or replace the weights — it just
    records the requested parameterization on the cloned patcher so
    samplers that honor ``transformer_options["model_sampling_type"]``
    dispatch to the right prediction mode.  ``zsnr`` is a boolean flag
    that tells downstream samplers to use a zero-terminal-SNR sigma
    schedule.
    """

    SAMPLING_TYPES = ["eps", "v_prediction", "lcm", "x0"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sampling": (cls.SAMPLING_TYPES,),
                "zsnr": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model: ModelPatcher, sampling: str, zsnr: bool):
        """Clone ``model`` and record the sampling-type flag.

        Returns a tuple ``(cloned_patcher,)`` so the node matches the
        ComfyUI contract.
        """
        out = model.clone()
        transformer_options = out.model_options.setdefault(
            "transformer_options", {}
        )
        transformer_options["model_sampling_type"] = sampling
        transformer_options["zsnr"] = bool(zsnr)
        return (out,)


class RescaleCFG:
    """Apply terminal-SNR-safe CFG rescaling via a post-cfg hook.

    The hook is registered on a *cloned* :class:`ModelPatcher` so the
    original patcher is left untouched.  At sampling time, the hook
    receives a dict of tensors (``cond``, ``uncond``, the post-CFG
    ``denoised`` prediction and the effective ``cond_scale``) and
    returns a blended version of the denoised prediction whose
    per-sample std matches the conditional prediction's std.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "multiplier": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model: ModelPatcher, multiplier: float):
        """Clone ``model`` and register the rescale hook.

        Returns a tuple ``(cloned_patcher,)``.  The hook is a closure
        over ``multiplier`` so each ``RescaleCFG`` node records its
        own strength setting.
        """
        out = model.clone()

        def _post_cfg(args):
            # ComfyUI's post_cfg hook convention: ``args`` is a dict
            # carrying at least ``cond`` (the conditional prediction)
            # and ``denoised`` (the post-CFG result).  Fall back to
            # ``cfg_result`` / ``cond_denoised`` if a caller uses the
            # older key names.
            cond_pred = args.get("cond")
            if cond_pred is None:
                cond_pred = args.get("cond_denoised")
            cfg_pred = args.get("denoised")
            if cfg_pred is None:
                cfg_pred = args.get("cfg_result")
            if cond_pred is None or cfg_pred is None:
                # Nothing to do if the caller didn't give us what we
                # need — pass the input through unchanged.
                return cfg_pred if cfg_pred is not None else cond_pred
            return _rescale_cfg(cfg_pred, cond_pred, multiplier)

        out.set_model_sampler_post_cfg_function(_post_cfg)
        return (out,)


# ----------------------------------------------------------------------
# Stubs kept for import-compat — workflows rarely instantiate these and
# the heavy lifting they do (custom schedulers, Flux-specific sampling)
# is already covered by ``compat/comfy/samplers.py`` and
# ``compat/comfy_extras/nodes_custom_sampler.py``.
# ----------------------------------------------------------------------


class ModelSamplingContinuousEDM:
    pass


class ModelSamplingDiscreteDistilled:
    pass


class ModelSamplingStableCascade:
    pass


class LCM:
    """Latent Consistency Model wrapper stub."""

    pass


class ModelSamplingFlux:
    pass


def rescale_zero_terminal_snr_sigmas(sigmas: torch.Tensor) -> torch.Tensor:
    """Rescale a sigma schedule for zero-terminal-SNR sampling.

    Implements the schedule remap described in *Common Diffusion Noise
    Schedules and Sample Steps Are Flawed* (Lin et al., 2024): the
    schedule is squeezed so the final sigma corresponds to
    ``alpha_cumprod == 0`` (pure noise) while the initial sigma is
    preserved, which keeps inference at ``cfg > 1`` stable on models
    that were trained with ``zsnr=True``.

    AnimateDiff and a few other ZSNR-aware packs call this helper on
    their own sigma tensors before handing them to the sampler.

    Args:
        sigmas: A 1-D tensor of sigmas in ascending noise order (the
            schedule ComfyUI hands to the sampler).

    Returns:
        A tensor of the same shape with the terminal sigma pushed to
        represent pure noise.
    """
    alpha_cumprod = 1.0 / (sigmas**2 + 1.0)
    alpha = torch.sqrt(alpha_cumprod)
    alpha_T = alpha[-1]
    alpha_0 = alpha[0]
    # Avoid dividing by zero if the schedule is degenerate (alpha_0 == alpha_T).
    denom = (alpha_0 - alpha_T).clamp(min=1e-10)
    alpha = (alpha - alpha_T) / denom
    alpha = (alpha * alpha_0).clamp(min=1e-10)
    new_sigmas = torch.sqrt((1.0 - alpha**2).clamp(min=0.0)) / alpha
    return new_sigmas
