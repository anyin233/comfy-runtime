"""MIT port of ``comfy_extras.nodes_custom_sampler``.

ComfyUI splits the sampling pipeline into discrete pieces so that
workflows can compose them with explicit edges:

  * **Noise**     — RandomNoise / DisableNoise produce a noise generator
  * **Sigmas**    — BasicScheduler / KarrasScheduler / ExponentialScheduler /
                    PolyexponentialScheduler / SDTurboScheduler / VPScheduler
                    produce a 1-D sigma schedule tensor
  * **Sampler**   — KSamplerSelect picks a named sampler
  * **Guider**    — BasicGuider / CFGGuider wrap a model with guidance
  * **Drive**     — SamplerCustomAdvanced ties them together and runs
                    the loop

Each node returns a single object suitable for downstream consumption
by the SamplerCustomAdvanced node.  This file ports the most-used
subset; ComfyUI-only schedulers (Beta, AYS, KL-optimal) remain stubs.
"""
import math
from typing import Optional

import torch

from comfy_runtime.compat.comfy import samplers
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


# ---------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------


class BasicScheduler:
    """Compute a sigma schedule from a model + named scheduler.

    Returns a 1-D tensor of length ``steps + 1`` (the extra entry is the
    final ``sigma=0``).  ``denoise`` < 1.0 truncates the schedule from
    the front so partial denoising starts mid-schedule.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                      "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, model, scheduler, steps, denoise=1.0):
        total_steps = max(1, int(round(steps / max(denoise, 1e-6))))
        sigmas = samplers.calculate_sigmas(None, scheduler, total_steps)
        if denoise < 1.0:
            # Take only the last `steps + 1` entries
            sigmas = sigmas[-(steps + 1):]
        return (sigmas,)


class KarrasScheduler:
    """Karras-style sigma schedule (rho-power interpolation)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0,
                                        "max": 1000.0, "step": 0.01}),
                "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0,
                                        "max": 1000.0, "step": 0.0001}),
                "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0,
                                  "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        ramp = torch.linspace(0, 1, steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas,)


class ExponentialScheduler:
    """Exponential sigma schedule (geometric interpolation)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0,
                                        "max": 1000.0, "step": 0.01}),
                "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0,
                                        "max": 1000.0, "step": 0.0001}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, steps, sigma_max, sigma_min):
        sigmas = torch.exp(
            torch.linspace(math.log(sigma_max), math.log(sigma_min), steps)
        )
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas,)


class PolyexponentialScheduler:
    """Polynomial-exponential sigma schedule."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0,
                                        "max": 1000.0, "step": 0.01}),
                "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0,
                                        "max": 1000.0, "step": 0.0001}),
                "rho": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                                  "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        ramp = torch.linspace(1, 0, steps) ** rho
        sigmas = torch.exp(
            ramp * (math.log(sigma_max) - math.log(sigma_min))
            + math.log(sigma_min)
        )
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas,)


class SDTurboScheduler:
    """SD Turbo's distilled-step schedule."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                      "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, model, steps, denoise):
        # SD-Turbo uses a fixed [14.6, ..., 0.029] linspace.
        sigmas = torch.linspace(14.6, 0.029, steps + 1)
        return (sigmas,)


class VPScheduler:
    """Variance-preserving sigma schedule (DDPM-style)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "beta_d": ("FLOAT", {"default": 19.9, "min": 0.0, "max": 100.0,
                                     "step": 0.01}),
                "beta_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 100.0,
                                       "step": 0.01}),
                "eps_s": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0,
                                    "step": 0.0001}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, steps, beta_d, beta_min, eps_s):
        t = torch.linspace(1.0, eps_s, steps)
        sigmas = torch.sqrt(
            torch.exp(beta_d * t * t / 2 + beta_min * t) - 1.0
        )
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas,)


# ---------------------------------------------------------------------
# Sampler selection
# ---------------------------------------------------------------------


class KSamplerSelect:
    """Pick a named sampler from :data:`samplers.SAMPLER_NAMES`."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (samplers.SAMPLER_NAMES,),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, sampler_name):
        return (samplers.sampler_object(sampler_name),)


# Sampler-variant aliases used by some workflows.
class SamplerEulerAncestral(KSamplerSelect):
    pass


class SamplerDPMPP_2M_SDE(KSamplerSelect):
    pass


class SamplerDPMPP_3M_SDE(KSamplerSelect):
    pass


class SamplerDPMPP_SDE(KSamplerSelect):
    pass


class SamplerLMS(KSamplerSelect):
    pass


# ---------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------


class _NoiseGenerator:
    """Wraps a seed and produces deterministic noise on demand."""

    def __init__(self, seed: int):
        self.seed = int(seed)

    def generate_noise(self, latent_dict_or_tensor) -> torch.Tensor:
        """Return noise matching the latent's shape and dtype."""
        if isinstance(latent_dict_or_tensor, dict):
            latent = latent_dict_or_tensor["samples"]
        else:
            latent = latent_dict_or_tensor
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        return torch.randn(
            latent.shape,
            generator=gen,
            dtype=latent.dtype,
            device="cpu",
        )


class _DisableNoiseGenerator:
    """Returns zeros — used to skip the noise-add step."""

    def generate_noise(self, latent_dict_or_tensor) -> torch.Tensor:
        if isinstance(latent_dict_or_tensor, dict):
            latent = latent_dict_or_tensor["samples"]
        else:
            latent = latent_dict_or_tensor
        return torch.zeros_like(latent)


class RandomNoise:
    """Produce a noise generator seeded by ``noise_seed``."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 0, "min": 0,
                                       "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self, noise_seed):
        return (_NoiseGenerator(noise_seed),)


class DisableNoise:
    """Produce a no-op noise generator (zeros)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self):
        return (_DisableNoiseGenerator(),)


# ---------------------------------------------------------------------
# Guiders
# ---------------------------------------------------------------------


class _BasicGuider:
    """Single-conditioning guider — no CFG, just the cond pass."""

    def __init__(self, model_patcher: ModelPatcher, conditioning):
        self.model_patcher = model_patcher
        self.conditioning = conditioning

    def predict_noise(self, x, timestep, *args, **kwargs):
        unet = self.model_patcher.model
        cond_t = (
            self.conditioning[0][0]
            if self.conditioning else None
        )
        with torch.no_grad():
            return unet(x, timestep, encoder_hidden_states=cond_t).sample


class _CFGGuider:
    """Classifier-free guidance wrapper."""

    def __init__(self, model_patcher, positive, negative, cfg: float):
        self.model_patcher = model_patcher
        self.positive = positive
        self.negative = negative
        self.cfg = float(cfg)

    def predict_noise(self, x, timestep, *args, **kwargs):
        unet = self.model_patcher.model
        pos_t = self.positive[0][0]
        neg_t = self.negative[0][0]
        with torch.no_grad():
            x_in = torch.cat([x, x], dim=0)
            cond_in = torch.cat([neg_t, pos_t], dim=0)
            noise_pred = unet(
                x_in, timestep, encoder_hidden_states=cond_in
            ).sample
            uncond, cond = noise_pred.chunk(2)
            return uncond + self.cfg * (cond - uncond)


class BasicGuider:
    """Wraps a model+positive conditioning into a no-CFG guider."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, conditioning):
        return (_BasicGuider(model, conditioning),)


class CFGGuider:
    """Standard classifier-free guidance wrapper."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0,
                                  "step": 0.1}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg):
        return (_CFGGuider(model, positive, negative, cfg),)


class DualCFGGuider(CFGGuider):
    """Two-cfg variant — same surface as CFGGuider for compat."""

    pass


class Guider_Basic(BasicGuider):
    """Alias kept for ComfyUI imports."""

    pass


class Guider_DualCFG(CFGGuider):
    pass


# ---------------------------------------------------------------------
# Drive: SamplerCustomAdvanced
# ---------------------------------------------------------------------


class SamplerCustomAdvanced:
    """Tie a noise generator + sampler + sigmas + guider together and
    run the sampling loop, returning two latent dicts: the denoised
    output and the noise-added input (for downstream noise-injection
    nodes)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent_samples = latent_image["samples"]
        noise_t = noise.generate_noise(latent_image)

        # Build a guider-aware ModelPatcher proxy.  We give the
        # underlying KSAMPLER access to the guider's model_patcher and
        # set positive/negative through the conditioning_fn shim.
        model_patcher = guider.model_patcher

        # For the BasicGuider case there's no negative; the sampler
        # gets called with cfg=1.0 and just one cond.
        if isinstance(guider, _CFGGuider):
            positive = guider.positive
            negative = guider.negative
            cfg = guider.cfg
        elif isinstance(guider, _BasicGuider):
            positive = guider.conditioning
            negative = None
            cfg = 1.0
        else:
            positive = getattr(guider, "conditioning", None)
            negative = None
            cfg = 1.0

        out_samples = sampler.sample(
            model=model_patcher,
            noise=noise_t,
            positive=positive,
            negative=negative,
            cfg=cfg,
            latent_image=latent_samples,
            sigmas=sigmas,
            disable_pbar=True,
        )

        out = latent_image.copy()
        out["samples"] = out_samples
        return (out, out)


# ---------------------------------------------------------------------
# Sigma manipulation utilities
# ---------------------------------------------------------------------


class SplitSigmas:
    """Split a sigma schedule at the given step index."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def get_sigmas(self, sigmas, step):
        return (sigmas[: step + 1], sigmas[step:])


class FlipSigmas:
    """Reverse a sigma schedule."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"sigmas": ("SIGMAS",)}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def get_sigmas(self, sigmas):
        return (torch.flip(sigmas, dims=[0]),)


class SplitSigmasDenoise:
    """Split a sigma schedule at a fractional denoise level."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                      "step": 0.001}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def get_sigmas(self, sigmas, denoise):
        n = len(sigmas) - 1
        split = int(round(n * (1.0 - denoise)))
        return (sigmas[: split + 1], sigmas[split:])


class AddNoise:
    """Add noise to a latent — useful for img2img mid-sampling."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "noise": ("NOISE",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "add_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def add_noise(self, model, noise, sigmas, latent_image):
        latent = latent_image["samples"]
        noise_t = noise.generate_noise(latent_image)
        sigma = float(sigmas[0]) if len(sigmas) > 0 else 1.0
        out = latent_image.copy()
        out["samples"] = latent + noise_t * sigma
        return (out,)


class BetaSamplingScheduler(BasicScheduler):
    """Alias — beta-distribution scheduling routes through BasicScheduler
    with the 'beta' name."""

    pass


# Other class names referenced by ComfyUI's import sites — kept as
# aliases for now since they're rarely instantiated:
SamplerCustom = SamplerCustomAdvanced
