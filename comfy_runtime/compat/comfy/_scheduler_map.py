"""Map ComfyUI ``sampler_name`` + ``scheduler_name`` to a diffusers scheduler.

Phase 1 covers only the 4 most common samplers — ``euler``,
``euler_ancestral``, ``dpmpp_2m``, ``ddim`` — which map cleanly onto
diffusers classes.  Phase 2 fills in the remaining 29 samplers from
ComfyUI's full list, including ComfyUI-only variants that have no
direct diffusers counterpart (``uni_pc_bh2``, ``res_multistep*``,
``er_sde``, ``gradient_estimation``).

The goal for this file is **name resolution**, not numerical parity:
we guarantee that a ComfyUI workflow saying ``sampler_name="euler",
scheduler="normal"`` gets a sensibly-initialized diffusers scheduler.
Exact bit-for-bit parity with the original ComfyUI sampling path is
Phase 5 work (benchmark and hash comparison).
"""
from typing import Dict, Tuple


# name → (diffusers class name, extra constructor kwargs)
_SAMPLER_SCHEDULER: Dict[str, Tuple[str, Dict]] = {
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_ancestral": ("EulerAncestralDiscreteScheduler", {}),
    "dpmpp_2m": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++"},
    ),
    "ddim": ("DDIMScheduler", {}),
}


# ComfyUI scheduler_name → diffusers beta_schedule.  Not every combination
# is exactly equivalent — ``karras`` in ComfyUI means "use Karras sigmas",
# which diffusers expresses as ``use_karras_sigmas=True`` on the scheduler
# rather than a beta schedule choice.  We flag that separately below.
_BETA_SCHEDULE: Dict[str, str] = {
    "normal": "scaled_linear",
    "karras": "scaled_linear",
    "exponential": "scaled_linear",
    "linear": "linear",
    "simple": "scaled_linear",
    "sgm_uniform": "scaled_linear",
    "ddim_uniform": "scaled_linear",
    "beta": "scaled_linear",
}


def make_diffusers_scheduler(sampler_name: str, scheduler_name: str = "normal"):
    """Return an initialized diffusers scheduler for ComfyUI sampler names.

    Args:
        sampler_name: One of Phase 1's supported samplers.
        scheduler_name: ComfyUI scheduler_name — affects beta schedule and
            Karras sigma flag.

    Raises:
        KeyError: If ``sampler_name`` is not yet implemented (Phase 1 whitelist).
    """
    if sampler_name not in _SAMPLER_SCHEDULER:
        raise KeyError(
            f"Sampler {sampler_name!r} is not yet implemented in Phase 1. "
            f"Phase 1 supports: {sorted(_SAMPLER_SCHEDULER)}"
        )

    class_name, extra_kwargs = _SAMPLER_SCHEDULER[sampler_name]
    import diffusers

    cls = getattr(diffusers, class_name)
    kwargs = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule=_BETA_SCHEDULE.get(scheduler_name, "scaled_linear"),
    )
    kwargs.update(extra_kwargs)

    if scheduler_name == "karras":
        # Only some classes accept use_karras_sigmas; guard against older
        # or stricter schedulers.
        try:
            return cls(**kwargs, use_karras_sigmas=True)
        except TypeError:
            return cls(**kwargs)
    return cls(**kwargs)
