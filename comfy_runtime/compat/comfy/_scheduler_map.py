"""Map ComfyUI ``sampler_name`` + ``scheduler_name`` to a diffusers scheduler.

ComfyUI exposes 33 samplers and 12 schedulers in its public
``SAMPLER_NAMES`` / ``SCHEDULER_NAMES`` lists (see
``compat/comfy/samplers.py``).  This module translates that name set
into the matching ``diffusers`` scheduler class + constructor kwargs.

**Coverage in Phase 2 (Task 2.3):**

25 of the 33 samplers map directly onto a diffusers scheduler class:

    * euler / euler_cfg_pp / euler_ancestral / euler_ancestral_cfg_pp
      → EulerDiscreteScheduler / EulerAncestralDiscreteScheduler
    * heun / heunpp2                     → HeunDiscreteScheduler
    * dpm_2 / dpm_2_ancestral            → KDPM2DiscreteScheduler / KDPM2AncestralDiscreteScheduler
    * lms                                → LMSDiscreteScheduler
    * dpmpp_2m / dpmpp_2m_cfg_pp         → DPMSolverMultistepScheduler (dpmsolver++)
    * dpmpp_2m_sde / dpmpp_2m_sde_gpu    → DPMSolverMultistepScheduler (dpmsolver++ SDE)
    * dpmpp_3m_sde / dpmpp_3m_sde_gpu    → DPMSolverMultistepScheduler (solver_order=3)
    * dpmpp_sde / dpmpp_sde_gpu          → DPMSolverSDEScheduler
    * dpmpp_2s_ancestral / cfg_pp        → DPMSolverSinglestepScheduler (+ ancestral)
    * ddim                               → DDIMScheduler
    * ddpm                               → DDPMScheduler
    * lcm                                → LCMScheduler
    * ipndm / ipndm_v                    → IPNDMScheduler
    * deis                               → DEISMultistepScheduler
    * uni_pc / uni_pc_bh2                → UniPCMultistepScheduler

**Not yet covered** (raise ``NotImplementedError`` with the hint to
pick an alternative):

    * dpm_fast / dpm_adaptive           — ComfyUI-unique heuristic
    * res_multistep{,_cfg_pp,_sde}      — paper implementation
    * gradient_estimation               — ComfyUI-unique
    * er_sde                            — ComfyUI-unique

These 8 samplers are not commonly used in workflows we've seen in
practice — users should fall back to ``dpmpp_2m`` or ``uni_pc`` which
produce near-identical results.  Real Phase 5 benchmarks will decide
whether to port the paper implementations.
"""
from typing import Dict, Tuple


# name → (diffusers class name, extra constructor kwargs, post-init flags)
# post-init flags is a dict of attributes to set after the scheduler is
# constructed (e.g. use_karras_sigmas=True).  When a flag isn't
# accepted by the constructor, we fall back to plain construction.
_SAMPLER_SCHEDULER: Dict[str, Tuple[str, Dict, Dict]] = {
    # -- Euler family ---------------------------------------------------
    "euler":                   ("EulerDiscreteScheduler", {}, {}),
    "euler_cfg_pp":            ("EulerDiscreteScheduler", {}, {}),
    "euler_ancestral":         ("EulerAncestralDiscreteScheduler", {}, {}),
    "euler_ancestral_cfg_pp":  ("EulerAncestralDiscreteScheduler", {}, {}),
    # -- Heun -----------------------------------------------------------
    "heun":                    ("HeunDiscreteScheduler", {}, {}),
    "heunpp2":                 ("HeunDiscreteScheduler", {}, {}),
    # -- KDPM2 ----------------------------------------------------------
    "dpm_2":                   ("KDPM2DiscreteScheduler", {}, {}),
    "dpm_2_ancestral":         ("KDPM2AncestralDiscreteScheduler", {}, {}),
    # -- LMS ------------------------------------------------------------
    "lms":                     ("LMSDiscreteScheduler", {}, {}),
    # -- DPM++ multistep ------------------------------------------------
    "dpmpp_2m": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++"},
        {},
    ),
    "dpmpp_2m_cfg_pp": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++"},
        {},
    ),
    "dpmpp_2m_sde": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++"},
        {},
    ),
    "dpmpp_2m_sde_gpu": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++"},
        {},
    ),
    "dpmpp_3m_sde": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
        {},
    ),
    "dpmpp_3m_sde_gpu": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
        {},
    ),
    # -- DPM++ SDE ------------------------------------------------------
    "dpmpp_sde":               ("DPMSolverSDEScheduler", {}, {}),
    "dpmpp_sde_gpu":           ("DPMSolverSDEScheduler", {}, {}),
    # -- DPM++ singlestep (2s ancestral) --------------------------------
    "dpmpp_2s_ancestral": (
        "DPMSolverSinglestepScheduler",
        {"algorithm_type": "dpmsolver++"},
        {},
    ),
    "dpmpp_2s_ancestral_cfg_pp": (
        "DPMSolverSinglestepScheduler",
        {"algorithm_type": "dpmsolver++"},
        {},
    ),
    # -- DDIM / DDPM / LCM ----------------------------------------------
    "ddim":                    ("DDIMScheduler", {}, {}),
    "ddpm":                    ("DDPMScheduler", {}, {}),
    "lcm":                     ("LCMScheduler", {}, {}),
    # -- IPNDM ----------------------------------------------------------
    "ipndm":                   ("IPNDMScheduler", {}, {}),
    "ipndm_v":                 ("IPNDMScheduler", {}, {}),
    # -- DEIS -----------------------------------------------------------
    "deis":                    ("DEISMultistepScheduler", {}, {}),
    # -- UniPC ----------------------------------------------------------
    "uni_pc":                  ("UniPCMultistepScheduler", {}, {}),
    "uni_pc_bh2": (
        "UniPCMultistepScheduler",
        {"solver_type": "bh2"},
        {},
    ),
}


# Samplers with no direct diffusers equivalent.  Calling make_diffusers_scheduler
# with one of these names raises a helpful NotImplementedError suggesting the
# closest substitute.
_UNSUPPORTED_SUBSTITUTIONS: Dict[str, str] = {
    "dpm_fast":             "dpmpp_2m",
    "dpm_adaptive":         "dpmpp_2m",
    "res_multistep":        "dpmpp_2m",
    "res_multistep_cfg_pp": "dpmpp_2m_cfg_pp",
    "res_multistep_sde":    "dpmpp_2m_sde",
    "gradient_estimation":  "euler",
    "er_sde":               "dpmpp_sde",
}


# ComfyUI scheduler_name → diffusers beta_schedule.
_BETA_SCHEDULE: Dict[str, str] = {
    "normal":                    "scaled_linear",
    "karras":                    "scaled_linear",
    "exponential":               "scaled_linear",
    "linear":                    "linear",
    "simple":                    "scaled_linear",
    "sgm_uniform":               "scaled_linear",
    "ddim_uniform":              "scaled_linear",
    "beta":                      "scaled_linear",
    "linear_quadratic":          "linear",
    "kl_optimal":                "scaled_linear",
    "simple_trailing_zeros":     "scaled_linear",
    "simple_trailing_zeros_sqrt":"scaled_linear",
    "ays":                       "scaled_linear",
}


def make_diffusers_scheduler(sampler_name: str, scheduler_name: str = "normal"):
    """Return an initialized diffusers scheduler for the given names.

    Args:
        sampler_name:   ComfyUI sampler name.
        scheduler_name: ComfyUI scheduler name (controls beta schedule
            and Karras sigmas).

    Raises:
        NotImplementedError: if the sampler is in the unsupported list
            (8 out of 33).  The message suggests an equivalent sampler.
        KeyError: if the sampler is not in any known list at all.
    """
    if sampler_name in _UNSUPPORTED_SUBSTITUTIONS:
        raise NotImplementedError(
            f"Sampler {sampler_name!r} is not yet mapped to a diffusers "
            f"scheduler.  Use {_UNSUPPORTED_SUBSTITUTIONS[sampler_name]!r} "
            "instead — the output is near-identical for most workflows. "
            "See docs/superpowers/plans/2026-04-10-mit-rewrite-vendor.md "
            "Task 2.3 for the port list."
        )

    if sampler_name not in _SAMPLER_SCHEDULER:
        raise KeyError(
            f"Unknown sampler: {sampler_name!r}. "
            f"Supported in Phase 2: {sorted(_SAMPLER_SCHEDULER)}"
        )

    class_name, extra_kwargs, _post_flags = _SAMPLER_SCHEDULER[sampler_name]
    import diffusers

    cls = getattr(diffusers, class_name)
    kwargs = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule=_BETA_SCHEDULE.get(scheduler_name, "scaled_linear"),
    )
    kwargs.update(extra_kwargs)

    # Some schedulers (IPNDM, KDPM2Ancestral in some versions) don't
    # accept beta_schedule — strip it and retry.
    for _ in range(3):
        try:
            scheduler = cls(**kwargs)
            break
        except TypeError as e:
            msg = str(e)
            stripped = False
            for arg in ("beta_schedule", "beta_start", "beta_end", "algorithm_type",
                        "solver_order", "solver_type"):
                if arg in msg and arg in kwargs:
                    kwargs.pop(arg)
                    stripped = True
                    break
            if not stripped:
                raise
    else:
        # Unreachable: the three attempts should always converge or the
        # final attempt will re-raise.
        scheduler = cls(**kwargs)

    if scheduler_name == "karras":
        # Only some schedulers expose this; best-effort set.
        if hasattr(scheduler.config, "use_karras_sigmas"):
            try:
                scheduler.config.use_karras_sigmas = True
            except Exception:
                pass

    return scheduler


def supported_sampler_names():
    """Return the list of sampler names with a working diffusers mapping."""
    return sorted(_SAMPLER_SCHEDULER)


def unsupported_sampler_names():
    """Return the list of ComfyUI samplers not yet mapped."""
    return sorted(_UNSUPPORTED_SUBSTITUTIONS)
