"""Stub for comfy.patcher_extension.

Provides WrappersMP enum for model-patcher wrapper registration.
"""

import enum


class WrappersMP(str, enum.Enum):
    """Enum of model-patcher wrapper mount points.

    Each value identifies a specific point in the inference pipeline
    where wrapper functions can be injected.
    """

    CALC_COND_BATCH = "calc_cond_batch"
    DIFFUSION_MODEL = "diffusion_model"
    PREDICT_NOISE = "predict_noise"
    OUTER_SAMPLE = "outer_sample"
