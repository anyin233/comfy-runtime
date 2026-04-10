"""Tests for the ported ``comfy_extras.nodes_model_advanced``.

Covers the two nodes that are part of Task 5.5 of the MIT rewrite:

* ``ModelSamplingDiscrete`` — switches a model between ``eps`` and
  ``v_prediction`` sampling parameterizations (with a ``zsnr`` toggle).
* ``RescaleCFG`` — applies terminal-SNR-safe CFG rescaling as a
  post-cfg hook on the patched model.

The remaining classes in the module (``ModelSamplingContinuousEDM``,
``LCM``, ``ModelSamplingFlux``, …) are import-compat stubs and are not
exercised here.
"""
import torch

from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat.comfy_extras import nodes_model_advanced as nma
from tests.fixtures.tiny_sd15 import make_tiny_sd15


def test_rescale_cfg_helper_preserves_shape_and_reduces_cfg():
    """``_rescale_cfg`` should blend the CFG prediction toward a
    normalized version of itself, preserving shape and changing values.

    At ``multiplier=0.7`` the output should be noticeably different
    from the raw CFG prediction (otherwise the node is a no-op).
    """
    torch.manual_seed(0)
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    cfg_scale = 8.0
    multiplier = 0.7

    cfg_pred = uncond + cfg_scale * (cond - uncond)
    rescaled = nma._rescale_cfg(cfg_pred, cond, multiplier)

    assert rescaled.shape == cfg_pred.shape
    # The rescale must actually change the tensor — otherwise the node
    # would be a no-op and the test wouldn't be verifying anything.
    assert not torch.allclose(rescaled, cfg_pred)


def test_model_sampling_discrete_returns_clone_with_sampling_type():
    """``ModelSamplingDiscrete.patch`` should return a cloned
    ``ModelPatcher`` that shares the same underlying model but carries
    the requested sampling-type flag in ``transformer_options``.
    """
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])

    (out,) = nma.ModelSamplingDiscrete().patch(
        model=model, sampling="v_prediction", zsnr=False,
    )

    # Clone: different ModelPatcher object, same underlying model.
    assert out is not model
    assert out.model is model.model

    # The sampling type flag should be readable from transformer_options
    # so downstream samplers can dispatch on it.
    to = out.model_options.get("transformer_options", {})
    assert to.get("model_sampling_type") == "v_prediction"
    assert to.get("zsnr") is False


def test_rescale_cfg_returns_patcher_with_post_cfg_hook():
    """``RescaleCFG.patch`` should return a cloned patcher whose
    ``model_options["sampler_post_cfg_function"]`` list contains a
    callable — the hook that performs the rescale at sampling time.

    This matches the ``ModelPatcher.set_model_sampler_post_cfg_function``
    API already used in the compat layer (see ``model_patcher.py``).
    """
    comp = make_tiny_sd15()
    model = ModelPatcher(comp["unet"])

    (out,) = nma.RescaleCFG().patch(model=model, multiplier=0.7)

    assert out is not model
    hooks = out.model_options.get("sampler_post_cfg_function", [])
    assert len(hooks) == 1
    assert callable(hooks[0])
