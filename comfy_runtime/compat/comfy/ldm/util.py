"""Stub for ``comfy.ldm.util``.

ComfyUI's ldm.util module exposes a handful of helper functions
(``instantiate_from_config``, ``count_params``, ``log_txt_as_img``)
that some custom nodes import at module-load time.
"""
from typing import Any, Dict


def instantiate_from_config(config: Dict[str, Any], *args, **kwargs):
    """Construct a class from a ComfyUI-style config dict.

    The config has the shape ``{"target": "module.path.ClassName",
    "params": {...}}``.  We import the target and instantiate it
    with the params; this is a faithful port of the original
    ComfyUI helper.
    """
    if "target" not in config:
        if config == "__is_first_stage__" or config == "__is_unconditional__":
            return None
        raise KeyError("Expected key 'target' to instantiate.")

    import importlib

    target = config["target"]
    module_path, _, class_name = target.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    params = config.get("params", {})
    return cls(**params, **kwargs)


def count_params(model, verbose: bool = False) -> int:
    """Count the trainable parameters of a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        import logging
        logging.getLogger(__name__).info(f"{model.__class__.__name__} has {total:,} params.")
    return total


def log_txt_as_img(*args, **kwargs):
    """Render text as an image — stub returning None."""
    return None


def ismap(x) -> bool:
    """Stub: True for 4D tensors with channels > 3."""
    if not hasattr(x, "shape"):
        return False
    return len(x.shape) == 4 and x.shape[1] > 3


def isimage(x) -> bool:
    """Stub: True for 4D tensors with 3 channels."""
    if not hasattr(x, "shape"):
        return False
    return len(x.shape) == 4 and x.shape[1] == 3


def exists(val) -> bool:
    """ComfyUI helper: True if ``val is not None``."""
    return val is not None


def default(val, d):
    """ComfyUI helper: ``val`` if not None, else ``d`` (or ``d()`` if callable)."""
    if val is not None:
        return val
    return d() if callable(d) else d


def mean_flat(tensor):
    """Mean over all dims except the batch dim."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
