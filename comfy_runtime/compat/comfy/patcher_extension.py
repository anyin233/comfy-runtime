"""Stub for comfy.patcher_extension.

Provides the wrapper / callback mount-point enums and helpers that
ComfyUI's patcher_extension module exposes.  Custom nodes that
register wrappers or callbacks (AnimateDiff-Evolved, ipadapter, etc.)
import these names at module-load time.
"""
import enum
from typing import Any, Callable, Dict, List


class WrappersMP(str, enum.Enum):
    """Enum of model-patcher wrapper mount points.

    Each value identifies a specific point in the inference pipeline
    where wrapper functions can be injected.
    """

    CALC_COND_BATCH = "calc_cond_batch"
    DIFFUSION_MODEL = "diffusion_model"
    PREDICT_NOISE = "predict_noise"
    OUTER_SAMPLE = "outer_sample"
    SAMPLER_SAMPLE = "sampler_sample"
    APPLY_MODEL = "apply_model"


class CallbacksMP(str, enum.Enum):
    """Enum of model-patcher callback mount points.

    Callbacks fire at specific points without altering the data flow
    (unlike wrappers which can modify the inputs / outputs).  Custom
    nodes use these to log, track, or adjust state during sampling.
    """

    ON_PRE_RUN = "pre_run"
    ON_PREPARE_STATE = "prepare_state"
    ON_APPLY_HOOKS = "apply_hooks"
    ON_REGISTER_ALL_HOOK_PATCHES = "register_all_hook_patches"
    ON_INJECT_MODEL = "inject_model"
    ON_EJECT_MODEL = "eject_model"
    ON_CLEANUP = "cleanup"


def add_wrapper(wrapper_type, wrapper, *args, **kwargs):
    """Stub for the wrapper-registration helper.

    Real implementation would dispatch the registered wrapper to the
    matching pipeline mount-point during sampling.  Compat layer
    accepts the call and silently records it (no-op).
    """
    return None


def add_callback(callback_type, callback, *args, **kwargs):
    """Stub for the callback-registration helper."""
    return None


def get_all_wrappers(wrapper_type, transformer_options=None) -> List[Callable]:
    """Stub: return an empty wrapper list."""
    return []


def get_all_callbacks(callback_type, transformer_options=None) -> List[Callable]:
    """Stub: return an empty callback list."""
    return []


def merge_nested_dicts(*dicts) -> Dict[str, Any]:
    """Shallow merge of nested dicts (later wins)."""
    result: Dict[str, Any] = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


class PatcherInjection:
    """Import-compat stub for the PatcherInjection helper.

    ComfyUI's PatcherInjection bundles a per-model inject/eject
    callback pair so that custom nodes (e.g. AnimateDiff-Evolved) can
    register them on a ModelPatcher and have them fire on
    inject_model / eject_model events.
    """

    def __init__(self, inject=None, eject=None):
        self.inject = inject
        self.eject = eject

    def call_inject(self, *args, **kwargs):
        if callable(self.inject):
            return self.inject(*args, **kwargs)
        return None

    def call_eject(self, *args, **kwargs):
        if callable(self.eject):
            return self.eject(*args, **kwargs)
        return None


def copy_nested_dicts(d: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow nested-dict copy."""
    import copy as _copy
    return _copy.deepcopy(d)


def add_wrapper_with_key(wrapper_type, key, wrapper, *args, **kwargs):
    """Stub: register a named wrapper.

    ComfyUI's add_wrapper_with_key is the keyed variant of
    :func:`add_wrapper` — the key lets the caller later remove or
    replace a specific wrapper.  The compat layer is a no-op.
    """
    return None


def add_callback_with_key(callback_type, key, callback, *args, **kwargs):
    """Stub: register a named callback."""
    return None


def remove_wrapper_by_key(wrapper_type, key, *args, **kwargs):
    """Stub: remove a previously-registered keyed wrapper."""
    return None


def remove_callback_by_key(callback_type, key, *args, **kwargs):
    """Stub: remove a previously-registered keyed callback."""
    return None


class WrapperExecutor:
    """Import-compat stub for the WrapperExecutor helper.

    ComfyUI's WrapperExecutor wraps a sequence of registered wrappers
    and exposes them as a single chained callable.  Custom nodes
    (AnimateDiff-Evolved) reference this name at module-load time;
    the stub here just calls the inner function directly with no
    wrapping.
    """

    def __init__(self, original=None, class_obj=None, wrappers=None,
                 idx: int = 0):
        self.original = original
        self.class_obj = class_obj
        self.wrappers = wrappers or []
        self.idx = idx

    def execute(self, *args, **kwargs):
        if callable(self.original):
            return self.original(*args, **kwargs)
        return None

    @classmethod
    def new_executor(cls, original=None, wrappers=None):
        return cls(original=original, wrappers=wrappers)


class CallbackExecutor:
    """Import-compat stub for the CallbackExecutor helper."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def execute(self, *args, **kwargs):
        for cb in self.callbacks:
            if callable(cb):
                cb(*args, **kwargs)
        return None
