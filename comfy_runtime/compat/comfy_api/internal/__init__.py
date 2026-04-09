"""Internal V3 node protocol infrastructure.

Provides the ``_ComfyNodeInternal`` base class that all V3 nodes
inherit from, used by executor.py for node-type detection.
Also provides helper functions used by the io type system.
"""

from dataclasses import asdict
from typing import Optional, Callable, Any


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def prune_dict(d: dict) -> dict:
    """Return a copy of *d* with None values removed."""
    return {k: v for k, v in d.items() if v is not None}


def prune_dict_dataclass(dataclass_obj) -> dict:
    """Return dict of dataclass object with pruned None values."""
    return prune_dict(asdict(dataclass_obj))


def is_class(obj) -> bool:
    """Return True if *obj* is a class type (not an instance)."""
    return isinstance(obj, type)


def copy_class(cls: type) -> type:
    """Create a shallow copy of a class."""
    if cls is None:
        return None
    cls_dict = {
        k: v
        for k, v in cls.__dict__.items()
        if k not in ("__dict__", "__weakref__", "__module__", "__doc__")
    }
    new_cls = type(cls.__name__, (cls,), cls_dict)
    new_cls.__module__ = cls.__module__
    new_cls.__doc__ = cls.__doc__
    return new_cls


def shallow_clone_class(cls, new_name=None):
    """Shallow clone a class while preserving super() functionality."""
    new_name = new_name or f"{cls.__name__}Clone"
    new_bases = (cls,) + cls.__bases__
    return type(new_name, new_bases, dict(cls.__dict__))


class classproperty:
    """Descriptor for class-level properties."""

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def first_real_override(
    cls: type, name: str, *, base: type = None
) -> Optional[Callable]:
    """Return the override of *name* on *cls* that is not the base placeholder."""
    if base is None:
        if not hasattr(cls, "GET_BASE_CLASS"):
            raise ValueError("base is required if cls does not have a GET_BASE_CLASS")
        base = cls.GET_BASE_CLASS()
    base_attr = getattr(base, name, None)
    if base_attr is None:
        return None
    base_func = base_attr.__func__
    for c in cls.mro():
        if c is base:
            break
        if name in c.__dict__:
            func = getattr(c, name).__func__
            if func is not base_func:
                return getattr(cls, name)
    return None


# ---------------------------------------------------------------------------
# API Registry stubs
# ---------------------------------------------------------------------------


class ComfyAPIBase:
    """Base class for versioned ComfyAPI."""

    VERSION = "base"
    STABLE = False

    def __init__(self):
        pass


class ComfyAPIWithVersion(ComfyAPIBase):
    """ComfyAPI with version tracking."""

    pass


def register_versions(*versions):
    """Register API versions (no-op in comfy_runtime)."""
    pass


def get_all_versions():
    """Return all registered API versions."""
    return {}


# ---------------------------------------------------------------------------
# Node base classes
# ---------------------------------------------------------------------------


class _ComfyNodeInternal:
    """Base class that all V3-based node APIs inherit from.

    Used by execution.py for V3 node detection via ``issubclass()``.
    """

    @classmethod
    def GET_NODE_INFO_V1(cls): ...


class _NodeOutputInternal:
    """Base class that all V3-based NodeOutput objects inherit from."""

    pass
