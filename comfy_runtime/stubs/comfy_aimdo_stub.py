"""Stub replacement for the ``comfy_aimdo`` family of modules.

ComfyUI's codebase has ~7 import sites that pull in ``comfy_aimdo``
sub-modules at the top level.  This stub registers lightweight
:class:`types.ModuleType` objects in ``sys.modules`` so those imports
succeed without the real native extension being present.

Each stub module returns a fresh empty module for any attribute access
(via ``__getattr__``), preventing ``AttributeError`` crashes.
"""

import sys
import types

_AIMDO_MODULES = [
    "comfy_aimdo",
    "comfy_aimdo.model_vbar",
    "comfy_aimdo.control",
    "comfy_aimdo.model_mmap",
    "comfy_aimdo.torch",
    "comfy_aimdo.host_buffer",
]


def _make_stub_module(name: str) -> types.ModuleType:
    """Create a stub module that returns a new empty module for any attr access.

    Args:
        name: Fully-qualified module name (e.g. ``comfy_aimdo.control``).

    Returns:
        A :class:`types.ModuleType` with a permissive ``__getattr__``.
    """
    mod = types.ModuleType(name)
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    mod.__path__ = []

    def _getattr(name_: str):  # noqa: N807 — overriding __getattr__
        if name_.startswith("__") and name_.endswith("__"):
            raise AttributeError(name_)
        child = types.ModuleType(f"{name}.{name_}")
        return child

    mod.__getattr__ = _getattr  # type: ignore[assignment]
    return mod


def install_aimdo_stubs() -> None:
    """Register all ``comfy_aimdo`` stub modules in ``sys.modules``.

    Safe to call multiple times — already-registered modules are skipped.
    """
    for mod_name in _AIMDO_MODULES:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub_module(mod_name)
