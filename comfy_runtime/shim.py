"""Shim layer that registers compat modules in sys.modules.

After ``install_shims()`` runs, user code can write::

    from comfy.model_management import get_torch_device
    import nodes

and it transparently resolves to the MIT reimplementations under ``compat/``.
"""

import importlib
import importlib.machinery
import importlib.util
import sys

_COMPAT_PKG = "comfy_runtime.compat"

# Full packages — Python's PathFinder resolves submodule imports via __path__
_PACKAGE_NAMES = ("comfy", "comfy_api", "comfy_execution")

# Standalone modules — registered via a meta-path finder
_STANDALONE_MODULES = (
    "nodes",
    "folder_paths",
    "node_helpers",
    "execution",
    "comfyui_version",
    "protocol",
)


class _CompatLoader:
    """Loader that imports the compat module and aliases it directly.

    Instead of creating a new module and copying attributes, we directly
    reuse the compat module object.  This ensures that functions like
    ``folder_paths.get_output_directory()`` — which rely on module-level
    globals — see changes made via ``folder_paths.output_directory = ...``.
    """

    def __init__(self, compat_name):
        self._compat_name = compat_name

    def create_module(self, spec):
        compat = importlib.import_module(self._compat_name)
        return compat

    def exec_module(self, module):
        sys.modules[module.__name__] = module


class _CompatFinder:
    """Meta-path finder that lazily redirects standalone module imports to compat/.

    Uses the modern ``find_spec`` protocol (PEP 451).
    """

    def find_spec(self, fullname, path, target=None):
        if fullname not in _STANDALONE_MODULES or fullname in sys.modules:
            return None

        compat_name = f"{_COMPAT_PKG}.{fullname}"
        loader = _CompatLoader(compat_name)
        return importlib.machinery.ModuleSpec(fullname, loader)


def _wire_attribute_chain(pkg_name, compat_mod):
    """Ensure pre-registered sub-modules are reachable via attribute access.

    Modules that were manually placed in ``sys.modules`` by the bootstrap
    (e.g. ``comfy.options``, ``comfy.cli_args``) need their full
    attribute chain set up so that dotted access works at runtime.
    """
    prefix = f"{pkg_name}."
    for key, mod in sorted(sys.modules.items()):
        if not key.startswith(prefix) or mod is None:
            continue

        parts = key[len(prefix) :].split(".")

        for depth in range(len(parts)):
            if depth == 0:
                parent = compat_mod
            else:
                parent_key = f"{pkg_name}.{'.'.join(parts[:depth])}"
                parent = sys.modules.get(parent_key)
                if parent is None:
                    continue

            child_name = parts[depth]

            if depth == len(parts) - 1:
                if not hasattr(parent, child_name):
                    setattr(parent, child_name, mod)
            else:
                if not hasattr(parent, child_name):
                    child_key = f"{pkg_name}.{'.'.join(parts[: depth + 1])}"
                    child_mod = sys.modules.get(child_key)
                    if child_mod is None:
                        try:
                            child_mod = importlib.import_module(
                                f"{_COMPAT_PKG}.{child_key}"
                            )
                            sys.modules[child_key] = child_mod
                        except ImportError:
                            continue
                    setattr(parent, child_name, child_mod)


# Backward-compatible aliases for tests
_VendorFinder = _CompatFinder
_VendorLoader = _CompatLoader


def install_shims():
    """Register all compat packages and modules in ``sys.modules``.

    For packages (``comfy``, ``comfy_api``, ``comfy_execution``), the compat
    package module is imported and placed in ``sys.modules`` under the short
    name.  Python's default ``PathFinder`` then resolves submodule imports
    (e.g. ``comfy.cli_args``) using the package's ``__path__``.

    For standalone modules (``nodes``, ``folder_paths``, ...), a lightweight
    ``sys.meta_path`` finder lazily imports them from ``compat/`` on first access.

    Attributes set on stub modules by earlier bootstrap steps (e.g.
    ``comfy.options``) are preserved by copying them onto the compat module.
    """
    for pkg_name in _PACKAGE_NAMES:
        compat_mod = importlib.import_module(f"{_COMPAT_PKG}.{pkg_name}")

        existing = sys.modules.get(pkg_name)
        if existing is not None and existing is not compat_mod:
            for attr_name, attr_val in vars(existing).items():
                if not attr_name.startswith("_") and not hasattr(compat_mod, attr_name):
                    setattr(compat_mod, attr_name, attr_val)

        sys.modules[pkg_name] = compat_mod

        _wire_attribute_chain(pkg_name, compat_mod)

    sys.meta_path.insert(0, _CompatFinder())
