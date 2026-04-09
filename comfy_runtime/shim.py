"""Shim layer that registers vendored ComfyUI modules in sys.modules.

After ``install_shims()`` runs, user code can write::

    from comfy.model_management import get_torch_device
    import nodes

and it transparently resolves to the copies under ``_vendor/``.
"""

import importlib
import importlib.machinery
import importlib.util
import sys

_VENDOR_PKG = "comfy_runtime._vendor"

_PACKAGE_NAMES = ("comfy", "comfy_api", "comfy_execution")

_STANDALONE_MODULES = (
    "nodes",
    "folder_paths",
    "node_helpers",
    "execution",
    "comfyui_version",
    "protocol",
)


class _VendorLoader:
    """Loader that imports the vendored module and aliases it directly.

    Instead of creating a new module and copying attributes, we directly
    reuse the vendored module object.  This ensures that functions like
    ``folder_paths.get_output_directory()`` — which rely on module-level
    globals — see changes made via ``folder_paths.output_directory = ...``.
    """

    def __init__(self, vendored_name):
        self._vendored_name = vendored_name

    def create_module(self, spec):
        # Import the vendored module and return it directly so that
        # sys.modules[short_name] IS the vendored module object.
        vendored = importlib.import_module(self._vendored_name)
        return vendored

    def exec_module(self, module):
        # Module was already fully initialised in create_module; register alias.
        sys.modules[module.__name__] = module


class _VendorFinder:
    """Meta-path finder that lazily redirects standalone module imports to _vendor.

    Uses the modern ``find_spec`` protocol (PEP 451) instead of the legacy
    ``find_module``/``load_module`` pair, which no longer works in Python 3.12+.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname not in _STANDALONE_MODULES or fullname in sys.modules:
            return None

        vendored_name = f"{_VENDOR_PKG}.{fullname}"
        loader = _VendorLoader(vendored_name)
        return importlib.machinery.ModuleSpec(fullname, loader)


def _wire_attribute_chain(pkg_name, vendored):
    """Ensure all pre-registered sub-modules are reachable via attribute access.

    Modules that were manually placed in ``sys.modules`` by the bootstrap
    (e.g. ``comfy.ldm.modules.diffusionmodules.mmdit``) need their full
    attribute chain set up so that dotted access like
    ``comfy.ldm.modules.diffusionmodules.mmdit.MMDiT`` works at runtime.
    """
    prefix = f"{pkg_name}."
    for key, mod in sorted(sys.modules.items()):
        if not key.startswith(prefix) or mod is None:
            continue

        parts = key[len(prefix) :].split(".")

        # Walk the chain: set each intermediate as an attribute on its parent.
        # For the full key, set the leaf module on its direct parent.
        for depth in range(len(parts)):
            if depth == 0:
                parent = vendored
            else:
                parent_key = f"{pkg_name}.{'.'.join(parts[:depth])}"
                parent = sys.modules.get(parent_key)
                if parent is None:
                    continue

            child_name = parts[depth]

            # The leaf level gets the actual module from sys.modules.
            # Intermediate levels should be namespace packages — import them
            # if not already present.
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
                                f"{_VENDOR_PKG}.{child_key[len(pkg_name) + 1 :]}"
                            )
                            sys.modules[child_key] = child_mod
                        except ImportError:
                            continue
                    setattr(parent, child_name, child_mod)


def install_shims():
    """Register all vendored ComfyUI packages and modules in ``sys.modules``.

    For packages (``comfy``, ``comfy_api``, ``comfy_execution``), the vendored
    package module is imported and placed in ``sys.modules`` under the short
    name.  Python's default ``PathFinder`` then resolves submodule imports
    (e.g. ``comfy.cli_args``) using the package's ``__path__``.

    For standalone modules (``nodes``, ``folder_paths``, …), a lightweight
    ``sys.meta_path`` finder is installed that lazily imports them from
    ``_vendor/`` on first access.

    Attributes set on stub modules by earlier bootstrap steps (e.g.
    ``comfy.options``) are preserved by copying them onto the vendored module.
    """
    for pkg_name in _PACKAGE_NAMES:
        vendored = importlib.import_module(f"{_VENDOR_PKG}.{pkg_name}")

        existing = sys.modules.get(pkg_name)
        if existing is not None and existing is not vendored:
            for attr_name, attr_val in vars(existing).items():
                if not attr_name.startswith("_") and not hasattr(vendored, attr_name):
                    setattr(vendored, attr_name, attr_val)

        sys.modules[pkg_name] = vendored

        _wire_attribute_chain(pkg_name, vendored)

    sys.meta_path.insert(0, _VendorFinder())
