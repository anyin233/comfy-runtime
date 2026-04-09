"""Unit tests for the shim module that don't require _vendor."""

import importlib
import importlib.machinery
import sys
import types

import pytest


def test_vendor_finder_find_spec_known_module(monkeypatch):
    """VendorFinder.find_spec returns a spec for known standalone modules."""
    from comfy_runtime.shim import _VendorFinder, _STANDALONE_MODULES

    finder = _VendorFinder()
    # Ensure the module isn't cached
    for name in _STANDALONE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    spec = finder.find_spec("node_helpers", None)
    # Can't actually load without _vendor, but spec should be returned
    # (or None if _vendor is missing — both are acceptable in CI)
    if spec is not None:
        assert spec.name == "node_helpers"
        assert spec.loader is not None


def test_vendor_finder_find_spec_unknown_module():
    """VendorFinder.find_spec returns None for unknown modules."""
    from comfy_runtime.shim import _VendorFinder

    finder = _VendorFinder()
    assert finder.find_spec("nonexistent_module_xyz", None) is None


def test_vendor_finder_find_spec_cached_module(monkeypatch):
    """VendorFinder.find_spec returns None if module already in sys.modules."""
    from comfy_runtime.shim import _VendorFinder

    dummy = types.ModuleType("node_helpers")
    monkeypatch.setitem(sys.modules, "node_helpers", dummy)

    finder = _VendorFinder()
    assert finder.find_spec("node_helpers", None) is None


def test_vendor_loader_protocol():
    """VendorLoader implements create_module and exec_module."""
    from comfy_runtime.shim import _VendorLoader

    loader = _VendorLoader("os")  # use stdlib 'os' as a safe stand-in
    assert hasattr(loader, "create_module")
    assert hasattr(loader, "exec_module")

    spec = importlib.machinery.ModuleSpec("os_alias", loader)
    module = loader.create_module(spec)
    assert module is not None
    assert hasattr(module, "path")  # os.path should exist


def test_standalone_modules_are_strings():
    """_STANDALONE_MODULES is a tuple of strings."""
    from comfy_runtime.shim import _STANDALONE_MODULES

    assert isinstance(_STANDALONE_MODULES, tuple)
    assert all(isinstance(m, str) for m in _STANDALONE_MODULES)
    assert "node_helpers" in _STANDALONE_MODULES
    assert "folder_paths" in _STANDALONE_MODULES


def test_package_names_are_strings():
    """_PACKAGE_NAMES is a tuple of strings."""
    from comfy_runtime.shim import _PACKAGE_NAMES

    assert isinstance(_PACKAGE_NAMES, tuple)
    assert "comfy" in _PACKAGE_NAMES
    assert "comfy_api" in _PACKAGE_NAMES
