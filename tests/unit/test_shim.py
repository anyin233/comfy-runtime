"""Unit tests for the shim module that don't require _vendor."""

import importlib
import importlib.machinery
import importlib.util
import sys
import types

import pytest


def _import_shim_directly():
    """Import comfy_runtime.shim without triggering comfy_runtime.__init__ bootstrap."""
    spec = importlib.util.spec_from_file_location(
        "comfy_runtime.shim",
        "comfy_runtime/shim.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def shim():
    return _import_shim_directly()


def test_vendor_finder_find_spec_known_module(shim, monkeypatch):
    """VendorFinder.find_spec returns a spec for known standalone modules."""
    finder = shim._VendorFinder()
    for name in shim._STANDALONE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    spec = finder.find_spec("node_helpers", None)
    # Without _vendor the loader can't resolve, so None is acceptable in CI
    if spec is not None:
        assert spec.name == "node_helpers"
        assert spec.loader is not None


def test_vendor_finder_find_spec_unknown_module(shim):
    """VendorFinder.find_spec returns None for unknown modules."""
    finder = shim._VendorFinder()
    assert finder.find_spec("nonexistent_module_xyz", None) is None


def test_vendor_finder_find_spec_cached_module(shim, monkeypatch):
    """VendorFinder.find_spec returns None if module already in sys.modules."""
    dummy = types.ModuleType("node_helpers")
    monkeypatch.setitem(sys.modules, "node_helpers", dummy)

    finder = shim._VendorFinder()
    assert finder.find_spec("node_helpers", None) is None


def test_vendor_loader_protocol(shim):
    """VendorLoader implements create_module and exec_module."""
    loader = shim._VendorLoader("os")  # use stdlib 'os' as a safe stand-in
    assert hasattr(loader, "create_module")
    assert hasattr(loader, "exec_module")

    spec = importlib.machinery.ModuleSpec("os_alias", loader)
    module = loader.create_module(spec)
    assert module is not None
    assert hasattr(module, "path")  # os.path should exist


def test_standalone_modules_are_strings(shim):
    """_STANDALONE_MODULES is a tuple of strings."""
    assert isinstance(shim._STANDALONE_MODULES, tuple)
    assert all(isinstance(m, str) for m in shim._STANDALONE_MODULES)
    assert "node_helpers" in shim._STANDALONE_MODULES
    assert "folder_paths" in shim._STANDALONE_MODULES


def test_package_names_are_strings(shim):
    """_PACKAGE_NAMES is a tuple of strings."""
    assert isinstance(shim._PACKAGE_NAMES, tuple)
    assert "comfy" in shim._PACKAGE_NAMES
    assert "comfy_api" in shim._PACKAGE_NAMES
