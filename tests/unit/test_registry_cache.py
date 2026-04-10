"""Tests for load_nodes_from_path mtime-based caching (Step 5)."""

import importlib.machinery
import os
import tempfile
import textwrap
from unittest.mock import patch

import comfy_runtime  # noqa: F401 — triggers bootstrap
from comfy_runtime import registry
from comfy_runtime.compat import nodes as compat_nodes


NODE_SRC = textwrap.dedent(
    """
    class _RegistryCacheProbe:
        FUNCTION = "run"
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
        RETURN_TYPES = ()
        def run(self):
            return ()

    NODE_CLASS_MAPPINGS = {"_RegistryCacheProbe": _RegistryCacheProbe}
    """
)


def _write_probe(path: str, name: str = "_RegistryCacheProbe"):
    src = NODE_SRC.replace("_RegistryCacheProbe", name)
    with open(path, "w") as f:
        f.write(src)


def test_load_nodes_from_file_caches_by_mtime():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe_node_a.py")
        _write_probe(path, "_RegistryCacheProbeA")
        registry._LOAD_CACHE.clear()

        call_count = {"n": 0}
        real_exec_module = importlib.machinery.SourceFileLoader.exec_module

        def spy_exec(self, module):
            call_count["n"] += 1
            return real_exec_module(self, module)

        with patch.object(
            importlib.machinery.SourceFileLoader, "exec_module", spy_exec
        ):
            first = registry.load_nodes_from_path(path)
            second = registry.load_nodes_from_path(path)

        assert first == second == ["_RegistryCacheProbeA"]
        assert call_count["n"] == 1, (
            f"module re-executed {call_count['n']} times; cache miss"
        )
        # Clean up registered class so other tests aren't polluted.
        compat_nodes.NODE_CLASS_MAPPINGS.pop("_RegistryCacheProbeA", None)


def test_load_nodes_cache_invalidated_when_file_modified():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe_node_b.py")
        _write_probe(path, "_RegistryCacheProbeB")
        registry._LOAD_CACHE.clear()

        registry.load_nodes_from_path(path)

        # Bump mtime forward to simulate an edit.
        new_time = os.path.getmtime(path) + 10
        os.utime(path, (new_time, new_time))

        call_count = {"n": 0}
        real_exec_module = importlib.machinery.SourceFileLoader.exec_module

        def spy_exec(self, module):
            call_count["n"] += 1
            return real_exec_module(self, module)

        with patch.object(
            importlib.machinery.SourceFileLoader, "exec_module", spy_exec
        ):
            registry.load_nodes_from_path(path)

        assert call_count["n"] == 1  # mtime changed → cache miss → exec again
        compat_nodes.NODE_CLASS_MAPPINGS.pop("_RegistryCacheProbeB", None)


def test_load_nodes_cache_miss_when_class_unregistered():
    """If the cached class was manually unregistered, reload must happen."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe_node_c.py")
        _write_probe(path, "_RegistryCacheProbeC")
        registry._LOAD_CACHE.clear()

        registry.load_nodes_from_path(path)
        assert "_RegistryCacheProbeC" in compat_nodes.NODE_CLASS_MAPPINGS

        # Manually unregister — simulates downstream code removing the class.
        compat_nodes.NODE_CLASS_MAPPINGS.pop("_RegistryCacheProbeC", None)

        call_count = {"n": 0}
        real_exec_module = importlib.machinery.SourceFileLoader.exec_module

        def spy_exec(self, module):
            call_count["n"] += 1
            return real_exec_module(self, module)

        with patch.object(
            importlib.machinery.SourceFileLoader, "exec_module", spy_exec
        ):
            registry.load_nodes_from_path(path)

        assert call_count["n"] == 1
        compat_nodes.NODE_CLASS_MAPPINGS.pop("_RegistryCacheProbeC", None)
