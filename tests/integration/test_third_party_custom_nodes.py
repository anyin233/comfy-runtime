"""Integration tests for third-party custom node compatibility.

Tests that popular ComfyUI custom nodes can be loaded by comfy_runtime.
Requires the custom node repos to be cloned to /tmp/custom_nodes_test/.

Skip if repos are not present (CI won't have them).
"""

import importlib
import importlib.util
import os
import sys

import pytest

import comfy_runtime

CUSTOM_NODES_DIR = "/tmp/custom_nodes_test"


def _setup_runtime():
    """Configure comfy_runtime for custom node testing."""
    comfy_runtime.configure(
        models_dir=os.path.join(CUSTOM_NODES_DIR, "models"),
        output_dir=os.path.join(CUSTOM_NODES_DIR, "output"),
    )


def _load_custom_node_dir(path):
    """Load a custom node directory, returning (node_count, error_or_None)."""
    init_path = os.path.join(path, "__init__.py")
    module_name = os.path.basename(path)

    # Add parent to sys.path so relative imports work
    parent = os.path.dirname(path)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[path],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        ncm = getattr(mod, "NODE_CLASS_MAPPINGS", {})
        return len(ncm), None
    except Exception as e:
        return 0, e


CUSTOM_NODES = [
    ("was-node-suite-comfyui", 100),  # WAS Node Suite: 200+ nodes
    ("ComfyUI_IPAdapter_plus", 20),  # IPAdapter Plus: 30+ nodes
    ("ComfyUI-KJNodes", 100),  # KJNodes: 200+ nodes
    ("ComfyUI-Advanced-ControlNet", 20),  # Advanced ControlNet: 40+ nodes
    ("ComfyUI-AnimateDiff-Evolved", 50),  # AnimateDiff: 140+ nodes
]


@pytest.fixture(scope="module", autouse=True)
def setup():
    if not os.path.isdir(CUSTOM_NODES_DIR):
        pytest.skip("Custom node repos not cloned to /tmp/custom_nodes_test/")
    _setup_runtime()


@pytest.mark.parametrize(
    "dirname,min_nodes", CUSTOM_NODES, ids=[n[0] for n in CUSTOM_NODES]
)
def test_custom_node_loads(dirname, min_nodes):
    """Each custom node pack loads and registers at least min_nodes nodes."""
    path = os.path.join(CUSTOM_NODES_DIR, dirname)
    if not os.path.isdir(path):
        pytest.skip(f"{dirname} not cloned")

    count, error = _load_custom_node_dir(path)
    if error is not None:
        # Check if it's a missing optional dependency (not a compat issue)
        err_str = str(error)
        optional_deps = ["numba", "aiohttp", "opencv", "cv2"]
        for dep in optional_deps:
            if dep in err_str:
                pytest.skip(f"Missing optional dependency: {dep}")

        pytest.fail(f"Failed to load {dirname}: {error}")

    assert count >= min_nodes, (
        f"{dirname} loaded {count} nodes, expected at least {min_nodes}"
    )
