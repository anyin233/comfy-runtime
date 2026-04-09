# pyright: reportMissingImports=false
"""Compatibility tests — verify comfy_extras node files load under comfy-runtime."""

import os
import sys

import pytest

import comfy_runtime

COMFY_ROOT = os.environ.get("COMFYUI_ROOT", "/home/yanweiye/Project/ComfyUI")
COMFY_EXTRAS_DIR = os.path.join(COMFY_ROOT, "comfy_extras")

# Some comfy_extras files cross-import from each other via `from comfy_extras.X import Y`,
# so the ComfyUI root must be on sys.path for those to resolve.
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

# Files that require optional / heavy dependencies not present in all environments.
OPTIONAL_DEPS_FILES = {
    "nodes_glsl.py",  # PyOpenGL
    "nodes_audio.py",  # torchaudio
    "nodes_audio_encoder.py",  # torchaudio
    "nodes_lt_audio.py",  # torchaudio
    "nodes_canny.py",  # kornia
    "nodes_morphology.py",  # kornia
    "nodes_upscale_model.py",  # spandrel
    "nodes_rtdetr.py",  # torchvision.transforms.ToPILImage
    "nodes_wanmove.py",  # torchvision.transforms.functional
}

# Files where 0 nodes registered is expected (utility / no NODE_CLASS_MAPPINGS).
ZERO_NODES_OK = {
    "nodes_replacements.py",
}


def get_node_files():
    files = []
    for f in sorted(os.listdir(COMFY_EXTRAS_DIR)):
        if f.endswith(".py") and not f.startswith("_"):
            files.append(f)
    return files


NODE_FILES = get_node_files()


@pytest.mark.parametrize("filename", NODE_FILES, ids=lambda f: f.replace(".py", ""))
def test_load_comfy_extras_file(filename):
    """Each comfy_extras file should load and register at least 1 node."""
    filepath = os.path.join(COMFY_EXTRAS_DIR, filename)

    try:
        registered = comfy_runtime.load_nodes_from_path(filepath)
    except Exception as e:
        if filename in OPTIONAL_DEPS_FILES:
            pytest.skip(f"Optional dependency missing: {e}")
        raise

    if (
        len(registered) == 0
        and filename not in OPTIONAL_DEPS_FILES
        and filename not in ZERO_NODES_OK
    ):
        pytest.fail(f"No nodes registered from {filename}")


def test_critical_files_all_load():
    """Critical node files must load successfully (not skip)."""
    critical_files = [
        "nodes_flux.py",
        "nodes_images.py",
        "nodes_cond.py",
        "nodes_custom_sampler.py",
        "nodes_latent.py",
        "nodes_model_advanced.py",
        "nodes_controlnet.py",
        "nodes_post_processing.py",
        "nodes_sd3.py",
    ]
    failed = []
    for f in critical_files:
        filepath = os.path.join(COMFY_EXTRAS_DIR, f)
        try:
            registered = comfy_runtime.load_nodes_from_path(filepath)
            if len(registered) == 0:
                failed.append((f, "0 nodes registered"))
        except Exception as e:
            failed.append((f, str(e)))

    assert len(failed) == 0, f"Critical files failed: {failed}"
