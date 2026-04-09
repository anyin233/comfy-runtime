"""Directory management for comfy_runtime.

MIT reimplementation of ComfyUI's ``folder_paths`` module.
Manages model directories, output/input/temp directories,
and filename resolution.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import time
from collections.abc import Collection
from typing import Literal

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

supported_pt_extensions: set[str] = {
    ".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft",
}

# ---------------------------------------------------------------------------
# Core directory registry
# ---------------------------------------------------------------------------

folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "models")

# Default model directories
folder_names_and_paths["checkpoints"] = ([os.path.join(models_dir, "checkpoints")], supported_pt_extensions)
folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], {".yaml"})
folder_names_and_paths["loras"] = ([os.path.join(models_dir, "loras")], supported_pt_extensions)
folder_names_and_paths["vae"] = ([os.path.join(models_dir, "vae")], supported_pt_extensions)
folder_names_and_paths["text_encoders"] = (
    [os.path.join(models_dir, "text_encoders"), os.path.join(models_dir, "clip")],
    supported_pt_extensions,
)
folder_names_and_paths["diffusion_models"] = (
    [os.path.join(models_dir, "unet"), os.path.join(models_dir, "diffusion_models")],
    supported_pt_extensions,
)
folder_names_and_paths["clip_vision"] = ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions)
folder_names_and_paths["style_models"] = ([os.path.join(models_dir, "style_models")], supported_pt_extensions)
folder_names_and_paths["embeddings"] = ([os.path.join(models_dir, "embeddings")], supported_pt_extensions)
folder_names_and_paths["diffusers"] = ([os.path.join(models_dir, "diffusers")], {"folder"})
folder_names_and_paths["vae_approx"] = ([os.path.join(models_dir, "vae_approx")], supported_pt_extensions)
folder_names_and_paths["controlnet"] = (
    [os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")],
    supported_pt_extensions,
)
folder_names_and_paths["gligen"] = ([os.path.join(models_dir, "gligen")], supported_pt_extensions)
folder_names_and_paths["upscale_models"] = ([os.path.join(models_dir, "upscale_models")], supported_pt_extensions)
folder_names_and_paths["latent_upscale_models"] = (
    [os.path.join(models_dir, "latent_upscale_models")],
    supported_pt_extensions,
)
folder_names_and_paths["custom_nodes"] = ([os.path.join(base_path, "custom_nodes")], set())
folder_names_and_paths["hypernetworks"] = ([os.path.join(models_dir, "hypernetworks")], supported_pt_extensions)
folder_names_and_paths["photomaker"] = ([os.path.join(models_dir, "photomaker")], supported_pt_extensions)
folder_names_and_paths["classifiers"] = ([os.path.join(models_dir, "classifiers")], {""})
folder_names_and_paths["model_patches"] = ([os.path.join(models_dir, "model_patches")], supported_pt_extensions)
folder_names_and_paths["audio_encoders"] = ([os.path.join(models_dir, "audio_encoders")], supported_pt_extensions)

# ---------------------------------------------------------------------------
# Standard directories
# ---------------------------------------------------------------------------

output_directory = os.path.join(base_path, "output")
temp_directory = os.path.join(base_path, "temp")
input_directory = os.path.join(base_path, "input")
user_directory = os.path.join(base_path, "user")

# ---------------------------------------------------------------------------
# File list cache
# ---------------------------------------------------------------------------

filename_list_cache: dict[str, tuple[list[str], dict[str, float], float]] = {}


class CacheHelper:
    """Helper for managing file list cache data."""

    def __init__(self):
        self.cache: dict[str, tuple[list[str], dict[str, float], float]] = {}
        self.active = False

    def get(self, key, default=None):
        if not self.active:
            return default
        return self.cache.get(key, default)

    def set(self, key, value):
        if self.active:
            self.cache[key] = value

    def clear(self):
        self.cache.clear()

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        self.clear()


cache_helper = CacheHelper()

extension_mimetypes_cache = {
    "webp": "image",
    "fbx": "model",
}


# ---------------------------------------------------------------------------
# Legacy name mapping
# ---------------------------------------------------------------------------

def map_legacy(folder_name: str) -> str:
    """Map legacy folder names to current names."""
    legacy = {"unet": "diffusion_models", "clip": "text_encoders"}
    return legacy.get(folder_name, folder_name)


# ---------------------------------------------------------------------------
# Directory getters / setters
# ---------------------------------------------------------------------------

def set_output_directory(output_dir: str) -> None:
    global output_directory
    output_directory = output_dir


def set_temp_directory(temp_dir: str) -> None:
    global temp_directory
    temp_directory = temp_dir


def set_input_directory(input_dir: str) -> None:
    global input_directory
    input_directory = input_dir


def get_output_directory() -> str:
    return output_directory


def get_temp_directory() -> str:
    return temp_directory


def get_input_directory() -> str:
    return input_directory


def get_user_directory() -> str:
    return user_directory


def set_user_directory(user_dir: str) -> None:
    global user_directory
    user_directory = user_dir


def get_directory_by_type(type_name: str) -> str | None:
    if type_name == "output":
        return get_output_directory()
    if type_name == "temp":
        return get_temp_directory()
    if type_name == "input":
        return get_input_directory()
    return None


# ---------------------------------------------------------------------------
# Model folder management
# ---------------------------------------------------------------------------

def add_model_folder_path(folder_name: str, full_folder_path: str, is_default: bool = False) -> None:
    """Add a folder path to the model directory registry."""
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name in folder_names_and_paths:
        paths, _exts = folder_names_and_paths[folder_name]
        if full_folder_path in paths:
            if is_default and paths[0] != full_folder_path:
                paths.remove(full_folder_path)
                paths.insert(0, full_folder_path)
        else:
            if is_default:
                paths.insert(0, full_folder_path)
            else:
                paths.append(full_folder_path)
    else:
        folder_names_and_paths[folder_name] = ([full_folder_path], set())


def get_folder_paths(folder_name: str) -> list[str]:
    """Return list of directory paths for a model category."""
    folder_name = map_legacy(folder_name)
    return folder_names_and_paths[folder_name][0][:]


# ---------------------------------------------------------------------------
# File search
# ---------------------------------------------------------------------------

def recursive_search(directory: str, excluded_dir_names: list[str] | None = None) -> tuple[list[str], dict[str, float]]:
    """Recursively search *directory* for files, returning ``(relative_paths, dir_mtimes)``."""
    if not os.path.isdir(directory):
        return [], {}

    if excluded_dir_names is None:
        excluded_dir_names = []

    result = []
    dirs: dict[str, float] = {}

    try:
        dirs[directory] = os.path.getmtime(directory)
    except FileNotFoundError:
        logging.warning(f"Unable to access {directory}. Skipping.")

    for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):
        subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
        for file_name in filenames:
            try:
                relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)
                result.append(relative_path)
            except Exception:
                continue

        for d in subdirs:
            path = os.path.join(dirpath, d)
            try:
                dirs[path] = os.path.getmtime(path)
            except FileNotFoundError:
                continue

    return result, dirs


def filter_files_extensions(files: Collection[str], extensions: Collection[str]) -> list[str]:
    """Filter *files* by their extensions."""
    return sorted(f for f in files if os.path.splitext(f)[-1].lower() in extensions or len(extensions) == 0)


def get_full_path(folder_name: str, filename: str) -> str | None:
    """Return full path of *filename* within *folder_name*, or None if not found."""
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path
        elif os.path.islink(full_path):
            logging.warning(f"Path {full_path} exists but doesn't link anywhere, skipping.")
    return None


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    """Return full path of *filename* or raise FileNotFoundError."""
    full_path = get_full_path(folder_name, filename)
    if full_path is None:
        raise FileNotFoundError(
            f"Model in folder '{folder_name}' with filename '{filename}' not found."
        )
    return full_path


def get_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    """Build a fresh file listing for *folder_name*."""
    folder_name = map_legacy(folder_name)
    global folder_names_and_paths
    output_list: set[str] = set()
    folders = folder_names_and_paths[folder_name]
    output_folders: dict[str, float] = {}
    for x in folders[0]:
        files, folders_all = recursive_search(x, excluded_dir_names=[".git"])
        output_list.update(filter_files_extensions(files, folders[1]))
        output_folders = {**output_folders, **folders_all}
    return sorted(output_list), output_folders, time.perf_counter()


def cached_filename_list_(folder_name: str):
    """Return cached file listing if still fresh, else None."""
    strong_cache = cache_helper.get(folder_name)
    if strong_cache is not None:
        return strong_cache

    global filename_list_cache, folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]

    for x, time_modified in out[1].items():
        try:
            if os.path.getmtime(x) != time_modified:
                return None
        except FileNotFoundError:
            return None

    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x) and x not in out[1]:
            return None

    return out


def get_filename_list(folder_name: str) -> list[str]:
    """Return sorted list of filenames for a model category (cached)."""
    folder_name = map_legacy(folder_name)
    out = cached_filename_list_(folder_name)
    if out is None:
        out = get_filename_list_(folder_name)
        global filename_list_cache
        filename_list_cache[folder_name] = out
    cache_helper.set(folder_name, out)
    return list(out[0])


# ---------------------------------------------------------------------------
# Save image path
# ---------------------------------------------------------------------------

def get_save_image_path(
    filename_prefix: str, output_dir: str, image_width=0, image_height=0
) -> tuple[str, str, int, str, str]:
    """Compute the next available save path for an image.

    Returns:
        (full_output_folder, filename, counter, subfolder, filename_prefix)
    """

    def map_filename(filename: str) -> tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except Exception:
            digits = 0
        return digits, prefix

    def compute_vars(text: str, image_width: int, image_height: int) -> str:
        text = text.replace("%width%", str(image_width))
        text = text.replace("%height%", str(image_height))
        now = time.localtime()
        text = text.replace("%year%", str(now.tm_year))
        text = text.replace("%month%", str(now.tm_mon).zfill(2))
        text = text.replace("%day%", str(now.tm_mday).zfill(2))
        text = text.replace("%hour%", str(now.tm_hour).zfill(2))
        text = text.replace("%minute%", str(now.tm_min).zfill(2))
        text = text.replace("%second%", str(now.tm_sec).zfill(2))
        return text

    if "%" in filename_prefix:
        filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))
    full_output_folder = os.path.join(output_dir, subfolder)

    if os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) != output_dir:
        raise Exception(
            f"Saving image outside the output folder is not allowed.\n"
            f"  full_output_folder: {os.path.abspath(full_output_folder)}\n"
            f"  output_dir: {output_dir}"
        )

    try:
        counter = (
            max(
                d
                for d, p in map(map_filename, os.listdir(full_output_folder))
                if os.path.normcase(p[:-1]) == os.path.normcase(filename) and p[-1] == "_"
            )
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1

    return full_output_folder, filename, counter, subfolder, filename_prefix


# ---------------------------------------------------------------------------
# Content type filtering
# ---------------------------------------------------------------------------

def filter_files_content_types(
    files: list[str],
    content_types: list[Literal["image", "video", "audio", "model"]],
) -> list[str]:
    """Filter files by MIME content type (image, video, audio, model)."""
    global extension_mimetypes_cache
    result = []
    for file in files:
        extension = file.split(".")[-1]
        if extension not in extension_mimetypes_cache:
            mime_type, _ = mimetypes.guess_type(file, strict=False)
            if not mime_type:
                continue
            content_type = mime_type.split("/")[0]
            extension_mimetypes_cache[extension] = content_type
        else:
            content_type = extension_mimetypes_cache[extension]
        if content_type in content_types:
            result.append(file)
    return result


# ---------------------------------------------------------------------------
# Annotated filepath helpers
# ---------------------------------------------------------------------------

def annotated_filepath(name: str) -> tuple[str, str | None]:
    """Parse ``filename [annotation]`` format."""
    if name.endswith("[output]"):
        return name[:-9], get_output_directory()
    if name.endswith("[input]"):
        return name[:-8], get_input_directory()
    if name.endswith("[temp]"):
        return name[:-7], get_temp_directory()
    return name, None


def get_annotated_filepath(name: str, default_dir: str | None = None) -> str:
    name, base_dir = annotated_filepath(name)
    if base_dir is None:
        base_dir = default_dir if default_dir is not None else get_input_directory()
    return os.path.join(base_dir, name)


def exists_annotated_filepath(name) -> bool:
    name, base_dir = annotated_filepath(name)
    if base_dir is None:
        base_dir = get_input_directory()
    return os.path.exists(os.path.join(base_dir, name))


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

SYSTEM_USER_PREFIX = "__"


def get_system_user_directory(name: str = "system") -> str:
    if not name or not isinstance(name, str):
        raise ValueError("System user name cannot be empty")
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid system user name: '{name}'")
    if name.startswith("_"):
        raise ValueError("System user name should not start with underscore")
    return os.path.join(get_user_directory(), f"{SYSTEM_USER_PREFIX}{name}")


def get_public_user_directory(user_id: str) -> str | None:
    if not user_id or not isinstance(user_id, str):
        return None
    if user_id.startswith(SYSTEM_USER_PREFIX):
        return None
    return os.path.join(get_user_directory(), user_id)


def get_input_subfolders() -> list[str]:
    """Return relative paths of all subdirectories in the input directory."""
    input_dir = get_input_directory()
    folders = []
    try:
        if not os.path.exists(input_dir):
            return []
        for root, dirs, _ in os.walk(input_dir):
            rel_path = os.path.relpath(root, input_dir)
            if rel_path != ".":
                folders.append(rel_path.replace(os.sep, "/"))
        return sorted(folders)
    except FileNotFoundError:
        return []
