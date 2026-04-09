"""Utility functions for comfy_runtime.

MIT reimplementation of the public API surface from comfy.utils.
All functions are implemented from scratch using standard PyTorch,
safetensors, PIL, and numpy operations.
"""

import json
import logging
import math
import os
import struct
import time
import threading
import warnings

import numpy as np
import safetensors.torch
import torch
from PIL import Image
from torch.nn.functional import interpolate


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

PROGRESS_BAR_HOOK = None
PROGRESS_BAR_ENABLED = True
PROGRESS_THROTTLE_MIN_INTERVAL = 0.02  # seconds
PROGRESS_THROTTLE_MIN_PERCENT = 0.5


def set_progress_bar_global_hook(function):
    """Set a global hook called on every progress update."""
    global PROGRESS_BAR_HOOK
    PROGRESS_BAR_HOOK = function


def set_progress_bar_enabled(enabled):
    """Enable / disable progress bar updates."""
    global PROGRESS_BAR_ENABLED
    PROGRESS_BAR_ENABLED = enabled


class ProgressBar:
    """Progress tracker compatible with the ComfyUI progress API."""

    def __init__(self, total, node_id=None):
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK
        self.node_id = node_id
        self._last_update_time = 0.0
        self._last_sent_value = -1

    def update_absolute(self, value, total=None, preview=None):
        """Set progress to an absolute *value* out of *total*."""
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            current_time = time.perf_counter()
            is_first = self._last_sent_value < 0
            is_final = value >= self.total
            has_preview = preview is not None

            if has_preview or is_first or is_final:
                self.hook(self.current, self.total, preview, node_id=self.node_id)
                self._last_update_time = current_time
                self._last_sent_value = value
                return

            if self.total > 0:
                percent_changed = (
                    (value - max(0, self._last_sent_value)) / self.total
                ) * 100
            else:
                percent_changed = 100
            time_elapsed = current_time - self._last_update_time

            if (
                time_elapsed >= PROGRESS_THROTTLE_MIN_INTERVAL
                and percent_changed >= PROGRESS_THROTTLE_MIN_PERCENT
            ):
                self.hook(self.current, self.total, preview, node_id=self.node_id)
                self._last_update_time = current_time
                self._last_sent_value = value

    def update(self, value):
        """Increment progress by *value*."""
        self.update_absolute(self.current + value)


# ---------------------------------------------------------------------------
# Tensor file I/O
# ---------------------------------------------------------------------------


def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    """Load a .safetensors or PyTorch checkpoint file.

    Args:
        ckpt: Path to the checkpoint file.
        safe_load: Ignored (kept for API compat).
        device: Target device for loaded tensors.
        return_metadata: If True, return ``(state_dict, metadata)`` tuple.

    Returns:
        State dict, or ``(state_dict, metadata)`` when *return_metadata* is True.
    """
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith((".safetensors", ".sft")):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {k: f.get_tensor(k) for k in f.keys()}
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError(
                        f"{message}\n\nFile path: {ckpt}\n\n"
                        "The safetensors file is corrupt or invalid."
                    )
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError(
                        f"{message}\n\nFile path: {ckpt}\n\n"
                        "The safetensors file is corrupt/incomplete."
                    )
            raise
    else:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd


def save_torch_file(sd, ckpt, metadata=None):
    """Save a state dict as a safetensors file."""
    if metadata is not None:
        safetensors.torch.save_file(sd, ckpt, metadata=metadata)
    else:
        safetensors.torch.save_file(sd, ckpt)


# ---------------------------------------------------------------------------
# State dict manipulation
# ---------------------------------------------------------------------------


def calculate_parameters(sd, prefix=""):
    """Count the total number of scalar parameters in *sd*."""
    params = 0
    for k in sd:
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def weight_dtype(sd, prefix=""):
    """Return the dominant dtype among weights with the given *prefix*."""
    dtypes: dict[torch.dtype, int] = {}
    for k in sd:
        if k.startswith(prefix):
            w = sd[k]
            dtypes[w.dtype] = dtypes.get(w.dtype, 0) + w.numel()
    if not dtypes:
        return None
    return max(dtypes, key=dtypes.get)


def state_dict_key_replace(state_dict, keys_to_replace):
    """Rename keys in *state_dict* according to the mapping."""
    for old_key, new_key in keys_to_replace.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    """Replace key prefixes in *state_dict*."""
    out = {} if filter_keys else state_dict
    for rp, new_prefix in replace_prefix.items():
        keys = [k for k in list(state_dict.keys()) if k.startswith(rp)]
        for k in keys:
            out[new_prefix + k[len(rp) :]] = state_dict.pop(k)
    return out


def convert_sd_to(state_dict, dtype):
    """Convert all tensors in *state_dict* to *dtype* in-place."""
    for k in list(state_dict.keys()):
        state_dict[k] = state_dict[k].to(dtype)
    return state_dict


def transformers_convert(sd, prefix_from, prefix_to, number):
    """Convert OpenAI CLIP state dict keys to HuggingFace format."""
    keys_to_replace = {
        f"{prefix_from}positional_embedding": f"{prefix_to}embeddings.position_embedding.weight",
        f"{prefix_from}token_embedding.weight": f"{prefix_to}embeddings.token_embedding.weight",
        f"{prefix_from}ln_final.weight": f"{prefix_to}final_layer_norm.weight",
        f"{prefix_from}ln_final.bias": f"{prefix_to}final_layer_norm.bias",
    }
    for k_from, k_to in keys_to_replace.items():
        if k_from in sd:
            sd[k_to] = sd.pop(k_from)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x, x_to in resblock_to_replace.items():
            for y in ("weight", "bias"):
                k = f"{prefix_from}transformer.resblocks.{resblock}.{x}.{y}"
                k_to = f"{prefix_to}encoder.layers.{resblock}.{x_to}.{y}"
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ("weight", "bias"):
            k_from = f"{prefix_from}transformer.resblocks.{resblock}.attn.in_proj_{y}"
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for i, proj in enumerate(
                    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
                ):
                    k_to = f"{prefix_to}encoder.layers.{resblock}.{proj}.{y}"
                    sd[k_to] = weights[shape_from * i : shape_from * (i + 1)]
    return sd


def clip_text_transformers_convert(sd, prefix_from, prefix_to):
    """Convert CLIP text model keys (OpenAI -> HF format)."""
    sd = transformers_convert(sd, prefix_from, f"{prefix_to}text_model.", 32)
    tp = f"{prefix_from}text_projection.weight"
    if tp in sd:
        sd[f"{prefix_to}text_projection.weight"] = sd.pop(tp)
    tp = f"{prefix_from}text_projection"
    if tp in sd:
        sd[f"{prefix_to}text_projection.weight"] = (
            sd.pop(tp).transpose(0, 1).contiguous()
        )
    return sd


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------


def repeat_to_batch_size(tensor, batch_size, dim=0):
    """Repeat or narrow *tensor* along *dim* to match *batch_size*."""
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        repeats = [1] * len(tensor.shape)
        repeats[dim] = math.ceil(batch_size / tensor.shape[dim])
        return tensor.repeat(*repeats).narrow(dim, 0, batch_size)
    return tensor


def resize_to_batch_size(tensor, batch_size):
    """Resize *tensor* to *batch_size* by nearest-neighbor resampling."""
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty(
        [batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device
    )
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]
    return output


def resize_list_to_batch_size(lst, batch_size):
    """Resize a list to *batch_size* by nearest-neighbor resampling."""
    n = len(lst)
    if n == batch_size or n == 0:
        return lst
    if batch_size <= 1:
        return lst[:batch_size]

    output = []
    if batch_size < n:
        scale = (n - 1) / (batch_size - 1)
        for i in range(batch_size):
            output.append(lst[min(round(i * scale), n - 1)])
    else:
        scale = n / batch_size
        for i in range(batch_size):
            output.append(lst[min(math.floor((i + 0.5) * scale), n - 1)])
    return output


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------

ATTR_UNSET = {}


def resolve_attr(obj, attr):
    """Resolve a dotted attribute path down to ``(parent, leaf_name)``."""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    return obj, attrs[-1]


def set_attr(obj, attr, value):
    """Set a dotted attribute on *obj*, returning the previous value."""
    obj, name = resolve_attr(obj, attr)
    prev = getattr(obj, name, ATTR_UNSET)
    if value is ATTR_UNSET:
        delattr(obj, name)
    else:
        setattr(obj, name, value)
    return prev


def set_attr_param(obj, attr, value):
    """Set *attr* on *obj* as a frozen ``nn.Parameter``."""
    if (not torch.is_inference_mode_enabled()) and value.is_inference():
        value = value.clone()
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))


def set_attr_buffer(obj, attr, value):
    """Set *attr* on *obj* as a registered buffer."""
    obj, name = resolve_attr(obj, attr)
    prev = getattr(obj, name, ATTR_UNSET)
    persistent = name not in getattr(obj, "_non_persistent_buffers_set", set())
    obj.register_buffer(name, value, persistent=persistent)
    return prev


def copy_to_param(obj, attr, value):
    """In-place copy *value* into the parameter at *attr*."""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    getattr(obj, attrs[-1]).data.copy_(value)


def get_attr(obj, attr: str):
    """Retrieve a nested attribute via dot notation.

    Args:
        obj: The root object.
        attr: Dotted path, e.g. ``"layer1.conv.weight"``.

    Returns:
        The attribute value.
    """
    for name in attr.split("."):
        obj = getattr(obj, name)
    return obj


# ---------------------------------------------------------------------------
# Image / latent upscaling
# ---------------------------------------------------------------------------


def bislerp(samples, width, height):
    """Attempt bi-slerp upscaling; falls back to bilinear on error."""
    try:
        # Slerp between bilinear and nearest for sharper upscaling
        bilinear = interpolate(
            samples, size=(height, width), mode="bilinear", align_corners=False
        )
        nearest = interpolate(samples, size=(height, width), mode="nearest-exact")

        # Blend: use bilinear for low frequencies, nearest for high
        t = 0.5
        dot = (bilinear * nearest).sum(dim=1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(dot)
        so = torch.sin(omega)
        # Avoid division by zero — fall back to bilinear where sin(omega) ~ 0
        mask = so.abs() < 1e-6
        out = torch.where(
            mask,
            bilinear,
            (torch.sin((1.0 - t) * omega) / so) * bilinear
            + (torch.sin(t * omega) / so) * nearest,
        )
        return out
    except Exception:
        return interpolate(
            samples, size=(height, width), mode="bilinear", align_corners=False
        )


def lanczos(samples, width, height):
    """Lanczos upscale via PIL (per-sample, per-channel)."""
    result = []
    for sample in samples:
        channels = []
        for c in range(sample.shape[0]):
            arr = sample[c].cpu().numpy()
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            img = img.resize((width, height), Image.LANCZOS)
            channels.append(torch.from_numpy(np.array(img)))
        result.append(torch.stack(channels))
    return torch.stack(result).to(samples.device)


def common_upscale(samples, width, height, upscale_method, crop):
    """Upscale *samples* tensor to *(width, height)*.

    Args:
        samples: Tensor of shape ``(B, C, H, W)`` or ``(B, C, D, H, W)``.
        width: Target width.
        height: Target height.
        upscale_method: One of ``"nearest-exact"``, ``"bilinear"``,
            ``"bislerp"``, ``"lanczos"``, ``"area"``.
        crop: ``"disabled"`` or ``"center"``.

    Returns:
        Upscaled tensor.
    """
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(
            samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1]
        )
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])

    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    if upscale_method == "bislerp":
        out = bislerp(s, width, height)
    elif upscale_method == "lanczos":
        out = lanczos(s, width, height)
    else:
        out = interpolate(s, size=(height, width), mode=upscale_method)

    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    """Return the number of tile steps for tiled processing."""
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols


# ---------------------------------------------------------------------------
# Safetensors header
# ---------------------------------------------------------------------------


def safetensors_header(safetensors_path, max_size=100 * 1024 * 1024):
    """Read and return the JSON header bytes of a safetensors file."""
    with open(safetensors_path, "rb") as f:
        header = f.read(8)
        length_of_header = struct.unpack("<Q", header)[0]
        if length_of_header > max_size:
            return None
        return f.read(length_of_header)


# ---------------------------------------------------------------------------
# Seed / hashing
# ---------------------------------------------------------------------------


def string_to_seed(data):
    """CRC32-based seed from byte data."""
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def reshape_mask(input_mask, output_shape):
    """Reshape a mask tensor to match *output_shape*."""
    dims = len(output_shape) - 2
    scale_mode = {1: "linear", 2: "bilinear", 3: "trilinear"}.get(dims)
    if scale_mode is None:
        return input_mask
    mask = input_mask.reshape((-1, 1) + input_mask.shape[-dims:])
    if mask.shape[-dims:] != output_shape[-dims:]:
        mask = interpolate(mask.float(), size=output_shape[-dims:], mode=scale_mode)
    if mask.shape[0] < output_shape[0]:
        mask = mask.repeat(
            [math.ceil(output_shape[0] / mask.shape[0])] + [1] * (dims + 1)
        )[: output_shape[0]]
    return mask


def deepcopy_list_dict(obj, memo=None):
    """Deep copy limited to dicts and lists (leaves shared)."""
    if memo is None:
        memo = {}
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(obj, dict):
        res = {
            deepcopy_list_dict(k, memo): deepcopy_list_dict(v, memo)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        res = [deepcopy_list_dict(i, memo) for i in obj]
    else:
        res = obj
    memo[obj_id] = res
    return res
