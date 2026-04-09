# pyright: reportMissingImports=false
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import comfy_runtime


COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "/home/yanweiye/Project/ComfyUI")
if COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, COMFYUI_ROOT)


NODE_FILES = [
    f"{COMFYUI_ROOT}/comfy_extras/nodes_flux.py",
    f"{COMFYUI_ROOT}/comfy_extras/nodes_images.py",
    f"{COMFYUI_ROOT}/comfy_extras/nodes_cond.py",
    f"{COMFYUI_ROOT}/comfy_extras/nodes_custom_sampler.py",
    f"{COMFYUI_ROOT}/comfy_extras/nodes_edit_model.py",
]

for node_file in NODE_FILES:
    comfy_runtime.load_nodes_from_path(node_file)

comfy_runtime.load_nodes_from_path(f"{COMFYUI_ROOT}/comfy_extras")


FLUX2_WORKFLOW_NODES = [
    "KSamplerSelect",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEEncode",
    "RandomNoise",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "CFGGuider",
    "EmptyFlux2LatentImage",
    "Flux2Scheduler",
    "ImageScaleToTotalPixels",
    "GetImageSize",
    "ReferenceLatent",
    "ConditioningZeroOut",
]

MODEL_NODES = [
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "CFGGuider",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEEncode",
]


def _unwrap_result(result: Any) -> Any:
    if isinstance(result, tuple):
        return result[0]
    return result


@pytest.mark.parametrize("class_type", FLUX2_WORKFLOW_NODES)
def test_node_exists(class_type):
    info = comfy_runtime.get_node_info(class_type)

    assert info is not None
    assert info["class_type"] == class_type
    assert (
        info["function"] is not None
        or comfy_runtime.get_node_class(class_type) is not None
    )


@pytest.mark.parametrize("class_type", FLUX2_WORKFLOW_NODES)
def test_node_instantiation(class_type):
    instance = comfy_runtime.create_node_instance(class_type)
    assert instance is not None


@pytest.mark.parametrize("class_type", FLUX2_WORKFLOW_NODES)
def test_node_has_valid_input_types(class_type):
    info = comfy_runtime.get_node_info(class_type)

    assert isinstance(info, dict)
    assert "input_types" in info
    assert isinstance(info["input_types"], dict)

    if class_type in MODEL_NODES:
        cls = comfy_runtime.get_node_class(class_type)
        input_types = cls.INPUT_TYPES()
        assert isinstance(input_types, dict)
        assert "required" in input_types


def test_empty_flux2_latent_image():
    latent = _unwrap_result(
        comfy_runtime.execute_node(
            "EmptyFlux2LatentImage",
            width=1024,
            height=1024,
            batch_size=1,
        )
    )

    assert "samples" in latent
    assert latent["samples"].shape == (1, 128, 64, 64)


def test_flux2_scheduler():
    sigmas = cast(
        torch.Tensor,
        _unwrap_result(
            comfy_runtime.execute_node(
                "Flux2Scheduler", steps=4, width=1024, height=1024
            )
        ),
    )

    assert isinstance(sigmas, torch.Tensor)
    assert len(sigmas) == 5
    assert sigmas.dtype == torch.float32


def test_ksampler_select():
    sampler = _unwrap_result(
        comfy_runtime.execute_node("KSamplerSelect", sampler_name="euler")
    )

    assert sampler is not None


def test_random_noise():
    noise = _unwrap_result(comfy_runtime.execute_node("RandomNoise", noise_seed=42))

    assert noise is not None
    assert hasattr(noise, "generate_noise")


def test_get_image_size():
    image = torch.rand(1, 512, 768, 3)
    comfy_runtime.get_node_class("GetImageSize").hidden = SimpleNamespace(
        unique_id=None
    )
    result = comfy_runtime.execute_node("GetImageSize", image=image)

    assert isinstance(result, tuple)
    width, height, batch_size = result
    assert width == 768
    assert height == 512
    assert batch_size == 1


def test_image_scale_to_total_pixels():
    image = torch.rand(1, 256, 256, 3)
    scaled = cast(
        torch.Tensor,
        _unwrap_result(
            comfy_runtime.execute_node(
                "ImageScaleToTotalPixels",
                image=image,
                upscale_method="nearest-exact",
                megapixels=1.0,
                resolution_steps=1,
            )
        ),
    )

    assert scaled.shape[-1] == 3
    pixels = scaled.shape[1] * scaled.shape[2]
    assert abs(pixels - 1048576) < 100000


def test_conditioning_zero_out():
    conditioning = [[torch.randn(1, 77, 768), {}]]
    output = _unwrap_result(
        comfy_runtime.execute_node("ConditioningZeroOut", conditioning=conditioning)
    )

    assert output is not None
    assert torch.all(output[0][0] == 0)


def test_reference_latent():
    conditioning = [[torch.randn(1, 77, 768), {}]]
    latent = {"samples": torch.randn(1, 128, 64, 64)}
    output = _unwrap_result(
        comfy_runtime.execute_node(
            "ReferenceLatent",
            conditioning=conditioning,
            latent=latent,
        )
    )

    assert output is not None
    assert "reference_latents" in output[0][1]
    assert len(output[0][1]["reference_latents"]) == 1
    assert output[0][1]["reference_latents"][0].shape == (1, 128, 64, 64)


@pytest.mark.parametrize("class_type", MODEL_NODES)
def test_model_node_instantiation(class_type):
    cls = comfy_runtime.get_node_class(class_type)
    instance = cls()

    assert instance is not None


@pytest.mark.parametrize("class_type", MODEL_NODES)
def test_model_node_has_input_types(class_type):
    cls = comfy_runtime.get_node_class(class_type)
    input_types = cls.INPUT_TYPES()

    assert isinstance(input_types, dict)
    assert "required" in input_types
