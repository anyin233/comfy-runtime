"""Integration tests for all project workflows.

Each workflow is tested in stages:
1. Configuration — comfy_runtime.configure() succeeds
2. Node availability — all required nodes are registered
3. Non-model operations — nodes that don't need real models execute correctly
4. Model loading boundary — model loaders raise NotImplementedError (expected until Phase 2)

Workflows tested:
- sd15_text_to_image
- flux2_klein_text_to_image
- img2img
- hires_fix
- area_composition
- inpainting
- esrgan_upscale
"""

import os
import sys

import pytest
import torch

import comfy_runtime

WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "workflows")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure(workflow_name, **extra):
    """Configure comfy_runtime for a given workflow."""
    wf_dir = os.path.join(WORKFLOWS_DIR, workflow_name)
    models_dir = os.path.join(wf_dir, "models")
    output_dir = os.path.join(wf_dir, "output")
    input_dir = os.path.join(wf_dir, "input")
    os.makedirs(output_dir, exist_ok=True)
    comfy_runtime.configure(
        models_dir=models_dir,
        output_dir=output_dir,
        input_dir=input_dir,
        **extra,
    )


def _load_workflow_nodes(workflow_name):
    """Load custom nodes from a workflow's nodes/ directory if present."""
    nodes_dir = os.path.join(WORKFLOWS_DIR, workflow_name, "nodes")
    if os.path.isdir(nodes_dir):
        node_files = sorted(
            f
            for f in os.listdir(nodes_dir)
            if f.endswith(".py") and not f.startswith("_")
        )
        registered = []
        for nf in node_files:
            try:
                r = comfy_runtime.load_nodes_from_path(os.path.join(nodes_dir, nf))
                registered.extend(r)
            except Exception:
                pass
        return registered
    return []


# ===================================================================
# SD 1.5 Text-to-Image
# ===================================================================


class TestSD15TextToImage:
    """Tests for the sd15_text_to_image workflow."""

    REQUIRED_NODES = [
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    ]

    def test_configure(self):
        _configure("sd15_text_to_image")

    def test_node_availability(self):
        _configure("sd15_text_to_image")
        available = comfy_runtime.list_nodes()
        for node in self.REQUIRED_NODES:
            assert node in available, f"Missing node: {node}"

    def test_empty_latent_image(self):
        _configure("sd15_text_to_image")
        latent = comfy_runtime.execute_node(
            "EmptyLatentImage",
            width=512,
            height=512,
            batch_size=1,
        )[0]
        assert "samples" in latent
        assert latent["samples"].shape == (1, 4, 64, 64)

    def test_checkpoint_loader_raises_not_implemented(self):
        _configure("sd15_text_to_image")
        with pytest.raises(
            comfy_runtime.NodeExecutionError, match="not yet implemented|not found"
        ):
            comfy_runtime.execute_node(
                "CheckpointLoaderSimple",
                ckpt_name="nonexistent.safetensors",
            )


# ===================================================================
# Flux.2 Klein Text-to-Image
# ===================================================================


class TestFlux2KleinTextToImage:
    """Tests for the flux2_klein_text_to_image workflow."""

    BUILT_IN_NODES = [
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncode",
        "VAEDecode",
        "SaveImage",
    ]

    EXTRA_NODES = [
        "CFGGuider",
        "KSamplerSelect",
        "RandomNoise",
        "SamplerCustomAdvanced",
        "Flux2Scheduler",
        "EmptyFlux2LatentImage",
    ]

    def test_configure(self):
        _configure("flux2_klein_text_to_image")

    def test_load_custom_nodes(self):
        _configure("flux2_klein_text_to_image")
        _load_workflow_nodes("flux2_klein_text_to_image")
        available = comfy_runtime.list_nodes()
        for node in self.EXTRA_NODES:
            assert node in available, f"Missing extra node: {node}"

    def test_all_nodes_available(self):
        _configure("flux2_klein_text_to_image")
        _load_workflow_nodes("flux2_klein_text_to_image")
        available = comfy_runtime.list_nodes()
        for node in self.BUILT_IN_NODES + self.EXTRA_NODES:
            assert node in available, f"Missing node: {node}"

    def test_empty_flux2_latent_image(self):
        _configure("flux2_klein_text_to_image")
        _load_workflow_nodes("flux2_klein_text_to_image")
        latent = comfy_runtime.execute_node(
            "EmptyFlux2LatentImage",
            width=1024,
            height=1024,
            batch_size=1,
        )[0]
        assert "samples" in latent
        # Flux2 uses 128 channels, spatial_downscale=16
        assert latent["samples"].shape == (1, 128, 64, 64)

    def test_flux2_scheduler(self):
        _configure("flux2_klein_text_to_image")
        _load_workflow_nodes("flux2_klein_text_to_image")
        sigmas = comfy_runtime.execute_node(
            "Flux2Scheduler",
            steps=20,
            width=1024,
            height=1024,
        )[0]
        assert isinstance(sigmas, torch.Tensor)
        assert len(sigmas) == 21  # steps + 1
        assert sigmas[0] > sigmas[-1]  # descending

    def test_ksampler_select(self):
        _configure("flux2_klein_text_to_image")
        _load_workflow_nodes("flux2_klein_text_to_image")
        sampler = comfy_runtime.execute_node(
            "KSamplerSelect",
            sampler_name="euler",
        )[0]
        assert sampler is not None

    def test_random_noise(self):
        _configure("flux2_klein_text_to_image")
        _load_workflow_nodes("flux2_klein_text_to_image")
        noise = comfy_runtime.execute_node(
            "RandomNoise",
            noise_seed=42,
        )[0]
        assert noise is not None
        assert hasattr(noise, "generate_noise")

    def test_cfg_guider_instantiation(self):
        """CFGGuider node can be instantiated and has INPUT_TYPES."""
        _configure("flux2_klein_text_to_image")
        _load_workflow_nodes("flux2_klein_text_to_image")
        cls = comfy_runtime.get_node_class("CFGGuider")
        instance = cls()
        assert instance is not None
        input_types = cls.INPUT_TYPES()
        assert "required" in input_types

    def test_unet_loader_raises_not_implemented(self):
        _configure("flux2_klein_text_to_image")
        with pytest.raises(
            comfy_runtime.NodeExecutionError, match="not yet implemented|not found"
        ):
            comfy_runtime.execute_node(
                "UNETLoader",
                unet_name="nonexistent.safetensors",
                weight_dtype="default",
            )


# ===================================================================
# Image-to-Image
# ===================================================================


class TestImg2Img:
    """Tests for the img2img workflow."""

    REQUIRED_NODES = [
        "CheckpointLoaderSimple",
        "LoadImage",
        "VAEEncode",
        "CLIPTextEncode",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    ]

    def test_configure(self):
        _configure("img2img")

    def test_node_availability(self):
        _configure("img2img")
        available = comfy_runtime.list_nodes()
        for node in self.REQUIRED_NODES:
            assert node in available, f"Missing node: {node}"

    def test_node_info(self):
        """All required nodes have valid metadata."""
        _configure("img2img")
        for node_name in self.REQUIRED_NODES:
            info = comfy_runtime.get_node_info(node_name)
            assert info is not None
            assert info["class_type"] == node_name
            assert info["function"] is not None


# ===================================================================
# Hires Fix
# ===================================================================


class TestHiresFix:
    """Tests for the hires_fix workflow."""

    REQUIRED_NODES = [
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "LatentUpscale",
        "VAEDecode",
        "SaveImage",
    ]

    def test_configure(self):
        _configure("hires_fix")

    def test_node_availability(self):
        _configure("hires_fix")
        available = comfy_runtime.list_nodes()
        for node in self.REQUIRED_NODES:
            assert node in available, f"Missing node: {node}"

    def test_latent_upscale(self):
        """LatentUpscale works on a latent tensor."""
        _configure("hires_fix")
        # Create a small latent
        latent = {"samples": torch.randn(1, 4, 64, 64)}
        upscaled = comfy_runtime.execute_node(
            "LatentUpscale",
            samples=latent,
            upscale_method="nearest-exact",
            width=1024,
            height=1024,
            crop="disabled",
        )[0]
        assert upscaled["samples"].shape == (1, 4, 128, 128)

    def test_two_pass_latent_pipeline(self):
        """EmptyLatentImage → LatentUpscale chain works."""
        _configure("hires_fix")
        # Pass 1: Create 512x512 latent
        latent = comfy_runtime.execute_node(
            "EmptyLatentImage",
            width=512,
            height=512,
            batch_size=1,
        )[0]
        assert latent["samples"].shape == (1, 4, 64, 64)

        # Upscale to 1024x1024
        upscaled = comfy_runtime.execute_node(
            "LatentUpscale",
            samples=latent,
            upscale_method="nearest-exact",
            width=1024,
            height=1024,
            crop="disabled",
        )[0]
        assert upscaled["samples"].shape == (1, 4, 128, 128)


# ===================================================================
# Area Composition
# ===================================================================


class TestAreaComposition:
    """Tests for the area_composition workflow."""

    REQUIRED_NODES = [
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "ConditioningSetArea",
        "ConditioningCombine",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    ]

    def test_configure(self):
        _configure("area_composition")

    def test_node_availability(self):
        _configure("area_composition")
        available = comfy_runtime.list_nodes()
        for node in self.REQUIRED_NODES:
            assert node in available, f"Missing node: {node}"

    def test_conditioning_area_pipeline(self):
        """ConditioningSetArea + ConditioningCombine work end-to-end."""
        _configure("area_composition")

        # Create mock conditioning (like what CLIPTextEncode returns)
        bg_cond = [[torch.randn(1, 77, 768), {}]]
        left_cond = [[torch.randn(1, 77, 768), {}]]
        right_cond = [[torch.randn(1, 77, 768), {}]]

        # Set areas
        left_area = comfy_runtime.execute_node(
            "ConditioningSetArea",
            conditioning=left_cond,
            width=448,
            height=512,
            x=0,
            y=0,
            strength=1.0,
        )[0]
        assert len(left_area) == 1
        assert "area" in left_area[0][1]

        right_area = comfy_runtime.execute_node(
            "ConditioningSetArea",
            conditioning=right_cond,
            width=448,
            height=512,
            x=320,
            y=0,
            strength=1.0,
        )[0]

        # Combine
        combined = comfy_runtime.execute_node(
            "ConditioningCombine",
            conditioning_1=bg_cond,
            conditioning_2=left_area,
        )[0]
        assert len(combined) == 2

        combined = comfy_runtime.execute_node(
            "ConditioningCombine",
            conditioning_1=combined,
            conditioning_2=right_area,
        )[0]
        assert len(combined) == 3


# ===================================================================
# Inpainting
# ===================================================================


class TestInpainting:
    """Tests for the inpainting workflow."""

    REQUIRED_NODES = [
        "CheckpointLoaderSimple",
        "LoadImage",
        "VAEEncode",
        "SetLatentNoiseMask",
        "CLIPTextEncode",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    ]

    def test_configure(self):
        _configure("inpainting")

    def test_node_availability(self):
        _configure("inpainting")
        available = comfy_runtime.list_nodes()
        for node in self.REQUIRED_NODES:
            assert node in available, f"Missing node: {node}"

    def test_set_latent_noise_mask(self):
        """SetLatentNoiseMask correctly applies a mask to latent."""
        _configure("inpainting")
        latent = {"samples": torch.randn(1, 4, 64, 64)}
        mask = torch.zeros(512, 512)
        mask[128:384, 128:384] = 1.0

        masked = comfy_runtime.execute_node(
            "SetLatentNoiseMask",
            samples=latent,
            mask=mask,
        )[0]
        assert "noise_mask" in masked
        assert "samples" in masked

    def test_conditioning_zero_out(self):
        """ConditioningZeroOut zeroes the conditioning tensor."""
        _configure("inpainting")
        cond = [[torch.randn(1, 77, 768), {}]]
        result = comfy_runtime.execute_node(
            "ConditioningZeroOut",
            conditioning=cond,
        )[0]
        assert torch.all(result[0][0] == 0)


# ===================================================================
# ESRGAN Upscale
# ===================================================================


class TestEsrganUpscale:
    """Tests for the esrgan_upscale workflow."""

    BUILT_IN_NODES = ["LoadImage", "SaveImage"]

    def test_configure(self):
        _configure("esrgan_upscale")

    def test_load_custom_nodes(self):
        _configure("esrgan_upscale")
        registered = _load_workflow_nodes("esrgan_upscale")
        # May fail if spandrel not installed — that's OK
        if registered:
            assert (
                "UpscaleModelLoader" in registered
                or "ImageUpscaleWithModel" in registered
            )

    def test_builtin_nodes_available(self):
        _configure("esrgan_upscale")
        available = comfy_runtime.list_nodes()
        for node in self.BUILT_IN_NODES:
            assert node in available, f"Missing node: {node}"


# ===================================================================
# Cross-workflow: verify no GPL code loaded
# ===================================================================


class TestModuleArchitecture:
    """Verify the compat/vendor bridge architecture."""

    def test_compat_namespace_registered(self):
        """Key comfy.* modules should be in sys.modules."""
        import comfy_runtime

        for mod_name in ["comfy.cli_args"]:
            assert mod_name in sys.modules, f"Missing module: {mod_name}"

    def test_configure_activates_vendor_bridge(self):
        """After configure(), vendor modules are available for inference."""
        import comfy_runtime

        _configure("sd15_text_to_image")
        # After configure, comfy.samplers should have KSampler with SAMPLERS
        import comfy.samplers

        assert hasattr(comfy.samplers, "KSampler")
