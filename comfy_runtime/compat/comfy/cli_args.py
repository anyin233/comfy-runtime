"""Minimal CLI args namespace for comfy_runtime.

Provides the same ``args`` object that ComfyUI's ``comfy.cli_args``
exposes, but without argparse.  Every flag starts at its safe default
and can be mutated at runtime by ``comfy_runtime.configure()``.
"""

import enum
from types import SimpleNamespace


class LatentPreviewMethod(enum.Enum):
    """Preview method enum (kept for import compatibility)."""
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        return None


class PerformanceFeature(enum.Enum):
    """Performance feature flags."""
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"
    AutoTune = "autotune"


def _default_args() -> SimpleNamespace:
    """Build the default argument namespace."""
    return SimpleNamespace(
        # Server (unused in comfy_runtime, but nodes may read them)
        listen="127.0.0.1",
        port=8188,

        # Directories
        base_directory=None,
        output_directory=None,
        temp_directory=None,
        input_directory=None,
        extra_model_paths_config=None,
        user_directory=None,

        # Device
        cuda_device=None,
        default_device=None,
        cuda_malloc=False,
        disable_cuda_malloc=False,
        directml=None,
        oneapi_device_selector=None,
        disable_ipex_optimize=False,
        supports_fp8_compute=False,

        # Precision — global
        force_fp32=False,
        force_fp16=False,

        # Precision — UNet / diffusion model
        fp32_unet=False,
        fp64_unet=False,
        bf16_unet=False,
        fp16_unet=False,
        fp8_e4m3fn_unet=False,
        fp8_e5m2_unet=False,
        fp8_e8m0fnu_unet=False,

        # Precision — VAE
        fp16_vae=False,
        fp32_vae=False,
        bf16_vae=False,
        cpu_vae=False,

        # Precision — text encoder
        fp8_e4m3fn_text_enc=False,
        fp8_e5m2_text_enc=False,
        fp16_text_enc=False,
        fp32_text_enc=False,
        bf16_text_enc=False,

        # Precision — intermediates
        fp16_intermediates=False,

        # Memory layout
        force_channels_last=False,

        # Attention
        use_split_cross_attention=False,
        use_quad_cross_attention=False,
        use_pytorch_cross_attention=False,
        use_sage_attention=False,
        use_flash_attention=False,
        disable_xformers=False,
        force_upcast_attention=False,
        dont_upcast_attention=False,

        # VRAM management
        gpu_only=False,
        highvram=False,
        normalvram=False,
        lowvram=False,
        novram=False,
        cpu=False,
        reserve_vram=None,
        async_offload=None,
        disable_async_offload=False,
        disable_dynamic_vram=False,
        enable_dynamic_vram=False,
        force_non_blocking=False,
        disable_smart_memory=False,

        # Performance
        fast=set(),
        deterministic=False,
        disable_pinned_memory=False,

        # File loading
        mmap_torch_files=False,
        disable_mmap=False,

        # Preview
        preview_method=LatentPreviewMethod.NoPreviews,
        preview_size=512,

        # Cache
        cache_classic=False,
        cache_lru=0,
        cache_none=False,
        cache_ram=0,

        # Hashing
        default_hashing_function="sha256",

        # Metadata
        disable_metadata=False,

        # Custom nodes
        disable_all_custom_nodes=False,
        whitelist_custom_nodes=[],
        disable_api_nodes=False,

        # Logging
        verbose="INFO",
        log_stdout=False,
        dont_print_server=False,

        # Misc
        auto_launch=False,
        disable_auto_launch=False,
        quick_test_for_ci=False,
        windows_standalone_build=False,
        multi_user=False,
        enable_manager=False,
        disable_manager_ui=False,
        enable_manager_legacy_ui=False,
        enable_compress_response_body=False,
        front_end_version="comfyanonymous/ComfyUI@latest",
        front_end_root=None,
        comfy_api_base="https://api.comfy.org",
        database_url=None,
        enable_assets=False,
    )


args = _default_args()

# Keep the same constant that the original module exposes.
CACHE_RAM_AUTO_GB = -1.0


def enables_dynamic_vram():
    """Return whether dynamic VRAM management should be active."""
    if args.enable_dynamic_vram:
        return True
    return (
        not args.disable_dynamic_vram
        and not args.highvram
        and not args.gpu_only
        and not args.novram
        and not args.cpu
    )
