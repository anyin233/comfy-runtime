"""Stub for comfy.memory_management.

Re-exports from comfy.model_management for nodes that reference this module.
"""

# Some custom nodes import from comfy.memory_management instead of
# comfy.model_management. Re-export the key symbols so those imports work.
from comfy_runtime.compat.comfy.model_management import (  # noqa: F401
    get_torch_device,
    get_free_memory,
    get_total_memory,
    soft_empty_cache,
    load_models_gpu,
    load_model_gpu,
    free_memory,
    unload_all_models,
    intermediate_device,
    unet_offload_device,
    vae_device,
    vae_offload_device,
    text_encoder_device,
    text_encoder_offload_device,
)
