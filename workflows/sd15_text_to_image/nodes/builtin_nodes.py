"""Built-in nodes used by the SD 1.5 Text to Image workflow.

All nodes listed here are built-in to comfy_runtime and do NOT need to be
loaded externally. This file serves as documentation of the workflow's
node dependencies.

Built-in nodes used:
    - CheckpointLoaderSimple: Loads a full SD checkpoint (model + clip + vae)
    - CLIPTextEncode: Encodes text into CLIP conditioning
    - EmptyLatentImage: Creates an empty latent image tensor
    - KSampler: Runs the diffusion sampling loop
    - VAEDecode: Decodes latent tensor into pixel image
    - SaveImage: Saves image tensor as PNG to disk
"""

BUILTIN_NODES = [
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
]
