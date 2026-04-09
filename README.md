# comfy-runtime

Run any ComfyUI node as a Python function. No server, no UI, no workflow graph.

## Install

```bash
pip install comfy-runtime
```

## Usage

### 1. Configure & run built-in nodes

```python
import comfy_runtime

comfy_runtime.configure(models_dir="/path/to/models", output_dir="./output")

# Text-to-image with SD 1.5
model, clip, vae = comfy_runtime.execute_node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
positive = comfy_runtime.execute_node("CLIPTextEncode", clip=clip, text="a castle on a hill, fantasy art")[0]
negative = comfy_runtime.execute_node("CLIPTextEncode", clip=clip, text="blurry, low quality")[0]
latent = comfy_runtime.execute_node("EmptyLatentImage", width=512, height=512, batch_size=1)[0]
sampled = comfy_runtime.execute_node("KSampler", model=model, positive=positive, negative=negative, latent_image=latent, seed=42, steps=20, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=1.0)[0]
images = comfy_runtime.execute_node("VAEDecode", samples=sampled, vae=vae)[0]
comfy_runtime.execute_node("SaveImage", images=images, filename_prefix="output")
```

### 2. Load any custom node

```python
# Single file
comfy_runtime.load_nodes_from_path("/path/to/my_node.py")

# Directory (with or without __init__.py)
comfy_runtime.load_nodes_from_path("/path/to/ComfyUI-AnimateDiff-Evolved")

# Then use it
result = comfy_runtime.execute_node("ADE_AnimateDiffLoRALoader", ...)
```

Both V1 (`NODE_CLASS_MAPPINGS`) and V3 (`comfy_entrypoint`) node formats are supported.

### 3. Discover nodes

```python
comfy_runtime.list_nodes()          # All registered node names
comfy_runtime.get_node_info("KSampler")  # Input/output types, category
```

## API

| Function | Description |
|---|---|
| `configure(models_dir, output_dir, ...)` | Set model paths and device config. Call before loading models. |
| `execute_node(class_type, **kwargs)` | Run a node and return its output tuple. |
| `load_nodes_from_path(path)` | Load nodes from a `.py` file or directory. Returns list of registered names. |
| `list_nodes()` | All registered node names. |
| `get_node_info(class_type)` | Node metadata (inputs, outputs, category). |
| `register_node(class_type, cls)` | Register a node class manually. |
| `unregister_node(class_type)` | Remove a node. |
| `get_config()` | Current runtime config. |

## Custom node compatibility

Tested with popular third-party nodes:

| Node Pack | Nodes | Status |
|---|---|---|
| WAS-Node-Suite (220 nodes) | Image/text utilities | Works |
| IPAdapter-Plus (37 nodes) | Style transfer | Works |
| KJNodes (232 nodes) | Utility pack | Works |
| Advanced-ControlNet (43 nodes) | ControlNet | Works |
| AnimateDiff-Evolved (145 nodes) | Video animation | Works |

Custom nodes that import from `comfy.*`, `comfy_api.*`, `comfy_extras.*`, `folder_paths`, or `nodes` all resolve correctly.

## Examples

See [`workflows/`](workflows/) for complete runnable examples:

- `sd15_text_to_image` — basic text-to-image
- `flux2_klein_text_to_image` — Flux.2 with custom sampler
- `img2img` — image-to-image style transfer
- `inpainting` — masked region replacement
- `hires_fix` — 2-pass high-resolution generation
- `area_composition` — spatial prompt composition
- `esrgan_upscale` — 4x super-resolution

## License

GPL-3.0
