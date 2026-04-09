---
name: port-workflow
description: Use when porting a ComfyUI workflow or custom node to comfy_runtime - guides the full process from workflow analysis through implementation and testing
---

# Port ComfyUI Workflow / Custom Node to comfy_runtime

You are an expert at porting ComfyUI workflows to the comfy_runtime single-node execution engine. Follow this guide precisely.

## Architecture Overview

comfy_runtime is a **single-node execution engine** — it has NO workflow graph executor. Each node is called individually via:

```python
result = comfy_runtime.execute_node("NodeClassName", **kwargs)
```

A workflow is ported by **manually chaining** `execute_node` calls in Python, passing outputs from one node as inputs to the next.

## Step 1: Analyze the Source Workflow

### If given a workflow JSON (ComfyUI API format):
- Each key is a node ID; each value has `class_type` and `inputs`
- Inputs referencing `["other_id", slot_index]` are connections to other nodes
- Topologically sort nodes by dependency order

### If given a workflow JSON (LiteGraph UI format):
- `nodes` array contains node objects with `type` (= class_type)
- `links` array defines connections: `[link_id, origin_node, origin_slot, target_node, target_slot, type]`
- `widgets_values` on each node contain the static parameter values
- Trace the data flow from source nodes (loaders) to sink nodes (SaveImage)

### If given a workflow URL from comfy.org:
- Fetch the JSON from `https://www.comfy.org/workflows/download/<slug>.json`
- Parse the `nodes` and `links` arrays

### Key information to extract:
1. **All unique `class_type` / `type` values** — these are the nodes you need
2. **Data flow graph** — which node outputs feed into which node inputs
3. **Widget values** — static parameters (steps, cfg, dimensions, model names)
4. **Model files** — found in node properties (`models` array with `directory`, `name`, `url`)

## Step 2: Identify Node Availability

### Built-in nodes (always available, no loading needed):
Read the file `comfy_runtime/_vendor/nodes.py` and check `NODE_CLASS_MAPPINGS` at the bottom. Common built-in nodes include:
- **Loaders**: CheckpointLoaderSimple, UNETLoader, CLIPLoader, DualCLIPLoader, VAELoader, LoraLoader, ControlNetLoader, LoadImage
- **Encoding**: CLIPTextEncode, CLIPVisionEncode
- **Sampling**: KSampler, KSamplerAdvanced
- **Latent**: EmptyLatentImage, LatentUpscale, SetLatentNoiseMask, VAEEncode, VAEDecode, VAEDecodeTiled
- **Conditioning**: ConditioningCombine, ConditioningSetArea, ConditioningZeroOut
- **Output**: SaveImage, PreviewImage
- **Image**: ImageScale, ImageInvert, ImageBatch

### Extra nodes (must be loaded from ComfyUI/comfy_extras):
These require `comfy_runtime.load_nodes_from_path()`. Check `/home/yanweiye/Project/ComfyUI/comfy_extras/` for the source file. Common extras:
- **nodes_custom_sampler.py**: KSamplerSelect, RandomNoise, SamplerCustomAdvanced, CFGGuider, BasicGuider
- **nodes_flux.py**: Flux2Scheduler, EmptyFlux2LatentImage
- **nodes_upscale_model.py**: UpscaleModelLoader, ImageUpscaleWithModel (needs `spandrel` dependency)
- **nodes_video_model.py**: ImageOnlyCheckpointLoader, SVD_img2vid_Conditioning, VideoLinearCFGGuidance
- **nodes_images.py**: SaveAnimatedWEBP, SaveSVGNode
- **nodes_model_advanced.py**: ModelSamplingSD3, ModelSamplingFlux
- **nodes_primitive.py**: PrimitiveInt, PrimitiveStringMultiline
- **nodes_controlnet.py**: ControlNet apply nodes
- **nodes_sd3.py**: SD3 specific nodes
- **nodes_cond.py**: ReferenceLatent

### API nodes (in ComfyUI/comfy_api_nodes):
These call external APIs and require auth tokens. They CANNOT be used directly via `execute_node` because they rely on hidden variable injection. If encountered, either:
- Create a simplified replacement that makes direct HTTP calls
- Or skip and note the limitation

### Nodes not found anywhere:
If a node's `class_type` doesn't exist in built-in or comfy_extras, it's a **third-party custom node**. Search PyPI or GitHub for the package, install it, and load via `load_nodes_from_path`.

## Step 3: Create the Workflow Folder

```
workflows/<workflow_name>/
  pyproject.toml          # uv project with comfy-runtime dependency
  nodes/                  # Extra node files (copied from comfy_extras)
    __init__.py
    nodes_xxx.py          # Only include files with nodes this workflow uses
  workflow_utils/          # NOT "utils/" — that name conflicts with comfy.utils
    __init__.py
    download_models.py    # Model download logic using huggingface_hub
  main.py                 # Workflow entry point
```

**CRITICAL**: Name the utilities directory `workflow_utils/`, NOT `utils/`. Importing `comfy_runtime` registers `comfy.utils` in `sys.modules["utils"]`, which shadows any local `utils` package.

### pyproject.toml template:
```toml
[project]
name = "<workflow-name>"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "comfy-runtime",
    "huggingface-hub",  # if models need downloading
    # Add extra deps needed by nodes (e.g., "spandrel" for ESRGAN)
]

[tool.uv.sources]
comfy-runtime = { path = "/home/yanweiye/Project/comfy_runtime", editable = true }
```

## Step 4: Implement main.py

### Structure:
```python
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import comfy_runtime
from workflow_utils.download_models import ensure_models

MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_models(MODELS_DIR)

    comfy_runtime.configure(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Load extra nodes
    for f in ["nodes_xxx.py"]:
        comfy_runtime.load_nodes_from_path(os.path.join(SCRIPT_DIR, "nodes", f))

    # Chain execute_node calls in topological order...
    # result = comfy_runtime.execute_node("NodeType", param=value, ...)

if __name__ == "__main__":
    main()
```

### execute_node return value conventions:
- **V1 nodes** (have `FUNCTION` attribute): returns a **tuple** of outputs matching `RETURN_TYPES`. Access by index: `result[0]`, `result[1]`, etc.
- **V3 nodes** (have `define_schema`/`execute`): return is automatically unwrapped from `NodeOutput`. Same tuple convention.
- **Output nodes** (SaveImage, etc.): return `{"ui": {...}}` dict or None. Usually don't need the return value.

### Unpacking convention:
```python
# Single output node:
model = comfy_runtime.execute_node("UNETLoader", unet_name="model.safetensors", weight_dtype="default")[0]

# Multi-output node:
model, clip, vae = comfy_runtime.execute_node("CheckpointLoaderSimple", ckpt_name="model.safetensors")

# Two-output node:
output_latent, denoised = comfy_runtime.execute_node("SamplerCustomAdvanced", ...)
```

### Input parameter names:
Parameter names must match the node's `INPUT_TYPES()` keys exactly. Check by:
```python
info = comfy_runtime.get_node_info("NodeClassName")
print(info["input_types"])
```

## Step 5: Handle Models

### download_models.py template:
```python
import os
import shutil
from huggingface_hub import hf_hub_download

MODELS = {
    "category_dir": [
        {"repo_id": "org/repo", "filename": "path/in/repo.safetensors", "local_name": "model.safetensors"},
    ],
}

def ensure_models(models_dir):
    for category, model_list in MODELS.items():
        cat_dir = os.path.join(models_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        for m in model_list:
            dest = os.path.join(cat_dir, m["local_name"])
            if os.path.exists(dest):
                continue
            downloaded = hf_hub_download(repo_id=m["repo_id"], filename=m["filename"], local_dir=cat_dir)
            if os.path.abspath(downloaded) != os.path.abspath(dest):
                shutil.move(downloaded, dest)
```

### Model category directories:
Models must be placed in subdirectories matching ComfyUI's `folder_paths` categories:
- `checkpoints/` — full model checkpoints (CheckpointLoaderSimple)
- `diffusion_models/` — UNET/diffusion models (UNETLoader)
- `text_encoders/` — CLIP/text encoder models (CLIPLoader)
- `vae/` — VAE models (VAELoader)
- `loras/` — LoRA files (LoraLoader)
- `controlnet/` — ControlNet models (ControlNetLoader)
- `upscale_models/` — ESRGAN/upscale models (UpscaleModelLoader)
- `clip_vision/` — CLIP vision models (CLIPVisionLoader)

### Reusing models across workflows:
Use absolute symlinks to avoid re-downloading:
```bash
ln -sf /absolute/path/to/existing/model.safetensors ./models/checkpoints/
```

## Step 6: Common Pitfalls

### 1. Tensor gradient errors on SaveImage
VAEDecode may return tensors with `requires_grad=True`. Detach before saving:
```python
images = comfy_runtime.execute_node("VAEDecode", samples=latent, vae=vae)[0]
if hasattr(images, "detach"):
    images = images.detach()
```

### 2. CLIPLoader `type` parameter
Some CLIP models require a specific `type` argument:
- SD 1.5 / SDXL: omit or use default
- Flux models: `type="flux"` or `type="flux2"`
- SD3: `type="sd3"`

### 3. V3 nodes with hidden variables
V3 nodes that access `cls.hidden` (like SaveSVGNode, API nodes) will fail because `execute_node` doesn't inject hidden context. Create simplified V1 replacements if needed.

### 4. Extra node dependencies
Some comfy_extras nodes require additional packages not bundled with comfy_runtime:
- `nodes_upscale_model.py` → needs `spandrel`
- `nodes_canny.py` → needs `kornia`
- `nodes_audio.py` → needs `torchaudio`
- `nodes_glsl.py` → needs `PyOpenGL`

Add these to `pyproject.toml` dependencies.

### 5. Bypassed nodes (mode=4)
In workflow JSON, nodes with `"mode": 4` are bypassed/muted. Skip them.

### 6. Subgraph nodes
Newer ComfyUI workflows may use "subgraphs" — nodes whose `type` is a UUID. The subgraph definition is in `definitions.subgraphs[]`. Flatten them: extract the internal nodes and inline them into the main execution chain.

## Step 7: Test

```bash
cd workflows/<name>
uv sync
uv run python main.py
```

### Verify:
1. No import errors
2. All nodes load successfully
3. Models download and load
4. Execution completes without error
5. Output files exist and are valid

### Debug with subagent:
If comfy_runtime raises errors during execution, launch an Agent subagent with the full traceback and the relevant node source code to diagnose and fix.

## Reference: Existing Workflow Examples

Study these tested examples in `workflows/` for patterns:

| Workflow | Pattern | Key Nodes |
|----------|---------|-----------|
| `sd15_text_to_image` | Basic text→image, all built-in | CheckpointLoaderSimple, KSampler |
| `flux2_klein_text_to_image` | Advanced sampler with extra nodes | UNETLoader, CLIPLoader, SamplerCustomAdvanced |
| `img2img` | Image input + VAEEncode + denoise | LoadImage, VAEEncode, KSampler(denoise<1) |
| `esrgan_upscale` | External model loader (spandrel) | UpscaleModelLoader, ImageUpscaleWithModel |
| `hires_fix` | Multi-pass pipeline | KSampler×2, LatentUpscale |
| `area_composition` | Spatial conditioning | ConditioningSetArea, ConditioningCombine |
| `inpainting` | Mask-based region editing | SetLatentNoiseMask, programmatic mask |
