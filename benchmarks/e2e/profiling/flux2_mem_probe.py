"""Walk the Flux2 workflow step-by-step and print GPU mem + current_loaded_models."""
import os
import sys
from pathlib import Path

# Repo root: this file lives at $REPO/benchmarks/e2e/profiling/, so three
# parents up is the repo root.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Path to upstream ComfyUI clone (needed for the comfyui-side comparison
# probes). Override via COMFYUI_PATH env var if your clone lives elsewhere.
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/home/yanweiye/Project/ComfyUI")

# Where each workflow's model files live. Defaults to the repo-local
# workflows/<name>/models/ directory. Override via WORKFLOW_MODELS env var
# if you share a single models tree across workflows.
WORKFLOW_MODELS = Path(os.environ.get("WORKFLOW_MODELS", REPO / "workflows"))

# Flux2-specific model locations derived from the above.
FLUX2_MODELS_DIR = str(WORKFLOW_MODELS / "flux2_klein_text_to_image" / "models")
FLUX2_NODES_DIR = str(REPO / "workflows" / "flux2_klein_text_to_image" / "nodes")

SIDE = sys.argv[1]

if SIDE == "runtime":
    sys.path.insert(0, str(REPO))
    import torch
    import comfy_runtime
    comfy_runtime.configure(
        models_dir=FLUX2_MODELS_DIR,
        output_dir="/tmp/flux2_out",
    )
    WORKFLOW_NODES_DIR = FLUX2_NODES_DIR
    comfy_runtime.load_nodes_from_path(f"{WORKFLOW_NODES_DIR}/nodes_custom_sampler.py")
    comfy_runtime.load_nodes_from_path(f"{WORKFLOW_NODES_DIR}/nodes_flux.py")
    EX = comfy_runtime.execute_node
else:
    sys.path.insert(0, COMFYUI_PATH)
    import torch
    import nodes
    import folder_paths
    import asyncio, inspect
    MODELS = FLUX2_MODELS_DIR
    for cat in ("diffusion_models", "text_encoders", "vae"):
        folder_paths.add_model_folder_path(cat, f"{MODELS}/{cat}")
    if hasattr(nodes, "init_extra_nodes"):
        fn = nodes.init_extra_nodes
        if inspect.iscoroutinefunction(fn):
            asyncio.run(fn(init_custom_nodes=False, init_api_nodes=False))
        else:
            fn()
    NCM = nodes.NODE_CLASS_MAPPINGS
    def EX(ct, **kw):
        cls = NCM[ct]
        return getattr(cls(), getattr(cls, "FUNCTION", None))(**kw)

import comfy.model_management as mm

def mb(b): return f"{b/1024/1024:8.0f}"

def show(label):
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    loaded = [type(m.model.model).__name__ for m in mm.current_loaded_models]
    print(f"[{SIDE}] {label:<30} alloc={mb(alloc)}MB peak={mb(peak)}MB reserved={mb(reserved)}MB loaded={loaded}")

torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

show("baseline")

model = EX("UNETLoader", unet_name="flux-2-klein-base-4b.safetensors", weight_dtype="default")[0]
show("after UNETLoader")

clip = EX("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")[0]
show("after CLIPLoader")

vae = EX("VAELoader", vae_name="flux2-vae.safetensors")[0]
show("after VAELoader")

pos = EX("CLIPTextEncode", text="a hedgehog wearing a tiny party hat", clip=clip)[0]
show("after CLIPTextEncode (pos)")

neg = EX("CLIPTextEncode", text="", clip=clip)[0]
show("after CLIPTextEncode (neg)")

guider = EX("CFGGuider", model=model, positive=pos, negative=neg, cfg=5.0)[0]
show("after CFGGuider")

sampler = EX("KSamplerSelect", sampler_name="euler")[0]
sigmas = EX("Flux2Scheduler", steps=20, width=1024, height=1024)[0]
latent = EX("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)[0]
noise = EX("RandomNoise", noise_seed=42)[0]
show("after sampler_prep")

try:
    output_latent, _ = EX("SamplerCustomAdvanced", noise=noise, guider=guider,
                          sampler=sampler, sigmas=sigmas, latent_image=latent)
    show("after SamplerCustomAdvanced")
except Exception as e:
    print(f"[{SIDE}] SAMPLER FAILED: {e}")
    sys.exit(1)

try:
    images = EX("VAEDecode", samples=output_latent, vae=vae)[0]
    show("after VAEDecode")
except Exception as e:
    print(f"[{SIDE}] VAEDECODE FAILED: {e}")
    sys.exit(1)

print(f"[{SIDE}] DONE, final alloc={mb(torch.cuda.memory_allocated())}MB peak={mb(torch.cuda.max_memory_allocated())}MB")

