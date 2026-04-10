"""Run Flux2 Klein via upstream ComfyUI directly, to see if it reproduces."""
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

MODELS = FLUX2_MODELS_DIR
for cat in ("diffusion_models", "text_encoders", "vae"):
    folder_paths.add_model_folder_path(cat, f"{MODELS}/{cat}")

torch.manual_seed(42); torch.cuda.manual_seed_all(42)
NCM = nodes.NODE_CLASS_MAPPINGS
def ex(ct, **kw):
    cls = NCM[ct]
    fn_name = getattr(cls, "FUNCTION", None)
    return getattr(cls(), fn_name)(**kw)

# Register comfy_extras (for Flux2Scheduler, SamplerCustomAdvanced, etc.)
import asyncio, inspect
if hasattr(nodes, "init_extra_nodes"):
    fn = nodes.init_extra_nodes
    if inspect.iscoroutinefunction(fn):
        asyncio.run(fn(init_custom_nodes=False, init_api_nodes=False))
    else:
        fn()

print("=== UNETLoader ===")
model = ex("UNETLoader", unet_name="flux-2-klein-base-4b.safetensors", weight_dtype="default")[0]
print(f"  -> model type: {type(model).__name__}")

print("=== CLIPLoader ===")
clip = ex("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")[0]
print(f"  -> clip type: {type(clip).__name__}")

print("=== VAELoader ===")
vae = ex("VAELoader", vae_name="flux2-vae.safetensors")[0]

print("=== CLIPTextEncode (positive) ===")
pos = ex("CLIPTextEncode", text="a hedgehog wearing a tiny party hat", clip=clip)[0]
print("=== CLIPTextEncode (negative empty) ===")
neg = ex("CLIPTextEncode", text="", clip=clip)[0]

print("=== CFGGuider ===")
guider = ex("CFGGuider", model=model, positive=pos, negative=neg, cfg=5.0)[0]
print("=== KSamplerSelect ===")
sampler = ex("KSamplerSelect", sampler_name="euler")[0]
print("=== Flux2Scheduler ===")
sigmas = ex("Flux2Scheduler", steps=20, width=1024, height=1024)[0]
print("=== EmptyFlux2LatentImage ===")
latent = ex("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)[0]
print("=== RandomNoise ===")
noise = ex("RandomNoise", noise_seed=42)[0]

print("=== SamplerCustomAdvanced ===")
try:
    out = ex("SamplerCustomAdvanced",
             noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)
    print("  -> SUCCESS")
except Exception as e:
    print(f"  -> FAIL: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

