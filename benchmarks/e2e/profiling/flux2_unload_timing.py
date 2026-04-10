"""Time each unload_all_models() call in isolation."""
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

comfy_runtime.configure(
    models_dir=FLUX2_MODELS_DIR,
    output_dir="/tmp/flux2_out",
)
WF = FLUX2_NODES_DIR
comfy_runtime.load_nodes_from_path(f"{WF}/nodes_custom_sampler.py")
comfy_runtime.load_nodes_from_path(f"{WF}/nodes_flux.py")

model = comfy_runtime.execute_node("UNETLoader", unet_name="flux-2-klein-base-4b.safetensors", weight_dtype="default")[0]
clip = comfy_runtime.execute_node("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")[0]
vae = comfy_runtime.execute_node("VAELoader", vae_name="flux2-vae.safetensors")[0]
pos = comfy_runtime.execute_node("CLIPTextEncode", text="a hedgehog wearing a tiny party hat", clip=clip)[0]
neg = comfy_runtime.execute_node("CLIPTextEncode", text="", clip=clip)[0]

print(f"before unload: alloc={torch.cuda.memory_allocated()/1024/1024:.0f}MB")
torch.cuda.synchronize()
t0 = time.perf_counter()
comfy_runtime.unload_all_models()
torch.cuda.synchronize()
print(f"unload_all_models #1: {(time.perf_counter()-t0)*1000:.1f}ms")
print(f"after unload: alloc={torch.cuda.memory_allocated()/1024/1024:.0f}MB")

# Now sample
guider = comfy_runtime.execute_node("CFGGuider", model=model, positive=pos, negative=neg, cfg=5.0)[0]
sampler = comfy_runtime.execute_node("KSamplerSelect", sampler_name="euler")[0]
sigmas = comfy_runtime.execute_node("Flux2Scheduler", steps=20, width=1024, height=1024)[0]
latent = comfy_runtime.execute_node("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)[0]
noise = comfy_runtime.execute_node("RandomNoise", noise_seed=42)[0]

t0 = time.perf_counter()
output_latent, _ = comfy_runtime.execute_node("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)
torch.cuda.synchronize()
print(f"sampler: {(time.perf_counter()-t0)*1000:.1f}ms")

torch.cuda.synchronize()
t0 = time.perf_counter()
comfy_runtime.unload_all_models()
torch.cuda.synchronize()
print(f"unload_all_models #2: {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
images = comfy_runtime.execute_node("VAEDecode", samples=output_latent, vae=vae)[0]
torch.cuda.synchronize()
print(f"decode: {(time.perf_counter()-t0)*1000:.1f}ms")

