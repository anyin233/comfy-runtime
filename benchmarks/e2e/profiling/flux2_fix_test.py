"""Test: does explicit free_memory before VAEDecode avoid the tiled fallback?"""
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
EX = comfy_runtime.execute_node

import comfy.model_management as mm
def mb(b): return f"{b/1024/1024:7.0f}"
def show(label):
    loaded = [type(m.model.model).__name__ for m in mm.current_loaded_models]
    print(f"[test] {label:<30} alloc={mb(torch.cuda.memory_allocated())}MB peak={mb(torch.cuda.max_memory_allocated())}MB loaded={loaded}")

torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

model = EX("UNETLoader", unet_name="flux-2-klein-base-4b.safetensors", weight_dtype="default")[0]
clip = EX("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")[0]
vae = EX("VAELoader", vae_name="flux2-vae.safetensors")[0]
pos = EX("CLIPTextEncode", text="a hedgehog wearing a tiny party hat", clip=clip)[0]
neg = EX("CLIPTextEncode", text="", clip=clip)[0]
show("after text_encode")

# EXPERIMENT: Evict text encoder before sampling
print("[test] calling free_memory(1e30, cuda:0) to evict text encoder...")
mm.free_memory(1e30, torch.device("cuda:0"))
torch.cuda.empty_cache()
show("after free_memory call")

guider = EX("CFGGuider", model=model, positive=pos, negative=neg, cfg=5.0)[0]
sampler = EX("KSamplerSelect", sampler_name="euler")[0]
sigmas = EX("Flux2Scheduler", steps=20, width=1024, height=1024)[0]
latent = EX("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)[0]
noise = EX("RandomNoise", noise_seed=42)[0]

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
output_latent, _ = EX("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent)
torch.cuda.synchronize()
print(f"[test] SamplerCustomAdvanced took: {(time.perf_counter()-t0)*1000:.1f}ms")
show("after sampling")

# EXPERIMENT: Evict diffusion model before decode
print("[test] calling free_memory(1e30, cuda:0) to evict diffusion model...")
mm.free_memory(1e30, torch.device("cuda:0"))
torch.cuda.empty_cache()
show("after free_memory call 2")

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
images = EX("VAEDecode", samples=output_latent, vae=vae)[0]
torch.cuda.synchronize()
print(f"[test] VAEDecode took:            {(time.perf_counter()-t0)*1000:.1f}ms")
show("after VAEDecode")

