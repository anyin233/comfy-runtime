"""probe_full_warm.py"""
import os
import sys
from pathlib import Path

# Repo root: this file is at $REPO/benchmarks/e2e/profiling/, so 3 parents up.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Upstream ComfyUI clone path (overridable via env var).
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/home/yanweiye/Project/ComfyUI")

# Workflow models root (overridable via env var).
WORKFLOW_MODELS = Path(os.environ.get("WORKFLOW_MODELS", REPO / "workflows"))
SD15_MODELS_DIR = str(WORKFLOW_MODELS / "sd15_text_to_image" / "models")
SD15_CHECKPOINTS_DIR = str(WORKFLOW_MODELS / "sd15_text_to_image" / "models" / "checkpoints")

# Eagerly load EVERYTHING at "import" time (simulating a fixed bootstrap)
t0 = time.perf_counter()
import comfy_runtime
print(f"import comfy_runtime:           {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
from comfy_runtime.compat.comfy._vendor_bridge import activate_vendor_bridge
activate_vendor_bridge()
print(f"activate_vendor_bridge:         {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
import comfy.sample
import comfy.samplers
import comfy.k_diffusion.sampling
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_base
import comfy.utils
print(f"pre-import sampler chain:       {(time.perf_counter()-t0)*1000:.1f}ms")

# Also pre-import text encoder & vae chains
t0 = time.perf_counter()
import comfy.sd
import comfy.ldm.models.autoencoder
print(f"pre-import sd/autoencoder:      {(time.perf_counter()-t0)*1000:.1f}ms")

# === Now run the workflow ===
import torch
comfy_runtime.configure(models_dir=SD15_MODELS_DIR)

torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

EX = comfy_runtime.execute_node

# Run the FULL workflow (cold, like benchmark would)
t_start = time.perf_counter()

t0 = time.perf_counter()
ckpts = EX("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
model, clip, vae = ckpts
torch.cuda.synchronize()
print(f"CheckpointLoaderSimple:         {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
pos = EX("CLIPTextEncode", clip=clip, text="a beautiful castle on a hill, fantasy art, highly detailed, dramatic lighting, digital painting")[0]
neg = EX("CLIPTextEncode", clip=clip, text="blurry, low quality, distorted, watermark, text")[0]
torch.cuda.synchronize()
print(f"CLIPTextEncode x2:              {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
latent = EX("EmptyLatentImage", width=512, height=512, batch_size=1)[0]
torch.cuda.synchronize()
print(f"EmptyLatentImage:               {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
sampled = EX("KSampler", model=model, positive=pos, negative=neg, latent_image=latent,
             seed=42, steps=20, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=1.0)[0]
torch.cuda.synchronize()
print(f"KSampler:                       {(time.perf_counter()-t0)*1000:.1f}ms")

t0 = time.perf_counter()
images = EX("VAEDecode", samples=sampled, vae=vae)[0]
torch.cuda.synchronize()
print(f"VAEDecode:                      {(time.perf_counter()-t0)*1000:.1f}ms")

print(f"---")
print(f"workflow total (excluding imports): {(time.perf_counter()-t_start)*1000:.1f}ms")
print(f"GPU peak: {torch.cuda.max_memory_allocated()/1024/1024:.0f}MB")

