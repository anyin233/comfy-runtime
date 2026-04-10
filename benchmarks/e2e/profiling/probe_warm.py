"""probe_warm.py"""
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

comfy_runtime.configure(models_dir=SD15_MODELS_DIR)

# Warmup: trigger bridge init
comfy_runtime.execute_node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")

# Warmup: pre-import the sampler chain
t0 = time.perf_counter()
import comfy.sample
import comfy.samplers
import comfy.k_diffusion.sampling
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_base
print(f"warm-import: {(time.perf_counter()-t0)*1000:.1f}ms")

# Now do the actual workflow with a fresh model
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.cuda.reset_peak_memory_stats()
EX = comfy_runtime.execute_node

ckpts = EX("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
model, clip, vae = ckpts
pos = EX("CLIPTextEncode", clip=clip, text="a beautiful castle on a hill")[0]
neg = EX("CLIPTextEncode", clip=clip, text="blurry")[0]
latent = EX("EmptyLatentImage", width=512, height=512, batch_size=1)[0]
torch.cuda.synchronize()

t0 = time.perf_counter()
sampled = EX("KSampler", model=model, positive=pos, negative=neg, latent_image=latent,
             seed=42, steps=20, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=1.0)
torch.cuda.synchronize()
print(f"KSampler (warm imports): {(time.perf_counter()-t0)*1000:.1f}ms")

# Run again - now everything's warm
torch.cuda.synchronize()
t0 = time.perf_counter()
sampled = EX("KSampler", model=model, positive=pos, negative=neg, latent_image=latent,
             seed=42, steps=20, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=1.0)
torch.cuda.synchronize()
print(f"KSampler (2nd call):     {(time.perf_counter()-t0)*1000:.1f}ms")

