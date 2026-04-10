"""profile_sample.py"""
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

SIDE = sys.argv[1]
if SIDE == "runtime":
    sys.path.insert(0, str(REPO))
    import torch
    import comfy_runtime
    comfy_runtime.configure(models_dir=SD15_MODELS_DIR)
    EX = comfy_runtime.execute_node
else:
    sys.path.insert(0, COMFYUI_PATH)
    import torch
    import nodes
    import folder_paths
    folder_paths.add_model_folder_path("checkpoints", SD15_CHECKPOINTS_DIR)
    NCM = nodes.NODE_CLASS_MAPPINGS
    def EX(class_type, **kwargs):
        cls = NCM[class_type]
        return getattr(cls(), getattr(cls, "FUNCTION", None))(**kwargs)

torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# Setup (not profiled)
ckpts = EX("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
model, clip, vae = ckpts
pos = EX("CLIPTextEncode", clip=clip, text="a beautiful castle on a hill")[0]
neg = EX("CLIPTextEncode", clip=clip, text="blurry")[0]
latent = EX("EmptyLatentImage", width=512, height=512, batch_size=1)[0]
torch.cuda.synchronize()

# Profile JUST the KSampler call
pr = cProfile.Profile()
pr.enable()
sampled = EX("KSampler", model=model, positive=pos, negative=neg, latent_image=latent,
             seed=42, steps=20, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=1.0)
torch.cuda.synchronize()
pr.disable()

pr.dump_stats(f"/tmp/profile_{SIDE}.prof")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())

