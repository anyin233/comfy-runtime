"""Test a fast-unload path that avoids moving weights CPU-ward."""
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
)
WF = FLUX2_NODES_DIR
comfy_runtime.load_nodes_from_path(f"{WF}/nodes_custom_sampler.py")

model = comfy_runtime.execute_node("UNETLoader", unet_name="flux-2-klein-base-4b.safetensors", weight_dtype="default")[0]
clip = comfy_runtime.execute_node("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")[0]
pos = comfy_runtime.execute_node("CLIPTextEncode", text="a hedgehog", clip=clip)[0]

import comfy.model_management as mm
print(f"before: alloc={torch.cuda.memory_allocated()/1024/1024:.0f}MB loaded={[type(m.model.model).__name__ for m in mm.current_loaded_models]}")

def fast_unload(device):
    """Pop loaded models + drop local clip ref + empty_cache (no CPU-ward copy)."""
    # 1. Pop entries for this device from current_loaded_models WITHOUT calling
    #    the expensive detach(unpatch_weights=True) path. We can't fully clean
    #    up the ModelPatcher's internal state without detach, but we can drop
    #    its strong refs to weights via unpatch_all=False which skips model.to().
    to_pop = []
    for i, lm in enumerate(mm.current_loaded_models):
        if lm.device == device:
            to_pop.append(i)
    for i in sorted(to_pop, reverse=True):
        lm = mm.current_loaded_models.pop(i)
        # unpatch_all=False: skip the 3.3s .to(offload_device) call. The weights
        # are left in their current state (on GPU), but since we're about to
        # drop all Python refs, they'll be freed when refcount hits 0.
        lm.model.detach(unpatch_all=False)
        if lm.model_finalizer is not None:
            lm.model_finalizer.detach()
        lm.model_finalizer = None
        lm.real_model = None

torch.cuda.synchronize()
t0 = time.perf_counter()
fast_unload(torch.device("cuda:0"))
# drop user-level refs
del clip
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"fast_unload + del + gc + empty_cache: {(time.perf_counter()-t0)*1000:.1f}ms")
print(f"after:  alloc={torch.cuda.memory_allocated()/1024/1024:.0f}MB loaded={[type(m.model.model).__name__ for m in mm.current_loaded_models]}")

