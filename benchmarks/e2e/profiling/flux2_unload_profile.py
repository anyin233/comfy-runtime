"""Profile the 3.3s unload cost to find where time goes."""
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
loaded = mm.current_loaded_models[0]
print(f"loaded model: {type(loaded.model.model).__name__}, current_used={loaded.currently_used}")

torch.cuda.synchronize()

# Step A: just empty_cache
t0 = time.perf_counter()
torch.cuda.empty_cache()
print(f"A) empty_cache only:                  {(time.perf_counter()-t0)*1000:.1f}ms")

# Step B: free_memory
torch.cuda.synchronize()
t0 = time.perf_counter()
mm.free_memory(1e30, torch.device("cuda:0"))
torch.cuda.synchronize()
print(f"B) free_memory(1e30) [moves to CPU]:  {(time.perf_counter()-t0)*1000:.1f}ms")

# Step C: empty_cache after
torch.cuda.synchronize()
t0 = time.perf_counter()
torch.cuda.empty_cache()
print(f"C) empty_cache post-unload:           {(time.perf_counter()-t0)*1000:.1f}ms")

print(f"loaded models after: {[type(m.model.model).__name__ for m in mm.current_loaded_models]}")
print(f"gpu alloc after: {torch.cuda.memory_allocated()/1024/1024:.0f}MB")

