"""Test the actual CLIPLoader code path after the vendor bridge fix."""
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

comfy_runtime.configure(models_dir=FLUX2_MODELS_DIR)

clip = comfy_runtime.execute_node("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")[0]
print(f"CLIP class: {type(clip.cond_stage_model).__name__}")
print(f"Success: {type(clip.cond_stage_model).__name__ == 'Flux2TEModel_'}")

# Encode a prompt and check shape
out = comfy_runtime.execute_node("CLIPTextEncode", clip=clip, text="test prompt")
cond = out[0][0][0]  # extract tensor
print(f"Conditioning shape: {tuple(cond.shape)}")
print(f"Expected 7680 features, got {cond.shape[-1]}")

