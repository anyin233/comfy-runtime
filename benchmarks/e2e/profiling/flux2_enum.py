""""""
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
    import comfy_runtime
    comfy_runtime.configure(models_dir=FLUX2_MODELS_DIR)
    # Trigger vendor bridge so `comfy.sd` resolves to the vendored version
    from comfy_runtime.compat.comfy._vendor_bridge import activate_vendor_bridge
    activate_vendor_bridge()
else:
    sys.path.insert(0, COMFYUI_PATH)

from comfy.sd import CLIPType, TEModel, detect_te_model
print(f"[{SIDE}] CLIPType.FLUX2 = {CLIPType.FLUX2!r} value={CLIPType.FLUX2.value}")
print(f"[{SIDE}] CLIPType.FLUX  = {CLIPType.FLUX!r} value={CLIPType.FLUX.value}")
print(f"[{SIDE}] TEModel.QWEN3_4B = {TEModel.QWEN3_4B!r} value={TEModel.QWEN3_4B.value}")
print(f"[{SIDE}] CLIPType module: {CLIPType.__module__}")
print(f"[{SIDE}] detect_te_model module: {detect_te_model.__module__}")

# Now actually load the qwen3 state dict and check detection + CLIPType comparison
import safetensors.torch as st
sd = st.load_file(os.path.join(FLUX2_MODELS_DIR, 'text_encoders', 'qwen_3_4b.safetensors'))
print(f"[{SIDE}] state_dict has {len(sd)} tensors")
detected = detect_te_model(sd)
print(f"[{SIDE}] detect_te_model(qwen_3_4b) -> {detected!r}")
print(f"[{SIDE}] detected == TEModel.QWEN3_4B: {detected == TEModel.QWEN3_4B}")

# Check comparison
ct = CLIPType.FLUX2
print(f"[{SIDE}] ct == CLIPType.FLUX: {ct == CLIPType.FLUX}")
print(f"[{SIDE}] ct == CLIPType.FLUX2: {ct == CLIPType.FLUX2}")
print(f"[{SIDE}] ct == CLIPType.FLUX or ct == CLIPType.FLUX2: {ct == CLIPType.FLUX or ct == CLIPType.FLUX2}")

