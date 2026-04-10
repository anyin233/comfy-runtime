"""Test the dual-import hypothesis: is there a CLIPType identity mismatch?"""
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
# The fix in bootstrap.py should already have activated the bridge

# Approach 1: import CLIPType via `comfy.sd` (sys.modules alias)
from comfy.sd import CLIPType as CT_alias
ct_a = CT_alias.FLUX2

# Approach 2: import CLIPType via full vendor path
from comfy_runtime._vendor.comfy.sd import CLIPType as CT_direct
ct_d = CT_direct.FLUX2

print(f"CT_alias class id:  {id(CT_alias)}")
print(f"CT_direct class id: {id(CT_direct)}")
print(f"Same class? {CT_alias is CT_direct}")
print(f"ct_a == ct_d? {ct_a == ct_d}")
print(f"type(ct_a) == type(ct_d)? {type(ct_a) is type(ct_d)}")

# Show the underlying modules
import comfy.sd as comfy_sd
import comfy_runtime._vendor.comfy.sd as vendor_sd
print(f"comfy.sd module id:                           {id(comfy_sd)}")
print(f"comfy_runtime._vendor.comfy.sd module id:     {id(vendor_sd)}")
print(f"Same module? {comfy_sd is vendor_sd}")
print(f"comfy.sd.__file__:                           {comfy_sd.__file__}")
print(f"comfy_runtime._vendor.comfy.sd.__file__:     {vendor_sd.__file__}")

# Check sys.modules
for k in sorted(sys.modules.keys()):
    if k in ("comfy", "comfy.sd", "comfy_runtime._vendor.comfy", "comfy_runtime._vendor.comfy.sd"):
        m = sys.modules[k]
        f = getattr(m, "__file__", "(none)")
        print(f"sys.modules[{k!r}] id={id(m)} file={f}")

