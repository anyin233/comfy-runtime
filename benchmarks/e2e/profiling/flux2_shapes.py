"""Dump shapes of all intermediates on both sides, stop right before SamplerCustomAdvanced."""
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
    import torch
    import comfy_runtime
    comfy_runtime.configure(
        models_dir=FLUX2_MODELS_DIR,
        output_dir="/tmp/flux2_out",
    )
    comfy_runtime.load_nodes_from_path(
        os.path.join(FLUX2_NODES_DIR, "nodes_custom_sampler.py")
    )
    comfy_runtime.load_nodes_from_path(
        os.path.join(FLUX2_NODES_DIR, "nodes_flux.py")
    )
    EX = comfy_runtime.execute_node
else:
    sys.path.insert(0, COMFYUI_PATH)
    import torch
    import nodes
    import folder_paths
    import asyncio, inspect
    MODELS = FLUX2_MODELS_DIR
    for cat in ("diffusion_models", "text_encoders", "vae"):
        folder_paths.add_model_folder_path(cat, f"{MODELS}/{cat}")
    if hasattr(nodes, "init_extra_nodes"):
        fn = nodes.init_extra_nodes
        if inspect.iscoroutinefunction(fn):
            asyncio.run(fn(init_custom_nodes=False, init_api_nodes=False))
        else:
            fn()
    NCM = nodes.NODE_CLASS_MAPPINGS
    def EX(ct, **kw):
        cls = NCM[ct]
        fn_name = getattr(cls, "FUNCTION", None)
        return getattr(cls(), fn_name)(**kw)

torch.manual_seed(42); torch.cuda.manual_seed_all(42)

def dump(label, obj, depth=0):
    prefix = "  " * depth
    if isinstance(obj, torch.Tensor):
        print(f"[{SIDE}] {prefix}{label}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}")
    elif isinstance(obj, dict):
        print(f"[{SIDE}] {prefix}{label}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            dump(k, v, depth+1)
    elif isinstance(obj, (list, tuple)):
        print(f"[{SIDE}] {prefix}{label}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj):
            if i > 5: break
            dump(f"[{i}]", v, depth+1)
    else:
        print(f"[{SIDE}] {prefix}{label}: {type(obj).__name__} repr={repr(obj)[:80]}")

print(f"===== {SIDE} =====")

model = EX("UNETLoader", unet_name="flux-2-klein-base-4b.safetensors", weight_dtype="default")[0]
print(f"[{SIDE}] model class: {type(model).__name__}")
if hasattr(model, 'model'): print(f"[{SIDE}] model.model class: {type(model.model).__name__}")
if hasattr(model.model, 'diffusion_model'):
    dm = model.model.diffusion_model
    print(f"[{SIDE}] diffusion_model class: {type(dm).__name__}")
    # Look at first few modules to understand arch
    for name, mod in list(dm.named_modules())[:5]:
        print(f"[{SIDE}]   {name}: {type(mod).__name__}")

clip = EX("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2")[0]
print(f"[{SIDE}] clip class: {type(clip).__name__}")
print(f"[{SIDE}] clip.cond_stage_model class: {type(clip.cond_stage_model).__name__}")

vae = EX("VAELoader", vae_name="flux2-vae.safetensors")[0]

pos = EX("CLIPTextEncode", text="a hedgehog wearing a tiny party hat", clip=clip)[0]
dump("pos_cond", pos)

neg = EX("CLIPTextEncode", text="", clip=clip)[0]
dump("neg_cond", neg)

guider = EX("CFGGuider", model=model, positive=pos, negative=neg, cfg=5.0)[0]
print(f"[{SIDE}] guider class: {type(guider).__name__}")

sampler = EX("KSamplerSelect", sampler_name="euler")[0]
print(f"[{SIDE}] sampler class: {type(sampler).__name__}")

sigmas = EX("Flux2Scheduler", steps=20, width=1024, height=1024)[0]
dump("sigmas", sigmas)

latent = EX("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)[0]
dump("latent", latent)

noise = EX("RandomNoise", noise_seed=42)[0]
print(f"[{SIDE}] noise class: {type(noise).__name__}")
if hasattr(noise, 'seed'): print(f"[{SIDE}]   seed: {noise.seed}")

