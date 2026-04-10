import sys, time
sys.path.insert(0, "/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark")

t0 = time.perf_counter()
import comfy_runtime
print(f"import comfy_runtime: {time.perf_counter()-t0:.3f}s")

comfy_runtime.configure(
    models_dir="/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark/workflows/sd15_text_to_image/models",
)

t0 = time.perf_counter()
model, clip, vae = comfy_runtime.execute_node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
print(f"first CheckpointLoaderSimple call: {time.perf_counter()-t0:.3f}s")

# Probe the model object
print(f"model type: {type(model).__name__}")
print(f"model.model dtype: {model.model.diffusion_model.dtype if hasattr(model.model, 'diffusion_model') else 'unknown'}")
print(f"model.model_dtype: {model.model_dtype() if callable(getattr(model, 'model_dtype', None)) else 'unknown'}")
print(f"model device: {model.load_device if hasattr(model, 'load_device') else 'unknown'}")
print(f"clip type: {type(clip).__name__}")
print(f"vae type: {type(vae).__name__}")

# Second call should be fast (already loaded? no - it reloads from disk)
t0 = time.perf_counter()
m2, c2, v2 = comfy_runtime.execute_node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
print(f"second CheckpointLoaderSimple call: {time.perf_counter()-t0:.3f}s")
