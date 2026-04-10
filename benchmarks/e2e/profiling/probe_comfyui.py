import sys, time
sys.path.insert(0, "/home/yanweiye/Project/ComfyUI")
import torch

t0 = time.perf_counter()
import nodes
print(f"import nodes: {time.perf_counter()-t0:.3f}s")

import folder_paths
folder_paths.add_model_folder_path("checkpoints", "/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark/workflows/sd15_text_to_image/models/checkpoints")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

t0 = time.perf_counter()
loader = nodes.NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
result = loader.load_checkpoint("v1-5-pruned-emaonly.safetensors")
torch.cuda.synchronize()
print(f"first CheckpointLoaderSimple call: {time.perf_counter()-t0:.3f}s")
print(f"GPU after: alloc={torch.cuda.memory_allocated()/1024/1024:.0f}MB peak={torch.cuda.max_memory_allocated()/1024/1024:.0f}MB")

model, clip, vae = result
def storage_summary(name, m):
    on_gpu = 0
    on_cpu = 0
    total_n = 0
    for p in m.parameters() if hasattr(m, 'parameters') else []:
        n = p.numel() * p.element_size()
        total_n += n
        if p.device.type == 'cuda': on_gpu += n
        else: on_cpu += n
    print(f"  {name}: total={total_n/1024/1024:.0f}MB on_gpu={on_gpu/1024/1024:.0f}MB on_cpu={on_cpu/1024/1024:.0f}MB")

print("--- model ---")
storage_summary("model.diffusion_model", model.model.diffusion_model)
print(f"  load_device: {model.load_device}")
print(f"  offload_device: {model.offload_device if hasattr(model, 'offload_device') else 'n/a'}")
print(f"  model_dtype: {model.model_dtype()}")

t0 = time.perf_counter()
result2 = loader.load_checkpoint("v1-5-pruned-emaonly.safetensors")
torch.cuda.synchronize()
print(f"second CheckpointLoaderSimple call: {time.perf_counter()-t0:.3f}s")
