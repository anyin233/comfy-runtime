import sys, time
sys.path.insert(0, "/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark")
import torch
import comfy_runtime
comfy_runtime.configure(models_dir="/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark/workflows/sd15_text_to_image/models")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

model, clip, vae = comfy_runtime.execute_node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
torch.cuda.synchronize()

print(f"GPU after CheckpointLoaderSimple: alloc={torch.cuda.memory_allocated()/1024/1024:.0f}MB peak={torch.cuda.max_memory_allocated()/1024/1024:.0f}MB")

# Inspect where the model weights actually live
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

# ModelPatcher wraps the actual model in .model attribute
print("--- model ---")
storage_summary("model.diffusion_model", model.model.diffusion_model)
print(f"  load_device: {model.load_device}")
print(f"  offload_device: {model.offload_device if hasattr(model, 'offload_device') else 'n/a'}")
print(f"  model_options keys: {list(model.model_options.keys()) if hasattr(model, 'model_options') else []}")

print("--- clip ---")
storage_summary("clip.cond_stage_model", clip.cond_stage_model)
print(f"  load_device: {clip.load_device if hasattr(clip, 'load_device') else 'n/a'}")

print("--- vae ---")
storage_summary("vae.first_stage_model", vae.first_stage_model)
