import sys, time, os
SIDE = sys.argv[1]  # "runtime" or "comfyui"

if SIDE == "runtime":
    sys.path.insert(0, "/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark")
    import torch
    import comfy_runtime
    comfy_runtime.configure(models_dir="/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark/workflows/sd15_text_to_image/models")
    EX = comfy_runtime.execute_node
else:
    sys.path.insert(0, "/home/yanweiye/Project/ComfyUI")
    import torch
    import nodes
    import folder_paths
    folder_paths.add_model_folder_path("checkpoints", "/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark/workflows/sd15_text_to_image/models/checkpoints")
    NCM = nodes.NODE_CLASS_MAPPINGS
    def EX(class_type, **kwargs):
        cls = NCM[class_type]
        instance = cls()
        func_name = getattr(cls, "FUNCTION", None)
        return getattr(instance, func_name)(**kwargs)

def mb(b): return f"{b/1024/1024:7.0f}"

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

ckpts = EX("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
model, clip, vae = ckpts if SIDE == "runtime" else ckpts
torch.cuda.synchronize()
print(f"[{SIDE}] after load:    alloc={mb(torch.cuda.memory_allocated())} peak={mb(torch.cuda.max_memory_allocated())}")

pos = EX("CLIPTextEncode", clip=clip, text="a beautiful castle on a hill")[0]
neg = EX("CLIPTextEncode", clip=clip, text="blurry")[0]
torch.cuda.synchronize()
print(f"[{SIDE}] after encode:  alloc={mb(torch.cuda.memory_allocated())} peak={mb(torch.cuda.max_memory_allocated())}")

latent = EX("EmptyLatentImage", width=512, height=512, batch_size=1)[0]
torch.cuda.synchronize()

t0 = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
sampled = EX("KSampler", model=model, positive=pos, negative=neg, latent_image=latent,
             seed=42, steps=20, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=1.0)
torch.cuda.synchronize()
sample_time = time.perf_counter() - t0
print(f"[{SIDE}] KSampler took:  {sample_time*1000:.1f}ms  alloc={mb(torch.cuda.memory_allocated())} peak={mb(torch.cuda.max_memory_allocated())}")

# Check where weights live now
def where(name, m):
    on_gpu = sum(p.numel()*p.element_size() for p in m.parameters() if p.device.type == 'cuda')
    on_cpu = sum(p.numel()*p.element_size() for p in m.parameters() if p.device.type == 'cpu')
    print(f"[{SIDE}] {name}: gpu={mb(on_gpu)} cpu={mb(on_cpu)}")
where("diffusion_model", model.model.diffusion_model)
where("clip", clip.cond_stage_model)
where("vae", vae.first_stage_model)
