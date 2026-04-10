import sys, time
SIDE = sys.argv[1]
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
        return getattr(cls(), getattr(cls, "FUNCTION", None))(**kwargs)

torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def show(label):
    print(f"[{SIDE}] {label}: alloc={torch.cuda.memory_allocated()/1024/1024:7.0f}MB peak={torch.cuda.max_memory_allocated()/1024/1024:7.0f}MB")

show("baseline")
ckpts = EX("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
model, clip, vae = ckpts
torch.cuda.synchronize()
show("after CheckpointLoaderSimple")

pos = EX("CLIPTextEncode", clip=clip, text="a beautiful castle on a hill, fantasy art, highly detailed, dramatic lighting, digital painting")[0]
neg = EX("CLIPTextEncode", clip=clip, text="blurry, low quality, distorted, watermark, text")[0]
torch.cuda.synchronize()
show("after CLIPTextEncode x2")

latent = EX("EmptyLatentImage", width=512, height=512, batch_size=1)[0]
torch.cuda.synchronize()
show("after EmptyLatentImage")

sampled = EX("KSampler", model=model, positive=pos, negative=neg, latent_image=latent,
             seed=42, steps=20, cfg=8.0, sampler_name="euler", scheduler="normal", denoise=1.0)[0]
torch.cuda.synchronize()
show("after KSampler")

images = EX("VAEDecode", samples=sampled, vae=vae)[0]
torch.cuda.synchronize()
show("after VAEDecode")
