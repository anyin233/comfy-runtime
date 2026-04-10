import sys
SIDE = sys.argv[1]
if SIDE == "runtime":
    sys.path.insert(0, "/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark")
    import comfy_runtime
    comfy_runtime.configure(models_dir="/home/yanweiye/Project/comfy_runtime/.worktrees/e2e-benchmark/workflows/sd15_text_to_image/models")
    # Trigger bridge init
    comfy_runtime.execute_node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
else:
    sys.path.insert(0, "/home/yanweiye/Project/ComfyUI")
    import nodes

# Now check what attention path is active
import comfy.ldm.modules.attention as attn_mod
print(f"[{SIDE}] attention module: {attn_mod.__file__}")
print(f"[{SIDE}] optimized_attention: {attn_mod.optimized_attention.__module__}.{attn_mod.optimized_attention.__name__}")
print(f"[{SIDE}] xformers available: {getattr(attn_mod, 'XFORMERS_IS_AVAILABLE', 'unknown')}")
print(f"[{SIDE}] sage available: {getattr(attn_mod, 'SAGE_ATTENTION_IS_AVAILABLE', 'unknown')}")
print(f"[{SIDE}] flash available: {getattr(attn_mod, 'FLASH_ATTENTION_IS_AVAILABLE', 'unknown')}")

import comfy.cli_args as cli_args
print(f"[{SIDE}] cli_args use_split_cross_attention: {getattr(cli_args.args, 'use_split_cross_attention', 'unknown')}")
print(f"[{SIDE}] cli_args use_pytorch_cross_attention: {getattr(cli_args.args, 'use_pytorch_cross_attention', 'unknown')}")
print(f"[{SIDE}] cli_args use_sage_attention: {getattr(cli_args.args, 'use_sage_attention', 'unknown')}")
print(f"[{SIDE}] cli_args use_flash_attention: {getattr(cli_args.args, 'use_flash_attention', 'unknown')}")
print(f"[{SIDE}] cli_args use_quad_cross_attention: {getattr(cli_args.args, 'use_quad_cross_attention', 'unknown')}")

import comfy.model_management as mm
print(f"[{SIDE}] vram_state: {mm.vram_state}")
print(f"[{SIDE}] cpu_state: {mm.cpu_state}")
print(f"[{SIDE}] xformers_enabled: {mm.xformers_enabled()}")
print(f"[{SIDE}] pytorch_attention_enabled: {mm.pytorch_attention_enabled()}")
print(f"[{SIDE}] vae_dtype: {mm.vae_dtype()}")
print(f"[{SIDE}] should_use_fp16: {mm.should_use_fp16()}")
