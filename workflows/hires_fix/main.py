"""2-Pass Hires Fix workflow using comfy_runtime.

Pipeline:
  Pass 1: Text → EmptyLatentImage(512x512) → KSampler → low-res latent
  Pass 2: LatentUpscale(2x) → KSampler(denoise=0.5) → VAEDecode → SaveImage

Generates at low resolution first, upscales in latent space, then
refines with a second sampling pass for crisp high-resolution output.
All nodes are built-in to comfy_runtime.
"""

import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import comfy_runtime

# --- Configuration ---
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
PROMPT = "a majestic snow-capped mountain landscape, alpine meadow with wildflowers, crystal clear lake reflection, golden hour lighting, photorealistic"
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, cartoon"

# Pass 1: Low-res generation
PASS1_WIDTH = 512
PASS1_HEIGHT = 512
PASS1_STEPS = 20
PASS1_CFG = 8.0

# Pass 2: High-res refinement
UPSCALE_FACTOR = 2.0  # 512→1024
PASS2_DENOISE = 0.5
PASS2_STEPS = 15
PASS2_CFG = 8.0

SEED = random.randint(0, 2**63)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Configuring comfy_runtime ===")
    comfy_runtime.configure(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Load checkpoint
    print(f"\n=== Loading checkpoint: {CHECKPOINT} ===")
    model, clip, vae = comfy_runtime.execute_node(
        "CheckpointLoaderSimple",
        ckpt_name=CHECKPOINT,
    )

    # Encode prompts
    print("\n=== Encoding prompts ===")
    positive = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=PROMPT,
    )[0]
    negative = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=NEGATIVE_PROMPT,
    )[0]

    # === Pass 1: Low-resolution generation ===
    print(f"\n=== Pass 1: Generate {PASS1_WIDTH}x{PASS1_HEIGHT} ===")
    latent = comfy_runtime.execute_node(
        "EmptyLatentImage",
        width=PASS1_WIDTH,
        height=PASS1_HEIGHT,
        batch_size=1,
    )[0]

    sampled_lr = comfy_runtime.execute_node(
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        seed=SEED,
        steps=PASS1_STEPS,
        cfg=PASS1_CFG,
        sampler_name="euler",
        scheduler="normal",
        denoise=1.0,
    )[0]
    print("  Pass 1 complete.")

    # === Upscale latent ===
    target_w = int(PASS1_WIDTH * UPSCALE_FACTOR)
    target_h = int(PASS1_HEIGHT * UPSCALE_FACTOR)
    print(f"\n=== Upscaling latent to {target_w}x{target_h} ===")
    upscaled_latent = comfy_runtime.execute_node(
        "LatentUpscale",
        samples=sampled_lr,
        upscale_method="nearest-exact",
        width=target_w,
        height=target_h,
        crop="disabled",
    )[0]

    # === Pass 2: High-resolution refinement ===
    print(
        f"\n=== Pass 2: Refine at {target_w}x{target_h} (denoise={PASS2_DENOISE}) ==="
    )
    sampled_hr = comfy_runtime.execute_node(
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=upscaled_latent,
        seed=SEED,
        steps=PASS2_STEPS,
        cfg=PASS2_CFG,
        sampler_name="euler",
        scheduler="normal",
        denoise=PASS2_DENOISE,
    )[0]
    print("  Pass 2 complete.")

    # Decode and save
    print("\n=== Decoding VAE ===")
    images = comfy_runtime.execute_node(
        "VAEDecode",
        samples=sampled_hr,
        vae=vae,
    )[0]
    if hasattr(images, "detach"):
        images = images.detach()
    print(f"  Output shape: {images.shape}")

    print("\n=== Saving image ===")
    comfy_runtime.execute_node(
        "SaveImage",
        images=images,
        filename_prefix="HiresFix",
    )
    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
