"""Area Composition workflow using comfy_runtime.

Pipeline:
  CLIPTextEncode (background) + ConditioningSetArea (left subject) +
  ConditioningSetArea (right subject) → ConditioningCombine →
  KSampler → VAEDecode → SaveImage

Creates a composed image with different prompts for different spatial regions.
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

# Scene composition: left side is day, right side is night
BACKGROUND_PROMPT = (
    "a beautiful landscape, wide panoramic view, highly detailed, masterpiece"
)
LEFT_PROMPT = (
    "sunny daytime scene, bright blue sky, green meadow, flowers, warm lighting"
)
RIGHT_PROMPT = (
    "nighttime scene, dark starry sky, moonlight, mysterious forest, cool blue tones"
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark"

WIDTH = 768
HEIGHT = 512
STEPS = 25
CFG = 8.0
SEED = random.randint(0, 2**63)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Configuring comfy_runtime ===")
    comfy_runtime.configure(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Load checkpoint
    print(f"\n=== Loading checkpoint ===")
    model, clip, vae = comfy_runtime.execute_node(
        "CheckpointLoaderSimple",
        ckpt_name=CHECKPOINT,
    )

    # Encode background prompt (full image)
    print("\n=== Encoding prompts ===")
    bg_cond = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=BACKGROUND_PROMPT,
    )[0]

    # Encode left area prompt
    left_cond = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=LEFT_PROMPT,
    )[0]

    # Encode right area prompt
    right_cond = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=RIGHT_PROMPT,
    )[0]

    # Encode negative
    negative = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=NEGATIVE_PROMPT,
    )[0]

    # Set areas: left half for daytime scene
    # ConditioningSetArea uses pixel coordinates: x, y, width, height
    print("\n=== Setting composition areas ===")
    left_area = comfy_runtime.execute_node(
        "ConditioningSetArea",
        conditioning=left_cond,
        width=WIDTH // 2 + 64,  # slight overlap for blending
        height=HEIGHT,
        x=0,
        y=0,
        strength=1.0,
    )[0]

    # Right half for nighttime scene
    right_area = comfy_runtime.execute_node(
        "ConditioningSetArea",
        conditioning=right_cond,
        width=WIDTH // 2 + 64,
        height=HEIGHT,
        x=WIDTH // 2 - 64,
        y=0,
        strength=1.0,
    )[0]

    # Combine all positive conditioning
    print("=== Combining conditioning ===")
    combined = comfy_runtime.execute_node(
        "ConditioningCombine",
        conditioning_1=bg_cond,
        conditioning_2=left_area,
    )[0]
    combined = comfy_runtime.execute_node(
        "ConditioningCombine",
        conditioning_1=combined,
        conditioning_2=right_area,
    )[0]

    # Create latent and sample
    print(f"\n=== Generating {WIDTH}x{HEIGHT} composed image ===")
    latent = comfy_runtime.execute_node(
        "EmptyLatentImage",
        width=WIDTH,
        height=HEIGHT,
        batch_size=1,
    )[0]

    sampled = comfy_runtime.execute_node(
        "KSampler",
        model=model,
        positive=combined,
        negative=negative,
        latent_image=latent,
        seed=SEED,
        steps=STEPS,
        cfg=CFG,
        sampler_name="euler",
        scheduler="normal",
        denoise=1.0,
    )[0]

    # Decode and save
    print("\n=== Decoding and saving ===")
    images = comfy_runtime.execute_node(
        "VAEDecode",
        samples=sampled,
        vae=vae,
    )[0]
    if hasattr(images, "detach"):
        images = images.detach()
    print(f"  Output shape: {images.shape}")

    comfy_runtime.execute_node(
        "SaveImage",
        images=images,
        filename_prefix="AreaCompose",
    )
    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
