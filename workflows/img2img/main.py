"""Image-to-Image style transfer workflow using comfy_runtime.

Pipeline: LoadImage → VAEEncode → KSampler (denoise<1) → VAEDecode → SaveImage

Takes an input image and transforms it using a text prompt while preserving
the overall structure (controlled by denoise strength).
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
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")

INPUT_IMAGE = "source.png"
CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
PROMPT = "a beautiful castle on a hill, oil painting style, impressionist, vibrant colors, thick brushstrokes"
NEGATIVE_PROMPT = "photo, realistic, blurry, low quality"
DENOISE = 0.75  # Lower = more faithful to input; higher = more creative
STEPS = 20
CFG = 7.5
SEED = random.randint(0, 2**63)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Configuring comfy_runtime ===")
    comfy_runtime.configure(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
        input_dir=INPUT_DIR,
    )

    # Step 1: Load checkpoint
    print(f"\n=== Loading checkpoint: {CHECKPOINT} ===")
    model, clip, vae = comfy_runtime.execute_node(
        "CheckpointLoaderSimple",
        ckpt_name=CHECKPOINT,
    )

    # Step 2: Load input image
    print(f"\n=== Loading input image: {INPUT_IMAGE} ===")
    image, mask = comfy_runtime.execute_node(
        "LoadImage",
        image=INPUT_IMAGE,
    )
    print(f"  Image shape: {image.shape}")

    # Step 3: Encode image to latent
    print("\n=== Encoding image to latent (VAEEncode) ===")
    latent = comfy_runtime.execute_node(
        "VAEEncode",
        pixels=image,
        vae=vae,
    )[0]

    # Step 4: Encode prompts
    print(f"\n=== Encoding prompts ===")
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

    # Step 5: KSampler with denoise < 1.0 (preserves input structure)
    print(f"\n=== Sampling (denoise={DENOISE}, steps={STEPS}, seed={SEED}) ===")
    sampled = comfy_runtime.execute_node(
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        seed=SEED,
        steps=STEPS,
        cfg=CFG,
        sampler_name="euler",
        scheduler="normal",
        denoise=DENOISE,
    )[0]

    # Step 6: Decode latent to image
    print("\n=== Decoding VAE ===")
    images = comfy_runtime.execute_node(
        "VAEDecode",
        samples=sampled,
        vae=vae,
    )[0]
    if hasattr(images, "detach"):
        images = images.detach()
    print(f"  Output shape: {images.shape}")

    # Step 7: Save image
    print("\n=== Saving image ===")
    comfy_runtime.execute_node(
        "SaveImage",
        images=images,
        filename_prefix="Img2Img",
    )
    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
