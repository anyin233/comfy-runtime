"""Image Inpainting workflow using comfy_runtime.

Pipeline:
  LoadImage → VAEEncode → SetLatentNoiseMask(mask) →
  KSampler(denoise=1.0) → VAEDecode → SaveImage

Replaces a masked region of an input image with newly generated content
guided by a text prompt. The mask is generated programmatically (center region).
All nodes are built-in to comfy_runtime.
"""

import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch

import comfy_runtime

# --- Configuration ---
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")

INPUT_IMAGE = "source.png"
CHECKPOINT = "v1-5-pruned-emaonly.safetensors"

# Inpaint prompt: what to fill the masked region with
PROMPT = "a magical glowing portal with blue and purple energy, fantasy, detailed"
NEGATIVE_PROMPT = "blurry, low quality, distorted"

STEPS = 25
CFG = 8.0
SEED = random.randint(0, 2**63)


def create_center_mask(
    height: int, width: int, margin_ratio: float = 0.25
) -> torch.Tensor:
    """Create a binary mask with 1s in the center region.

    Args:
        height: Image height.
        width: Image width.
        margin_ratio: Fraction of each edge to leave unmasked.

    Returns:
        Mask tensor of shape (height, width) with values 0 or 1.
    """
    mask = torch.zeros(height, width)
    y_start = int(height * margin_ratio)
    y_end = int(height * (1 - margin_ratio))
    x_start = int(width * margin_ratio)
    x_end = int(width * (1 - margin_ratio))
    mask[y_start:y_end, x_start:x_end] = 1.0
    return mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Configuring comfy_runtime ===")
    comfy_runtime.configure(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
        input_dir=INPUT_DIR,
    )

    # Load checkpoint
    print(f"\n=== Loading checkpoint: {CHECKPOINT} ===")
    model, clip, vae = comfy_runtime.execute_node(
        "CheckpointLoaderSimple",
        ckpt_name=CHECKPOINT,
    )

    # Load input image
    print(f"\n=== Loading input image: {INPUT_IMAGE} ===")
    image, _ = comfy_runtime.execute_node(
        "LoadImage",
        image=INPUT_IMAGE,
    )
    print(f"  Image shape: {image.shape}")
    h, w = image.shape[1], image.shape[2]

    # Create center mask programmatically
    print(f"\n=== Creating center mask ({h}x{w}) ===")
    mask = create_center_mask(h, w, margin_ratio=0.25)
    print(f"  Mask shape: {mask.shape}, masked pixels: {mask.sum().item():.0f}")

    # Encode image to latent
    print("\n=== Encoding image to latent ===")
    latent = comfy_runtime.execute_node(
        "VAEEncode",
        pixels=image,
        vae=vae,
    )[0]

    # Set noise mask on latent (tells KSampler which region to regenerate)
    print("=== Setting noise mask ===")
    masked_latent = comfy_runtime.execute_node(
        "SetLatentNoiseMask",
        samples=latent,
        mask=mask,
    )[0]

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

    # Sample: regenerate masked region
    print(f"\n=== Inpainting (steps={STEPS}, seed={SEED}) ===")
    sampled = comfy_runtime.execute_node(
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=masked_latent,
        seed=SEED,
        steps=STEPS,
        cfg=CFG,
        sampler_name="euler",
        scheduler="normal",
        denoise=1.0,
    )[0]

    # Decode and save
    print("\n=== Decoding VAE ===")
    images = comfy_runtime.execute_node(
        "VAEDecode",
        samples=sampled,
        vae=vae,
    )[0]
    if hasattr(images, "detach"):
        images = images.detach()
    print(f"  Output shape: {images.shape}")

    print("\n=== Saving image ===")
    comfy_runtime.execute_node(
        "SaveImage",
        images=images,
        filename_prefix="Inpaint",
    )
    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
