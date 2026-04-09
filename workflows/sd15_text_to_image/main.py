"""SD 1.5 Text to Image workflow using comfy_runtime.

Executes the standard ComfyUI default text-to-image pipeline:
    CheckpointLoaderSimple → CLIPTextEncode (x2) → EmptyLatentImage →
    KSampler → VAEDecode → SaveImage

All nodes are built-in to comfy_runtime. No external node loading required.
"""

import os
import random
import sys

# Ensure project root is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import comfy_runtime

from workflow_utils.download_models import ensure_models

# --- Configuration ---
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

PROMPT = (
    "a beautiful castle on a hill surrounded by a moat, "
    "fantasy art, highly detailed, dramatic lighting, digital painting"
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text"

CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
WIDTH = 512
HEIGHT = 512
STEPS = 20
CFG = 8.0
SAMPLER = "euler"
SCHEDULER = "normal"
SEED = random.randint(0, 2**63)


def main():
    """Run the SD 1.5 text-to-image pipeline."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 0: Download models if needed
    print("=== Checking models ===")
    ensure_models(MODELS_DIR)

    # Step 1: Configure comfy_runtime
    print("\n=== Configuring comfy_runtime ===")
    comfy_runtime.configure(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Verify built-in nodes are available
    available = comfy_runtime.list_nodes()
    required = [
        "CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage",
        "KSampler", "VAEDecode", "SaveImage",
    ]
    for node_name in required:
        assert node_name in available, f"Missing built-in node: {node_name}"
    print(f"All {len(required)} required nodes available.")

    # Step 2: Load checkpoint (model + clip + vae)
    print(f"\n=== Loading checkpoint: {CHECKPOINT} ===")
    model, clip, vae = comfy_runtime.execute_node(
        "CheckpointLoaderSimple",
        ckpt_name=CHECKPOINT,
    )
    print("Checkpoint loaded successfully.")

    # Step 3: Encode positive prompt
    print(f"\n=== Encoding positive prompt ===")
    print(f"  Prompt: {PROMPT[:80]}...")
    positive = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=PROMPT,
    )[0]

    # Step 4: Encode negative prompt
    print(f"=== Encoding negative prompt ===")
    negative = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=NEGATIVE_PROMPT,
    )[0]

    # Step 5: Create empty latent image
    print(f"\n=== Creating empty latent ({WIDTH}x{HEIGHT}) ===")
    latent = comfy_runtime.execute_node(
        "EmptyLatentImage",
        width=WIDTH,
        height=HEIGHT,
        batch_size=1,
    )[0]
    print(f"Latent shape: {latent['samples'].shape}")

    # Step 6: Run KSampler
    print(f"\n=== Sampling (steps={STEPS}, cfg={CFG}, seed={SEED}) ===")
    sampled = comfy_runtime.execute_node(
        "KSampler",
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        seed=SEED,
        steps=STEPS,
        cfg=CFG,
        sampler_name=SAMPLER,
        scheduler=SCHEDULER,
        denoise=1.0,
    )[0]
    print("Sampling complete.")

    # Step 7: Decode latent to image
    print("\n=== Decoding VAE ===")
    images = comfy_runtime.execute_node(
        "VAEDecode",
        samples=sampled,
        vae=vae,
    )[0]
    print(f"Image shape: {images.shape}")

    # Step 8: Save image
    print("\n=== Saving image ===")
    # Detach from computation graph if needed (VAEDecode may return grad-enabled tensor)
    if hasattr(images, "detach"):
        images = images.detach()
    comfy_runtime.execute_node(
        "SaveImage",
        images=images,
        filename_prefix="SD15",
    )
    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
