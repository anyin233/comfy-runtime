"""ESRGAN Image Upscaling workflow using comfy_runtime.

Pipeline: LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage

Takes an input image and upscales it 4x using Real-ESRGAN.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import comfy_runtime

from workflow_utils.download_models import ensure_models

# --- Configuration ---
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
NODES_DIR = os.path.join(SCRIPT_DIR, "nodes")

INPUT_IMAGE = "source.png"
UPSCALE_MODEL = "RealESRGAN_x4.pth"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download model if needed
    print("=== Checking models ===")
    ensure_models(MODELS_DIR)

    print("\n=== Configuring comfy_runtime ===")
    comfy_runtime.configure(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR,
        input_dir=INPUT_DIR,
    )

    # Load extra nodes for upscaling
    print("\n=== Loading upscale nodes ===")
    comfy_runtime.load_nodes_from_path(os.path.join(NODES_DIR, "nodes_upscale_model.py"))

    # Step 1: Load input image
    print(f"\n=== Loading input image: {INPUT_IMAGE} ===")
    image, mask = comfy_runtime.execute_node(
        "LoadImage", image=INPUT_IMAGE,
    )
    print(f"  Input shape: {image.shape}")

    # Step 2: Load upscale model
    print(f"\n=== Loading upscale model: {UPSCALE_MODEL} ===")
    upscale_model = comfy_runtime.execute_node(
        "UpscaleModelLoader", model_name=UPSCALE_MODEL,
    )[0]

    # Step 3: Upscale image
    print("\n=== Upscaling image (4x) ===")
    upscaled = comfy_runtime.execute_node(
        "ImageUpscaleWithModel", upscale_model=upscale_model, image=image,
    )[0]
    print(f"  Output shape: {upscaled.shape}")

    # Step 4: Save result
    print("\n=== Saving image ===")
    comfy_runtime.execute_node(
        "SaveImage", images=upscaled, filename_prefix="ESRGAN_4x",
    )
    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
