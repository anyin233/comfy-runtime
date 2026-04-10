"""Flux.2 Klein 4B Text to Image workflow using comfy_runtime.

Executes the Flux.2 Klein text-to-image pipeline with advanced sampler:
    UNETLoader + CLIPLoader + VAELoader → CLIPTextEncode (x2) →
    CFGGuider → KSamplerSelect + Flux2Scheduler + EmptyFlux2LatentImage +
    RandomNoise → SamplerCustomAdvanced → VAEDecode → SaveImage

Built-in nodes: UNETLoader, CLIPLoader, VAELoader, CLIPTextEncode, VAEDecode, SaveImage
Extra nodes (loaded from nodes/): CFGGuider, KSamplerSelect, RandomNoise,
    SamplerCustomAdvanced, Flux2Scheduler, EmptyFlux2LatentImage
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
NODES_DIR = os.path.join(SCRIPT_DIR, "nodes")

PROMPT = (
    "A hedgehog wearing a tiny party hat surrounded by confetti, "
    "early digital camera style, slight noise, flash photography, "
    "candid moment, 2000s digicam aesthetic, festive birthday celebration atmosphere"
)

UNET_MODEL = "flux-2-klein-base-4b.safetensors"
CLIP_MODEL = "qwen_3_4b.safetensors"
VAE_MODEL = "flux2-vae.safetensors"

WIDTH = 1024
HEIGHT = 1024
STEPS = 20
CFG = 5.0
SAMPLER_NAME = "euler"
SEED = random.randint(0, 2**63)


def main():
    """Run the Flux.2 Klein text-to-image pipeline."""
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

    # Step 2: Load extra nodes from nodes/ directory
    print("\n=== Loading extra nodes ===")
    for node_file in ["nodes_custom_sampler.py", "nodes_flux.py"]:
        path = os.path.join(NODES_DIR, node_file)
        comfy_runtime.load_nodes_from_path(path)
        print(f"  Loaded: {node_file}")

    # Verify all required nodes are available
    available = comfy_runtime.list_nodes()
    required = [
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncode",
        "VAEDecode",
        "SaveImage",  # built-in
        "CFGGuider",
        "KSamplerSelect",
        "RandomNoise",
        "SamplerCustomAdvanced",
        "Flux2Scheduler",
        "EmptyFlux2LatentImage",  # extra
    ]
    for node_name in required:
        assert node_name in available, f"Missing node: {node_name}"
    print(f"All {len(required)} required nodes available.")

    # Step 3: Load UNET (diffusion model)
    print(f"\n=== Loading UNET: {UNET_MODEL} ===")
    model = comfy_runtime.execute_node(
        "UNETLoader",
        unet_name=UNET_MODEL,
        weight_dtype="default",
    )[0]
    print("UNET loaded.")

    # Step 4: Load CLIP (text encoder)
    print(f"\n=== Loading CLIP: {CLIP_MODEL} ===")
    clip = comfy_runtime.execute_node(
        "CLIPLoader",
        clip_name=CLIP_MODEL,
        type="flux2",
    )[0]
    print("CLIP loaded.")

    # Step 5: Load VAE
    print(f"\n=== Loading VAE: {VAE_MODEL} ===")
    vae = comfy_runtime.execute_node(
        "VAELoader",
        vae_name=VAE_MODEL,
    )[0]
    print("VAE loaded.")

    # Step 6: Encode positive prompt
    print(f"\n=== Encoding positive prompt ===")
    print(f"  Prompt: {PROMPT[:80]}...")
    positive = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text=PROMPT,
    )[0]

    # Step 7: Encode negative prompt (empty for Flux)
    print("=== Encoding negative prompt (empty) ===")
    negative = comfy_runtime.execute_node(
        "CLIPTextEncode",
        clip=clip,
        text="",
    )[0]

    # Evict the text encoder (Qwen 3 4B, ~8 GB) before sampling loads the
    # diffusion model weights. Without this, the 24 GB VRAM ceiling forces
    # VAE decode into tiled fallback mode and the whole workflow pays a
    # ~700 ms penalty. See docs/benchmarks/profiling_findings.md.
    #
    # Order matters: drop the user-level ``clip`` ref BEFORE calling
    # unload_all_models(). comfy_runtime.free_memory() ends with
    # soft_empty_cache(); that call can only return blocks to the driver
    # if no strong refs still pin them. Doing ``del clip`` after the
    # unload would leave the cached pool looking full to the VAE decode
    # that follows, re-triggering the tiled-decode fallback it's meant
    # to avoid.
    print("=== Releasing text encoder ===")
    del clip
    comfy_runtime.unload_all_models()

    # Step 8: Create CFG Guider
    print(f"\n=== Creating CFG Guider (cfg={CFG}) ===")
    guider = comfy_runtime.execute_node(
        "CFGGuider",
        model=model,
        positive=positive,
        negative=negative,
        cfg=CFG,
    )[0]

    # Step 9: Select sampler
    print(f"=== Selecting sampler: {SAMPLER_NAME} ===")
    sampler = comfy_runtime.execute_node(
        "KSamplerSelect",
        sampler_name=SAMPLER_NAME,
    )[0]

    # Step 10: Compute sigmas (Flux2 scheduler)
    print(f"=== Computing sigmas (steps={STEPS}, {WIDTH}x{HEIGHT}) ===")
    sigmas = comfy_runtime.execute_node(
        "Flux2Scheduler",
        steps=STEPS,
        width=WIDTH,
        height=HEIGHT,
    )[0]

    # Step 11: Create empty latent image
    print(f"=== Creating empty Flux2 latent ({WIDTH}x{HEIGHT}) ===")
    latent = comfy_runtime.execute_node(
        "EmptyFlux2LatentImage",
        width=WIDTH,
        height=HEIGHT,
        batch_size=1,
    )[0]

    # Step 12: Generate random noise
    print(f"=== Generating noise (seed={SEED}) ===")
    noise = comfy_runtime.execute_node(
        "RandomNoise",
        noise_seed=SEED,
    )[0]

    # Step 13: Run advanced sampler
    print(f"\n=== Running SamplerCustomAdvanced ===")
    output_latent, denoised_output = comfy_runtime.execute_node(
        "SamplerCustomAdvanced",
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=latent,
    )
    print("Sampling complete.")

    # Evict the diffusion model (Flux.2 Klein 4B, ~8 GB) before VAE decode.
    # Without this, VAE decode hits the 24 GB VRAM ceiling and falls back
    # to tiled decoding, which is ~4x slower.
    #
    # ``guider`` holds a strong ref to ``model`` via CFGGuider.model, so
    # both must be dropped before the unload call. ``positive`` and
    # ``negative`` are conditioning tensors, not model refs, but we drop
    # them here for symmetry — they're small but reachable via ``guider``
    # so cleanup is cheaper if we release the whole chain at once.
    print("=== Releasing diffusion model ===")
    del guider
    del model
    del positive
    del negative
    comfy_runtime.unload_all_models()

    # Step 14: Decode latent to image
    print("\n=== Decoding VAE ===")
    images = comfy_runtime.execute_node(
        "VAEDecode",
        samples=output_latent,
        vae=vae,
    )[0]
    print(f"Image shape: {images.shape}")

    # Step 15: Save image
    print("\n=== Saving image ===")
    comfy_runtime.execute_node(
        "SaveImage",
        images=images,
        filename_prefix="Flux2-Klein",
    )
    print(f"\nDone! Output saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
