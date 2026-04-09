"""Download Flux.2 Klein model files from HuggingFace."""

import os
import shutil

from huggingface_hub import hf_hub_download


MODELS = {
    "diffusion_models": [
        {
            "repo_id": "Comfy-Org/flux2-klein",
            "filename": "split_files/diffusion_models/flux-2-klein-base-4b.safetensors",
            "local_name": "flux-2-klein-base-4b.safetensors",
        },
    ],
    "text_encoders": [
        {
            "repo_id": "Comfy-Org/flux2-klein",
            "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
            "local_name": "qwen_3_4b.safetensors",
        },
    ],
    "vae": [
        {
            "repo_id": "Comfy-Org/flux2-dev",
            "filename": "split_files/vae/flux2-vae.safetensors",
            "local_name": "flux2-vae.safetensors",
        },
    ],
}


def ensure_models(models_dir: str) -> None:
    """Download all required models if not already present.

    Args:
        models_dir: Root directory for model storage.
    """
    for category, model_list in MODELS.items():
        category_dir = os.path.join(models_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for model in model_list:
            local_name = model.get("local_name", os.path.basename(model["filename"]))
            dest = os.path.join(category_dir, local_name)
            if os.path.exists(dest):
                print(f"[OK] {category}/{local_name} already exists")
                continue

            print(f"[DL] Downloading {local_name} from {model['repo_id']}...")
            downloaded_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                local_dir=category_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may place file in subdirectory matching repo path
            if os.path.abspath(downloaded_path) != os.path.abspath(dest):
                shutil.move(downloaded_path, dest)
            print(f"[OK] Saved to {dest}")

    # Clean up any empty subdirectories left by hf_hub_download
    for category in MODELS:
        category_dir = os.path.join(models_dir, category)
        split_dir = os.path.join(category_dir, "split_files")
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir, ignore_errors=True)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    ensure_models(models_dir)
