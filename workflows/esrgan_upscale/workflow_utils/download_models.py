"""Download ESRGAN upscale model from HuggingFace."""

import os
import shutil

from huggingface_hub import hf_hub_download

MODELS = {
    "upscale_models": [
        {
            "repo_id": "ai-forever/Real-ESRGAN",
            "filename": "RealESRGAN_x4.pth",
            "local_name": "RealESRGAN_x4.pth",
        },
    ],
}


def ensure_models(models_dir: str) -> None:
    """Download ESRGAN model if not already present."""
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
            if os.path.abspath(downloaded_path) != os.path.abspath(dest):
                shutil.move(downloaded_path, dest)
            print(f"[OK] Saved to {dest}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    ensure_models(models_dir)
