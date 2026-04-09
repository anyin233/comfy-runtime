"""Download SD 1.5 model checkpoint from HuggingFace."""

import os
import shutil

from huggingface_hub import hf_hub_download


MODELS = {
    "checkpoints": [
        {
            "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "filename": "v1-5-pruned-emaonly.safetensors",
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
            dest = os.path.join(category_dir, model["filename"])
            if os.path.exists(dest):
                print(f"[OK] {category}/{model['filename']} already exists")
                continue

            print(f"[DL] Downloading {model['filename']} from {model['repo_id']}...")
            downloaded_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                local_dir=category_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may place the file in a subdirectory; move it if needed
            if os.path.abspath(downloaded_path) != os.path.abspath(dest):
                shutil.move(downloaded_path, dest)
            print(f"[OK] Saved to {dest}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    ensure_models(models_dir)
