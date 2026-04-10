"""Standalone wheel verification — build, install into a fresh venv, run.

This integration test builds the current ``comfy_runtime`` wheel, installs
it into a fresh ``uv venv`` that has no access to the dev-time source tree
or any vendored ComfyUI copy, then runs the SD1.5 happy path through the
MIT compat layer.

The test proves that ``pip install comfy-runtime`` alone is enough to run
the core pipeline — no GPL code, no manual ``_vendor/`` population, no
pre-existing ComfyUI install on the host.

Gated by ``COMFY_RUNTIME_TEST_WHEEL=1`` because building a wheel + creating
a venv is ~30 s and we don't want it in every ``pytest tests/unit/`` run.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("COMFY_RUNTIME_TEST_WHEEL") != "1",
    reason="Set COMFY_RUNTIME_TEST_WHEEL=1 to run the wheel standalone test",
)


_LORA_SCRIPT = r'''
"""Standalone LoRA roundtrip test — runs inside a fresh venv with only
the installed comfy_runtime wheel on sys.path.  Proves ModelPatcher
weight deltas + LoRA apply/unpatch work end-to-end without any source
tree access."""
import torch
import torch.nn as nn

import comfy_runtime
assert "site-packages" in comfy_runtime.__file__
comfy_runtime.configure()

from comfy_runtime.compat.comfy._lora_peft import (
    apply_lora_to_patcher,
    extract_lora_deltas,
)
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher


class _Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.proj = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.diffusion_model.proj.weight.copy_(torch.zeros(8, 8))


target = _Target()
patcher = ModelPatcher(target)
original = target.diffusion_model.proj.weight.clone()

# Tiny LoRA with rank=2 / alpha=2 / all-ones factors
lora_sd = {
    "diffusion_model.proj.lora_up.weight": torch.ones(8, 2),
    "diffusion_model.proj.lora_down.weight": torch.ones(2, 8),
    "diffusion_model.proj.alpha": torch.tensor(2.0),
}
n_added = apply_lora_to_patcher(patcher, lora_sd, strength=1.0)
assert n_added == 1, f"expected 1 delta registered, got {n_added}"

patcher.patch_model()
assert not torch.allclose(target.diffusion_model.proj.weight, original), (
    "weight should change after patch_model"
)
patched = target.diffusion_model.proj.weight.clone()
# delta = (1@1)*2=2, scale alpha/rank = 2/2 = 1.0 → delta=2
expected = torch.full((8, 8), 2.0)
assert torch.allclose(patched, expected), f"unexpected patched: {patched[0]}"

patcher.unpatch_model()
assert torch.allclose(target.diffusion_model.proj.weight, original), (
    "unpatch_model must restore the original weights"
)

print("OK lora-roundtrip")
'''


# The inline smoke-test script that runs inside the fresh venv.  We build
# a tiny diffusers pipeline from scratch (no fixtures from the repo) so
# the test proves the installed wheel can run without touching the source.
_SMOKE_SCRIPT = r'''
import sys
import torch

import comfy_runtime
assert "site-packages" in comfy_runtime.__file__, (
    f"Expected installed wheel, got dev source: {comfy_runtime.__file__}"
)

comfy_runtime.configure()

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

unet = UNet2DConditionModel(
    sample_size=8, in_channels=4, out_channels=4,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    block_out_channels=(32, 64), layers_per_block=1,
    cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8,
).eval()
vae = AutoencoderKL(
    in_channels=3, out_channels=3,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(16, 32), layers_per_block=1,
    latent_channels=4, norm_num_groups=8,
).eval()
te_cfg = CLIPTextConfig(
    vocab_size=49408, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=77,
)
text_encoder = CLIPTextModel(te_cfg).eval()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

from comfy_runtime.compat.comfy.sd import CLIP, VAE
from comfy_runtime.compat.comfy.model_patcher import ModelPatcher
from comfy_runtime.compat import nodes

model = ModelPatcher(unet)
clip = CLIP(clip_model=text_encoder, tokenizer=tokenizer)
vae_w = VAE(vae_model=vae)

(pos,) = nodes.CLIPTextEncode().encode(clip, "hello")
(neg,) = nodes.CLIPTextEncode().encode(clip, "")
latent = {"samples": torch.zeros(1, 4, 8, 8)}
(sampled,) = nodes.KSampler().sample(
    model=model, seed=42, steps=2, cfg=1.0,
    sampler_name="euler", scheduler="normal",
    positive=pos, negative=neg,
    latent_image=latent, denoise=1.0,
)
(image,) = nodes.VAEDecode().decode(vae=vae_w, samples=sampled)
assert image.shape[-1] == 3
assert not torch.isnan(image).any()
print(f"OK image={tuple(image.shape)}")
'''


def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    while here != here.parent:
        if (here / "pyproject.toml").exists():
            return here
        here = here.parent
    raise RuntimeError("Could not find repo root containing pyproject.toml")


def _build_wheel_and_install(tmp_dir: str):
    """Build the wheel and install it in a fresh venv under ``tmp_dir``.

    Returns:
        Path to the venv's python executable.
    """
    repo_root = _find_repo_root()
    dist = repo_root / "dist"

    # Build the wheel fresh (idempotent — remove stale builds first)
    if dist.exists():
        shutil.rmtree(dist)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
    )
    wheels = list(dist.glob("comfy_runtime-*.whl"))
    assert len(wheels) == 1, f"Expected exactly one wheel, got {wheels}"
    wheel = wheels[0]

    venv = Path(tmp_dir) / "venv"
    subprocess.run(
        ["uv", "venv", str(venv), "--python", "3.12"],
        check=True,
        capture_output=True,
    )
    python = venv / "bin" / "python"

    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        capture_output=True,
    )
    return python


def _run_in_venv(python: Path, script: str, cwd: str) -> subprocess.CompletedProcess:
    """Run a Python script in the venv from a directory that doesn't have
    the repo source on sys.path."""
    return subprocess.run(
        [str(python), "-c", script],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def test_wheel_builds_installs_and_runs_sd15_happy_path():
    """Build wheel, install in fresh venv, run SD1.5 through MIT compat."""
    with tempfile.TemporaryDirectory() as tmp:
        python = _build_wheel_and_install(tmp)
        result = _run_in_venv(python, _SMOKE_SCRIPT, cwd=tmp)
        assert result.returncode == 0, (
            f"Standalone smoke test failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert "OK image=" in result.stdout, (
            f"Smoke test did not print expected marker:\n{result.stdout}"
        )


def test_wheel_lora_roundtrip_in_fresh_venv():
    """LoRA apply/unpatch roundtrip runs from the installed wheel."""
    with tempfile.TemporaryDirectory() as tmp:
        python = _build_wheel_and_install(tmp)
        result = _run_in_venv(python, _LORA_SCRIPT, cwd=tmp)
        assert result.returncode == 0, (
            f"LoRA roundtrip failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert "OK lora-roundtrip" in result.stdout


def test_wheel_has_no_vendor_entries():
    """The built wheel must not contain any comfy_runtime/_vendor/ files
    and must not have a compat/comfy/_vendor_bridge.py entry."""
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        python = _build_wheel_and_install(tmp)
        del python  # only needed to trigger the build

        repo_root = _find_repo_root()
        wheels = list((repo_root / "dist").glob("comfy_runtime-*.whl"))
        assert len(wheels) == 1
        wheel = wheels[0]

        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
        vendor_entries = [n for n in names if "_vendor" in n]
        assert vendor_entries == [], f"Wheel has _vendor entries: {vendor_entries}"
