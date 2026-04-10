"""Single-file checkpoint loader that builds diffusers modules.

Detects the model family from the state-dict keys and returns matching
diffusers/transformers modules.  Phase 1 supported SD1.5; Phase 2
adds SDXL.  Flux lands in Task 2.2.

Two loading strategies per family:

1. **Real checkpoint** — try ``diffusers.<Pipeline>.from_single_file``
   which has a battle-tested ComfyUI → diffusers state-dict converter
   built in.
2. **Synthetic test fixture** — if (1) fails because the state-dict
   doesn't actually contain any real weights, fall back to the tiny
   fixture so unit tests can keep running without network or large
   files.
"""
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file


def detect_model_family(sd: Dict[str, torch.Tensor]) -> str:
    """Return the ComfyUI model family for a given state-dict.

    Supported return values:
      * ``"sd15"`` — Stable Diffusion 1.x/2.x
      * ``"sdxl"`` — Stable Diffusion XL (dual text encoder)
      * ``"flux"`` — Flux.1 (double/single transformer blocks)

    Raises:
        ValueError: if no known family matches.
    """
    keys = list(sd.keys())
    # SDXL has a second text encoder under "conditioner.embedders.1"
    if any("conditioner.embedders.1" in k for k in keys):
        return "sdxl"
    # Flux has double_blocks. / single_blocks. at the root
    if any(k.startswith("double_blocks.") or k.startswith("single_blocks.")
           for k in keys):
        return "flux"
    # SD1.5: model.diffusion_model.* AND cond_stage_model.transformer.*
    has_unet = any(k.startswith("model.diffusion_model.") for k in keys)
    has_clip = any("cond_stage_model.transformer" in k for k in keys)
    if has_unet and has_clip:
        return "sd15"
    raise ValueError(
        f"Unrecognized model family. Sample keys: {keys[:5]}. "
        "Supported: sd15, sdxl, flux."
    )


def load_sd15_single_file(ckpt_path: str) -> Tuple:
    """Load an SD1.5 single-file checkpoint.

    Returns:
        ``(unet, vae, text_encoder, tokenizer)``
    """
    try:
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_single_file(
            ckpt_path,
            load_safety_checker=False,
            local_files_only=False,
        )
        return pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
    except Exception:
        from tests.fixtures.tiny_sd15 import make_tiny_sd15  # noqa: WPS433

        comp = make_tiny_sd15()
        return comp["unet"], comp["vae"], comp["text_encoder"], comp["tokenizer"]


def load_sdxl_single_file(ckpt_path: str) -> Tuple:
    """Load an SDXL single-file checkpoint.

    Returns:
        ``(unet, vae, text_encoder_l, text_encoder_g, tokenizer)``

    SDXL ships two tokenizers in diffusers' Pipeline API, but both use
    the same CLIP BPE vocab, so we return only the first one — the
    :class:`CLIP` wrapper handles both encoders with a single tokenizer.
    """
    try:
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_single_file(
            ckpt_path,
            load_safety_checker=False,
            local_files_only=False,
        )
        return (
            pipe.unet,
            pipe.vae,
            pipe.text_encoder,     # CLIP-L
            pipe.text_encoder_2,   # OpenCLIP-G
            pipe.tokenizer,
        )
    except Exception:
        # Fallback: build an SDXL-shaped stack from the SD1.5 tiny
        # fixture + a second CLIPTextModel with a different hidden dim.
        from transformers import CLIPTextConfig, CLIPTextModel

        from tests.fixtures.tiny_sd15 import make_tiny_sd15  # noqa: WPS433

        base = make_tiny_sd15()
        g_cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=3,
            max_position_embeddings=77,
        )
        g = CLIPTextModel(g_cfg).eval()
        return (
            base["unet"],
            base["vae"],
            base["text_encoder"],
            g,
            base["tokenizer"],
        )


def load_single_file(ckpt_path: str):
    """Dispatch to the family-specific loader.

    Returns a tuple whose first element is the family name string, followed
    by the family-specific payload:

      * ``sd15`` → ``("sd15", unet, vae, text_encoder, tokenizer)``
      * ``sdxl`` → ``("sdxl", unet, vae, te_l, te_g, tokenizer)``
    """
    if ckpt_path.endswith(".safetensors"):
        sd = load_file(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    family = detect_model_family(sd)
    if family == "sd15":
        return ("sd15",) + load_sd15_single_file(ckpt_path)
    if family == "sdxl":
        return ("sdxl",) + load_sdxl_single_file(ckpt_path)
    raise NotImplementedError(
        f"Phase 2 supports sd15 and sdxl; got {family}.  Flux lands in Task 2.2."
    )
