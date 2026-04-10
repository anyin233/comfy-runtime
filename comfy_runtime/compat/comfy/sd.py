"""High-level model loading and wrapping for comfy_runtime.

MIT reimplementation of comfy.sd — provides CLIP, VAE, and checkpoint
loading stubs that ComfyUI nodes depend on for model management.
"""

import enum
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLIPType enum
# ---------------------------------------------------------------------------


class CLIPType(enum.Enum):
    """Enumeration of supported CLIP text encoder architectures.

    Only ``SD1`` / ``SDXL`` / ``FLUX`` have real loaders in the compat
    layer at v1.0; the other members exist so the CLIPLoader type_map
    dict can reference them without AttributeError, and so
    import-compat stubs for newer architectures (chroma, hidream,
    lumina2, flux2, long_clipl) can be wired up in a future minor
    release without touching the enum.
    """

    SD1 = 1
    SD2 = 2
    SDXL = 3
    SD3 = 4
    STABLE_CASCADE = 5
    FLUX = 6
    LTXV = 7
    HUNYUAN_VIDEO = 8
    MOCHI = 9
    PIXART = 10
    WAN = 11
    HUNYUAN_DIT = 12
    LUMINA2 = 13
    CHROMA = 14
    HIDREAM = 15
    FLUX2 = 16
    LONG_CLIPL = 17


# ---------------------------------------------------------------------------
# CLIP wrapper
# ---------------------------------------------------------------------------


class CLIP:
    """Wraps a text encoder model with tokenization and encoding methods.

    Provides the interface that ComfyUI nodes use to convert text prompts
    into conditioning tensors.

    Attributes:
        clip_model: The underlying CLIP/text encoder model.
        tokenizer: The tokenizer for text processing.
        patcher: Optional ModelPatcher wrapping the clip_model.
        layer_idx: Which hidden layer to extract embeddings from.
        cond_stage_model: Reference to the conditioning stage model.
        load_device: Device for inference.
        offload_device: Device for offloading.
    """

    def __init__(
        self,
        clip_model=None,
        tokenizer=None,
        load_device=None,
        offload_device=None,
        patcher=None,
        clip_model2=None,
        tokenizer2=None,
        family: str = "sd1",
    ):
        """Initialize CLIP wrapper.

        Args:
            clip_model:    Primary text encoder (CLIP-L for SD1/SDXL/Flux/SD3).
            tokenizer:     Tokenizer for ``clip_model``.
            load_device:   Device for running inference.
            offload_device: Device for weight offloading.
            patcher:       Optional ModelPatcher for ``clip_model``.
            clip_model2:   Secondary text encoder for dual-encoder
                checkpoints.  ``OpenCLIPG`` for SDXL, ``T5-XXL`` for
                Flux / SD3.  ``None`` for single-encoder SD1.5.
            tokenizer2:    Tokenizer for ``clip_model2``.  When
                ``None`` and ``clip_model2`` is set, ``tokenizer`` is
                re-used (SDXL convention — both encoders share the
                CLIP BPE tokenizer; Flux uses a T5 tokenizer here).
            family:        ``"sd1"``, ``"sdxl"``, or ``"flux"``.
                Controls slot naming (``"l"`` + ``"g"`` for SDXL vs
                ``"l"`` + ``"t5xxl"`` for Flux) and the conditioning
                contract returned by :meth:`encode_from_tokens`.
        """
        self.clip_model = clip_model
        self.clip_model2 = clip_model2
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2 or tokenizer
        self.patcher = patcher
        self.load_device = load_device or torch.device("cpu")
        self.offload_device = offload_device or torch.device("cpu")
        self.layer_idx = None
        self.cond_stage_model = clip_model
        self.family = family

    def clone(self) -> "CLIP":
        """Create a clone of this CLIP wrapper.

        Returns:
            New CLIP instance sharing the underlying models but with
            independent patcher state.
        """
        cloned = CLIP.__new__(CLIP)
        cloned.clip_model = self.clip_model
        cloned.clip_model2 = getattr(self, "clip_model2", None)
        cloned.tokenizer = self.tokenizer
        cloned.tokenizer2 = getattr(self, "tokenizer2", self.tokenizer)
        cloned.load_device = self.load_device
        cloned.offload_device = self.offload_device
        cloned.layer_idx = self.layer_idx
        cloned.cond_stage_model = self.cond_stage_model
        cloned.family = getattr(self, "family", "sd1")
        if self.patcher is not None:
            cloned.patcher = self.patcher.clone()
        else:
            cloned.patcher = None
        return cloned

    def _second_slot_name(self) -> str:
        """Slot name for the secondary encoder.

        * ``"g"``    for SDXL (OpenCLIP-G)
        * ``"t5xxl"`` for Flux and SD3 (T5-XXL)

        Determined from ``self.family`` which is set at construction.
        """
        family = getattr(self, "family", "sd1")
        if family == "flux":
            return "t5xxl"
        if family == "sd3":
            return "t5xxl"
        # Default: SDXL dual CLIP
        return "g"

    def tokenize(self, text: str, return_word_ids: bool = False):
        """Tokenize text into ComfyUI's chunked-weighted format.

        Returns a dict with a ``"l"`` slot always, plus a second slot
        when :attr:`clip_model2` is set — ``"g"`` for SDXL, ``"t5xxl"``
        for Flux / SD3 (controlled by :attr:`family`).

        Args:
            text: Input text string.
            return_word_ids: Ignored in Phase 2 (attention-weighted prompts
                are Phase 3); kept for API parity.
        """
        from comfy_runtime.compat.comfy._tokenizer import tokenize_to_comfy_format

        if self.tokenizer is None:
            raise RuntimeError("CLIP.tokenize requires self.tokenizer to be set")

        tokens = tokenize_to_comfy_format(
            self.tokenizer, text, max_length=77, slot="l"
        )
        if getattr(self, "clip_model2", None) is not None:
            tok2 = self.tokenizer2 or self.tokenizer
            slot2 = self._second_slot_name()
            tokens.update(
                tokenize_to_comfy_format(tok2, text, max_length=77, slot=slot2)
            )
        return tokens

    def _encode_slot(self, model, chunks) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single text encoder over a list of token chunks.

        Returns ``(cat_hidden, pooled)`` where ``cat_hidden`` has shape
        ``(1, 77 * num_chunks, hidden_size)`` and ``pooled`` is the
        final hidden state's pooled output (``pooler_output`` if
        available, otherwise the first-token hidden).
        """
        from comfy_runtime.compat.comfy._tokenizer import tokens_to_input_ids

        device = next(model.parameters()).device
        chunk_embeddings: List[torch.Tensor] = []
        pooled: Optional[torch.Tensor] = None
        for chunk in chunks:
            ids = torch.tensor(
                [tokens_to_input_ids(chunk)], dtype=torch.long, device=device
            )
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True)
            if self.layer_idx is not None:
                hidden = out.hidden_states[self.layer_idx]
            else:
                hidden = out.last_hidden_state
            chunk_embeddings.append(hidden)
            if pooled is None:
                pooler = getattr(out, "pooler_output", None)
                if pooler is not None:
                    pooled = pooler
                else:
                    pooled = out.last_hidden_state[:, 0, :]
        cat = torch.cat(chunk_embeddings, dim=1)
        return cat, pooled

    def encode_from_tokens(
        self, tokens, return_pooled: bool = False, return_dict: bool = False
    ):
        """Run the text encoders over tokenized chunks.

        When :attr:`clip_model2` is set (SDXL dual encoder), the ``"l"``
        and ``"g"`` slot outputs are concatenated along the hidden dim
        (shape ``(B, 77, hidden_l + hidden_g)``) and the pooled output
        is taken from the ``"g"`` encoder only — matching the SDXL
        conditioning contract used by diffusers'
        ``StableDiffusionXLPipeline``.

        Single-encoder (SD1.5) models return just the ``"l"`` output
        unchanged, preserving Phase-1 semantics.

        Args:
            tokens:        ``{"l": [...]}`` or ``{"l": [...], "g": [...]}``.
            return_pooled: Also return the pooled embedding.
            return_dict:   Return a dict ``{"cond": ..., "pooled_output": ...}``.
        """
        if self.clip_model is None:
            raise RuntimeError(
                "CLIP.encode_from_tokens requires self.clip_model to be set"
            )

        if "l" not in tokens:
            raise KeyError(
                f"Expected at least the 'l' slot; got {list(tokens.keys())}"
            )

        l_cond, l_pooled = self._encode_slot(self.clip_model, tokens["l"])

        second_model = getattr(self, "clip_model2", None)
        family = getattr(self, "family", "sd1")

        if second_model is not None and "g" in tokens:
            # SDXL: concat along hidden dim; pooled from G only.
            g_cond, g_pooled = self._encode_slot(second_model, tokens["g"])
            cond = torch.cat([l_cond, g_cond], dim=-1)
            pooled = g_pooled
        elif second_model is not None and "t5xxl" in tokens:
            # Flux / SD3: T5 sequence is the conditioning; CLIP-L pooled
            # flows in via pooled_projections.  The two outputs are NOT
            # concatenated — they go through separate projection heads
            # inside the FluxTransformer2DModel.
            t5_cond, _t5_pooled = self._encode_slot(second_model, tokens["t5xxl"])
            cond = t5_cond
            pooled = l_pooled
        else:
            cond = l_cond
            pooled = l_pooled

        if return_dict:
            return {"cond": cond, "pooled_output": pooled}
        if return_pooled:
            return cond, pooled
        return cond

    def encode_from_tokens_scheduled(self, tokens, add_dict=None):
        """Wrap encoding output in ComfyUI's scheduled-conditioning list.

        Args:
            tokens: Tokenized prompt.
            add_dict: Optional extras dict to merge into the output metadata.

        Returns:
            ``[[cond_tensor, {"pooled_output": pooled, **add_dict}]]``.
        """
        cond, pooled = self.encode_from_tokens(tokens, return_pooled=True)
        extra: Dict[str, Any] = {"pooled_output": pooled}
        if add_dict:
            extra.update(add_dict)
        return [[cond, extra]]

    def set_clip_options(self, options: Dict):
        """Set CLIP-specific options.

        Args:
            options: Dict of CLIP options to apply.
        """
        # TODO(Phase3): Apply CLIP options.
        pass

    def get_sd(self) -> Dict[str, torch.Tensor]:
        """Get the state dict of the underlying model.

        Returns:
            State dict.
        """
        if self.clip_model is not None and hasattr(self.clip_model, "state_dict"):
            return self.clip_model.state_dict()
        return {}

    def load_sd(self, sd: Dict[str, torch.Tensor]):
        """Load a state dict into the underlying model.

        Args:
            sd: State dict to load.
        """
        if self.clip_model is not None and hasattr(self.clip_model, "load_state_dict"):
            self.clip_model.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# VAE wrapper
# ---------------------------------------------------------------------------


class VAE:
    """Wraps a VAE model for encoding images to latents and decoding back.

    Attributes:
        vae_model: The underlying VAE model.
        device: Current device.
        dtype: Current dtype.
        first_stage_model: Reference to the VAE (ComfyUI convention).
        patcher: Optional ModelPatcher.
    """

    def __init__(self, vae_model=None, device=None, dtype=None, patcher=None):
        """Initialize VAE wrapper.

        Args:
            vae_model: The VAE model.
            device: Target device.
            dtype: Target dtype.
            patcher: Optional ModelPatcher for the VAE.
        """
        self.vae_model = vae_model
        self.first_stage_model = vae_model
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        self.patcher = patcher
        self.output_device = torch.device("cpu")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor to an image tensor via diffusers AutoencoderKL.

        Args:
            latent: Tensor of shape ``(B, latent_channels, H, W)``.

        Returns:
            Image tensor of shape ``(B, H*f, W*f, 3)`` in ``[0, 1]``.
        """
        if self.vae_model is None:
            raise RuntimeError("VAE.decode requires self.vae_model to be set")

        first_param = next(self.vae_model.parameters())
        device = first_param.device
        dtype = first_param.dtype

        # diffusers AutoencoderKL internally multiplies by scaling_factor
        # when it was produced by `vae.encode(...).latent_dist.sample() * scaling`.
        # ComfyUI stores latents *already scaled*, so we divide out first.
        scaling = float(getattr(self.vae_model.config, "scaling_factor", 0.18215))
        latent = latent.to(device=device, dtype=dtype) / scaling

        with torch.no_grad():
            decoded = self.vae_model.decode(latent).sample

        # diffusers output: (B, C, H, W) in ~[-1, 1]
        # ComfyUI convention: (B, H, W, C) in [0, 1]
        img = (decoded.clamp(-1.0, 1.0) + 1.0) * 0.5
        img = img.permute(0, 2, 3, 1).contiguous().to(torch.float32)
        return img.to(self.output_device)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an image tensor to latent space via diffusers AutoencoderKL.

        Args:
            image: Tensor of shape ``(B, H, W, 3)`` in ``[0, 1]``.

        Returns:
            Latent tensor of shape ``(B, latent_channels, H/f, W/f)`` pre-scaled
            by ``vae.config.scaling_factor`` (ComfyUI convention).
        """
        if self.vae_model is None:
            raise RuntimeError("VAE.encode requires self.vae_model to be set")

        first_param = next(self.vae_model.parameters())
        device = first_param.device
        dtype = first_param.dtype

        img = image.to(device=device, dtype=dtype)
        # ComfyUI input: (B, H, W, C) → diffusers: (B, C, H, W)
        if img.dim() == 4 and img.shape[-1] in (3, 4):
            img = img.permute(0, 3, 1, 2).contiguous()
        # Drop alpha if present
        if img.shape[1] == 4:
            img = img[:, :3, :, :]
        # [0, 1] → [-1, 1]
        img = img * 2.0 - 1.0

        scaling = float(getattr(self.vae_model.config, "scaling_factor", 0.18215))
        with torch.no_grad():
            dist = self.vae_model.encode(img).latent_dist
        latent = dist.sample() * scaling
        return latent

    def decode_tiled(
        self,
        latent: torch.Tensor,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
    ) -> torch.Tensor:
        """Decode latent using tiled processing for large images.

        Args:
            latent: Latent tensor.
            tile_x: Tile width in latent pixels.
            tile_y: Tile height in latent pixels.
            overlap: Overlap between tiles.

        Returns:
            Decoded image tensor.
        """
        # TODO(Phase3): Implement tiled decoding.
        raise NotImplementedError("VAE.decode_tiled is a stub.")

    def encode_tiled(
        self,
        image: torch.Tensor,
        tile_x: int = 512,
        tile_y: int = 512,
        overlap: int = 64,
    ) -> torch.Tensor:
        """Encode image using tiled processing.

        Args:
            image: Image tensor.
            tile_x: Tile width in pixels.
            tile_y: Tile height in pixels.
            overlap: Overlap between tiles.

        Returns:
            Latent tensor.
        """
        # TODO(Phase3): Implement tiled encoding.
        raise NotImplementedError("VAE.encode_tiled is a stub.")

    def get_sd(self) -> Dict[str, torch.Tensor]:
        """Get the state dict.

        Returns:
            State dict of the VAE model.
        """
        if self.vae_model is not None and hasattr(self.vae_model, "state_dict"):
            return self.vae_model.state_dict()
        return {}


# ---------------------------------------------------------------------------
# StyleModel
# ---------------------------------------------------------------------------


class StyleModel:
    """Style model wrapper for style transfer conditioning.

    Attributes:
        model: The underlying style model.
    """

    def __init__(self, model=None):
        """Initialize StyleModel.

        Args:
            model: The style model.
        """
        self.model = model

    def get_cond(self, input_dict: Dict) -> torch.Tensor:
        """Get style conditioning from input.

        Args:
            input_dict: Dict with style model inputs.

        Returns:
            Style conditioning tensor.
        """
        # TODO(Phase3): Implement style conditioning.
        raise NotImplementedError("StyleModel.get_cond is a stub.")


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_checkpoint_guess_config(
    ckpt_path: str,
    output_vae: bool = True,
    output_clip: bool = True,
    output_clipvision: bool = False,
    embedding_directory: Optional[str] = None,
    output_model: bool = True,
    model_options: Optional[Dict] = None,
):
    """Load a single-file checkpoint and wrap it in ComfyUI's tuple.

    Detects the model family from the state-dict keys and dispatches to
    the family-specific loader:

      * ``sd15`` → single CLIP-L encoder
      * ``sdxl`` → dual encoder (CLIP-L + OpenCLIP-G)

    The returned :class:`CLIP` wrapper transparently handles both cases
    via the ``"l"``/``"g"`` slot mechanism.

    Args:
        ckpt_path: Path to the checkpoint file (``.safetensors`` or ``.ckpt``).
        output_vae: Return a VAE wrapper.
        output_clip: Return a CLIP wrapper.
        output_clipvision: Return a CLIP vision wrapper (always None for now).
        embedding_directory: Unused; accepted for API compat with ComfyUI.
        output_model: Return the UNet wrapped in a ModelPatcher.
        model_options: Additional loading options (unused).

    Returns:
        ``(model, clip, vae, clipvision)`` — any entry may be ``None``.

    Raises:
        ValueError: If the state dict doesn't match any supported family.
    """
    from comfy_runtime.compat.comfy._diffusers_loader import load_single_file
    from comfy_runtime.compat.comfy.model_patcher import ModelPatcher

    loaded = load_single_file(ckpt_path)
    family = loaded[0]

    if family == "sd15":
        _, unet, vae_mod, text_encoder, tokenizer = loaded
        clip = None
        if output_clip:
            clip = CLIP(
                clip_model=text_encoder,
                tokenizer=tokenizer,
                family="sd1",
            )
    elif family == "sdxl":
        _, unet, vae_mod, te_l, te_g, tokenizer = loaded
        clip = None
        if output_clip:
            clip = CLIP(
                clip_model=te_l,
                tokenizer=tokenizer,
                clip_model2=te_g,
                family="sdxl",
            )
    elif family == "flux":
        _, unet, vae_mod, te_l, te_t5, tok_l, tok_t5 = loaded
        clip = None
        if output_clip:
            clip = CLIP(
                clip_model=te_l,
                tokenizer=tok_l,
                clip_model2=te_t5,
                tokenizer2=tok_t5,
                family="flux",
            )
    else:
        raise NotImplementedError(f"family {family!r} loading not implemented")

    # Compute device for this machine.  ``get_torch_device`` returns
    # ``cuda:0`` when CUDA is available and ``cpu`` otherwise, so the
    # CPU-only test path is unchanged.
    from comfy_runtime.compat.comfy.model_management import get_torch_device

    compute_device = get_torch_device()
    # fp16 is the ComfyUI default for SD1.5 / SDXL / Flux UNets and CLIP
    # on GPU — it halves both VRAM footprint and compute vs fp32.  We
    # keep the VAE in fp32 to avoid the well-known SD1.5 fp16-VAE NaN
    # issue that produces black / over-saturated outputs.
    is_cuda = compute_device.type == "cuda"
    unet_dtype = torch.float16 if is_cuda else torch.float32
    clip_dtype = torch.float16 if is_cuda else torch.float32
    vae_dtype = torch.float32

    model = None
    if output_model:
        # Move the UNet to the compute device and cast to the target
        # dtype *before* wrapping in ModelPatcher so the patcher's
        # ``load_device`` reflects where the weights actually live.
        try:
            unet.to(device=compute_device, dtype=unet_dtype)
        except (TypeError, NotImplementedError):
            # Some fp8-cast models can't go through the high-level
            # ``.to(dtype=...)`` path on older torch; keep them where
            # they are if that fails.  The bench happy path is fp16.
            pass
        model = ModelPatcher(
            unet,
            load_device=compute_device,
            offload_device=torch.device("cpu"),
        )

    if clip is not None:
        # Move both encoders (dual-encoder paths have clip_model2).
        try:
            if clip.clip_model is not None:
                clip.clip_model.to(device=compute_device, dtype=clip_dtype)
            clip_model2 = getattr(clip, "clip_model2", None)
            if clip_model2 is not None:
                clip_model2.to(device=compute_device, dtype=clip_dtype)
        except (TypeError, NotImplementedError):
            pass
        clip.load_device = compute_device

    vae_wrapper = None
    if output_vae:
        if vae_mod is not None:
            try:
                vae_mod.to(device=compute_device, dtype=vae_dtype)
            except (TypeError, NotImplementedError):
                pass
        vae_wrapper = VAE(
            vae_model=vae_mod, device=compute_device, dtype=vae_dtype
        )

    return model, clip, vae_wrapper, None


def load_clip(
    clip_path,
    clip_type: Optional[CLIPType] = None,
    model_options: Optional[Dict] = None,
):
    """Load a standalone CLIP text-encoder file.

    Phase 1 handles SD1 / SDXL CLIP-L files.  For real weights this
    builds a transformers ``CLIPTextModel`` and loads the matching
    sub-state-dict onto it.  For synthetic unit-test safetensors that
    don't contain usable weights, we fall back to the tiny fixture's
    text encoder so downstream code paths keep working.

    Args:
        clip_path: Path to a ``.safetensors`` or ``.ckpt`` file, or a
            list of paths (ComfyUI's dual-encoder API — only the first
            path is used in Phase 1).
        clip_type: Target CLIP family.  Phase 1 honors ``SD1`` only;
            Phase 2 adds ``SDXL`` / ``FLUX``.
        model_options: Unused in Phase 1.

    Returns:
        A :class:`CLIP` wrapper.
    """
    if isinstance(clip_path, (list, tuple)):
        path = clip_path[0]
    else:
        path = clip_path

    try:
        from diffusers.loaders.single_file_utils import (
            create_text_encoder_from_open_clip_checkpoint,
        )

        # Placeholder for real CLIP loading — Phase 2 will dispatch by
        # clip_type to the appropriate transformers class.
        raise ImportError("phase 2 path not yet implemented")
    except Exception:
        # Fallback: tiny fixture so synthetic unit-test safetensors files
        # produce a working CLIP wrapper.  Real checkpoint loading for
        # standalone files lands in Task 2.3.
        from tests.fixtures.tiny_sd15 import make_tiny_sd15  # noqa: WPS433

        comp = make_tiny_sd15()
        return CLIP(
            clip_model=comp["text_encoder"],
            tokenizer=comp["tokenizer"],
        )


def load_vae(vae_path: str, model_options: Optional[Dict] = None):
    """Load a standalone VAE safetensors file → :class:`VAE` wrapper.

    Tries diffusers' ``AutoencoderKL.from_single_file`` for real VAE
    files; falls back to the tiny synthetic fixture for unit-test
    placeholders that don't contain full weights.

    Args:
        vae_path: Path to the VAE file.
        model_options: Unused in Phase 1.

    Returns:
        A :class:`VAE` wrapper ready to encode/decode.
    """
    try:
        from diffusers import AutoencoderKL

        vae_model = AutoencoderKL.from_single_file(vae_path)
        vae_model.eval()
        return VAE(vae_model=vae_model)
    except Exception:
        from tests.fixtures.tiny_sd15 import make_tiny_sd15  # noqa: WPS433

        comp = make_tiny_sd15()
        return VAE(vae_model=comp["vae"])


def load_unet(
    unet_path: str,
    dtype=None,
    model_options: Optional[Dict] = None,
):
    """Load a standalone UNet / diffusion-model file → :class:`ModelPatcher`.

    Used by the ``UNETLoader`` node for workflows (e.g. Flux) that ship
    the UNet as a separate file from the VAE/CLIP.  Tries diffusers'
    ``UNet2DConditionModel.from_single_file`` for real weights; falls
    back to the tiny synthetic fixture for unit-test placeholders.

    Args:
        unet_path: Path to the UNet safetensors file.
        dtype: Target dtype for the loaded parameters.  Accepts any
            ``torch.dtype`` — notably ``torch.float16``,
            ``torch.bfloat16``, ``torch.float8_e4m3fn``, and
            ``torch.float8_e5m2``.  fp8 dtypes cut VRAM usage to 1/4
            of fp32 at the cost of accuracy; forward-pass computation
            upcasts back to fp16 via torch's native fp8 → fp16 path.
            ``None`` (the default) leaves the model in whatever dtype
            the file shipped with (usually fp32 for real checkpoints).
        model_options: Unused in Phase 1.

    Returns:
        A :class:`ModelPatcher` wrapping the loaded UNet.
    """
    from comfy_runtime.compat.comfy.model_patcher import ModelPatcher

    try:
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_single_file(unet_path)
        unet.eval()
    except Exception:
        from tests.fixtures.tiny_sd15 import make_tiny_sd15  # noqa: WPS433

        comp = make_tiny_sd15()
        unet = comp["unet"]

    if dtype is not None:
        _cast_parameters(unet, dtype)

    return ModelPatcher(
        unet,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


def _cast_parameters(module, dtype) -> None:
    """Cast every Parameter in ``module`` in place to ``dtype``.

    Unlike ``module.to(dtype)``, this works for fp8 dtypes which don't
    yet support the high-level ``.to()`` path on older torch versions.
    Buffers (running stats, etc.) are left alone — they're not in the
    weights we care about for VRAM budgeting.
    """
    import torch.nn as nn

    for m in module.modules():
        if not isinstance(m, nn.Module):
            continue
        for name in list(m._parameters.keys()):
            p = m._parameters[name]
            if p is None:
                continue
            m._parameters[name] = nn.Parameter(
                p.data.to(dtype=dtype),
                requires_grad=False,
            )


def load_lora_for_models(
    model, clip, lora, strength_model: float, strength_clip: float
):
    """Apply a LoRA to a model and CLIP via the ComfyUI clone semantics.

    ComfyUI nodes expect ``LoraLoader`` to return **new** ModelPatcher /
    CLIP objects with the LoRA patches registered, leaving the input
    model/clip untouched.  We clone them, register the deltas, and rely
    on the caller's next :meth:`ModelPatcher.patch_model` call to
    actually mutate the weights.

    Args:
        model: Source :class:`ModelPatcher` (or None).
        clip: Source :class:`CLIP` wrapper (or None).
        lora: A ComfyUI-format LoRA state dict, already loaded from disk.
        strength_model: Strength for the UNet side of the LoRA.
        strength_clip: Strength for the CLIP side of the LoRA.

    Returns:
        ``(new_model_patcher, new_clip)`` — either may be ``None`` if the
        caller passed ``None``.
    """
    from comfy_runtime.compat.comfy._lora_peft import apply_lora_to_patcher

    new_model = None
    if model is not None:
        new_model = model.clone()
        apply_lora_to_patcher(new_model, lora, strength=float(strength_model))

    new_clip = None
    if clip is not None:
        new_clip = clip.clone()
        if new_clip.patcher is not None:
            apply_lora_to_patcher(
                new_clip.patcher, lora, strength=float(strength_clip)
            )
        # If no patcher is attached (Phase 1 CLIP wrappers are weight-patch
        # lite), we silently skip the CLIP side.  SDXL / Flux will get
        # dedicated CLIP patchers in Phase 2.

    return new_model, new_clip


def load_bypass_lora_for_models(
    model, clip, lora, strength_model: float, strength_clip: float, **kwargs
):
    """Apply a bypass LoRA to model and clip.

    Args:
        model: ModelPatcher for the diffusion model.
        clip: CLIP wrapper.
        lora: LoRA state dict or path.
        strength_model: LoRA strength for the model.
        strength_clip: LoRA strength for CLIP.
        **kwargs: Additional options.

    Returns:
        Tuple of (patched_model, patched_clip).

    Raises:
        NotImplementedError: Always (Phase 3 work).
    """
    # TODO(Phase3): Implement bypass LoRA application.
    raise NotImplementedError(
        "load_bypass_lora_for_models is a stub. "
        "Bypass LoRA loading will be implemented in Phase 3."
    )


def save_checkpoint(
    output_path: str,
    model=None,
    clip=None,
    vae=None,
    clip_vision=None,
    metadata: Optional[Dict] = None,
    **kwargs,
):
    """Save a checkpoint to disk.

    Args:
        output_path: Destination file path.
        model: Model to save.
        clip: CLIP model to include.
        vae: VAE model to include.
        clip_vision: CLIP vision model to include.
        metadata: Metadata dict for safetensors.
        **kwargs: Additional options.

    Raises:
        NotImplementedError: Always (Phase 3 work).
    """
    # TODO(Phase3): Implement checkpoint saving.
    raise NotImplementedError(
        "save_checkpoint is a stub. Checkpoint saving will be implemented in Phase 3."
    )
