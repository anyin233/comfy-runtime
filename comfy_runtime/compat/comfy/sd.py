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
    """Enumeration of supported CLIP text encoder architectures."""

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
    ):
        """Initialize CLIP wrapper.

        Args:
            clip_model: The text encoder model.
            tokenizer: Text tokenizer.
            load_device: Device for running inference.
            offload_device: Device for weight offloading.
            patcher: Optional ModelPatcher for the clip model.
        """
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.patcher = patcher
        self.load_device = load_device or torch.device("cpu")
        self.offload_device = offload_device or torch.device("cpu")
        self.layer_idx = None
        self.cond_stage_model = clip_model

    def clone(self) -> "CLIP":
        """Create a clone of this CLIP wrapper.

        Returns:
            New CLIP instance sharing the model but with independent state.
        """
        cloned = CLIP.__new__(CLIP)
        cloned.clip_model = self.clip_model
        cloned.tokenizer = self.tokenizer
        cloned.load_device = self.load_device
        cloned.offload_device = self.offload_device
        cloned.layer_idx = self.layer_idx
        cloned.cond_stage_model = self.cond_stage_model
        if self.patcher is not None:
            cloned.patcher = self.patcher.clone()
        else:
            cloned.patcher = None
        return cloned

    def tokenize(self, text: str, return_word_ids: bool = False):
        """Tokenize text into ComfyUI's chunked-weighted format.

        Args:
            text: Input text string.
            return_word_ids: Ignored in Phase 1 (attention-weighted prompts
                are Phase 2); kept for API parity.

        Returns:
            ``{"l": [[(id, weight), ...77...], ...]}``.
        """
        from comfy_runtime.compat.comfy._tokenizer import tokenize_to_comfy_format

        if self.tokenizer is None:
            raise RuntimeError("CLIP.tokenize requires self.tokenizer to be set")
        return tokenize_to_comfy_format(
            self.tokenizer, text, max_length=77, slot="l"
        )

    def encode_from_tokens(
        self, tokens, return_pooled: bool = False, return_dict: bool = False
    ):
        """Run the CLIP text encoder over tokenized chunks.

        Args:
            tokens: ``{"l": [[(id, weight), ...77...], ...]}`` from
                :meth:`tokenize`.
            return_pooled: Also return the pooled [CLS] embedding.
            return_dict: Return a dict ``{"cond": ..., "pooled_output": ...}``.

        Returns:
            Embedding tensor of shape ``(B, 77*num_chunks, hidden_size)``,
            optionally paired with the pooled output or wrapped in a dict.
        """
        from comfy_runtime.compat.comfy._tokenizer import tokens_to_input_ids

        if self.clip_model is None:
            raise RuntimeError(
                "CLIP.encode_from_tokens requires self.clip_model to be set"
            )

        # Phase 1: only the "l" slot.  SDXL's "g" and Flux's "t5xxl" are Phase 2.
        if "l" not in tokens:
            raise KeyError(
                f"Phase 1 only supports the 'l' slot; got {list(tokens.keys())}"
            )

        chunks = tokens["l"]
        device = next(self.clip_model.parameters()).device

        chunk_embeddings: List[torch.Tensor] = []
        pooled: Optional[torch.Tensor] = None
        for chunk in chunks:
            ids = torch.tensor(
                [tokens_to_input_ids(chunk)], dtype=torch.long, device=device
            )
            with torch.no_grad():
                out = self.clip_model(
                    input_ids=ids,
                    output_hidden_states=True,
                )
            if self.layer_idx is not None:
                hidden = out.hidden_states[self.layer_idx]
            else:
                hidden = out.last_hidden_state
            chunk_embeddings.append(hidden)
            if pooled is None:
                # CLIP pooled output is the [EOS] row of the last hidden state.
                # We use the first token here for determinism with the tiny
                # fixture (which has no real EOS).  Real CLIP models will
                # diverge slightly; this is Phase 1 and good enough for the
                # SD1.5 happy path.  Phase 2 will switch to pooler_output when
                # available.
                pooler = getattr(out, "pooler_output", None)
                if pooler is not None:
                    pooled = pooler
                else:
                    pooled = out.last_hidden_state[:, 0, :]

        cond = torch.cat(chunk_embeddings, dim=1)

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
        """Decode latent tensor to image tensor.

        Args:
            latent: Latent tensor of shape (B, C, H, W).

        Returns:
            Image tensor of shape (B, H*f, W*f, 3) in [0, 1] range.
        """
        # TODO(Phase3): Implement actual VAE decoding.
        raise NotImplementedError(
            "VAE.decode is a stub. Decoding will be implemented in Phase 3."
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image tensor to latent tensor.

        Args:
            image: Image tensor of shape (B, H, W, 3) in [0, 1] range.

        Returns:
            Latent tensor of shape (B, C, H/f, W/f).
        """
        # TODO(Phase3): Implement actual VAE encoding.
        raise NotImplementedError(
            "VAE.encode is a stub. Encoding will be implemented in Phase 3."
        )

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
    """Load a checkpoint and guess its model configuration.

    Args:
        ckpt_path: Path to the checkpoint file.
        output_vae: Whether to return a VAE.
        output_clip: Whether to return a CLIP model.
        output_clipvision: Whether to return a CLIP vision model.
        embedding_directory: Directory for text embeddings.
        output_model: Whether to return the diffusion model.
        model_options: Additional model loading options.

    Returns:
        Tuple of (model, clip, vae, clipvision) — any may be None.

    Raises:
        NotImplementedError: Always (Phase 3 work).
    """
    # TODO(Phase3): Implement checkpoint loading with config detection.
    raise NotImplementedError(
        "load_checkpoint_guess_config is a stub. "
        "Checkpoint loading will be implemented in Phase 3."
    )


def load_clip(
    clip_path: str,
    clip_type: Optional[CLIPType] = None,
    model_options: Optional[Dict] = None,
):
    """Load a standalone CLIP model.

    Args:
        clip_path: Path to the CLIP checkpoint.
        clip_type: Type of CLIP model to load.
        model_options: Additional loading options.

    Returns:
        CLIP instance.

    Raises:
        NotImplementedError: Always (Phase 3 work).
    """
    # TODO(Phase3): Implement CLIP loading.
    raise NotImplementedError(
        "load_clip is a stub. CLIP loading will be implemented in Phase 3."
    )


def load_lora_for_models(
    model, clip, lora, strength_model: float, strength_clip: float
):
    """Apply a LoRA to model and clip.

    Args:
        model: ModelPatcher for the diffusion model.
        clip: CLIP wrapper.
        lora: LoRA state dict or path.
        strength_model: LoRA strength for the model.
        strength_clip: LoRA strength for CLIP.

    Returns:
        Tuple of (patched_model, patched_clip).

    Raises:
        NotImplementedError: Always (Phase 3 work).
    """
    # TODO(Phase3): Implement LoRA application.
    raise NotImplementedError(
        "load_lora_for_models is a stub. LoRA loading will be implemented in Phase 3."
    )


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
