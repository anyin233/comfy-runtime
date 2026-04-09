"""Model base classes for comfy_runtime.

MIT reimplementation of comfy.model_base — provides the ModelType enum
and base model class hierarchy that ComfyUI uses to identify and
configure different diffusion model architectures.
"""

import enum
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelType enum
# ---------------------------------------------------------------------------

class ModelType(enum.IntEnum):
    """Enumeration of supported diffusion model types.

    Maps each architecture to an integer for identification during
    checkpoint loading and model configuration.
    """
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8
    IMG_TO_IMG = 9
    FLOW_COSMOS = 10
    IMG_TO_IMG_FLOW = 11


# ---------------------------------------------------------------------------
# BaseModel
# ---------------------------------------------------------------------------

class BaseModel:
    """Base class for all diffusion model wrappers.

    Provides the shared interface that ComfyUI nodes expect: model type,
    configuration, latent format, and device management.

    Attributes:
        model_type: The ModelType enum value.
        model_config: Configuration dict/object for the model.
        latent_format: LatentFormat instance for encoding/decoding.
        model_sampling: Sampling schedule object.
        adm_channels: Number of ADM (class embedding) channels.
        inpaint_model: Whether this is an inpaint variant.
        model: The underlying PyTorch module (set after loading).
    """

    model_type = ModelType.EPS

    def __init__(self, model_config=None, model_type=None, device=None,
                 unet_model=None):
        """Initialize BaseModel.

        Args:
            model_config: Configuration for the model architecture.
            model_type: Override for the model type enum.
            device: Target device.
            unet_model: Pre-built UNet module.
        """
        if model_type is not None:
            self.model_type = model_type
        self.model_config = model_config
        self.latent_format = None
        self.model_sampling = None
        self.adm_channels = 0
        self.inpaint_model = False
        self.model = unet_model
        self.device = device

        # Extract latent_format from config if available
        if model_config is not None:
            if hasattr(model_config, "latent_format"):
                self.latent_format = model_config.latent_format
            if hasattr(model_config, "unet_config"):
                unet_cfg = model_config.unet_config
                if isinstance(unet_cfg, dict):
                    self.adm_channels = unet_cfg.get("adm_in_channels", 0)

    def apply_model(self, x, t, c_concat=None, c_crossattn=None,
                    c_adm=None, control=None, transformer_options=None,
                    **kwargs):
        """Run the diffusion model forward pass.

        Args:
            x: Noisy latent input.
            t: Timestep tensor.
            c_concat: Concatenation conditioning.
            c_crossattn: Cross-attention conditioning.
            c_adm: ADM (class) conditioning.
            control: ControlNet signals.
            transformer_options: Transformer patch options.
            **kwargs: Additional model arguments.

        Returns:
            Model prediction tensor.
        """
        # TODO(Phase3): Implement model forward pass dispatch.
        raise NotImplementedError(
            "BaseModel.apply_model is a stub. Will be implemented in Phase 3."
        )

    def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
        """Pre-process latent before feeding to model.

        Args:
            latent: Raw latent tensor.

        Returns:
            Processed latent tensor.
        """
        if self.latent_format is not None:
            return self.latent_format.process_in(latent)
        return latent

    def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
        """Post-process model output latent.

        Args:
            latent: Model output latent tensor.

        Returns:
            Processed latent tensor.
        """
        if self.latent_format is not None:
            return self.latent_format.process_out(latent)
        return latent

    def memory_required(self, input_shape) -> int:
        """Estimate memory required for inference with given input shape.

        Args:
            input_shape: Tuple describing the input tensor shape.

        Returns:
            Estimated bytes required.
        """
        # Rough heuristic: 4 bytes per element, 4x for activations
        if input_shape is not None:
            elements = 1
            for dim in input_shape:
                elements *= dim
            return elements * 4 * 4
        return 0

    def state_dict_for_saving(self, clip_state_dict=None, vae_state_dict=None,
                              clip_vision_state_dict=None):
        """Build a state dict for checkpoint saving.

        Args:
            clip_state_dict: Optional CLIP state dict to include.
            vae_state_dict: Optional VAE state dict to include.
            clip_vision_state_dict: Optional CLIP vision state dict.

        Returns:
            Combined state dict.
        """
        # TODO(Phase3): Implement proper state dict assembly.
        sd = {}
        if self.model is not None and hasattr(self.model, "state_dict"):
            sd.update(self.model.state_dict())
        if clip_state_dict:
            sd.update(clip_state_dict)
        if vae_state_dict:
            sd.update(vae_state_dict)
        if clip_vision_state_dict:
            sd.update(clip_vision_state_dict)
        return sd


# ---------------------------------------------------------------------------
# Architecture-specific model stubs
# ---------------------------------------------------------------------------

class SD15(BaseModel):
    """Stable Diffusion 1.5 model.

    Attributes:
        model_type: Defaults to EPS.
    """

    model_type = ModelType.EPS

    def __init__(self, model_config=None, model_type=None, device=None,
                 unet_model=None):
        """Initialize SD 1.5 model.

        Args:
            model_config: Model configuration.
            model_type: Override model type.
            device: Target device.
            unet_model: Pre-built UNet.
        """
        super().__init__(model_config, model_type, device, unet_model)


class SDXL(BaseModel):
    """Stable Diffusion XL model.

    Attributes:
        model_type: Defaults to EPS.
    """

    model_type = ModelType.EPS

    def __init__(self, model_config=None, model_type=None, device=None,
                 unet_model=None):
        """Initialize SDXL model.

        Args:
            model_config: Model configuration.
            model_type: Override model type.
            device: Target device.
            unet_model: Pre-built UNet.
        """
        super().__init__(model_config, model_type, device, unet_model)


class SDXLRefiner(BaseModel):
    """SDXL Refiner model.

    Attributes:
        model_type: Defaults to EPS.
    """

    model_type = ModelType.EPS

    def __init__(self, model_config=None, model_type=None, device=None,
                 unet_model=None):
        """Initialize SDXL Refiner model.

        Args:
            model_config: Model configuration.
            model_type: Override model type.
            device: Target device.
            unet_model: Pre-built UNet.
        """
        super().__init__(model_config, model_type, device, unet_model)


class SD3(BaseModel):
    """Stable Diffusion 3 model.

    Attributes:
        model_type: Defaults to FLOW.
    """

    model_type = ModelType.FLOW

    def __init__(self, model_config=None, model_type=None, device=None,
                 unet_model=None):
        """Initialize SD3 model.

        Args:
            model_config: Model configuration.
            model_type: Override model type.
            device: Target device.
            unet_model: Pre-built UNet.
        """
        super().__init__(model_config, model_type, device, unet_model)


class SVD_img2vid(BaseModel):
    """Stable Video Diffusion image-to-video model.

    Attributes:
        model_type: Defaults to V_PREDICTION.
    """

    model_type = ModelType.V_PREDICTION

    def __init__(self, model_config=None, model_type=None, device=None,
                 unet_model=None):
        """Initialize SVD img2vid model.

        Args:
            model_config: Model configuration.
            model_type: Override model type.
            device: Target device.
            unet_model: Pre-built UNet.
        """
        super().__init__(model_config, model_type, device, unet_model)


class SDXL_instructpix2pix(BaseModel):
    """SDXL InstructPix2Pix model.

    Attributes:
        model_type: Defaults to EPS.
    """

    model_type = ModelType.EPS

    def __init__(self, model_config=None, model_type=None, device=None,
                 unet_model=None):
        """Initialize SDXL InstructPix2Pix model.

        Args:
            model_config: Model configuration.
            model_type: Override model type.
            device: Target device.
            unet_model: Pre-built UNet.
        """
        super().__init__(model_config, model_type, device, unet_model)
