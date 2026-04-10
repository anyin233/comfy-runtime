"""Model patching infrastructure for comfy_runtime.

MIT reimplementation of comfy.model_patcher — provides ModelPatcher for
wrapping PyTorch models with LoRA/weight patching support, plus helper
functions for manipulating model_options dicts.
"""

import copy
import logging
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _extract_delta(patch_data) -> Optional[torch.Tensor]:
    """Normalize a single patch payload into a delta tensor.

    ComfyUI's patch payload format is heterogeneous.  Phase 1 handles:

    * A raw ``torch.Tensor`` → treated as a direct delta
    * A 1-tuple ``(delta_tensor,)`` → equivalent to the raw form

    Phase 2 will also handle LoRA factored tuples such as
    ``(up_tensor, down_tensor, alpha, mid_tensor, strength_patch_scale)``
    via a dedicated ``_lora_peft`` helper.  Until then any multi-element
    tuple returns ``None`` and the patch is skipped.
    """
    if isinstance(patch_data, torch.Tensor):
        return patch_data
    if isinstance(patch_data, tuple):
        if len(patch_data) == 1 and isinstance(patch_data[0], torch.Tensor):
            return patch_data[0]
        # Phase 2: LoRA factored form.  Signal "skip" for now.
        return None
    return None


def _resolve_weight_slot(model, key: str):
    """Return ``(parent_module, attr_name)`` for a dotted state-dict key.

    Returns ``(None, None)`` if any segment is missing.
    """
    parts = key.split(".")
    obj = model
    for part in parts[:-1]:
        if not hasattr(obj, part):
            return None, None
        obj = getattr(obj, part)
    attr = parts[-1]
    if not hasattr(obj, attr):
        return None, None
    return obj, attr


# ---------------------------------------------------------------------------
# ModelPatcher
# ---------------------------------------------------------------------------


class ModelPatcher:
    """Wraps a PyTorch model with support for weight patching (LoRA, etc.).

    Manages patches, object patches, and model options while providing
    clone/copy semantics needed by ComfyUI's node graph.

    Attributes:
        model: The wrapped PyTorch model.
        load_device: Device for inference.
        offload_device: Device for offloading when not in use.
        size: Estimated model size in bytes.
        weight_inplace_update: Whether to update weights in-place.
        patches: Dict mapping weight keys to lists of patch tuples.
        object_patches: Dict mapping attribute names to replacement objects.
        model_options: Dict of model configuration options.
        model_keys: Set of all parameter/buffer keys in the model.
        backup: Dict of original weight values before patching.
        object_patches_backup: Backup of object patches.
    """

    def __init__(
        self,
        model,
        load_device=None,
        offload_device=None,
        size: int = 0,
        weight_inplace_update: bool = False,
    ):
        """Initialize ModelPatcher.

        Args:
            model: The PyTorch model to wrap.
            load_device: Device for running inference.
            offload_device: Device for weight offloading (typically CPU).
            size: Pre-computed model size in bytes (0 = auto-compute).
            weight_inplace_update: If True, modify weights in-place.
        """
        self.model = model
        self.load_device = (
            load_device if load_device is not None else torch.device("cpu")
        )
        self.offload_device = (
            offload_device if offload_device is not None else torch.device("cpu")
        )
        self.size = size
        self.weight_inplace_update = weight_inplace_update

        self.patches: Dict[str, list] = {}
        self.object_patches: Dict[str, Any] = {}
        self.model_options: Dict[str, Any] = {"transformer_options": {}}
        self.model_keys: Set[str] = set()
        self.backup: Dict[str, torch.Tensor] = {}
        self.object_patches_backup: Dict[str, Any] = {}

        self.current_device = (
            offload_device if offload_device is not None else torch.device("cpu")
        )
        self.is_patched = False

        # Populate model_keys from the model's state dict
        if model is not None and hasattr(model, "state_dict"):
            try:
                self.model_keys = set(model.state_dict().keys())
            except Exception:
                pass

    def model_size(self) -> int:
        """Return the model size in bytes.

        Returns:
            Size in bytes, either pre-set or computed from parameters.
        """
        if self.size > 0:
            return self.size
        if self.model is not None and hasattr(self.model, "parameters"):
            return sum(p.nelement() * p.element_size() for p in self.model.parameters())
        return 0

    def clone(self) -> "ModelPatcher":
        """Create a shallow clone of this ModelPatcher.

        The underlying model is shared; patches and options are copied
        so they can be modified independently.

        Returns:
            A new ModelPatcher sharing the same model but with independent
            patches and options.
        """
        cloned = ModelPatcher.__new__(ModelPatcher)
        cloned.model = self.model
        cloned.load_device = self.load_device
        cloned.offload_device = self.offload_device
        cloned.size = self.size
        cloned.weight_inplace_update = self.weight_inplace_update
        cloned.current_device = self.current_device
        cloned.is_patched = False

        # Deep copy mutable state
        cloned.patches = {k: list(v) for k, v in self.patches.items()}
        cloned.object_patches = dict(self.object_patches)
        cloned.model_options = create_model_options_clone(self.model_options)
        cloned.model_keys = set(self.model_keys)
        cloned.backup = {}
        cloned.object_patches_backup = {}

        return cloned

    def add_patches(
        self, patches: Dict, strength_patch: float = 1.0, strength_model: float = 1.0
    ) -> Set[str]:
        """Register weight patches to be applied before inference.

        Args:
            patches: Dict mapping weight keys to patch data.
            strength_patch: Multiplier for the patch delta.
            strength_model: Multiplier for the original weights.

        Returns:
            Set of keys that were actually added (present in model_keys).
        """
        added = set()
        for key, patch_data in patches.items():
            if key in self.model_keys:
                if key not in self.patches:
                    self.patches[key] = []
                self.patches[key].append((strength_patch, patch_data, strength_model))
                added.add(key)
        return added

    def model_dtype(self) -> torch.dtype:
        """Return the primary dtype of the wrapped model.

        Returns:
            The dtype of the first model parameter, or float32 as fallback.
        """
        if self.model is not None and hasattr(self.model, "parameters"):
            try:
                return next(self.model.parameters()).dtype
            except StopIteration:
                pass
        return torch.float32

    def get_model_object(self, name: str) -> Any:
        """Retrieve a named sub-object from the model.

        Checks object_patches first, then falls back to attribute lookup
        on the wrapped model.

        Args:
            name: Dotted attribute path (e.g. "model_sampling").

        Returns:
            The requested object.

        Raises:
            AttributeError: If the name cannot be resolved.
        """
        if name in self.object_patches:
            return self.object_patches[name]
        obj = self.model
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def set_model_patch(self, patch, name: str):
        """Set a model patch in model_options.

        Args:
            patch: The patch callable or data.
            name: Name for the patch in transformer_options.
        """
        to = self.model_options.setdefault("transformer_options", {})
        if name not in to:
            to[name] = []
        to[name].append(patch)

    def set_model_patch_replace(
        self,
        patch,
        name: str,
        block_name: str,
        number: int,
        transformer_index: Optional[int] = None,
    ):
        """Set a replacement patch for a specific transformer block.

        Args:
            patch: The replacement callable.
            name: Patch category name.
            block_name: Block identifier (e.g. "input", "middle", "output").
            number: Block number.
            transformer_index: Optional transformer layer index.
        """
        self.model_options = set_model_options_patch_replace(
            self.model_options, patch, name, block_name, number, transformer_index
        )

    def set_model_attn1_patch(self, patch):
        """Set a self-attention patch.

        Args:
            patch: The attention patch callable.
        """
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        """Set a cross-attention patch.

        Args:
            patch: The attention patch callable.
        """
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(
        self, patch, block_name, number, transformer_index=None
    ):
        """Replace self-attention in a specific block.

        Args:
            patch: Replacement callable.
            block_name: Block identifier.
            number: Block number.
            transformer_index: Optional transformer layer index.
        """
        self.set_model_patch_replace(
            patch, "attn1", block_name, number, transformer_index
        )

    def set_model_attn2_replace(
        self, patch, block_name, number, transformer_index=None
    ):
        """Replace cross-attention in a specific block.

        Args:
            patch: Replacement callable.
            block_name: Block identifier.
            number: Block number.
            transformer_index: Optional transformer layer index.
        """
        self.set_model_patch_replace(
            patch, "attn2", block_name, number, transformer_index
        )

    def set_model_attn1_output_patch(self, patch):
        """Set a post-self-attention output patch.

        Args:
            patch: The output patch callable.
        """
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        """Set a post-cross-attention output patch.

        Args:
            patch: The output patch callable.
        """
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch):
        """Set an input block patch.

        Args:
            patch: The block patch callable.
        """
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch):
        """Set a patch applied after skip connection in input blocks.

        Args:
            patch: The patch callable.
        """
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch):
        """Set an output block patch.

        Args:
            patch: The block patch callable.
        """
        self.set_model_patch(patch, "output_block_patch")

    def set_model_sampler_cfg_function(
        self, sampler_cfg_function, disable_cfg1_optimization=False
    ):
        """Set a custom CFG function.

        Args:
            sampler_cfg_function: Custom CFG callable.
            disable_cfg1_optimization: Disable optimization when cfg=1.
        """
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True
        to = self.model_options.setdefault("sampler_cfg_function", [])
        to.append(sampler_cfg_function)

    def set_model_sampler_post_cfg_function(
        self, post_cfg_function, disable_cfg1_optimization=False
    ):
        """Set a post-CFG callback function.

        Args:
            post_cfg_function: Post-CFG callable.
            disable_cfg1_optimization: Disable optimization when cfg=1.
        """
        self.model_options = set_model_options_post_cfg_function(
            self.model_options, post_cfg_function, disable_cfg1_optimization
        )

    def set_model_sampler_pre_cfg_function(
        self, pre_cfg_function, disable_cfg1_optimization=False
    ):
        """Set a pre-CFG callback function.

        Args:
            pre_cfg_function: Pre-CFG callable.
            disable_cfg1_optimization: Disable optimization when cfg=1.
        """
        self.model_options = set_model_options_pre_cfg_function(
            self.model_options, pre_cfg_function, disable_cfg1_optimization
        )

    def set_model_denoise_mask_function(self, denoise_mask_function):
        """Set a custom denoise mask function.

        Args:
            denoise_mask_function: Custom mask callable.
        """
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def patch_model(self, device_to=None, patch_weights=True):
        """Apply all registered patches to the model weights.

        Each patch entry is a triple
        ``(strength_patch, patch_data, strength_model)`` produced by
        :meth:`add_patches`.  For every key the update is::

            new_weight = original * strength_model + delta * strength_patch

        Multiple patches on the same key accumulate in registration order,
        where each subsequent patch starts from the already-accumulated
        intermediate (matching ComfyUI's semantics of stacking LoRAs).

        Backups of the original tensors are stored in ``self.backup`` so
        :meth:`unpatch_model` can restore the model bit-for-bit.

        Args:
            device_to: Optional device to move the model to *after* patching.
            patch_weights: When ``False``, skip weight modification and only
                do the device move (used by partial loaders in Phase 3).

        Returns:
            The underlying model (for chaining).
        """
        if self.model is None:
            self.is_patched = True
            return self.model

        if patch_weights and self.patches:
            with torch.no_grad():
                for key, patch_list in self.patches.items():
                    parent, attr_name = _resolve_weight_slot(self.model, key)
                    if parent is None:
                        continue

                    original = getattr(parent, attr_name)
                    if not isinstance(original, torch.Tensor):
                        continue

                    # Backup once so multiple patch/unpatch cycles are
                    # idempotent across repeated add_patches calls.
                    if key not in self.backup:
                        self.backup[key] = original.detach().clone()

                    new_weight = original.clone()
                    for strength_patch, patch_data, strength_model in patch_list:
                        delta = _extract_delta(patch_data)
                        if delta is None:
                            continue
                        delta_cast = delta.to(
                            device=new_weight.device, dtype=new_weight.dtype
                        )
                        new_weight = (
                            new_weight * float(strength_model)
                            + delta_cast * float(strength_patch)
                        )

                    if (
                        isinstance(parent, nn.Module)
                        and attr_name in parent._parameters
                    ):
                        parent._parameters[attr_name].data.copy_(new_weight)
                    else:
                        setattr(parent, attr_name, new_weight)

        if device_to is not None and hasattr(self.model, "to"):
            self.model.to(device_to)
            self.current_device = device_to
        self.is_patched = True
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """Restore original weights from :attr:`backup` and move device.

        After :meth:`patch_model` has stored backups in ``self.backup``,
        this method copies each original tensor back in-place so further
        sampling runs see an unmodified model.  Backups are cleared after
        restoration so the next patch_model pass starts fresh.

        Args:
            device_to: Optional device to move the model to after restoration.
            unpatch_weights: When ``False``, skip the weight restore and only
                do the device move.
        """
        if unpatch_weights and self.backup and self.model is not None:
            with torch.no_grad():
                for key, original in self.backup.items():
                    parent, attr_name = _resolve_weight_slot(self.model, key)
                    if parent is None:
                        continue
                    if (
                        isinstance(parent, nn.Module)
                        and attr_name in parent._parameters
                    ):
                        parent._parameters[attr_name].data.copy_(original)
                    else:
                        setattr(parent, attr_name, original)
            self.backup.clear()

        if device_to is not None and hasattr(self.model, "to"):
            self.model.to(device_to)
            self.current_device = device_to
        self.is_patched = False

    def partially_load(self, device, extra_memory=0):
        """Partially load model weights onto device.

        Args:
            device: Target device.
            extra_memory: Extra memory budget in bytes.
        """
        # TODO(Phase3): Implement partial loading.
        pass

    def partially_unload(self, device, extra_memory=0):
        """Partially unload model weights from device.

        Args:
            device: Offload device.
            extra_memory: Memory to free in bytes.
        """
        # TODO(Phase3): Implement partial unloading.
        pass


# ---------------------------------------------------------------------------
# model_options helper functions
# ---------------------------------------------------------------------------


def create_model_options_clone(model_options: Dict) -> Dict:
    """Deep copy a model_options dict.

    Uses copy.deepcopy to ensure all nested structures are independent.

    Args:
        model_options: The model options dict to clone.

    Returns:
        A deep copy of model_options.
    """
    if model_options is None:
        return {"transformer_options": {}}
    return copy.deepcopy(model_options)


def set_model_options_patch_replace(
    model_options: Dict,
    patch,
    name: str,
    block_name: str,
    number: int,
    transformer_index: Optional[int] = None,
) -> Dict:
    """Set a patch replacement in model_options for a specific block.

    Args:
        model_options: The model options dict (modified in-place).
        patch: The replacement callable.
        name: Patch category name (e.g. "attn1", "attn2").
        block_name: Block identifier (e.g. "input", "middle", "output").
        number: Block number.
        transformer_index: Optional transformer layer index within the block.

    Returns:
        The modified model_options dict.
    """
    to = model_options.setdefault("transformer_options", {})
    patch_replace = to.setdefault("patches_replace", {})
    category = patch_replace.setdefault(name, {})

    if transformer_index is not None:
        key = (block_name, number, transformer_index)
    else:
        key = (block_name, number)

    category[key] = patch
    return model_options


def set_model_options_post_cfg_function(
    model_options: Dict, post_cfg_function, disable_cfg1_optimization: bool = False
) -> Dict:
    """Add a post-CFG callback to model_options.

    Args:
        model_options: The model options dict (modified in-place).
        post_cfg_function: Callable invoked after CFG is applied.
        disable_cfg1_optimization: If True, disable the cfg=1 optimization.

    Returns:
        The modified model_options dict.
    """
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    lst = model_options.setdefault("sampler_post_cfg_function", [])
    lst.append(post_cfg_function)
    return model_options


def set_model_options_pre_cfg_function(
    model_options: Dict, pre_cfg_function, disable_cfg1_optimization: bool = False
) -> Dict:
    """Add a pre-CFG callback to model_options.

    Args:
        model_options: The model options dict (modified in-place).
        pre_cfg_function: Callable invoked before CFG is applied.
        disable_cfg1_optimization: If True, disable the cfg=1 optimization.

    Returns:
        The modified model_options dict.
    """
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    lst = model_options.setdefault("sampler_pre_cfg_function", [])
    lst.append(pre_cfg_function)
    return model_options
