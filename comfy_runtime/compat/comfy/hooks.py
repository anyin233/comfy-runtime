"""Hook system for comfy_runtime.

MIT reimplementation of comfy.hooks — provides the hook classes and
helper functions that ComfyUI's nodes_hooks.py uses for dynamic
conditioning modification during sampling.
"""

import enum
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnumWeightTarget(enum.Enum):
    """Target for hook weight application."""

    Clip = "clip"
    Model = "model"


class InterpolationMethod(enum.Enum):
    """Interpolation method for keyframe transitions."""

    LINEAR = "linear"


# ---------------------------------------------------------------------------
# HookKeyframe
# ---------------------------------------------------------------------------


class HookKeyframe:
    """A single timing keyframe for hook scheduling.

    Defines when and with what strength a hook should be active.

    Attributes:
        strength: Hook strength at this keyframe (0.0 to 1.0).
        start_percent: When this keyframe activates (0.0 to 1.0).
        guarantee_steps: Minimum steps this keyframe stays active.
    """

    def __init__(
        self,
        strength: float = 1.0,
        start_percent: float = 0.0,
        guarantee_steps: int = 1,
    ):
        """Initialize HookKeyframe.

        Args:
            strength: Hook strength at this keyframe.
            start_percent: Activation point as fraction of total steps.
            guarantee_steps: Minimum active steps.
        """
        self.strength = strength
        self.start_percent = start_percent
        self.guarantee_steps = guarantee_steps

    def clone(self) -> "HookKeyframe":
        """Create a copy of this keyframe.

        Returns:
            New HookKeyframe with the same parameters.
        """
        return HookKeyframe(
            strength=self.strength,
            start_percent=self.start_percent,
            guarantee_steps=self.guarantee_steps,
        )


# ---------------------------------------------------------------------------
# HookKeyframeGroup
# ---------------------------------------------------------------------------


class HookKeyframeGroup:
    """Group of keyframes defining a hook's schedule over time.

    Attributes:
        keyframes: List of HookKeyframe instances.
    """

    def __init__(self):
        """Initialize an empty HookKeyframeGroup."""
        self.keyframes: List[HookKeyframe] = []

    def add(self, keyframe: HookKeyframe) -> "HookKeyframeGroup":
        """Add a keyframe to the group.

        Args:
            keyframe: HookKeyframe to add.

        Returns:
            Self for chaining.
        """
        self.keyframes.append(keyframe)
        # Keep sorted by start_percent
        self.keyframes.sort(key=lambda kf: kf.start_percent)
        return self

    def clone(self) -> "HookKeyframeGroup":
        """Create a deep copy of this group.

        Returns:
            New HookKeyframeGroup with cloned keyframes.
        """
        group = HookKeyframeGroup()
        group.keyframes = [kf.clone() for kf in self.keyframes]
        return group

    def is_empty(self) -> bool:
        """Check if the group has no keyframes.

        Returns:
            True if empty.
        """
        return len(self.keyframes) == 0

    def get_strength(self, percent: float) -> float:
        """Get the interpolated strength at a given percentage.

        Args:
            percent: Current progress fraction (0.0 to 1.0).

        Returns:
            Interpolated strength value.
        """
        if not self.keyframes:
            return 1.0

        # Find the active keyframe (latest one with start_percent <= percent)
        active = None
        for kf in self.keyframes:
            if kf.start_percent <= percent:
                active = kf
            else:
                break

        if active is None:
            return 0.0
        return active.strength


# ---------------------------------------------------------------------------
# HookGroup
# ---------------------------------------------------------------------------


class HookGroup:
    """Group of hooks to be applied together during sampling.

    Attributes:
        hooks: List of hook objects.
    """

    def __init__(self):
        """Initialize an empty HookGroup."""
        self.hooks: List[Any] = []

    def add(self, hook) -> "HookGroup":
        """Add a hook to the group.

        Args:
            hook: Hook object to add.

        Returns:
            Self for chaining.
        """
        self.hooks.append(hook)
        return self

    def clone(self) -> "HookGroup":
        """Create a copy of this group.

        Returns:
            New HookGroup with the same hooks.
        """
        group = HookGroup()
        group.hooks = list(self.hooks)
        return group

    def is_empty(self) -> bool:
        """Check if the group has no hooks.

        Returns:
            True if empty.
        """
        return len(self.hooks) == 0

    @classmethod
    def combine_all_hooks(
        cls, hooks_list: List[Optional["HookGroup"]], require_count: int = 0
    ) -> Optional["HookGroup"]:
        """Combine multiple HookGroups into one.

        Args:
            hooks_list: List of HookGroup instances (may contain None).
            require_count: Minimum number of non-None groups required.

        Returns:
            Combined HookGroup, or None if not enough groups.
        """
        valid = [h for h in hooks_list if h is not None and not h.is_empty()]
        if len(valid) < require_count:
            return None
        if not valid:
            return None

        combined = HookGroup()
        seen = set()
        for group in valid:
            for hook in group.hooks:
                hook_id = id(hook)
                if hook_id not in seen:
                    seen.add(hook_id)
                    combined.hooks.append(hook)
        return combined


# ---------------------------------------------------------------------------
# Hook factory functions
# ---------------------------------------------------------------------------


def create_hook_lora(
    lora: Any,
    strength_model: float = 1.0,
    strength_clip: float = 1.0,
    keyframe_group: Optional[HookKeyframeGroup] = None,
    **kwargs,
) -> Any:
    """Create a LoRA-based hook.

    Args:
        lora: LoRA weights or path.
        strength_model: LoRA strength for the model.
        strength_clip: LoRA strength for CLIP.
        keyframe_group: Optional schedule keyframes.
        **kwargs: Additional options.

    Returns:
        Hook object.
    """
    # TODO(Phase3): Implement LoRA hook creation.
    hook = {
        "type": "lora",
        "lora": lora,
        "strength_model": strength_model,
        "strength_clip": strength_clip,
        "keyframe_group": keyframe_group,
    }
    hook.update(kwargs)
    return hook


def create_hook_model_as_lora(
    model: Any,
    strength: float = 1.0,
    keyframe_group: Optional[HookKeyframeGroup] = None,
    **kwargs,
) -> Any:
    """Create a hook that treats a full model as a LoRA.

    Args:
        model: Model to use as LoRA source.
        strength: Application strength.
        keyframe_group: Optional schedule keyframes.
        **kwargs: Additional options.

    Returns:
        Hook object.
    """
    # TODO(Phase3): Implement model-as-lora hook.
    hook = {
        "type": "model_as_lora",
        "model": model,
        "strength": strength,
        "keyframe_group": keyframe_group,
    }
    hook.update(kwargs)
    return hook


def get_patch_weights_from_model(
    model: Any, discard_model_sampling: bool = True
) -> Dict:
    """Extract patchable weights from a model.

    Args:
        model: Model to extract weights from.
        discard_model_sampling: Whether to exclude model_sampling weights.

    Returns:
        Dict of weight name -> tensor.
    """
    # TODO(Phase3): Implement weight extraction.
    if (
        model is not None
        and hasattr(model, "model")
        and hasattr(model.model, "state_dict")
    ):
        sd = model.model.state_dict()
        if discard_model_sampling:
            sd = {k: v for k, v in sd.items() if "model_sampling" not in k}
        return sd
    return {}


# ---------------------------------------------------------------------------
# Conditioning helper functions
# ---------------------------------------------------------------------------


def set_hooks_for_conditioning(cond: Any, hooks: Optional[HookGroup]) -> Any:
    """Attach hooks to conditioning data.

    Args:
        cond: Conditioning data (list of tuples or dicts).
        hooks: HookGroup to attach.

    Returns:
        Modified conditioning with hooks attached.
    """
    if cond is None:
        return cond
    if hooks is None or hooks.is_empty():
        return cond

    # Conditioning is typically a list of [tensor, dict] pairs
    if isinstance(cond, list):
        new_cond = []
        for c in cond:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                d = dict(c[1]) if isinstance(c[1], dict) else {}
                d["hooks"] = hooks
                new_cond.append([c[0], d])
            else:
                new_cond.append(c)
        return new_cond
    return cond


def set_conds_props(cond: Any, **kwargs) -> Any:
    """Set properties on conditioning data.

    Args:
        cond: Conditioning data.
        **kwargs: Properties to set on each conditioning entry.

    Returns:
        Modified conditioning.
    """
    if cond is None:
        return cond
    if isinstance(cond, list):
        new_cond = []
        for c in cond:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                d = dict(c[1]) if isinstance(c[1], dict) else {}
                d.update(kwargs)
                new_cond.append([c[0], d])
            else:
                new_cond.append(c)
        return new_cond
    return cond


def set_conds_props_and_combine(
    conds: List[Any], new_conds: List[Any], **kwargs
) -> List[Any]:
    """Set properties on new_conds and combine with existing conds.

    Args:
        conds: Existing conditioning list.
        new_conds: New conditioning entries to modify and append.
        **kwargs: Properties to set on new_conds.

    Returns:
        Combined conditioning list.
    """
    modified = set_conds_props(new_conds, **kwargs) if new_conds else []
    if conds is None:
        return modified
    if modified is None:
        return conds
    return list(conds) + list(modified)


def set_default_conds_and_combine(
    conds: Optional[List], new_conds: Optional[List], **kwargs
) -> List:
    """Set default properties and combine conditioning lists.

    Args:
        conds: Existing conditioning (may be None).
        new_conds: New conditioning to add (may be None).
        **kwargs: Default properties for new_conds.

    Returns:
        Combined conditioning list.
    """
    if new_conds is None:
        return conds if conds is not None else []
    modified = set_conds_props(new_conds, **kwargs)
    if conds is None:
        return modified if modified is not None else []
    return list(conds) + list(modified)
