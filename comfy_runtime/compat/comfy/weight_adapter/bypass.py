"""Bypass weight adapter for comfy_runtime.

MIT reimplementation of comfy.weight_adapter.bypass — provides the
bypass LoRA adapter that applies weight modifications through an
alternative pathway.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BypassWeightAdapter:
    """Bypass weight adapter for alternative LoRA application.

    Applies LoRA-like weight modifications through a bypass pathway
    that can differ from standard LoRA application.

    Attributes:
        name: Adapter identifier.
        strength: Application strength.
        patches: Dict of weight patches.
    """

    def __init__(
        self, name: str = "", strength: float = 1.0, patches: Optional[Dict] = None
    ):
        """Initialize BypassWeightAdapter.

        Args:
            name: Adapter identifier.
            strength: Strength multiplier.
            patches: Dict of weight key -> patch data.
        """
        self.name = name
        self.strength = strength
        self.patches = patches or {}

    def calculate_weight(self, weight: Any, key: str, **kwargs) -> Any:
        """Calculate the adapted weight via bypass path.

        Args:
            weight: Original weight tensor.
            key: Weight key name.
            **kwargs: Additional arguments.

        Returns:
            Adapted weight tensor.
        """
        # TODO(Phase3): Implement bypass weight calculation.
        return weight

    @staticmethod
    def load_bypass_lora(lora_sd: Dict, strength: float = 1.0) -> "BypassWeightAdapter":
        """Load a bypass LoRA from a state dict.

        Args:
            lora_sd: LoRA state dict.
            strength: Application strength.

        Returns:
            BypassWeightAdapter instance.
        """
        # TODO(Phase3): Implement bypass LoRA loading.
        return BypassWeightAdapter(
            name="bypass_lora", strength=strength, patches=lora_sd
        )


class BypassInjectionManager:
    """Manages bypass injection for weight adapters.

    Stub — full implementation in Phase 3.
    """

    def __init__(self):
        self.injections = []

    def register(self, model, adapter):
        """Register a bypass injection."""
        self.injections.append((model, adapter))

    def apply_all(self):
        """Apply all registered injections."""
        pass

    def clear(self):
        """Clear all registered injections."""
        self.injections.clear()
