"""Weight adapter infrastructure for comfy_runtime.

MIT reimplementation of comfy.weight_adapter — provides the base
interfaces for weight adaptation methods (LoRA, bypass, etc.).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WeightAdapterBase:
    """Base class for weight adapters.

    Weight adapters modify model weights during inference, supporting
    techniques like LoRA, DoRA, and other low-rank adaptations.

    Attributes:
        name: Identifier for this adapter.
        strength: Application strength multiplier.
    """

    def __init__(self, name: str = "", strength: float = 1.0):
        """Initialize WeightAdapterBase.

        Args:
            name: Adapter identifier.
            strength: Strength multiplier for the adaptation.
        """
        self.name = name
        self.strength = strength

    def calculate_weight(self, weight: Any, key: str, **kwargs) -> Any:
        """Calculate the adapted weight.

        Args:
            weight: Original weight tensor.
            key: Weight key name.
            **kwargs: Additional arguments.

        Returns:
            Adapted weight tensor.
        """
        # TODO(Phase3): Implement weight calculation.
        return weight


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

# List of all registered adapter classes.
adapters: list = []

# Mapping from adapter name/key to adapter class or factory.
adapter_maps: Dict[str, Any] = {}
