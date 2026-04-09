"""Conditioning wrapper classes for comfy_runtime.

MIT reimplementation of comfy.conds — provides the conditioning container
classes that ComfyUI uses to pass different types of conditioning data
through the sampling pipeline.
"""

import logging

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONDRegular
# ---------------------------------------------------------------------------


class CONDRegular:
    """Regular conditioning wrapper for non-attention conditioning.

    Wraps a tensor that is concatenated or added to the model input
    rather than used as cross-attention context.

    Attributes:
        cond: The conditioning tensor.
    """

    def __init__(self, cond: torch.Tensor):
        """Initialize CONDRegular.

        Args:
            cond: Conditioning tensor.
        """
        self.cond = cond

    def _copy_with(self, cond: torch.Tensor) -> "CONDRegular":
        """Create a copy with a different tensor.

        Args:
            cond: New conditioning tensor.

        Returns:
            New CONDRegular wrapping cond.
        """
        return CONDRegular(cond)

    def can_concat_cond(self, other: "CONDRegular") -> bool:
        """Check if this conditioning can be concatenated with other.

        Args:
            other: Another CONDRegular instance.

        Returns:
            True if shapes are compatible for concatenation.
        """
        if not isinstance(other, CONDRegular):
            return False
        # Must match on all dims except batch
        if self.cond.shape[1:] != other.cond.shape[1:]:
            return False
        return True

    def concat_cond(self, others: list) -> "CONDRegular":
        """Concatenate with other CONDRegular instances along batch dim.

        Args:
            others: List of CONDRegular instances to concatenate with.

        Returns:
            New CONDRegular with concatenated tensors.
        """
        conds = [self.cond] + [o.cond for o in others]
        return CONDRegular(torch.cat(conds, dim=0))

    def process_cond(self, batch_size: int, device, **kwargs) -> "CONDRegular":
        """Process conditioning for a specific batch size.

        Args:
            batch_size: Target batch size.
            device: Target device.
            **kwargs: Additional processing options.

        Returns:
            Processed CONDRegular.
        """
        cond = self.cond
        if cond.shape[0] < batch_size:
            # Repeat to match batch size
            repeats = [1] * len(cond.shape)
            repeats[0] = -(-batch_size // cond.shape[0])  # ceil division
            cond = cond.repeat(*repeats)[:batch_size]
        return self._copy_with(cond.to(device))


# ---------------------------------------------------------------------------
# CONDCrossAttn
# ---------------------------------------------------------------------------


class CONDCrossAttn:
    """Cross-attention conditioning wrapper.

    Wraps a tensor used as key/value in cross-attention layers of
    the diffusion model (typically text encoder output).

    Attributes:
        cond: The conditioning tensor (B, seq_len, dim).
    """

    def __init__(self, cond: torch.Tensor):
        """Initialize CONDCrossAttn.

        Args:
            cond: Cross-attention conditioning tensor.
        """
        self.cond = cond

    def _copy_with(self, cond: torch.Tensor) -> "CONDCrossAttn":
        """Create a copy with a different tensor.

        Args:
            cond: New conditioning tensor.

        Returns:
            New CONDCrossAttn wrapping cond.
        """
        return CONDCrossAttn(cond)

    def can_concat_cond(self, other: "CONDCrossAttn") -> bool:
        """Check if this conditioning can be concatenated with other.

        Args:
            other: Another CONDCrossAttn instance.

        Returns:
            True if shapes are compatible.
        """
        if not isinstance(other, CONDCrossAttn):
            return False
        # For cross-attention, seq_len and dim must match
        if self.cond.shape[1:] != other.cond.shape[1:]:
            return False
        return True

    def concat_cond(self, others: list) -> "CONDCrossAttn":
        """Concatenate with other CONDCrossAttn instances along batch dim.

        Args:
            others: List of CONDCrossAttn instances.

        Returns:
            New CONDCrossAttn with concatenated tensors.
        """
        conds = [self.cond] + [o.cond for o in others]
        return CONDCrossAttn(torch.cat(conds, dim=0))

    def process_cond(self, batch_size: int, device, **kwargs) -> "CONDCrossAttn":
        """Process conditioning for a specific batch size.

        Args:
            batch_size: Target batch size.
            device: Target device.
            **kwargs: Additional processing options.

        Returns:
            Processed CONDCrossAttn.
        """
        cond = self.cond
        if cond.shape[0] < batch_size:
            repeats = [1] * len(cond.shape)
            repeats[0] = -(-batch_size // cond.shape[0])
            cond = cond.repeat(*repeats)[:batch_size]
        return self._copy_with(cond.to(device))


# ---------------------------------------------------------------------------
# CONDNoiseShape
# ---------------------------------------------------------------------------


class CONDNoiseShape:
    """Noise shape conditioning wrapper.

    Carries noise shape information used by some model architectures
    to determine the expected noise dimensions.

    Attributes:
        cond: The noise shape tensor or shape tuple.
    """

    def __init__(self, cond):
        """Initialize CONDNoiseShape.

        Args:
            cond: Noise shape as a tensor or tuple.
        """
        self.cond = cond

    def _copy_with(self, cond) -> "CONDNoiseShape":
        """Create a copy with different shape data.

        Args:
            cond: New noise shape data.

        Returns:
            New CONDNoiseShape.
        """
        return CONDNoiseShape(cond)

    def can_concat_cond(self, other: "CONDNoiseShape") -> bool:
        """Check if this can be concatenated with other.

        Args:
            other: Another CONDNoiseShape.

        Returns:
            True if compatible.
        """
        if not isinstance(other, CONDNoiseShape):
            return False
        if isinstance(self.cond, torch.Tensor) and isinstance(other.cond, torch.Tensor):
            return self.cond.shape[1:] == other.cond.shape[1:]
        return True

    def concat_cond(self, others: list) -> "CONDNoiseShape":
        """Concatenate noise shapes.

        Args:
            others: List of CONDNoiseShape instances.

        Returns:
            New CONDNoiseShape with combined data.
        """
        if isinstance(self.cond, torch.Tensor):
            conds = [self.cond] + [
                o.cond for o in others if isinstance(o.cond, torch.Tensor)
            ]
            return CONDNoiseShape(torch.cat(conds, dim=0))
        return self

    def process_cond(self, batch_size: int, device, **kwargs) -> "CONDNoiseShape":
        """Process noise shape for a specific batch size.

        Args:
            batch_size: Target batch size.
            device: Target device.
            **kwargs: Additional options.

        Returns:
            Processed CONDNoiseShape.
        """
        cond = self.cond
        if isinstance(cond, torch.Tensor):
            if cond.shape[0] < batch_size:
                repeats = [1] * len(cond.shape)
                repeats[0] = -(-batch_size // cond.shape[0])
                cond = cond.repeat(*repeats)[:batch_size]
            cond = cond.to(device)
        return self._copy_with(cond)
