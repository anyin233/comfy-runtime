"""LoRA state-dict → ModelPatcher delta conversion.

ComfyUI LoRA state dicts use a simple naming convention based on the
underlying kohya-ss / sd-scripts format that the community standardized
on::

    <target>.lora_up.weight     — shape (out_features, rank)
    <target>.lora_down.weight   — shape (rank, in_features)
    <target>.alpha              — scalar (defaults to ``rank``)

The effective delta applied to the target weight is::

    delta = (lora_up @ lora_down) * (alpha / rank) * strength

where ``strength`` is the slider value the user passes on the
``LoraLoader`` node.

Phase 1's :class:`ModelPatcher` understands **raw delta** patches, so
this helper's job is to:

1. Group ``.lora_up / .lora_down / .alpha`` triples back into single
   entries keyed by the target parameter name.
2. Compute the fused delta tensor per triple.
3. Register the delta on the patcher via :meth:`ModelPatcher.add_patches`.

This file is named ``_lora_peft`` to match the plan's filename, even
though the current Phase-1 implementation does not depend on the
``peft`` library (we just do the matmul directly).  Phase 2 will add a
``peft``-based path for HF-style LoRA dicts that use ``lora_A`` /
``lora_B`` keys and for composite LoCon/LoKr/LoHa families — those
really do benefit from ``peft``'s tuner machinery.
"""
from typing import Dict

import torch


def _target_key_from_lora_key(lora_key: str) -> str:
    """Return the model's state-dict key for a given LoRA key.

    ``diffusion_model.proj.lora_up.weight`` → ``diffusion_model.proj.weight``
    """
    if lora_key.endswith(".lora_up.weight"):
        return lora_key[: -len(".lora_up.weight")] + ".weight"
    if lora_key.endswith(".lora_down.weight"):
        return lora_key[: -len(".lora_down.weight")] + ".weight"
    if lora_key.endswith(".alpha"):
        return lora_key[: -len(".alpha")] + ".weight"
    return lora_key


def extract_lora_deltas(
    lora_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Parse a ComfyUI LoRA state dict into raw delta tensors.

    Args:
        lora_sd: ComfyUI/kohya LoRA state dict with ``.lora_up.weight``,
            ``.lora_down.weight``, and optionally ``.alpha`` entries.

    Returns:
        Dict mapping target weight keys to fused delta tensors of the
        same shape as the target weight.  Keys without matching up/down
        pairs are silently dropped.
    """
    # First pass: group by target key
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in lora_sd.items():
        if (
            key.endswith(".lora_up.weight")
            or key.endswith(".lora_down.weight")
            or key.endswith(".alpha")
        ):
            target = _target_key_from_lora_key(key)
            slot = grouped.setdefault(target, {})
            if key.endswith(".lora_up.weight"):
                slot["up"] = value
            elif key.endswith(".lora_down.weight"):
                slot["down"] = value
            elif key.endswith(".alpha"):
                slot["alpha"] = value

    # Second pass: compute fused delta per target
    deltas: Dict[str, torch.Tensor] = {}
    for target, parts in grouped.items():
        up = parts.get("up")
        down = parts.get("down")
        if up is None or down is None:
            continue

        # Rank is the shared inner dim.
        rank = up.shape[1]

        # Alpha defaults to ``rank`` when absent, meaning the scale is 1.0.
        # The kohya convention: scale = alpha / rank.
        alpha_tensor = parts.get("alpha")
        if alpha_tensor is None:
            alpha = float(rank)
        elif isinstance(alpha_tensor, torch.Tensor):
            alpha = float(alpha_tensor.item())
        else:
            alpha = float(alpha_tensor)
        scale = alpha / float(rank)

        # Cast to fp32 for the matmul to avoid precision loss in fp16 LoRAs,
        # then the patcher will cast back to the target's dtype at apply time.
        delta = (up.float() @ down.float()) * scale
        deltas[target] = delta

    return deltas


def apply_lora_to_patcher(
    patcher,
    lora_sd: Dict[str, torch.Tensor],
    strength: float = 1.0,
) -> int:
    """Register all LoRA deltas onto a :class:`ModelPatcher`.

    Args:
        patcher: The target :class:`ModelPatcher`.
        lora_sd: ComfyUI-format LoRA state dict.
        strength: Slider value scaling every delta.

    Returns:
        Number of deltas that matched a weight key in the model.  Keys
        not present in ``patcher.model_keys`` are silently skipped.
    """
    deltas = extract_lora_deltas(lora_sd)

    # Wrap each delta in a 1-tuple so ModelPatcher._extract_delta sees
    # a "raw delta" rather than the future LoRA-factored form.
    patches = {key: (delta,) for key, delta in deltas.items()}

    added = patcher.add_patches(
        patches,
        strength_patch=float(strength),
        strength_model=1.0,
    )
    return len(added)
