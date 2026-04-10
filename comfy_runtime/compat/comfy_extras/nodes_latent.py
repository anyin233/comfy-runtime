"""Import-compat stub for ``comfy_extras.nodes_latent``."""
import torch


def reshape_latent_to(target_shape, latent, repeat_batch=True):
    """Reshape ``latent`` to ``target_shape``, optionally repeating the batch.

    Faithful port of ComfyUI's helper used by latent post-processing.
    Interpolates spatial dims and repeats/truncates the batch.
    """
    if latent is None:
        return None
    target_h, target_w = target_shape[-2], target_shape[-1]
    if latent.shape[-2] != target_h or latent.shape[-1] != target_w:
        latent = torch.nn.functional.interpolate(
            latent, size=(target_h, target_w), mode="bilinear"
        )
    if repeat_batch and latent.shape[0] != target_shape[0]:
        if latent.shape[0] < target_shape[0]:
            repeats = (target_shape[0] + latent.shape[0] - 1) // latent.shape[0]
            latent = latent.repeat(repeats, *([1] * (latent.dim() - 1)))
        latent = latent[: target_shape[0]]
    return latent


class LatentAdd:
    pass


class LatentSubtract:
    pass


class LatentMultiply:
    pass


class LatentInterpolate:
    pass


class LatentBatch:
    pass


class LatentBatchSeedBehavior:
    pass


class LatentApplyOperation:
    pass
