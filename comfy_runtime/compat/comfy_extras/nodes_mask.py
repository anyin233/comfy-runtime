"""MIT port of ``comfy_extras.nodes_mask``.

Provides the mask manipulation node classes plus the ``composite``
helper function used by AnimateDiff and several other custom nodes.

Mask convention: ``(B, H, W)`` float tensors in ``[0, 1]``.
Image convention: ``(B, H, W, 3)`` float tensors in ``[0, 1]``.
"""
import torch
import torch.nn.functional as F


def composite(destination: torch.Tensor, source: torch.Tensor, x: int, y: int,
              mask=None, multiplier: int = 8, resize_source: bool = False) -> torch.Tensor:
    """Composite ``source`` onto ``destination`` at ``(x, y)``.

    Faithful port of ComfyUI's mask composite helper that AnimateDiff
    references for motion overlap.  Supports both image-shaped
    ``(B, C, H, W)`` and mask-shaped ``(B, H, W)`` inputs.
    """
    destination = destination.clone()
    source = source.clone()

    if resize_source:
        # Resize source to match destination spatial dims
        if source.dim() == 4:
            source = F.interpolate(
                source, size=destination.shape[-2:], mode="bilinear"
            )
        else:
            source = F.interpolate(
                source.unsqueeze(0), size=destination.shape[-2:], mode="bilinear"
            ).squeeze(0)

    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))

    left, top = (x // multiplier), (y // multiplier)
    right = left + source.shape[-1]
    bottom = top + source.shape[-2]

    # Clip to destination bounds
    visible_left = max(0, left)
    visible_top = max(0, top)
    visible_right = min(destination.shape[-1], right)
    visible_bottom = min(destination.shape[-2], bottom)

    if visible_right <= visible_left or visible_bottom <= visible_top:
        return destination

    src_left = visible_left - left
    src_top = visible_top - top
    src_right = src_left + (visible_right - visible_left)
    src_bottom = src_top + (visible_bottom - visible_top)

    src_slice = source[..., src_top:src_bottom, src_left:src_right]
    dst_view = destination[..., visible_top:visible_bottom, visible_left:visible_right]

    if mask is None:
        # Pure overlay
        if dst_view.shape != src_slice.shape:
            return destination
        destination[..., visible_top:visible_bottom, visible_left:visible_right] = src_slice
    else:
        mask_slice = mask[..., src_top:src_bottom, src_left:src_right]
        if mask_slice.dim() < src_slice.dim():
            mask_slice = mask_slice.unsqueeze(-3)
        destination[..., visible_top:visible_bottom, visible_left:visible_right] = (
            dst_view * (1 - mask_slice) + src_slice * mask_slice
        )

    return destination


class SolidMask:
    """Generate a constant-value mask of the requested size."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                    "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "solid"
    CATEGORY = "mask"

    def solid(self, value, width, height):
        return (torch.full((1, height, width), float(value)),)


class InvertMask:
    """1.0 − mask."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "invert"
    CATEGORY = "mask"

    def invert(self, mask):
        return (1.0 - mask,)


class GrowMask:
    """Dilate (or erode) a mask by the given pixel count.

    Positive ``expand`` dilates outward, negative shrinks inward.
    The ``tapered_corners`` flag uses a cross-shaped structuring
    element to avoid square corners.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -16384, "max": 16384}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand"
    CATEGORY = "mask"

    def expand(self, mask, expand, tapered_corners):
        if expand == 0:
            return (mask,)

        # Use repeated 3x3 max-pool / min-pool for dilate/erode.
        m = mask.clone().float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        m = m.unsqueeze(1)  # (B, 1, H, W)

        if tapered_corners:
            kernel = torch.tensor(
                [[0.0, 1.0, 0.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 0.0]],
                dtype=m.dtype,
            ).view(1, 1, 3, 3)
        else:
            kernel = torch.ones(1, 1, 3, 3, dtype=m.dtype)

        if expand > 0:
            for _ in range(expand):
                m = F.conv2d(m, kernel, padding=1).clamp(max=1.0)
                m = (m > 0).to(m.dtype)
        else:
            # Erosion = dilate the inverted mask, but pad with 1 (outside
            # is "not mask") so the implicit image boundary eats inward.
            inverted = 1.0 - m
            for _ in range(-expand):
                padded = F.pad(inverted, (1, 1, 1, 1), mode="constant", value=1.0)
                inverted = F.conv2d(padded, kernel, padding=0).clamp(max=1.0)
                inverted = (inverted > 0).to(m.dtype)
            m = 1.0 - inverted

        return (m.squeeze(1),)


class FeatherMask:
    """Soften the edges of a mask by linearly fading inward."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "left": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "top": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "right": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 16384}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "feather"
    CATEGORY = "mask"

    def feather(self, mask, left, top, right, bottom):
        out = mask.clone().float()
        if out.dim() == 2:
            out = out.unsqueeze(0)
        b, h, w = out.shape

        # Fade in from each edge by linear ramp
        if left > 0:
            ramp = torch.linspace(0, 1, left + 1)[1:].view(1, 1, -1)
            out[:, :, :left] = out[:, :, :left] * ramp
        if right > 0:
            ramp = torch.linspace(1, 0, right + 1)[:-1].view(1, 1, -1)
            out[:, :, w - right:] = out[:, :, w - right:] * ramp
        if top > 0:
            ramp = torch.linspace(0, 1, top + 1)[1:].view(1, -1, 1)
            out[:, :top, :] = out[:, :top, :] * ramp
        if bottom > 0:
            ramp = torch.linspace(1, 0, bottom + 1)[:-1].view(1, -1, 1)
            out[:, h - bottom:, :] = out[:, h - bottom:, :] * ramp

        return (out,)


class MaskComposite:
    """Combine two masks with a per-pixel operation at offset ``(x, y)``."""

    OPERATIONS = ["add", "subtract", "multiply", "and", "or", "xor"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("MASK",),
                "source": ("MASK",),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "operation": (s.OPERATIONS,),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine"
    CATEGORY = "mask"

    def combine(self, destination, source, x, y, operation):
        out = destination.clone().float()
        if out.dim() == 2:
            out = out.unsqueeze(0)
        src = source.float()
        if src.dim() == 2:
            src = src.unsqueeze(0)

        b, dh, dw = out.shape
        sh, sw = src.shape[-2], src.shape[-1]

        x = max(0, min(x, dw))
        y = max(0, min(y, dh))
        right = min(x + sw, dw)
        bottom = min(y + sh, dh)

        if right <= x or bottom <= y:
            return (out,)

        src_w = right - x
        src_h = bottom - y
        sub = out[:, y:bottom, x:right]
        src_sub = src[:, :src_h, :src_w]

        if operation == "add":
            sub = (sub + src_sub).clamp(0.0, 1.0)
        elif operation == "subtract":
            sub = (sub - src_sub).clamp(0.0, 1.0)
        elif operation == "multiply":
            sub = sub * src_sub
        elif operation == "and":
            sub = ((sub > 0.5) & (src_sub > 0.5)).float()
        elif operation == "or":
            sub = ((sub > 0.5) | (src_sub > 0.5)).float()
        elif operation == "xor":
            sub = ((sub > 0.5) ^ (src_sub > 0.5)).float()

        out[:, y:bottom, x:right] = sub
        return (out,)


class ImageToMask:
    """Extract a single channel from an image as a mask."""

    CHANNELS = ["red", "green", "blue", "alpha"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (s.CHANNELS,),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "image_to_mask"
    CATEGORY = "mask"

    def image_to_mask(self, image, channel):
        idx = self.CHANNELS.index(channel)
        if idx >= image.shape[-1]:
            # Alpha requested but image is RGB → return all-ones
            return (torch.ones(image.shape[:-1]),)
        return (image[..., idx],)


class MaskToImage:
    """Broadcast a single-channel mask into a 3-channel image."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"
    CATEGORY = "mask"

    def mask_to_image(self, mask):
        return (mask.unsqueeze(-1).expand(*mask.shape, 3).clone(),)


class CropMask:
    """Crop a mask to the given rectangle."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "crop"
    CATEGORY = "mask"

    def crop(self, mask, x, y, width, height):
        return (mask[..., y:y + height, x:x + width].clone(),)
