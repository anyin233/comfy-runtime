"""Tests for the ported comfy_extras.nodes_mask."""
import torch

from comfy_runtime.compat.comfy_extras import nodes_mask as nm


def test_solid_mask_returns_filled_tensor():
    (mask,) = nm.SolidMask().solid(value=0.5, width=64, height=32)
    assert mask.shape == (1, 32, 64)
    assert torch.allclose(mask, torch.full_like(mask, 0.5))


def test_invert_mask_negates_values():
    src = torch.tensor([[[0.0, 0.25, 0.75, 1.0]]])
    (out,) = nm.InvertMask().invert(mask=src)
    expected = torch.tensor([[[1.0, 0.75, 0.25, 0.0]]])
    assert torch.allclose(out, expected)


def test_grow_mask_expands_unit_pixels():
    """A 1-pixel hot mask grown by 1 should produce a 3x3 block."""
    src = torch.zeros(1, 5, 5)
    src[0, 2, 2] = 1.0
    (out,) = nm.GrowMask().expand(mask=src, expand=1, tapered_corners=False)
    # Center 3x3 should be 1
    block = out[0, 1:4, 1:4]
    assert (block > 0.5).all()


def test_grow_mask_negative_shrinks():
    src = torch.ones(1, 5, 5)
    (out,) = nm.GrowMask().expand(mask=src, expand=-1, tapered_corners=False)
    # Border ring should be 0; center 3x3 should be 1
    assert out[0, 0, 0] == 0
    assert out[0, 2, 2] == 1


def test_feather_mask_softens_edges():
    """Feather a full-image mask: image-edge pixels should fade to 0,
    interior pixels stay at 1.

    ComfyUI's FeatherMask multiplies the outermost L/T/R/B rows of the
    mask by a linear ramp from 0 to 1.  When the source mask is all
    ones, the post-feather output has a soft border around the image.
    """
    src = torch.ones(1, 16, 16)
    (out,) = nm.FeatherMask().feather(
        mask=src, left=2, top=2, right=2, bottom=2
    )
    # The outermost top-left pixel should be 0 (or very close)
    assert out[0, 0, 0] < 0.5
    # The center should still be 1
    assert out[0, 8, 8] == 1.0
    # At least one fully-faded pixel and at least one interior pixel
    assert (out[0] < 0.5).any()
    assert (out[0] >= 0.999).any()


def test_mask_composite_blends_with_destination():
    dest = torch.zeros(1, 8, 8)
    src = torch.ones(1, 4, 4)
    (out,) = nm.MaskComposite().combine(
        destination=dest, source=src, x=2, y=2, operation="add"
    )
    assert out[0, 4, 4] == 1.0
    assert out[0, 0, 0] == 0.0


def test_mask_composite_subtract_operation():
    dest = torch.ones(1, 8, 8)
    src = torch.ones(1, 4, 4)
    (out,) = nm.MaskComposite().combine(
        destination=dest, source=src, x=2, y=2, operation="subtract"
    )
    # Subtracted area should be 0
    assert out[0, 4, 4] == 0.0
    # Outside the subtraction area should still be 1
    assert out[0, 0, 0] == 1.0


def test_image_to_mask_extracts_channel():
    image = torch.zeros(1, 8, 8, 3)
    image[..., 0] = 0.7  # red channel
    (mask,) = nm.ImageToMask().image_to_mask(image=image, channel="red")
    assert mask.shape == (1, 8, 8)
    assert torch.allclose(mask, torch.full_like(mask, 0.7))


def test_mask_to_image_broadcasts_channel():
    mask = torch.full((1, 8, 8), 0.4)
    (image,) = nm.MaskToImage().mask_to_image(mask=mask)
    assert image.shape == (1, 8, 8, 3)
    assert torch.allclose(image, torch.full_like(image, 0.4))


def test_composite_helper_pure_function():
    """The composite() helper used by AnimateDiff."""
    dest = torch.zeros(1, 3, 8, 8)
    src = torch.ones(1, 3, 4, 4)
    out = nm.composite(dest, src, x=2, y=2)
    assert out[0, :, 2:6, 2:6].sum() > 0
