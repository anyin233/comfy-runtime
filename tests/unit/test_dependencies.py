"""Verify MIT-licensed inference dependencies are importable.

These packages replace the functionality currently provided by the
GPL-licensed vendored ComfyUI code under comfy_runtime/_vendor/. See
docs/superpowers/plans/2026-04-10-mit-rewrite-vendor.md Task 0.1.
"""


def test_diffusers_importable():
    import diffusers

    parts = tuple(int(x) for x in diffusers.__version__.split(".")[:2])
    assert parts >= (0, 32), f"diffusers too old: {diffusers.__version__}"


def test_peft_importable():
    import peft

    parts = tuple(int(x) for x in peft.__version__.split(".")[:2])
    assert parts >= (0, 14), f"peft too old: {peft.__version__}"


def test_accelerate_importable():
    import accelerate

    parts = tuple(int(x) for x in accelerate.__version__.split(".")[:2])
    assert parts >= (1, 3), f"accelerate too old: {accelerate.__version__}"


def test_huggingface_hub_importable():
    import huggingface_hub

    parts = tuple(int(x) for x in huggingface_hub.__version__.split(".")[:2])
    assert parts >= (0, 27), f"huggingface_hub too old: {huggingface_hub.__version__}"
