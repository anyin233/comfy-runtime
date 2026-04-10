"""Tests for the MIT CLIP.tokenize / encode_from_tokens implementation.

Phase 1 of the MIT rewrite plan: replaces the ``raise NotImplementedError``
stubs in ``compat/comfy/sd.py::CLIP`` with a real implementation backed
by transformers CLIPTokenizer / CLIPTextModel.
"""
import pytest
import torch

from tests.fixtures.tiny_sd15 import make_tiny_sd15
from comfy_runtime.compat.comfy.sd import CLIP


@pytest.fixture
def tiny_clip():
    comp = make_tiny_sd15()
    return CLIP(
        clip_model=comp["text_encoder"],
        tokenizer=comp["tokenizer"],
    )


def test_tokenize_returns_dict_with_l_slot(tiny_clip):
    """ComfyUI convention: tokens = {'l': [[(token_id, weight), ...]]}."""
    tokens = tiny_clip.tokenize("a cat")
    assert isinstance(tokens, dict)
    assert "l" in tokens
    assert isinstance(tokens["l"], list)
    assert len(tokens["l"]) >= 1  # at least one chunk


def test_tokenize_pads_chunk_to_77(tiny_clip):
    tokens = tiny_clip.tokenize("a cat")
    first_chunk = tokens["l"][0]
    assert len(first_chunk) == 77
    # Each entry is (id, weight) tuple
    assert isinstance(first_chunk[0], tuple)
    assert len(first_chunk[0]) == 2


def test_tokenize_empty_string(tiny_clip):
    """Empty prompt must still produce a valid 77-length padded chunk."""
    tokens = tiny_clip.tokenize("")
    assert "l" in tokens
    assert len(tokens["l"]) == 1
    assert len(tokens["l"][0]) == 77


def test_encode_from_tokens_shape(tiny_clip):
    tokens = tiny_clip.tokenize("a cat")
    cond = tiny_clip.encode_from_tokens(tokens)
    # Tiny text encoder has hidden_size=32
    assert cond.shape == (1, 77, 32)
    assert cond.dtype == torch.float32


def test_encode_from_tokens_with_pooled(tiny_clip):
    tokens = tiny_clip.tokenize("a cat")
    cond, pooled = tiny_clip.encode_from_tokens(tokens, return_pooled=True)
    assert cond.shape == (1, 77, 32)
    assert pooled.shape == (1, 32)


def test_encode_from_tokens_scheduled_returns_list_pair(tiny_clip):
    """ComfyUI convention: [[cond_tensor, {'pooled_output': pooled}]]."""
    tokens = tiny_clip.tokenize("a cat")
    result = tiny_clip.encode_from_tokens_scheduled(tokens)
    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0]) == 2
    cond, extra = result[0]
    assert cond.shape == (1, 77, 32)
    assert isinstance(extra, dict)
    assert "pooled_output" in extra


def test_encode_is_deterministic(tiny_clip):
    """Same tokens → same embedding."""
    tokens = tiny_clip.tokenize("a cat")
    out1 = tiny_clip.encode_from_tokens(tokens)
    out2 = tiny_clip.encode_from_tokens(tokens)
    assert torch.allclose(out1, out2)
