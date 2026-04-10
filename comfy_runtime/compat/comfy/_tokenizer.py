"""Tokenizer wrapper producing ComfyUI-style token dicts.

ComfyUI convention::

    tokens = {
        "l": [[(token_id, weight_float), ...77...], ...chunks...],
    }

The outer key ("l", "g", "t5xxl", ...) identifies the encoder slot:

* ``"l"``     — CLIP-L (SD1, SDXL slot 1, Flux slot 1, SD3 slot 1)
* ``"g"``     — OpenCLIP-G (SDXL slot 2, SD3 slot 2)
* ``"t5xxl"`` — T5-XXL (Flux slot 2, SD3 slot 3)

Each chunk is a list of (token_id, weight) tuples of exactly the fixed
chunk length (77 for CLIP, 256 for T5).  Short prompts are padded to the
chunk length using the tokenizer's pad token at weight 1.0; long prompts
are split across multiple chunks.  The per-token float weight field
supports attention-weighted prompts like ``(masterpiece:1.2)`` — Phase 1
always emits weight 1.0; attention-weighted parsing is Phase 2 work.
"""
from typing import Dict, List, Sequence, Tuple


def tokenize_to_comfy_format(
    tokenizer,
    text: str,
    max_length: int = 77,
    slot: str = "l",
) -> Dict[str, List[List[Tuple[int, float]]]]:
    """Tokenize text and wrap it in ComfyUI's chunked weighted format.

    Args:
        tokenizer: A HuggingFace tokenizer (CLIPTokenizer, T5Tokenizer, ...).
        text:       Input prompt.
        max_length: Chunk size (77 for CLIP, 256 for T5).
        slot:       Encoder slot name.

    Returns:
        ``{slot: [chunk1, chunk2, ...]}`` where each chunk is a list of
        exactly ``max_length`` ``(token_id, weight)`` tuples.
    """
    # Use the tokenizer directly to get plain ID lists.  We handle padding
    # and chunking ourselves so the output always matches the ComfyUI layout.
    ids = tokenizer.encode(
        text if text else "",
        add_special_tokens=True,
        truncation=False,
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    chunks: List[List[Tuple[int, float]]] = []
    # Always emit at least one chunk, even for empty strings.
    start = 0
    while True:
        chunk_ids = ids[start : start + max_length]
        if not chunk_ids and chunks:
            break
        # Right-pad with pad_id at weight 1.0
        while len(chunk_ids) < max_length:
            chunk_ids.append(pad_id)
        chunks.append([(int(tid), 1.0) for tid in chunk_ids])
        start += max_length
        if start >= len(ids):
            break

    return {slot: chunks}


def tokens_to_input_ids(tokens_chunk: Sequence[Tuple[int, float]]) -> List[int]:
    """Extract only the integer ID list from a weighted chunk.

    Args:
        tokens_chunk: A list of ``(token_id, weight)`` pairs.

    Returns:
        List of ``int`` token IDs.
    """
    return [int(pair[0]) for pair in tokens_chunk]
