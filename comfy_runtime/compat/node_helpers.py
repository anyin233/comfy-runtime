"""Small utility functions used by built-in nodes.

MIT reimplementation — contains only generic helper utilities.
"""

import torch
from PIL import Image


def conditioning_set_values(conditioning, values: dict, append=False):
    """Return a copy of *conditioning* with extra *values* merged into each item.

    Args:
        conditioning: List of ``(tensor, dict)`` tuples.
        values: Key/value pairs to merge into each dict.
        append: If True, append list values instead of replacing.

    Returns:
        New conditioning list.
    """
    output = []
    for cond_tensor, cond_dict in conditioning:
        new_dict = dict(cond_dict)
        if append:
            for k, v in values.items():
                if k in new_dict and isinstance(new_dict[k], list) and isinstance(v, list):
                    new_dict[k] = new_dict[k] + v
                elif k in new_dict and isinstance(new_dict[k], list):
                    new_dict[k] = new_dict[k] + [v]
                else:
                    new_dict[k] = v
        else:
            new_dict.update(values)
        output.append((cond_tensor, new_dict))
    return output


def pillow(fn, arg):
    """Apply a PIL operation, converting to RGB first if needed."""
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, ValueError) as e:
        prev_value = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        x = fn(arg)
    finally:
        if prev_value is not None:
            Image.MAX_IMAGE_PIXELS = prev_value
    return x
