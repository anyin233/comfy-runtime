"""Shared fixtures for benchmark scripts.

Kept empty-by-default; individual bench files add helpers here when
they're reused across multiple benches.
"""

import os
import tempfile


def make_temp_dirs():
    """Create models/output/input temp dirs for configure() benches."""
    base = tempfile.mkdtemp(prefix="comfy_bench_")
    models = os.path.join(base, "models")
    output = os.path.join(base, "output")
    inp = os.path.join(base, "input")
    for p in (models, output, inp):
        os.makedirs(p, exist_ok=True)
    return base, models, output, inp
