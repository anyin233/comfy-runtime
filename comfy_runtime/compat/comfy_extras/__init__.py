"""Namespace stub for ``comfy_extras``.

ComfyUI ships a sibling package ``comfy_extras`` containing supplemental
built-in node modules (``nodes_mask``, ``nodes_post_processing``,
``nodes_custom_sampler``, …).  Many custom nodes import from it at
load time — e.g. ``from comfy_extras.nodes_mask import MaskComposite``.

Our MIT compat layer doesn't ship the full set of these modules yet.
Instead, we provide an empty namespace package so ``import
comfy_extras`` succeeds at import time, and rely on custom nodes that
need specific submodules to trigger clearer ``ImportError`` messages
(e.g. "cannot import name 'MaskComposite' from 'comfy_extras.nodes_mask'").

Phase 5 will port the commonly-used ``comfy_extras.nodes_*`` modules
as MIT compat stubs once the benchmark suite identifies which ones
workflows actually call into.
"""
