"""Stub package for comfy_api.torch_helpers."""


def set_torch_compile_wrapper(wrapper):
    """Register a torch.compile wrapper (no-op in comfy_runtime)."""
    pass


def get_torch_compile_wrapper():
    """Return the current torch.compile wrapper (None in comfy_runtime)."""
    return None
